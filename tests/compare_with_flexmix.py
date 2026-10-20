"""Compare the MoP Gaussian-variant fit against flexmix on the same data.

Usage:
    python tests/compare_with_flexmix.py --write-data      # emit the data and the R script
    Rscript tests/compare_with_flexmix.R                   # fit with flexmix
    python tests/compare_with_flexmix.py --compare         # fit with MoP and compare
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_CSV = os.path.join(HERE, "_flexmix_data.csv")
MOP_JSON = os.path.join(HERE, "_flexmix_mop.json")
FMX_JSON = os.path.join(HERE, "_flexmix_flexmix.json")
R_SCRIPT = os.path.join(HERE, "compare_with_flexmix.R")

N_ALT = 4
N_COMP = 2
PI_TRUE = np.array([0.6, 0.4])
U_TRUE = np.array([
    [0.9, -0.4, 0.2],
    [-0.5, 0.7, -0.6],
])
PHI_TRUE = np.array([0.25, 0.16])
N_OBS = 3000
SEED = 20261005

PAIRS = [(i, j) for i in range(1, N_ALT + 1) for j in range(i + 1, N_ALT + 1)]


def design_matrix():
    """The six transfer vectors, taken from the library rather than re-derived."""
    from mop import inference, pairwise_comparison

    pcs = [pairwise_comparison.PairwiseComparison(
        former_id=i, latter_id=j, agent_id=1, judgment=0.0)
        for i, j in PAIRS]
    X, _, _ = inference.comparisons2regression(
        pcs, [lambda s: np.asarray(s, dtype=float)])
    return X


def generate():
    rng = np.random.default_rng(SEED)
    X_pairs = design_matrix()
    rows = []
    for k in range(N_OBS):
        pair = k % len(X_pairs)
        c = rng.choice(N_COMP, p=PI_TRUE)
        x = X_pairs[pair]
        y = float(x @ U_TRUE[c] + rng.normal(0.0, np.sqrt(PHI_TRUE[c])))
        rows.append((pair + 1, c + 1, *x, y))
    return X_pairs, np.array(rows)


def write_data():
    X_pairs, rows = generate()
    header = (["pair", "state"] + [f"x{i + 1}" for i in range(N_ALT - 1)] + ["y"])
    with open(DATA_CSV, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(
                [str(int(r[0])), str(int(r[1]))] + [f"{v:.12g}" for v in r[2:]]) + "\n")
    print(f"  wrote {os.path.relpath(DATA_CSV, ROOT)}  ({len(rows)} rows)")


def fit_mop():
    from mop import inference, pairwise_comparison

    records = {}
    with open(DATA_CSV, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            pair, _state, *rest = line.strip().split(",")
            records.setdefault(int(pair), []).append(float(rest[-1]))

    pcs = []
    for pair, ys in sorted(records.items()):
        i, j = PAIRS[pair - 1]
        for y in ys:
            pcs.append(pairwise_comparison.PairwiseComparison(
                former_id=i, latter_id=j, agent_id=1, judgment=float(y)))

    result = inference.mop(pcs, [lambda s: np.asarray(s, dtype=float)], N_COMP,
                           lambda_u=1e-10, lambda_pi=1.0,
                           lambda_sigma=2.0 + 1e-6, s2=0.2)
    w = np.array([p.priority_vector for p in result.priorities])
    u = np.log(w[:, :-1] / w[:, [-1]])
    phi = np.array([p.inconsistency_level for p in result.priorities])
    pi = np.array(result.mixture_probabilities)
    out = {"prior": pi.tolist(), "u": u.tolist(), "sigma": phi.tolist()}
    with open(MOP_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-data", action="store_true")
    ap.add_argument("--run-flexmix", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if not (args.write_data or args.run_flexmix or args.compare):
        args.write_data = True
        args.run_flexmix = True
        args.compare = True

    if args.write_data:
        write_data()

    if args.run_flexmix:
        print("  running flexmix ...")
        proc = subprocess.run(
            ["Rscript", R_SCRIPT, DATA_CSV, FMX_JSON, str(N_COMP), str(SEED)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            raise SystemExit("flexmix failed")
        print("  flexmix done")

    if args.compare:
        mop_out = fit_mop()
        with open(FMX_JSON, encoding="utf-8") as fh:
            fmx = json.load(fh)
        report(mop_out, fmx)


def report(mop_out, fmx):
    u_m = np.array(mop_out["u"])
    u_f = np.array(fmx["u"])
    phi_m = np.array(mop_out["sigma"])
    phi_f = np.array(fmx["sigma"]) ** 2
    pi_m = np.array(mop_out["prior"])
    pi_f = np.array(fmx["prior"])

    print()
    print("  === label alignment ===")
    best, best_perm = None, None
    for perm in itertools.permutations(range(N_COMP)):
        cost = np.abs(u_m[list(perm)] - u_f).max()
        if best is None or cost < best:
            best, best_perm = cost, perm
    print(f"    permutation {best_perm} minimises the utility gap ({best:.3e})")
    p = list(best_perm)
    true_of = [int(np.argmin([np.abs(u_f[c] - t).max() for t in U_TRUE]))
               for c in range(N_COMP)]

    print()
    print("  === parameters, MoP against flexmix ===")
    print(f"    {'comp':>5} {'u (MoP)':>32} {'u (flexmix)':>32} {'u (true)':>30}")
    for c in range(N_COMP):
        print(f"    {c:>5} {np.array2string(u_m[p[c]], precision=6):>32} "
              f"{np.array2string(u_f[c], precision=6):>32} "
              f"{np.array2string(U_TRUE[true_of[c]], precision=6):>30}")
    print()
    print(f"    {'comp':>5} {'phi (MoP)':>12} {'phi (flexmix)':>14} {'phi (true)':>12} "
          f"{'pi (MoP)':>10} {'pi (flexmix)':>13}")
    for c in range(N_COMP):
        print(f"    {c:>5} {phi_m[p[c]]:>12.6f} {phi_f[c]:>14.6f} "
              f"{PHI_TRUE[true_of[c]]:>12.4f} {pi_m[p[c]]:>10.6f} {pi_f[c]:>13.6f}")

    du = np.abs(u_m[p] - u_f).max()
    dphi = np.abs(phi_m[p] - phi_f).max()
    dpi = np.abs(pi_m[p] - pi_f).max()
    print()
    print("  === agreement ===")
    print(f"    max |du|   = {du:.3e}")
    print(f"    max |dphi| = {dphi:.3e}")
    print(f"    max |dpi|  = {dpi:.3e}")
    print(f"    flexmix converged = {fmx.get('converged')}, "
          f"logLik = {fmx.get('loglik'):.6f}, iterations = {fmx.get('niter')}")
    ok = du < 5e-3 and dphi < 5e-3 and dpi < 5e-3
    print()
    print("  " + ("AGREES with flexmix" if ok else "DOES NOT agree with flexmix"))
    return ok


if __name__ == "__main__":
    main()
