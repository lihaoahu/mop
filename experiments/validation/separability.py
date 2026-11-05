"""Figure 1: what holding the dispersion fixed costs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.em import predictive_ll, select                            # noqa: E402
from common.generate import priorities, separated_pair, simulate_gaussian  # noqa: E402
from common.metrics import cosine_similarity                          # noqa: E402
from common.parallel import run, seeds                                # noqa: E402
from common.paths import result                                       # noqa: E402

RESULTS = result("separability.json")
A = 4
MODES = ("free", "one")


def _kw(X, y):
    """The paper's prior rule.

    ``phi_0`` is the pooled-residual mode, ``prior_mode_phi0(X, y)``, not the
    dispersion used to generate the data.  Setting it to the truth would hand the
    free-dispersion estimator the answer and turn the comparison into a test of a
    prior it was given.
    """
    from mop.inference import prior_mode_phi0

    return {"lambda_u": 1e-6, "lambda_pi": 1.0,
            "lambda_sigma": 0.02 * float(len(y)),
            "s2": float(prior_mode_phi0(X, y))}


def _gap(curve):
    v = sorted(curve.values())
    return float(v[1] - v[0]) if len(v) > 1 else float("nan")


def _one_rep(job):
    """A single replication; its seed fully determines its output."""
    kind, n_obs, phi, delta, seed = job
    rng = np.random.default_rng(seed)
    if kind == "single":
        u_true = np.array([[0.9, 0.4, -0.5]])
        pi_true = np.array([1.0])
        phi_true = np.array([phi])
    else:
        u_true = separated_pair(A, delta, rng)
        pi_true = np.array([0.5, 0.5])
        phi_true = np.array([phi, phi])
    k_true = len(pi_true)

    d = simulate_gaussian(u_true, pi_true, phi_true, n_obs, rng)
    idx = rng.permutation(n_obs)
    n_te = int(round(0.3 * n_obs))
    tr, te = idx[n_te:], idx[:n_te]
    Xtr, ytr, Xte, yte = d["X"][tr], d["y"][tr], d["X"][te], d["y"][te]

    kw = _kw(Xtr, ytr)
    row = {"k_true": k_true}
    for mode in MODES:
        t0 = time.perf_counter()
        best, _, curve = select(Xtr, ytr, (1, 2, 3, 4), phi_mode=mode, **kw)
        row[f"k_{mode}"] = best.n_components
        row[f"gap_{mode}"] = _gap(curve)
        row[f"ll_{mode}"] = predictive_ll(best, Xte, yte) / len(te)
        row[f"phi_{mode}"] = best.phi.tolist()
        row[f"cos_{mode}"] = (cosine_similarity(priorities(u_true), best.priorities())
                              if k_true > 1 else float("nan"))
        row[f"t_{mode}"] = time.perf_counter() - t0
    return row


def one_cell(kind, reps, n_obs, phi, delta, seed, jobs=None):
    rows = run(_one_rep,
               [(kind, n_obs, phi, delta, s) for s in seeds(seed, reps)],
               jobs=jobs)

    cell = {"truth": kind, "phi": phi, "delta": delta if kind == "pair" else None,
            "reps": reps, "n": n_obs, "k_true": rows[0]["k_true"]}
    for mode in MODES:
        k = np.array([x[f"k_{mode}"] for x in rows])
        cell[f"hit_{mode}"] = float(np.mean(k == rows[0]["k_true"]))
        cell[f"over_{mode}"] = float(np.mean(k > rows[0]["k_true"]))
        cell[f"under_{mode}"] = float(np.mean(k < rows[0]["k_true"]))
        cell[f"k_counts_{mode}"] = {str(v): int((k == v).sum()) for v in sorted(set(k))}
        cell[f"ll_{mode}"] = float(np.mean([x[f"ll_{mode}"] for x in rows]))
        cell[f"gap_{mode}"] = float(np.nanmean([x[f"gap_{mode}"] for x in rows]))
        cell[f"phi_{mode}"] = float(np.mean([np.mean(x[f"phi_{mode}"]) for x in rows]))
        cell[f"cos_{mode}"] = (float(np.nanmean([x[f"cos_{mode}"] for x in rows]))
                               if kind == "pair" else None)
        cell[f"t_{mode}"] = float(np.mean([x[f"t_{mode}"] for x in rows]))
    return cell, rows


def _row(cell):
    """One line per truth, one column pair per restriction in ``MODES``."""
    dl = "-" if cell["delta"] is None else f"{cell['delta']:.1f}"
    head = f"  {cell['truth']:<8} {cell['phi']:>5.2f} {dl:>6} {cell['k_true']:>3} "
    hits = "".join(f"{cell[f'hit_{m}']:>9.2f}" for m in MODES)
    over = "".join(f"{cell[f'over_{m}']:>9.2f}" for m in MODES)
    return head + hits + over


def _untimed(out):
    """The artifact carries no wall-clock reading: fits are timed and printed,
    never stored, which is what lets two runs of the same seeds be compared
    byte for byte."""
    drop = {f"t_{m}" for m in MODES}

    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items() if k not in drop}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    return walk(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--seed", type=int, default=20260301)
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes; 1 forces the serial path")
    args = ap.parse_args()

    singles = [0.5, 1.0, 2.0, 4.0]
    pair_cells = [(0.1, 1.0), (0.1, 2.0), (0.1, 3.0), (0.25, 2.0)]

    out = {"design": {"A": A, "n": args.n, "reps": args.reps,
                      "grid": [1, 2, 3, 4], "test_fraction": 0.3,
                      "modes": list(MODES), "seed": args.seed,
                      "seed_stride": 1000003},
           "cells": [], "per_replicate": []}

    print(f"Separability -- {args.reps} replicates per cell, n={args.n}\n")
    cols = "".join(f"{'hit ' + m:>9}" for m in MODES)
    more = "".join(f"{'over ' + m:>9}" for m in MODES)
    print(f"  {'truth':<8} {'phi':>5} {'delta':>6} {'K*':>3} {cols} {more}")
    print("  " + "-" * (34 + 9 * len(MODES) * 2))

    for i, phi in enumerate(singles):
        cell, rows = one_cell("single", args.reps, args.n, phi, 0.0,
                              args.seed + i, jobs=args.jobs)
        out["cells"].append(cell)
        out["per_replicate"].append(rows)
        print(_row(cell))

    for j, (phi, delta) in enumerate(pair_cells):
        cell, rows = one_cell("pair", args.reps, args.n, phi, delta,
                              args.seed + 100 + j, jobs=args.jobs)
        out["cells"].append(cell)
        out["per_replicate"].append(rows)
        print(_row(cell))

    print("\n  held-out log-likelihood per judgment (higher is better)")
    print(f"  {'truth':<8} {'phi':>5} {'delta':>6} "
          + "".join(f"{m:>10}" for m in MODES))
    for c in out["cells"]:
        dl = "-" if c["delta"] is None else f"{c['delta']:.1f}"
        print(f"  {c['truth']:<8} {c['phi']:>5.2f} {dl:>6} "
              + "".join(f"{c[f'll_{m}']:>10.4f}" for m in MODES))

    Path(args.out).write_text(json.dumps(_untimed(out), indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
