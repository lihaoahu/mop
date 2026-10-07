"""Separability of intrinsic variability and structural inconsistency.

The framework claims that, with a locked variance function, dispersion beyond
n * mu * (1 - mu) has nowhere to go but into extra mixture components, so the
selected number of components inflates. Freeing the overdispersion rho_p should
let each mechanism be recovered on its own.

This script generates data under both regimes and fits both models, so the
claim is tested in both directions:

  regime "binomial"     m_k | c_k=p ~ Binomial(n_k, mu_kp)
                        every extra bit of dispersion is genuine
                        state switching -> the mixture should recover P_true
                        and rho should be estimated at zero.

  regime "beta-binomial" m_k | c_k=p ~ BetaBinomial(n_k, mu_kp, rho_gen)
                        part of the dispersion is within-state noise -> a
                        locked-rho fit should over-select components, while a
                        free-rho fit should recover both P_true and rho_gen.

Grid: P_true in {1,2,3} x rho_gen in {0, 0.10, 0.25}, N_REP replicates each.
Each replicate uses A = 5 alternatives (10 pairs), 45 repetitions per pair.

Usage:
    uv run python experiments/v3/step2_separability.py
    uv run python experiments/v3/step2_separability.py --n-rep 5 --p-max 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import betabinom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mop import inference_binomial as ib  # noqa: E402

HERE = Path(__file__).resolve().parent
N_ALT = 6          # 15 pairs; 5 alternatives (10 pairs) is too few to identify
                   # three components and a dispersion parameter at once
PAIRS = [(i, j) for i in range(N_ALT) for j in range(i + 1, N_ALT)]
N_TRIALS = 45
MODES = ("binomial", "beta-binomial")


def simulate(u_true, pi_true, rho_gen, rng):
    """Draw counts for every pair: pick one component per pair, then judge.

    rho_gen = 0 draws from a Binomial, rho_gen > 0 from a BetaBinomial with
    that overdispersion, i.e. the same state but noisier execution.
    """
    counts = []
    n_comp = u_true.shape[0]
    u_full = np.hstack([u_true, np.zeros((n_comp, 1))])   # reference alternative at 0
    for i, j in PAIRS:
        c = int(rng.choice(n_comp, p=pi_true))
        mu = float(expit(u_full[c, i] - u_full[c, j]))
        if rho_gen <= 0.0:
            m = int(rng.binomial(N_TRIALS, mu))
        else:
            conc = (1.0 - rho_gen) / rho_gen
            m = int(betabinom.rvs(N_TRIALS, mu * conc, (1.0 - mu) * conc, random_state=rng))
        counts.append({"former": i, "latter": j,
                       "successes": float(m), "trials": N_TRIALS})
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=12)
    ap.add_argument("--p-max", type=int, default=4)
    ap.add_argument("--n-starts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--out", type=Path, default=HERE / "separability_results.json")
    args = ap.parse_args()

    p_true_vals = [1, 2, 3]
    rho_vals = [0.0, 0.10, 0.25]
    cells = []

    for p_true in p_true_vals:
        for rho_gen in rho_vals:
            regime = "binomial" if rho_gen == 0.0 else "beta-binomial"
            for rep in range(args.n_rep):
                rng = np.random.default_rng(args.seed + 10007 * rep + 101 * p_true + int(rho_gen * 1000))
                u_true = rng.normal(0.0, 1.0, (p_true, N_ALT - 1))
                pi_true = (np.full(p_true, 1.0 / p_true) if p_true == 1
                           else rng.dirichlet(np.full(p_true, 2.0)))
                counts = simulate(u_true, pi_true, rho_gen, rng)

                record = {"P_true": p_true, "rho_gen": rho_gen, "replicate": rep,
                          "regime": regime}
                for mode in MODES:
                    best_p, results = ib.select_components(
                        counts, N_ALT, p_max=args.p_max, phi_mode=mode,
                        n_starts=args.n_starts)
                    model = results[best_p]["model"]
                    record[mode] = {
                        "selected_P": int(best_p),
                        "mean_rho": float(np.mean(model.rho)),
                        "max_rho": float(np.max(model.rho)),
                        "loglik": float(model.loglik),
                    }
                cells.append(record)
            print(f"  P_true={p_true}  rho_gen={rho_gen:<5} ({regime:<13}) done: "
                  f"{args.n_rep} replicates", flush=True)

    print()
    print("=" * 86)
    print("selected number of components (mean over replicates)")
    print("=" * 86)
    print(f"{'P_true':>6} {'rho_gen':>8} {'regime':>14} | "
          f"{'fit: binomial':>22} | {'fit: beta-binomial':>22}")
    print(f"{'':>6} {'':>8} {'':>14} | {'mean P':>8} {'= P_true':>12} | "
          f"{'mean P':>8} {'= P_true':>12}")
    print("-" * 86)

    summary = []
    for p_true in p_true_vals:
        for rho_gen in rho_vals:
            sub = [r for r in cells if r["P_true"] == p_true and r["rho_gen"] == rho_gen]
            row = {"P_true": p_true, "rho_gen": rho_gen, "regime": sub[0]["regime"]}
            for mode in MODES:
                ps = np.array([r[mode]["selected_P"] for r in sub])
                rhos = np.array([r[mode]["mean_rho"] for r in sub])
                row[mode] = {"mean_P": float(ps.mean()),
                             "P_distribution": {int(k): int((ps == k).sum())
                                                for k in range(1, args.p_max + 1)},
                             "frac_correct_P": float((ps == p_true).mean()),
                             "mean_rho_hat": float(rhos.mean())}
            summary.append(row)
            print(f"{p_true:>6} {rho_gen:>8.2f} {row['regime']:>14} | "
                  f"{row['binomial']['mean_P']:>8.2f} {row['binomial']['frac_correct_P']:>12.2f} | "
                  f"{row['beta-binomial']['mean_P']:>8.2f} {row['beta-binomial']['frac_correct_P']:>12.2f}")

    print()
    print("=" * 86)
    print("estimated rho (beta-binomial fit)")
    print("=" * 86)
    print(f"{'P_true':>6} {'rho_gen':>8} | {'mean rho_hat':>13} {'max rho_hat':>12}")
    print("-" * 86)
    for row in summary:
        sub = [r for r in cells if r["P_true"] == row["P_true"] and r["rho_gen"] == row["rho_gen"]]
        mr = np.mean([r["beta-binomial"]["mean_rho"] for r in sub])
        xr = np.mean([r["beta-binomial"]["max_rho"] for r in sub])
        print(f"{row['P_true']:>6} {row['rho_gen']:>8.2f} | {mr:>13.4f} {xr:>12.4f}")

    # the headline contrast
    infl = [row for row in summary if row["rho_gen"] > 0]
    print()
    print("=" * 86)
    print("headline: over-selection under a locked variance function")
    print("=" * 86)
    if infl:
        d_binom = np.mean([r["binomial"]["mean_P"] - r["P_true"] for r in infl])
        d_bb = np.mean([r["beta-binomial"]["mean_P"] - r["P_true"] for r in infl])
        print(f"  when rho_gen > 0: mean(P_hat - P_true) = {d_binom:+.3f} with locked rho, "
              f"{d_bb:+.3f} with free rho")
    zero = [row for row in summary if row["rho_gen"] == 0.0]
    if zero:
        d0 = np.mean([r["binomial"]["mean_P"] - r["P_true"] for r in zero])
        d0b = np.mean([r["beta-binomial"]["mean_P"] - r["P_true"] for r in zero])
        print(f"  when rho_gen = 0: mean(P_hat - P_true) = {d0:+.3f} with locked rho, "
              f"{d0b:+.3f} with free rho")

    payload = {"n_rep": args.n_rep, "p_max": args.p_max, "n_starts": args.n_starts,
               "seed": args.seed, "summary": summary, "replicates": cells}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
