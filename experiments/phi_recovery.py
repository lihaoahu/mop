"""Recovering the structural inconsistency: the simulation the abstract assumes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parent
CASE = EXPERIMENTS / "case_study"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(EXPERIMENTS.parent / "src"))

import _mop_params as MP                                            # noqa: E402
from mop.pairwise_comparison import comparisons2regression          # noqa: E402
from mop.pairwise_comparison import PairwiseComparison              # noqa: E402

ORDERS = [
    [1.60, 0.80, 0.20, 0.00],
    [0.00, 0.20, 0.80, 1.60],
]


def make_pairs(n_pairs, weights, rng):
    """Pairwise comparisons drawn from the given state weights."""
    pairs = []
    for _ in range(n_pairs):
        p = rng.choice(len(ORDERS), p=weights)
        i, j = rng.choice(4, size=2, replace=False)
        pairs.append((int(i), int(j), p))
    return pairs


def simulate(n_pairs, true_phi, rng):
    """Synthetic ratio judgments from a two-state mixture with known dispersion."""
    weights = np.array([0.5, 0.5])
    pairs = make_pairs(n_pairs, weights, rng)
    y = np.empty(n_pairs)
    states = np.empty(n_pairs, dtype=int)
    for k, (i, j, p) in enumerate(pairs):
        y[k] = ORDERS[p][i] - ORDERS[p][j] + rng.normal(0.0, np.sqrt(true_phi))
        states[k] = p
    return pairs, y, states, weights


def design(pairs):
    """The regression design the repo's own pipeline uses: x_k = e_i - e_j, A-1 columns."""
    pcv = [PairwiseComparison(former_id=i + 1, latter_id=j + 1, agent_id=1,
                              judgment=0.0) for i, j, _ in pairs]
    eps = 1e-10
    X, _, _ = comparisons2regression(pcv, [lambda s, _e=eps: s])
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--phi", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--grid", type=float, nargs="+", default=None,
                    help="fixed-phi values evaluated; defaults to the true values")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from step9_phi_identification import fit, criteria

    grid = args.grid if args.grid else list(args.phi)
    rng = np.random.default_rng(args.seed)
    nJ = args.pairs

    print(f"phi recovery on synthetic Gaussian data, {args.reps} replicates per setting")
    print(f"  {nJ} comparisons, 2 true states with opposite orderings, A = 4")
    print(f"  {MP.summary()}")
    print(f"  lambda_sigma = {MP.LAMBDA_SIGMA_OVER_N * nJ:.2f}, "
          f"phi_0 = prior_mode_phi0 (computed per replicate)\n")

    print(f"{'phi_true':>9} {'phi_hat mean':>13} {'sd':>8} {'ratio':>8} "
          f"{'|log ratio|':>12} | {'P chosen (fixed phi)':>21} {'(free phi)':>12}")
    print("-" * 92)

    results = {}
    for tp in args.phi:
        hats, ratios = [], []
        p_fixed, p_free, p1_fixed, p1_free = [], [], [], []
        true_phat_best = 0
        for _ in range(args.reps):
            pairs, y, states, w_true = simulate(nJ, tp, rng)
            X = design(pairs)
            s2 = 1.0

            r_free = fit(X, y, 2, free_phi=True, s2=s2)
            hats.append(float(np.mean(r_free["phi"])))
            ratios.append(float(np.mean(r_free["phi"])) / tp)

            plls = {}
            for g in grid:
                plls[g] = fit(X, y, 2, free_phi=False, s2=s2, phi_fixed=g)["pll"]
            best_g = max(plls, key=plls.get)
            if abs(best_g - tp) < 1e-9:
                true_phat_best += 1

            def best_p(free):
                out = {}
                for k in (1, 2, 3):
                    r = fit(X, y, k, free_phi=free, s2=s2)
                    out[k] = criteria(r, X, y, 4, k, free)["pBIC"]
                return min(out, key=out.get)
            kf = best_p(False)
            kg = best_p(True)
            p_fixed.append(kf)
            p_free.append(kg)
            if kf == 1:
                p1_fixed.append(1)
            if kg == 1:
                p1_free.append(1)

        hats = np.array(hats)
        results[str(tp)] = {
            "phi_true": float(tp),
            "phi_hat_mean": float(hats.mean()),
            "phi_hat_sd": float(hats.std()),
            "ratio_mean": float(np.mean(ratios)),
            "abs_log_ratio_mean": float(np.mean(np.abs(np.log(ratios)))),
            "P2_fixed_phi": int(sum(1 for k in p_fixed if k == 2)),
            "P2_free_phi": int(sum(1 for k in p_free if k == 2)),
            "P1_fixed_phi": int(len(p1_fixed)),
            "P1_free_phi": int(len(p1_free)),
            "reps": args.reps,
        }
        print(f"{tp:>9.2f} {hats.mean():>13.3f} {hats.std():>8.3f} "
              f"{np.mean(ratios):>8.3f} {np.mean(np.abs(np.log(ratios))):>12.3f} | "
              f"{sum(1 for k in p_fixed if k == 2):>2d}/{args.reps} P=2"
              f"{'':>11} {sum(1 for k in p_free if k == 2):>2d}/{args.reps} P=2")

    print()
    print("Reading: 'ratio' is phi_hat / phi_true. The estimator should track the truth")
    print("and the ratio should sit near one across the range; a ratio that collapses")
    print("towards small values as phi_true grows would mean the mixture is absorbing")
    print("the dispersion instead of the dispersion measuring it. The last two columns")
    print("report whether the correct number of states survives, which is the question")
    print("Section 2.5 raises for the binomial protocol.")

    out = Path(args.out) if args.out else EXPERIMENTS / "phi_recovery.json"
    out.write_text(json.dumps({"seed": args.seed, "reps": args.reps,
                               "pairs": args.pairs, "grid": grid,
                               "settings": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
