"""Generalization check on a published choice-under-uncertainty dataset.

Regenwetter, M., & Davis-Stober, C. P. (2012). Behavioral variability of
choices versus structural inconsistency of preferences. Psychological Review,
119(2), 408-416. doi:10.1037/a0027372

The dataset holds 30 participants x 3 gamble sets. Within each unit, every one
of the 10 gamble pairs was judged 45 times, and a judgment could be
"a > b", "b > a", or "a = b" (indifference). That repetition structure is what
makes the data useful here: the mixture is asked to separate intrinsic
variability (preference states that switch) from structural inconsistency
(residual variability in executing one state).

Each unit is modelled as a repeated binary comparison,

    m_k | c_k = p  ~  BetaBinomial(n_k, mu_kp, rho_p),   mu_kp = logistic(x_k' u_p)

and fitted under both available likelihoods:

    binomial        rho_p = 0. The variance function mu(1-mu) is locked, so any
                    dispersion beyond it has to be absorbed by extra components.
    beta-binomial   rho_p free in [0, 0.9], carrying structural inconsistency.

The comparison is a falsifiable prediction of the framework: freeing rho should
move the selected number of components downward, because dispersion that was
being explained by extra components is now carried by rho.

Ties are scored as half a success, m_k = n_{a>b} + 0.5 * n_{a=b}, with
n_k = 45.

Usage:
    uv run python experiments/v3/step1_regenwetter.py
    uv run python experiments/v3/step1_regenwetter.py --p-max 3 --out results.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mop import inference_binomial as ib  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "regenwetter2012.csv"
GAMBLES = ["a", "b", "c", "d", "e"]
PAIRS = [(i, j) for i in range(len(GAMBLES)) for j in range(i + 1, len(GAMBLES))]
MODES = ("binomial", "beta-binomial")


def load_units():
    """Yield (participant, gamble_set, counts) for every unit in the dataset."""
    with open(DATA, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts = []
            for i, j in PAIRS:
                gi, gj = GAMBLES[i], GAMBLES[j]
                gt = int(row[f"{gi}>{gj}"])
                lt = int(row[f"{gj}>{gi}"])
                eq = int(row[f"{gi}={gj}"])
                counts.append({
                    "former": i, "latter": j,
                    "successes": gt + 0.5 * eq,       # a tie counts as half a win
                    "trials": gt + lt + eq,
                })
            yield int(row["participant"]), int(row["gamble_set"]), counts


def variability(counts):
    """0 = every pair is a coin flip, 1 = every pair has a unanimous winner."""
    return float(np.mean([abs(c["successes"] / c["trials"] - 0.5) * 2 for c in counts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-max", type=int, default=4,
                    help="largest number of components considered (default 4)")
    ap.add_argument("--n-starts", type=int, default=2)
    ap.add_argument("--out", type=Path, default=HERE / "regenwetter_results.json")
    args = ap.parse_args()

    rows = []
    print(f"{'unit':>9} {'var':>6} | " + " | ".join(f"{m:>22}" for m in MODES))
    print("-" * 68)
    for participant, gset, counts in load_units():
        record = {"participant": participant, "gamble_set": gset,
                  "variability": variability(counts)}
        cells = []
        for mode in MODES:
            best_p, results = ib.select_components(
                counts, len(GAMBLES), p_max=args.p_max,
                phi_mode=mode, n_starts=args.n_starts)
            model = results[best_p]["model"]
            record[mode] = {
                "selected_P": best_p,
                "bic": float(results[best_p]["bic"]),
                "rho": [float(x) for x in model.rho],
                "pi": [float(x) for x in model.pi],
                "rankings": model.rankings(GAMBLES),
            }
            cells.append(f"P={best_p}  mean rho={np.mean(model.rho):.4f}")
        rows.append(record)
        print(f"{participant:>4}-{gset:<4} {record['variability']:>6.3f} | " + " | ".join(cells))

    print()
    print("=" * 68)
    print(f"selected number of components, {len(rows)} units")
    print("=" * 68)
    summary = {}
    for mode in MODES:
        ps = np.array([r[mode]["selected_P"] for r in rows])
        dist = {int(k): int((ps == k).sum()) for k in range(1, args.p_max + 1)}
        rho = np.array([np.mean(r[mode]["rho"]) for r in rows])
        summary[mode] = {"distribution": dist, "mean_P": float(ps.mean()),
                         "mean_rho": float(rho.mean()), "median_rho": float(np.median(rho))}
        print(f"  {mode:<15} " + "  ".join(f"P={k}: {v:>2}" for k, v in dist.items())
              + f"   mean P={ps.mean():.3f}")
    ps_b = np.array([r["binomial"]["selected_P"] for r in rows])
    ps_bb = np.array([r["beta-binomial"]["selected_P"] for r in rows])
    var = np.array([r["variability"] for r in rows])
    print()
    print(f"  mean P: binomial {ps_b.mean():.3f} -> beta-binomial {ps_bb.mean():.3f}"
          f"   ({int((ps_bb < ps_b).sum())} units moved down, {int((ps_bb > ps_b).sum())} up)")
    print(f"  corr(variability, P): binomial {np.corrcoef(var, ps_b)[0,1]:+.3f}, "
          f"beta-binomial {np.corrcoef(var, ps_bb)[0,1]:+.3f}")

    payload = {"p_max": args.p_max, "n_starts": args.n_starts,
               "summary": summary, "units": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
