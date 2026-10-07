"""Population-level fit of the published dataset.

`step1_regenwetter.py` fits one mixture per participant x set unit, i.e. ten
observations each. Pooling all 30 participants of a gamble set gives 300
observations for a single mixture, which is the population-level reading of the
same data: the components then describe preference states that recur *across*
participants, exactly the way the case study pools thousands of authors.

This matters for the separability question, because `step3_scaling.py` predicts
that the two likelihoods diverge as the number of observations grows: with a
locked variance function the extra dispersion gets more confidently read as
extra components, while a free rho absorbs it. This script checks that
prediction on the real data.

Usage:
    uv run python experiments/v3/step4_regenwetter_pooled.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mop import inference_binomial as ib  # noqa: E402
from step1_regenwetter import GAMBLES, PAIRS, load_units  # noqa: E402

MODES = ("binomial", "beta-binomial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-max", type=int, default=4)
    ap.add_argument("--n-starts", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE / "regenwetter_pooled_results.json")
    args = ap.parse_args()

    units = list(load_units())
    by_set = {}
    for participant, gset, counts in units:
        by_set.setdefault(gset, []).extend(counts)

    results = {}
    print("=" * 78)
    print("population-level fit: all participants of a gamble set pooled")
    print("=" * 78)
    for gset in sorted(by_set):
        counts = by_set[gset]
        n_participants = sum(1 for _, g, _ in units if g == gset)
        print(f"\ngamble set {gset}:  {n_participants} participants, "
              f"{len(counts)} observations ({len(PAIRS)} pairs)")
        results[gset] = {"n_participants": n_participants, "n_obs": len(counts)}
        for mode in MODES:
            best_p, table = ib.select_components(counts, len(GAMBLES),
                                                 p_max=args.p_max, phi_mode=mode,
                                                 n_starts=args.n_starts)
            model = table[best_p]["model"]
            print(f"   {mode:<15} P_hat={best_p}   BIC={table[best_p]['bic']:.1f}   "
                  f"mean rho={np.mean(model.rho):.4f}")
            for q, (rank, pi, rho) in enumerate(zip(model.rankings(GAMBLES),
                                                    model.pi, model.rho)):
                print(f"      C{q+1}: pi={pi:.4f}  rho={rho:.4f}  {rank}")
            results[gset][mode] = {
                "selected_P": int(best_p),
                "bic": float(table[best_p]["bic"]),
                "bic_by_P": {int(k): float(v["bic"]) for k, v in table.items()},
                "pi": [float(x) for x in model.pi],
                "rho": [float(x) for x in model.rho],
                "rankings": model.rankings(GAMBLES),
            }

    print()
    print("=" * 78)
    print("summary")
    print("=" * 78)
    print(f"{'set':>4} {'locked rho':>12} {'free rho':>10} {'rho_hat':>10}")
    for gset in sorted(results):
        r = results[gset]
        print(f"{gset:>4} {r['binomial']['selected_P']:>12} "
              f"{r['beta-binomial']['selected_P']:>10} "
              f"{np.mean(r['beta-binomial']['rho']):>10.4f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
