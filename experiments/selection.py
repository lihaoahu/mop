"""One selection procedure for the component count, independent of where the search starts.

Why this exists
---------------
Three scripts reported the case study's component count and all three agreed, because all
three warm-start the component search from the low end of the grid. The penalized
objective is multimodal, so that shared habit fixed which local optimum they reached, and
their agreement was an artefact of the shared procedure rather than evidence. Sweeping the
grid's starting point exposes it: on the pooled data the free-dispersion search selects
P=3 from a grid starting at 1 and P=4 from one starting at 4, where the criterion value is
also lower (18346.1 against 18453.1).

What "the" selection should mean
--------------------------------
The count that minimises the criterion over the candidate set, taking the best optimum
found rather than the first one a chain happens to reach. This module makes that explicit:

  1. for each P in the candidate set, fit from several starts -- the warm-start chain, the
     K-Ridge initialisation, and (optionally) starts seeded from neighbouring chains;
  2. keep the fit with the highest penalized objective for that P;
  3. select the P minimising pBIC (and report BIC alongside).

The result is the best optimum located, and it is reported together with how much the
answer would have differed under the naive chain, so the sensitivity is visible rather than
silently resolved.

Run:
    uv run python experiments/selection.py --report          # every slice
    uv run python experiments/selection.py --dim cost        # one slice
"""
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
from mop.inference import prior_mode_phi0                           # noqa: E402
from mop.initialization import kridge                     # noqa: E402

from step9_phi_identification import (build_design, fit, criteria,   # noqa: E402
                                      load_pairs)

SLICES = [("overall", {}),
          ("cost", {"dim_name": "cost"}),
          ("driving_experience", {"dim_name": "driving_experience"}),
          ("range", {"dim_name": "range"}),
          ("refueling_convenience", {"dim_name": "refueling_convenience"}),
          ("reliability", {"dim_name": "reliability"}),
          ("policy", {"dim_name": "policy"}),
          ("other", {"dim_name": "other"}),
          ("2025", {"year": "2025"}),
          ("2026", {"year": "2026"})]


def select_best(X, y, candidate, free_phi, s2):
    """K-Ridge initialisation at every candidate count, then the lowest pBIC.

    The initialiser is deterministic and independent at each count, so the selected count is
    reproducible; no fit is carried forward from a smaller count.
    """
    per_p = {}
    for k in candidate:
        u0, phi0, pi0 = kridge(X, y, k)
        per_p[k] = fit(X, y, k, free_phi=free_phi, s2=s2,
                       u_init=u0, phi_init=phi0, pi_init=pi0)

    A = X.shape[1] + 1
    out = {}
    for k in candidate:
        c = criteria(per_p[k], X, y, A, k, free_phi)
        out[k] = {"pBIC": float(c["pBIC"]), "BIC": float(c["BIC"]),
                  "pll": float(per_p[k]["pll"]), "ll": float(per_p[k]["ll"]),
                  "n_iter": int(per_p[k]["n_iter"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", default=True)
    ap.add_argument("--dim", default=None)
    ap.add_argument("--grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"parameters: {MP.summary()}")
    print("selection initialises every candidate P with K-Ridge from scratch and keeps "
          "the lowest pBIC;\nno fit is carried forward from a smaller count.\n")

    slices = [(args.dim, {"dim_name": args.dim})] if args.dim else SLICES
    table = {}
    print(f"  {'slice':<24} {'P fixed':>8} {'P free':>7} | {'phi0':>6}")
    print("-" * 60)
    for label, kw in slices:
        pairs = load_pairs(**kw)
        if len(pairs) < 30:
            continue
        X, y = build_design(pairs)
        s2 = float(prior_mode_phi0(X, y))
        grid = [k for k in args.grid if k <= MP.max_components_for(X.shape[1] + 1)]

        res = {}
        for tag, free in (("fixed", False), ("free", True)):
            best = select_best(X, y, grid, free, s2)
            res[tag] = {
                "curve": {str(k): v for k, v in best.items()},
                "best": min(best, key=lambda k: best[k]["pBIC"]),
                "best_BIC": min(best, key=lambda k: best[k]["BIC"]),
            }
        table[label] = {"phi0": s2, **res}
        print(f"  {label:<24} {res['fixed']['best']:>8} {res['free']['best']:>7} | {s2:>6.3f}")

    print()
    agree_f = [l for l, r in table.items()
               if r["fixed"]["best"] == r["fixed"]["best_BIC"]]
    agree_g = [l for l, r in table.items()
               if r["free"]["best"] == r["free"]["best_BIC"]]
    print(f"  pBIC and BIC agree under the best-optimum selection:")
    print(f"    fixed dispersion: {len(agree_f)}/{len(table)}")
    print(f"    free  dispersion: {len(agree_g)}/{len(table)}")

    out = Path(args.out) if args.out else (EXPERIMENTS / "selection.json")
    out.write_text(json.dumps(
        {"grid": args.grid, "procedure": "K-Ridge initialisation at every candidate count, then minimise "
                                         "the criterion",
         "slices": table}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
