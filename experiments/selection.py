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
from mop.initialization import warm_init, kridge                     # noqa: E402

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


def _fit_best_over_starts(X, y, k, free_phi, s2, starts):
    """The best (highest penalized objective) fit at ``k`` across the given starts."""
    best = None
    for kw in starts:
        r = fit(X, y, k, free_phi=free_phi, s2=s2, **kw)
        if best is None or r["pll"] > best["pll"]:
            best = r
    return best


def _starts_for(X, y, k, per_p, donor_order):
    """Initialisations that are structurally valid for a fit with ``k`` components.

    A start is valid when it actually has ``k`` components to begin with. Two ways to get
    one: a K-Ridge fit at ``k``, or a warm start from a fit with at least ``k``
    components, which ``warm_init`` reduces to ``k``.

    A warm start from a fit with FEWER than ``k`` components does not produce a ``k``
    component start -- it produces the donor at site ``k`` -- and including one was a real
    bug here. It let a P=2 solution be scored as a P=1 fit, since the EM took two steps
    from an already-converged point and stopped; the pooled cost slice then appeared to
    select P=1 at a penalized objective equal to the P=2 optimum. The simplest valid start
    is the K-Ridge fit, so it is always present even when no donor qualifies.
    """
    out = [{}]
    donors = [j for j in donor_order if j >= k and j in per_p]
    for j in donors:
        d = per_p[j]
        try:
            ui, si, mi = warm_init(X, y, d["u"], d["phi"], d["pi"],
                                   min_weight_share=MP.min_weight_share,
                                   n_components=k)
        except Exception:
            continue
        # warm_init only ADDS components; it does not reduce them. Asked to turn a
        # P=2 fit into a P=1 one it returns two rows, fit() adopts that count, and the
        # result is a P=2 fit labelled P=1 -- which then appeared to be the best single
        # component fit on the cost slice, at the P=2 optimum. Anything whose shape does
        # not match k is refused.
        ui = np.atleast_2d(np.asarray(ui, dtype=float))
        if ui.shape[0] != k:
            continue
        out.append({"u_init": ui, "phi_init": si, "pi_init": mi})
    return out


def select_best(X, y, candidate, free_phi, s2, chain_from=None, restarts=0, seed=0):
    """Selection over ``candidate``, taking the best valid optimum found at each P.

    Starts come from three families for every P:

      * a K-Ridge initialisation at that P, which depends on no other fit;
      * warm starts from fits with at least as many components, which is what the paper
        uses;
      * optionally, random restarts seeded from the K-Ridge fit at that P with jitter, so
        that a comparison between two counts is not settled by whichever of them happened
        to start nearer a good optimum.

    Taking the best over all of them means the P selected is the one with the lowest
    criterion among the valid optima located, not the one a single chain reached.
    """
    A = X.shape[1] + 1
    keys = list(chain_from) if chain_from is not None else list(candidate)
    keys = [k for k in keys if k in candidate] or list(candidate)
    rng = np.random.default_rng(seed)

    # Pass 1: independent K-Ridge starts, plus jittered restarts when asked.
    per_p = {}
    for k in candidate:
        starts = [{}]
        if restarts:
            u0, _, pi0 = kridge(X, y, k)
            u0 = np.atleast_2d(u0)
            pi0 = np.atleast_1d(pi0).astype(float).copy()
            for _ in range(restarts):
                uj = u0 * (1.0 + 0.25 * rng.standard_normal(u0.shape))
                pj = np.maximum(pi0 * (1.0 + 0.5 * rng.standard_normal(pi0.shape)), 1e-6)
                pj = pj / pj.sum()
                starts.append({"u_init": uj, "phi_init": np.full(k, float(s2)),
                               "pi_init": pj})
        per_p[k] = _fit_best_over_starts(X, y, k, free_phi, s2, starts)

    # Pass 2: warm starts from the other counts, iterated so a good optimum at one P can
    # seed an improvement at another.
    changed, rounds = True, 0
    while changed and rounds < 3:
        changed = False
        rounds += 1
        for k in list(candidate):
            for kw in _starts_for(X, y, k, per_p, keys):
                if not kw:
                    continue
                r = fit(X, y, k, free_phi=free_phi, s2=s2, **kw)
                if r["pll"] > per_p[k]["pll"] + 1e-8:
                    per_p[k] = r
                    changed = True

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
    ap.add_argument("--restarts", type=int, default=0,
                    help="jittered restarts per component count, on top of the K-Ridge "
                         "and warm starts; used to check that a comparison between two "
                         "counts is not settled by which one started nearer a good "
                         "optimum")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"parameters: {MP.summary()}")
    print(f"selection takes the best valid optimum per P across K-Ridge starts, warm")
    print(f"starts from counts with at least as many components, and {args.restarts} "
          f"jittered restarts.\n")

    slices = [(args.dim, {"dim_name": args.dim})] if args.dim else SLICES
    table = {}
    print(f"  {'slice':<24} {'P fixed':>8} {'P free':>7} | "
          f"{'phi0':>6} | {'naive-chain fixed':>18} {'naive-chain free':>17}")
    print("-" * 92)
    for label, kw in slices:
        pairs = load_pairs(**kw)
        if len(pairs) < 30:
            continue
        X, y = build_design(pairs)
        s2 = float(prior_mode_phi0(X, y))

        res = {}
        for tag, free in (("fixed", False), ("free", True)):
            best = select_best(X, y, args.grid, free, s2, restarts=args.restarts, seed=args.seed)
            # The naive chain, low to high, which is what the earlier scripts did.
            naive = select_best(X, y, args.grid, free, s2, chain_from=args.grid)
            res[tag] = {
                "curve": {str(k): v for k, v in best.items()},
                "best": min(best, key=lambda k: best[k]["pBIC"]),
                "best_BIC": min(best, key=lambda k: best[k]["BIC"]),
                "naive_chain_best": min(naive, key=lambda k: naive[k]["pBIC"]),
            }
        table[label] = {"phi0": s2, **res}
        print(f"  {label:<24} {res['fixed']['best']:>8} {res['free']['best']:>7} | "
              f"{s2:>6.3f} | {res['fixed']['naive_chain_best']:>18} "
              f"{res['free']['naive_chain_best']:>17}")

    print()
    flips_f = [l for l, r in table.items()
               if r["fixed"]["best"] != r["fixed"]["naive_chain_best"]]
    flips_g = [l for l, r in table.items()
               if r["free"]["best"] != r["free"]["naive_chain_best"]]
    print(f"  where the naive chain disagrees with the best-optimum selection:")
    print(f"    fixed dispersion: {len(flips_f)}/{len(table)} slices {flips_f}")
    print(f"    free  dispersion: {len(flips_g)}/{len(table)} slices {flips_g}")
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
        {"grid": args.grid, "procedure": "best optimum per P over independent and warm "
                                         "starts, then minimise the criterion",
         "slices": table}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
