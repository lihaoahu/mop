"""The two selection facts the case study's Limitations paragraph will quote.

Before any number goes into the manuscript it is computed here, so the paragraph cites
a run rather than a recollection. Two claims are checked:

  1. pBIC and an ordinary BIC disagree on the selected component count. Both are
     evaluated on the same fits and with the same degrees of freedom, so the
     difference between them is the penalized posterior against the likelihood -- the
     priors. That is a real difference and it is the one a reader who recomputes BIC
     on the same fit will meet, but the penalty COUNT is shared, so this does not test
     the count independently.
  2. The count depends on the candidate set: on the cost dimension the full grid picks
     P = 2, and excluding P = 2 makes P = 3 the selection.

Also reports the fixed-versus-free dispersion selections per slice, which is the
headline of the paragraph.

Run: uv run python experiments/case_study/_selection_facts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import _mop_params as MP                                            # noqa: E402
from mop.inference import prior_mode_phi0                           # noqa: E402

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

GRID = [1, 2, 3, 4, 5]

S2_MODES = {
    "data": "prior_mode_phi0(X, y)  -- the paper's rule, what step7_analyze uses",
    "one": "pinned at 1 -- what _bootstrap_pbic.py defaults to",
}


def warm_chain(X, y, grid, free_phi, s2):
    """Fit ``grid`` low to high, warm-starting each from the previous one.

    This is what step7_analyze.py does, and a single pass over the grid at fixed P
    does not reproduce it: the two differ whenever a fit has local optima. Both are
    reported so the difference is visible rather than assumed away.
    """
    from mop.initialization import warm_init

    out = {}
    prev = None            # step9's fit returns a dict, not a model object
    for k in grid:
        kw = {}
        if prev is not None:
            ui, si, mi = warm_init(X, y, prev["u"], prev["phi"], prev["pi"],
                                   min_weight_share=MP.min_weight_share,
                                   n_components=k)
            kw = {"u_init": ui, "phi_init": si, "pi_init": mi}
        r = fit(X, y, k, free_phi=free_phi, s2=s2, **kw)
        A = X.shape[1] + 1
        out[k] = criteria(r, X, y, A, k, free_phi)
        prev = r
    return out


def select(pairs, grid, free_phi, initialisation="single", s2_mode="data"):
    """pBIC and BIC selection over ``grid``, plus the curves.

    ``s2_mode`` defaults to the paper's rule. Pinning it at one is a DIFFERENT prior
    and is offered only so the prior-comparison mode can vary it: an earlier version of
    this file silently used 1.0 while calling itself the paper's setting, and reported
    P = 5 for the fixed-dispersion fit where the paper's own pathway gives P = 4.
    """
    X, y = build_design(pairs)
    nJ, A1 = X.shape
    A = A1 + 1
    s2 = (float(prior_mode_phi0(X, y)) if s2_mode == "data" else 1.0)
    if initialisation == "warm":
        return warm_chain(X, y, grid, free_phi, s2), s2
    out = {}
    for k in grid:
        r = fit(X, y, k, free_phi=free_phi, s2=s2)
        out[k] = criteria(r, X, y, A, k, free_phi)
    return out, s2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grid", "priors", "candidates"],
                    default="priors")
    args = ap.parse_args()

    print(f"parameters: {MP.summary()}\n")

    if args.mode == "priors":
        # The decisive run: does the paper's own prior mode reproduce the paper's own
        # selection, and does the bootstrap's pinned prior mode reproduce the
        # bootstrap's? Four combinations, so the difference is attributed rather than
        # guessed at.
        print("selection under each prior mode and each initialisation, free phi")
        print(f"  {'slice':<24} {'phi0':>7} | "
              f"{'data, single':>13} {'data, warm':>11} | "
              f"{'one, single':>12} {'one, warm':>10}")
        print("-" * 92)
        out = {}
        for label, kw in SLICES:
            pairs = load_pairs(**kw)
            if len(pairs) < 30:
                continue
            row, phi0 = {}, None
            for mode in ("data", "one"):
                for warm in (False, True):
                    cur, s2 = select(pairs, GRID, True, initialisation="warm" if warm else "single", s2_mode=mode)
                    row[(mode, warm)] = min(cur, key=lambda k: cur[k]["pBIC"])
                # keep each mode's own prior mode: assigning outside the mode loop
                # reports the last mode's value under both columns.
                if mode == "data":
                    phi0 = s2
            out[label] = {"phi0": phi0,
                          **{f"{m}_{'warm' if w else 'single'}": v
                             for (m, w), v in row.items()}}
            print(f"  {label:<24} {phi0:>7.3f} | "
                  f"{row[('data', False)]:>13} {row[('data', True)]:>11} | "
                  f"{row[('one', False)]:>12} {row[('one', True)]:>10}")
        print()
        counts = {}
        for key in ("data_single", "data_warm", "one_single", "one_warm"):
            counts[key] = sum(1 for v in out.values() if v[key] == 4)
        print("how often P = 4 is selected:")
        for k, v in counts.items():
            print(f"  {k:<12} {v}/{len(out)}")
        (HERE / "selection_priors.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwritten: selection_priors.json")
        return

    if args.mode == "candidates":
        print("candidate-set sensitivity on the cost dimension (free phi, data prior)")
        pairs = load_pairs(dim_name="cost")
        full, _ = select(pairs, GRID, True, s2_mode="data")
        minus2, _ = select(pairs, [1, 3, 4, 5], True, s2_mode="data")
        fb = min(full, key=lambda k: full[k]["pBIC"])
        mb = min(minus2, key=lambda k: minus2[k]["pBIC"])
        print(f"  full grid {GRID}:  selects P={fb}")
        print(f"  grid without P=2:   selects P={mb}")
        for k in sorted(full):
            print(f"    P={k}: pBIC={full[k]['pBIC']:.2f}")
        return

    # mode == "grid": the per-slice table with the paper's prior and warm start.
    print(f"{'slice':<24} {'P fixed':>8} {'P free':>7} | "
          f"{'pBIC fix':>9} {'BIC fix':>8} | {'pBIC free':>10} {'BIC free':>9} "
          f"| {'agree':>6}")
    print("-" * 100)
    rows = {}
    for label, kw in SLICES:
        pairs = load_pairs(**kw)
        if len(pairs) < 30:
            continue
        res = {}
        for tag, free in (("fixed", False), ("free", True)):
            cur, _ = select(pairs, GRID, free, initialisation="warm", s2_mode="data")
            res[tag] = {"curve": cur,
                        "pBIC_best": min(cur, key=lambda k: cur[k]["pBIC"]),
                        "BIC_best": min(cur, key=lambda k: cur[k]["BIC"])}
        rows[label] = res
        print(f"{label:<24} {res['fixed']['pBIC_best']:>8} "
              f"{res['free']['pBIC_best']:>7} | "
              f"{res['fixed']['pBIC_best']:>9} {res['fixed']['BIC_best']:>8} | "
              f"{res['free']['pBIC_best']:>10} {res['free']['BIC_best']:>9} | "
              f"{'yes' if res['fixed']['pBIC_best'] == res['fixed']['BIC_best'] else 'NO':>6}")
    print()
    for tag in ("fixed", "free"):
        bad = [l for l, r in rows.items() if r[tag]["pBIC_best"] != r[tag]["BIC_best"]]
        print(f"pBIC vs BIC, {tag:<6} disagree on {len(bad)}/{len(rows)}: {bad}")
    (HERE / "selection_facts.json").write_text(json.dumps(
        {"grid": GRID,
         "slices": {l: {t: {"pBIC_best": r[t]["pBIC_best"],
                            "BIC_best": r[t]["BIC_best"]}
                        for t in ("fixed", "free")} for l, r in rows.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwritten: selection_facts.json")


if __name__ == "__main__":
    main()
