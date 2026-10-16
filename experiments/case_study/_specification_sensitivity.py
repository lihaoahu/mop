"""The case study's selection sensitivity, recomputed under the paper's own specification.

Three claims were removed from the Limitations paragraph because they came from a script
that mixed the estimated- and fixed-dispersion specifications. This recomputes them under the
estimated dispersion only -- which is what `inference.mop` does and therefore what the paper
reports -- so that each can be restored, corrected, or left out on evidence rather than on
which script happened to produce it.

The claims:

  1. the selected count depends on the candidate set;
  2. the penalized criterion and an ordinary BIC disagree on the selected count;
  3. the selection survives resampling, or does not.

Everything here is `free_phi=True` and `phi_0 = prior_mode_phi0(X, y)`. Nothing is compared
against a fixed-dispersion value, which is the mistake this file exists to avoid.

Run:
    uv run python experiments/case_study/_specification_sensitivity.py
    uv run python experiments/case_study/_specification_sensitivity.py --reps 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import _mop_params as MP                                            # noqa: E402
import selection as SEL                                             # noqa: E402
from mop.inference import prior_mode_phi0                           # noqa: E402

from step9_phi_identification import (build_design, criteria, fit,   # noqa: E402
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


def curve_via_chain(X, y, grid, s2):
    """The paper's search order: warm start low to high, then take the criterion minimum."""
    from mop.initialization import warm_init
    A = X.shape[1] + 1
    prev, out = None, {}
    for k in grid:
        kw = {}
        if prev is not None:
            ui, si, mi = warm_init(X, y, prev["u"], prev["phi"], prev["pi"],
                                   min_weight_share=MP.min_weight_share,
                                   n_components=k)
            ui = np.atleast_2d(ui)
            if ui.shape[0] == k:
                kw = {"u_init": ui, "phi_init": si, "pi_init": mi}
        r = fit(X, y, k, free_phi=True, s2=s2, **kw)
        out[k] = criteria(r, X, y, A, k, True)
        prev = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("specification: free_phi = True, phi_0 = prior_mode_phi0(X, y)")
    print(f"parameters: {MP.summary()}\n")

    # ── 1. candidate set, and 2. pBIC against BIC, per slice ──
    print(f"{'slice':<24} {'chain P':>8} {'no P=2':>8} {'pBIC':>10} {'BIC':>10} "
          f"{'agree':>6}")
    print("-" * 74)
    per_slice = {}
    disagree = []
    for label, kw in SLICES:
        pairs = load_pairs(**kw)
        if len(pairs) < 30:
            continue
        X, y = build_design(pairs)
        s2 = float(prior_mode_phi0(X, y))
        full = curve_via_chain(X, y, GRID, s2)
        best_p = min(full, key=lambda k: full[k]["pBIC"])
        best_b = min(full, key=lambda k: full[k]["BIC"])
        # drop P=2 from the candidate set
        if 2 in GRID and best_p == 2:
            without = curve_via_chain(X, y, [k for k in GRID if k != 2], s2)
            best_w = min(without, key=lambda k: without[k]["pBIC"])
        else:
            best_w = None
        per_slice[label] = {
            "phi0": s2,
            "chain_best_pBIC": best_p, "chain_best_BIC": best_b,
            "without_P2": best_w,
            "curve": {str(k): {"pBIC": float(v["pBIC"]), "BIC": float(v["BIC"])}
                      for k, v in full.items()},
        }
        if best_p != best_b:
            disagree.append(label)
        print(f"{label:<24} {best_p:>8} "
              f"{(str(best_w) if best_w is not None else '-'):>8} "
              f"{full[best_p]['pBIC']:>10.1f} {full[best_p]['BIC']:>10.1f} "
              f"{'yes' if best_p == best_b else 'NO':>6}")

    print()
    print(f"  the BIC column is -2*loglik + log(n)*df at the pBIC-selected count, so the")
    print(f"  two differ only through the prior terms; they disagree on "
          f"{len(disagree)}/{len(per_slice)} slices")
    if disagree:
        print(f"    {disagree}")
    print()
    print("  candidate set: of the slices whose chain selects P=2, dropping P=2 from the")
    print("  grid selects:")
    for label, r in per_slice.items():
        if r["chain_best_pBIC"] == 2 and r["without_P2"] is not None:
            print(f"    {label:<24} -> P={r['without_P2']}")

    # ── 3. resampling ──
    print(f"\nresampling the judgments, {args.reps} replicates, chain selection each time")
    rng = np.random.default_rng(args.seed)
    pairs = load_pairs()
    X, y = build_design(pairs)
    nJ = X.shape[0]
    s2_full = float(prior_mode_phi0(X, y))
    counts = Counter()
    for rep in range(args.reps):
        idx = rng.integers(0, nJ, size=nJ)
        Xb, yb = X[idx], y[idx]
        s2b = float(prior_mode_phi0(Xb, yb))
        cur = curve_via_chain(Xb, yb, GRID, s2b)
        b = min(cur, key=lambda k: cur[k]["pBIC"])
        counts[b] += 1
        if rep < 10 or rep == args.reps - 1:
            print(f"  rep {rep + 1:>3}/{args.reps}  P={b}", flush=True)
    print()
    pretty = ", ".join(f"P={k}: {100 * c / args.reps:.0f}%" for k, c in sorted(counts.items()))
    print(f"  selection frequencies: {pretty}")

    out = Path(args.out) if args.out else (HERE / "specification_sensitivity.json")
    out.write_text(json.dumps(
        {"specification": "free_phi=True, phi0=prior_mode_phi0",
         "grid": GRID, "reps": args.reps,
         "slices": per_slice,
         "pBIC_BIC_disagree": disagree,
         "bootstrap": {str(k): c / args.reps for k, c in sorted(counts.items())}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
