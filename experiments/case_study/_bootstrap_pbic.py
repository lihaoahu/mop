"""Is the P = 3 versus P = 4 choice a finding, or a coin toss?

On the full case data with phi free, pBIC puts P = 4 ahead of P = 3 by about nine
units. Nine out of eighteen thousand is not obviously a decision, and the case
study's headline structural claim is that there are four components. Before that
claim is reported, the gap needs to be shown to survive resampling.

Method: resample the judgments with replacement (a nonparametric bootstrap over
judgments), refit, and count how often each component count wins. If the winner
changes across replicates the selection is not identified by the data, whatever the
point estimate says.

Both parameterisations are run, because the two can be unstable for different
reasons and the comparison is the point.

Run:
    uv run python experiments/case_study/_bootstrap_pbic.py --reps 20
    ... --dim cost
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
sys.path.insert(0, str(HERE.parents[1] / "src"))

from step9_phi_identification import (build_design, criteria, fit,  # noqa: E402
                                      load_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--grid", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--dim", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s2", default="data",
                    help="prior mode phi_0: 'data' (default, and what "
                         "step7_analyze.py does) computes it per replicate from the "
                         "resampled data via prior_mode_phi0; a number pins it. The "
                         "earlier default was 1.0, which is a DIFFERENT prior from the "
                         "one the paper's fits use, so the bootstrap was describing a "
                         "different model from step7_analyze.py's -- see "
                         "PHI_IDENTIFICATION.md.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mop.inference import prior_mode_phi0

    def resolve_s2(Xa, ya):
        return float(prior_mode_phi0(Xa, ya)) if args.s2 == "data" else float(args.s2)

    pairs = load_pairs(dim_name=args.dim) if args.dim else load_pairs()
    X, y = build_design(pairs)
    nJ, A1 = X.shape
    A = A1 + 1
    label = args.dim or "overall"
    print(f"[{label}] n={nJ}, bootstrap reps={args.reps}, grid={args.grid}")

    rng = np.random.default_rng(args.seed)

    # Point estimate first, so the bootstrap has a reference.
    point = {}
    for tag, free in (("fixed_phi", False), ("free_phi", True)):
        vals = {}
        for k in args.grid:
            r = fit(X, y, k, free, s2=resolve_s2(X, y))
            vals[k] = criteria(r, X, y, A, k, free)["pBIC"]
        best = min(vals, key=vals.get)
        point[tag] = {"pBIC": vals, "best": best}
        order = sorted(vals, key=vals.get)
        print(f"  point estimate {tag}: best P={best}, "
              f"gap to runner-up = {vals[order[1]] - vals[order[0]]:.1f}")

    counts = {tag: Counter() for tag in ("fixed_phi", "free_phi")}
    gaps = {tag: [] for tag in ("fixed_phi", "free_phi")}
    for rep in range(args.reps):
        idx = rng.integers(0, nJ, size=nJ)
        Xb, yb = X[idx], y[idx]
        line = []
        for tag, free in (("fixed_phi", False), ("free_phi", True)):
            vals = {}
            for k in args.grid:
                r = fit(Xb, yb, k, free, s2=resolve_s2(Xb, yb))
                vals[k] = criteria(r, Xb, yb, A, k, free)["pBIC"]
            best = min(vals, key=vals.get)
            order = sorted(vals, key=vals.get)
            counts[tag][best] += 1
            gaps[tag].append(vals[order[1]] - vals[order[0]])
            line.append(f"{tag}: P={best} (gap {vals[order[1]] - vals[order[0]]:5.1f})")
        print(f"  rep {rep + 1:>3}/{args.reps}  " + "   ".join(line), flush=True)

    print(f"\n=== bootstrap selection frequencies, {args.reps} replicates ===")
    result = {"label": label, "n": int(nJ), "reps": args.reps,
              "s2_setting": args.s2, "phi0_point": resolve_s2(X, y),
              "grid": list(args.grid), "point_estimate": {
                  t: {"best": int(v["best"]),
                      "pBIC": {str(k): float(x) for k, x in v["pBIC"].items()}}
                  for t, v in point.items()},
              "frequencies": {}, "gap_mean": {}, "gap_sd": {}}
    for tag in ("fixed_phi", "free_phi"):
        total = sum(counts[tag].values())
        freq = {int(k): c / total for k, c in sorted(counts[tag].items())}
        result["frequencies"][tag] = {str(k): v for k, v in freq.items()}
        g = np.array(gaps[tag])
        result["gap_mean"][tag] = float(g.mean())
        result["gap_sd"][tag] = float(g.std(ddof=1)) if g.size > 1 else None
        pretty = ", ".join(f"P={k}: {100 * v:.0f}%" for k, v in freq.items())
        print(f"  {tag:9s}  {pretty}")
        print(f"    best-vs-second gap across replicates: "
              f"{g.mean():.1f} +- {g.std(ddof=1) if g.size > 1 else 0:.1f}")

    outpath = Path(args.out) if args.out else (
        HERE / f"bootstrap_pbic_{label.split(' ')[0].strip('()')}.json")
    outpath.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nwritten: {outpath}")


if __name__ == "__main__":
    main()
