"""The pBIC curve from a search that finds each count's best optimum.

**Read the specification before comparing any two numbers here.** The first version of this
script plotted the fixed-dispersion curve beside the chain's values and made it look as
though the paper's minimum at P = 4 were an artefact of its search. It was not: the paper
estimates the dispersion, `inference.mop` estimates it, and the chain's own mean phi runs
1.88, 1.80, 1.20, 1.00, 1.02 across P = 1..5, so the paper's curve IS the estimated-dispersion
curve. Comparing it against a fixed-dispersion curve compared two different models.

Under the estimated dispersion the best-optimum search agrees with the paper: the minimum is
at P = 4. That is the headline result and it is a confirmation, not a correction.

Figure `fig:pbic` in the paper is drawn from `step7_analyze.run_mop`, which warm-starts the
component search from P = 1 upward. This script checks that curve against the best optimum
available at each count, over K-Ridge starts, warm starts from counts with at least as many
components, the sequential chain, and jittered restarts -- and reports both specifications
side by side so the two are never confused.

Run:
    uv run python experiments/case_study/_pbic_curve.py
    uv run python experiments/case_study/_pbic_curve.py --restarts 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import selection as SEL                                              # noqa: E402
from mop.inference import prior_mode_phi0                           # noqa: E402

from _common import PROJECT                                          # noqa: E402
from step7_analyze import load_pairs, run_mop                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--p-max", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = load_pairs()
    X, y = SEL.build_design(pairs)
    A = X.shape[1] + 1
    s2 = float(prior_mode_phi0(X, y))
    grid = list(range(1, args.p_max + 1))

    # The paper's path, as a reproduction check.
    trace = {}
    best_k, _, _, n_j = run_mop(pairs, trace=trace)
    chain = dict(trace["bic_vals"])
    print(f"n = {n_j}, A = {A}, phi_0 = {s2:.4f}")
    print(f"paper's path (run_mop, warm start from P=1): selects P={best_k}\n")

    curve = {}
    for tag, free in (("fixed", False), ("free", True)):
        curve[tag] = SEL.select_best(X, y, grid, free, s2,
                                     restarts=args.restarts, seed=1)

    print(f"  {'P':>3} | {'chain (paper)':>14} | {'fixed best':>12} {'free best':>12} "
          f"| {'chain-fixed':>12}")
    print("-" * 74)
    rows = []
    for k in grid:
        c = chain.get(k)
        f = curve["fixed"][k]
        g = curve["free"][k]
        rows.append({"P": k, "chain": c, "fixed_best": f["pBIC"],
                     "free_best": g["pBIC"], "fixed_pll": f["pll"],
                     "free_pll": g["pll"], "chain_minus_best": (c - f["pBIC"])
                     if c is not None else None})
        cs = f"{c:,.1f}" if c is not None else "---"
        d = f"{c - f['pBIC']:+,.1f}" if c is not None else "---"
        print(f"  {k:>3} | {cs:>14} | {f['pBIC']:>12,.1f} {g['pBIC']:>12,.1f} "
              f"| {d:>12}")

    best_fixed = min(grid, key=lambda k: curve["fixed"][k]["pBIC"])
    best_free = min(grid, key=lambda k: curve["free"][k]["pBIC"])
    chain_best = min(chain, key=chain.get)
    print()
    print(f"  paper's chain minimum      : P={chain_best}")
    print(f"  best-optimum fixed minimum : P={best_fixed}")
    print(f"  best-optimum free minimum  : P={best_free}")
    print()
    moved = (chain_best != best_fixed)
    print(f"  the minimum {'MOVES' if moved else 'does not move'} between the two "
          f"searches (fixed dispersion)")
    if moved:
        gap = curve["fixed"][best_fixed]["pBIC"] - curve["fixed"][chain_best]["pBIC"]
        print(f"  at P={best_fixed} the best-optimum curve is {abs(gap):,.1f} units "
              f"{'below' if gap < 0 else 'above'} the chain's value at P={chain_best}")

    # Report where the chain is materially worse than the best optimum found.
    worse = [(k, chain[k] - curve["fixed"][k]["pBIC"]) for k in grid
             if k in chain and chain[k] - curve["fixed"][k]["pBIC"] > 5]
    if worse:
        print()
        print("  counts where the chain is more than 5 units worse than the best found:")
        for k, d in worse:
            print(f"    P={k}: chain is {d:,.1f} worse")

    out = Path(args.out) if args.out else (HERE / "pbic_curve.json")
    out.write_text(json.dumps(
        {"n": int(n_j), "A": int(A), "phi0": s2, "restarts": args.restarts,
         "chain_selected": int(chain_best), "chain": {str(k): v for k, v in chain.items()},
         "best_optimum_selected": {"fixed": int(best_fixed), "free": int(best_free)},
         "rows": [{**r} for r in rows]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")

    # The corrected figure, as a separate file. The paper's figure is left alone.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ks = grid
        ax.plot(ks, [x["fixed_best"] for x in rows], "o-", color="#2171b5",
                markersize=6, linewidth=1.5, label="best optimum at each $\\mathcal{P}$")
        ax.plot(ks, [x["chain"] for x in rows], "s--", color="#969696", markersize=5,
                linewidth=1.2, label="warm start from $\\mathcal{P}=1$")
        ax.scatter([best_fixed], [curve["fixed"][best_fixed]["pBIC"]],
                   color="#d73027", s=90, zorder=5)
        ax.set_xlabel("Number of components ($\\mathcal{P}$)", fontsize=11)
        ax.set_ylabel("pBIC", fontsize=11)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        figout = HERE / "pbic_curves.pdf"
        figout.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figout, dpi=200, bbox_inches="tight",
                    metadata={"CreationDate": None})
        print(f"wrote {figout}  (the paper's pbic_vs_p.pdf is NOT overwritten)")
    except ImportError:
        print("matplotlib unavailable; figure skipped")


if __name__ == "__main__":
    main()
