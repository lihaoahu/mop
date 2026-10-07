"""Figure: distribution of dimension-labeled comparisons.

Draws the donut chart and writes manuscript/figures/dim_distribution.pdf. The
counts come from the same load_pairs the tables use, so the figure and the
percentages quoted in the text cannot disagree. The previous version of this
file was drawn from an earlier data snapshot and reported counts totalling
6,706 against the current 5,238.

Usage:
    uv run python _plot_dim_distribution.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import DIMS, DIM_LABELS, PROJECT
from step7_analyze import load_pairs

OUT = PROJECT / "manuscript" / "figures" / "dim_distribution.pdf"

TITLE = {
    "cost": "Cost",
    "driving_experience": "Driving Experience",
    "range": "Range",
    "refueling_convenience": "Refueling Convenience",
    "reliability": "Reliability",
    "policy": "Policy",
    "other": "Other",
}


def main():
    counts = {}
    for dim in DIMS:
        n = len(load_pairs(dim_name=dim))
        if n:
            counts[dim] = n
    total = sum(counts.values())

    # largest first, as the published figure orders them
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    labels = [TITLE.get(d, DIM_LABELS.get(d, d)) for d, _ in items]
    sizes = [n for _, n in items]

    print(f"  total dimension-labeled comparisons: {total}")
    for label, n in zip(labels, sizes):
        print(f"    {label:<22} {n:>6}  {100.0 * n / total:5.1f}%")

    fig, ax = plt.subplots(figsize=(7.36, 4.0))
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.axis("equal")
    ax.legend(
        wedges,
        [f"{lab} ({100.0 * n / total:.1f}%)" for lab, n in zip(labels, sizes)],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  wrote {OUT}")
    print(f"  wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
