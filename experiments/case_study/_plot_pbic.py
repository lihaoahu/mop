"""Figure: pBIC against the number of mixture components, overall dataset.

The curve is computed from the same fits that step7_analyze.run_mop performs for
the tables, with the same hyperparameters, so the figure cannot drift from the
numbers in the text. The values used to be hard-coded, which is how the plotted
minimum came to sit at a pBIC the text no longer quoted.

Usage:
    uv run python _plot_pbic.py            # K = 1..8, as plotted in the paper
    uv run python _plot_pbic.py --p-max 9
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import PROJECT
from step7_analyze import load_pairs, run_mop

OUT = PROJECT / "manuscript" / "figures" / "pbic_vs_p.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-max", type=int, default=8,
                    help="largest number of components to plot (default 8)")
    args = ap.parse_args()

    trace = {}
    best_k, best, _, n_j = run_mop(load_pairs(), trace=trace)
    bic = trace["bic_vals"]
    ks = list(range(1, args.p_max + 1))
    ys = [bic[k] for k in ks]

    print(f"  n = {n_j}")
    for k, v in zip(ks, ys):
        mark = "   <- minimum" if k == min(bic, key=bic.get) else ""
        print(f"  P={k}  pBIC={v:,.1f}{mark}")
    print(f"  selected by pBIC: P={min(bic, key=bic.get)}")

    k_min = min(bic, key=bic.get)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ks, ys, "o-", color="#2171b5", markersize=6, linewidth=1.5)
    ax.scatter([k_min], [bic[k_min]], color="#d73027", s=80, zorder=5)
    ax.set_xlabel("Number of components ($\\mathcal{P}$)", fontsize=11)
    ax.set_ylabel("pBIC", fontsize=11)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # CreationDate is omitted so that regenerating the figure leaves the
    # committed file unchanged; matplotlib writes the current time into it
    # otherwise, which made this the one artefact a re-run could never
    # reproduce byte for byte.
    plt.savefig(OUT, dpi=200, bbox_inches="tight",
                metadata={"CreationDate": None})
    print(f"  wrote {OUT}")
    if k_min != best_k:
        print(f"  WARNING: run_mop selected P={best_k} but the plotted "
              f"minimum is P={k_min}")


if __name__ == "__main__":
    main()
