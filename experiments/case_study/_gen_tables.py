"""Emit the LaTeX tables of section 3.2 (case study).

Every number comes from ``step7_analyze.run_mop``, so the tables and the
running text quote the same fits and the same hyperparameters. The aggregate
row is the expected utility defined in section 2 of the manuscript
(``_common.expected_utility``), not a vote score.

Usage:
    uv run python _gen_tables.py                   # print the LaTeX
    uv run python _gen_tables.py --json out.json   # also dump the raw numbers

The three tables written are tab:mop_overall, tab:dimension_mop and
tab:year_overall.
"""

import argparse, json
import numpy as np

from _common import T, DIMS, expected_utility  # noqa: F401
from step7_analyze import load_pairs, run_mop

ABBREV = {"纯电": "BEV", "插混": "PHEV", "增程": "REEV", "燃油": "ICEV"}
LABELS = [ABBREV[t] for t in T]

# a thousands separator that does not trip LaTeX's math-mode comma spacing
def thousands(n):
    """5,401 -> 5{,}401 (int) or 18345.8 -> 18{,}345.8 (float, one decimal)."""
    s = f"{n:,.1f}" if isinstance(n, float) else f"{n:,}"
    return s.replace(",", "{,}")


def fit_block(pairs):
    """Fit one dataset and return everything the tables need."""
    best_k, best, u_bar, nJ = run_mop(pairs)
    mix = best["mixture"]
    return {
        "n": int(nJ),
        "k": int(best_k),
        "pbic": float(best["bic"]),
        "components": [
            {"weights": [float(x) for x in pri],
             "pi": float(pi),
             "phi": float(phi)}
            for pri, pi, phi in zip(mix["priorities"], mix["pis"],
                                    mix["inconsistencies"])
        ],
        "expected_utility": [float(x) for x in u_bar],
    }


def _cells(values, bold_top):
    top = int(np.argmax(values)) if bold_top else -1
    out = []
    for i, v in enumerate(values):
        s = f"{v:.3f}"
        out.append(f"\\textbf{{{s}}}" if i == top else s)
    return " & ".join(out)


def component_row(name, c, bold_top=True):
    return (f"{name} & {_cells(c['weights'], bold_top)} & "
            f"{c['pi']:.4f} & {c['phi']:.4f} \\\\")


def eu_row(u_bar):
    return (f"\\textbf{{Expected Utility}} & {_cells(u_bar, True)} & "
            f"--- & --- \\\\")


HEADER = "Component & BEV & PHEV & REEV & ICEV & $\\pi_p$ & $\\phi_p$ \\\\"


def table_overall(blk):
    lines = [
        "\\begin{table}[ht]",
        "\t\\centering",
        (f"\t\\caption{{Four-component mixture of preferences for the overall "
         f"dataset ($n={thousands(blk['n'])}$, pBIC$={thousands(blk['pbic'])}$, "
         f"$\\lambda_u{{=}}10^{{-6}}$, $\\lambda_\\pi{{=}}1$, "
         f"$\\lambda_\\sigma{{=}}n/50{{\\approx}}{blk['n'] // 50}$).}}"
         f"\\label{{tab:mop_overall}}"),
        "\t\\begin{tabular}{lcccccc}\\toprule",
        "\t\t" + HEADER,
        "\t\t\\midrule",
    ]
    for i, c in enumerate(blk["components"], start=1):
        lines.append(component_row(f"C{i}", c))
    lines.append("\\midrule")
    lines.append(eu_row(blk["expected_utility"]))
    lines += ["\\bottomrule", "\t\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_dimensions(blocks):
    lines = [
        "\\begin{table}[ht]",
        "\t\\centering",
        ("\t\\caption{Dimensional decomposition of the MoP model "
         "($\\lambda_u{=}10^{-6}$, $\\lambda_\\pi{=}1$, $\\lambda_\\sigma{=}n/50$).}"
         "\\label{tab:dimension_mop}\\small"),
        "\t\\begin{tabular}{lcccccc}\\toprule",
        "\t\t" + HEADER,
        "\t\t\\midrule",
    ]
    for j, (dim, blk) in enumerate(blocks):
        title = dim.replace("_", " ").title()
        lines.append(
            f"\t\t\\multicolumn{{7}}{{l}}{{\\textbf{{{title}}}  "
            f"($n{{=}}{thousands(blk['n'])}$, $\\mathcal{{P}}^*{{=}}{blk['k']}$, "
            f"pBIC${{=}}{thousands(blk['pbic'])}$)}} \\\\")
        for i, c in enumerate(blk["components"], start=1):
            lines.append(component_row(f"C{i}", c))
        lines.append(eu_row(blk["expected_utility"]))
        if j < len(blocks) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\t\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_years(years):
    lines = [
        "\\begin{table}[ht]",
        "\t\\centering",
        "\t\\caption{MoP for 2025 and 2026: overall component-level results.}",
        "\t\\label{tab:year_overall}",
        "\t\\begin{tabular}{lcccccc}",
        "\t\t\\toprule",
        "\t\t" + HEADER,
        "\t\t\\midrule",
    ]
    for j, (year, blk) in enumerate(years):
        lines.append(
            f"\t\t\\multicolumn{{7}}{{l}}{{\\textbf{{{year}}}  "
            f"($n={thousands(blk['n'])}$, $\\mathcal{{P}}^*={blk['k']}$, "
            f"pBIC$={thousands(blk['pbic'])}$)}} \\\\")
        for i, c in enumerate(blk["components"], start=1):
            lines.append(component_row(f"C{i}        ", c))
        lines.append(eu_row(blk["expected_utility"]))
        if j < len(years) - 1:
            lines.append("\t\t\\midrule")
    lines += ["\t\t\\bottomrule", "\t\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the raw numbers here")
    args = ap.parse_args()

    out = {}

    banner = lambda s: print("%" + "=" * 74 + f"\n%% {s}\n" + "%" + "=" * 74)  # noqa: E731

    banner("tab:mop_overall")
    overall = fit_block(load_pairs())
    out["overall"] = overall
    print(table_overall(overall))
    print()

    banner("tab:dimension_mop")
    blocks = []
    for dim in DIMS:
        pairs = load_pairs(dim_name=dim)
        if len(pairs) < 30:
            continue
        blocks.append((dim, fit_block(pairs)))
    out["dimensions"] = {d: b for d, b in blocks}
    print(table_dimensions(blocks))
    print()

    banner("tab:year_overall")
    years = []
    for year in ["2025", "2026"]:
        pairs = load_pairs(year=year)
        years.append((year, fit_block(pairs)))
    out["years"] = {y: b for y, b in years}
    print(table_years(years))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n% raw numbers -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
