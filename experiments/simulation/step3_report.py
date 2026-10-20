"""Step 3: Report — text + LaTeX from simulation (A/C/D) + holdout.

Usage:
    uv run python experiments/simulation/step3_report.py
"""

import json
import numpy as np
from pathlib import Path

from _common import (
    SIM_DIR, SIM_RESULTS, HOLDOUT_RESULTS, REPORT_DIR,
    priority_vector_from_logratios,
    generate_cancellation, generate_noise_ground_truth,
)


def _mean_std_str(d, key):
    if key in d and "mean" in d[key]:
        return f"{d[key]['mean']:.4f} ± {d[key]['std']:.4f}"
    return "---"


def _val(d, key):
    if key in d and "mean" in d[key]:
        return d[key]["mean"]
    return None


def _format_delta(mop_val, sk1_val, lower_is_better=False):
    if mop_val is None or sk1_val is None:
        return "---"
    delta = mop_val - sk1_val
    better = (delta < 0) if lower_is_better else (delta > 0)
    sign = "+" if delta >= 0 else ""
    marker = "\\checkmark" if better else ""
    return f"{sign}{delta:.4f} {marker}"


def _latex_table(caption, label, headers, rows, notes=None):
    ncol = len(headers)
    colspec = "l" + "c" * (ncol - 1)
    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
        "    " + " & ".join(headers) + " \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(str(c) for c in row) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    if notes:
        lines.append(f"  \\smallskip\n  \\footnotesize{{{notes}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_text_report(sim_data, holdout_data):
    lines = ["=" * 70, "MoP Validation — Summary", "=" * 70]

    labels = {
        "cancellation": "Cancellation — opposite preferences cancel",
        "general": "General recovery — MoP never worse",
        "noise": "Noise — MoP isolates random raters",
    }
    for key in ["cancellation", "general", "noise"]:
        if key not in sim_data:
            continue
        sd = sim_data[key]
        keys = sorted(sd.keys())
        mid_key = keys[len(keys) // 2]
        agg = sd[mid_key]
        lines.append(f"\n## {labels.get(key, key)}")
        lines.append(f"  ({mid_key})")
        for method in ["MoP", "P=1", "BTL"]:
            if method in agg and "cosine_sim" in agg[method]:
                m = agg[method]
                ka = f"{m['k_accuracy']:.3f}" if "k_accuracy" in m else "  ---"
                kl = f"{m['kl_pi']['mean']:.4f}" if "kl_pi" in m else "  ---"
                lines.append(
                    f"  {method:<14} CosSim={m['cosine_sim']['mean']:.4f}  "
                    f"KL(pi)={kl}  "
                    f"Kendall={m['kendall_tau']['mean']:.4f}  "
                    f"K_acc={ka}  k_mean={m.get('k_mean',0):.2f}"
                )

    if "results" in holdout_data or "overall" in holdout_data:
        hr = holdout_data.get("overall", holdout_data.get("results", {}))
        n_pairs = holdout_data.get("n_pairs_total", holdout_data.get("n_pairs", "?"))
        lines.append(f"\n## Holdout ({n_pairs} pairs)")
        lines.append("  SUPERSEDED by step5_holdout.py. step2's rule picks the component")
        lines.append("  whose prediction is closest to the observed value, so it reads the")
        lines.append("  label it predicts and its MoP column is an oracle selection, not a")
        lines.append("  forecast. The paper reports step5's marginal-likelihood numbers.")
        for method in ["MoP", "P=1", "BTL"]:
            if method in hr and "mse" in hr[method]:
                m = hr[method]
                lines.append(
                    f"  {method:<14} MSE={m['mse']['mean']:.5f}±{m['mse']['std']:.5f}  "
                    f"MAE={m['mae']['mean']:.5f}  r={m['pearson_r']['mean']:.4f}  "
                    f"ρ={m['spearman_rho']['mean']:.4f}"
                )
        if "MoP" in hr and "P=1" in hr:
            mop_mse = _val(hr["MoP"], "mse")
            p1_mse = _val(hr["P=1"], "mse")
            if mop_mse and p1_mse:
                imp = (p1_mse - mop_mse) / p1_mse * 100
                lines.append(f"\n  → MoP MSE improvement: {imp:.1f}%")

    return "\n".join(lines)


def generate_simulation_tables(sim_data):
    all_latex = ["% ── Ground truth illustrations ──"]

    u_a, pi_a = generate_cancellation()
    w_a = priority_vector_from_logratios(u_a)
    all_latex.append(_scenario_table("Scenario A: Cancellation", w_a, pi_a))
    all_latex.append("")

    u_d, pi_d = generate_noise_ground_truth()
    w_d = priority_vector_from_logratios(u_d)
    all_latex.append(_scenario_table("Scenario D: Noise (true preference)", w_d, pi_d))
    all_latex.append("")

    for key in sorted(sim_data.keys()):
        sd = sim_data[key]
        label = {"cancellation": "Cancellation", "general": "General",
                 "noise": "Noise"}.get(key, key)
        all_latex.append(f"% ── {label} ──")
        for cond in sorted(sd.keys()):
            agg = sd[cond]
            header = ["Condition", "Method", "CosSim↑", "KL(π)↓",
                      "Kendall τ↑", "K acc.", r"$\hat{K}$"]
            rows = []
            for method in ["MoP", "P=1", "BTL"]:
                if method in agg and "cosine_sim" in agg[method]:
                    m = agg[method]
                    ka = f"{m['k_accuracy']:.3f}" if "k_accuracy" in m else "---"
                    rows.append([
                        cond if method == "MoP" else "", method,
                        f"{m['cosine_sim']['mean']:.4f}",
                        f"{m['kl_pi']['mean']:.4f}" if 'kl_pi' in m else "---",
                        f"{m['kendall_tau']['mean']:.4f}",
                        ka,
                        f"{m.get('k_mean',0):.2f}",
                    ])
            cap = f"{label}, {cond}."
            lbl = f"tab:sim_{key}_{cond.replace('=','').replace('.','_')}"
            all_latex.append(_latex_table(cap, lbl, header, rows))
            all_latex.append("")
    return "\n".join(all_latex)


def _scenario_table(name, w, pi):
    nA = w.shape[1]
    labels = [f"$A_{{{i+1}}}$" for i in range(nA)]
    header = ["Component"] + labels + ["$\\pi_p$"]
    rows = []
    for p in range(len(pi)):
        row = [f"$C_{{{p+1}}}$"] + [f"{w[p,i]:.3f}" for i in range(nA)] + [f"{pi[p]:.3f}"]
        rows.append(row)
    w_avg = np.average(w, axis=0, weights=pi)
    rows.append(["Weighted"] + [f"{w_avg[i]:.3f}" for i in range(nA)] + ["1.000"])
    return _latex_table(f"Ground truth: {name}.", f"tab:gt_{name.lower().replace(' ','_')[:30]}", header, rows)


def generate_holdout_table(holdout_data):
    results = holdout_data.get("overall", holdout_data.get("results", {}))
    n_folds = holdout_data.get("n_folds", "?")
    n_pairs = holdout_data.get("n_pairs_total", holdout_data.get("n_pairs", "?"))
    header = ["Method", "MSE↓", "MAE↓", "Pearson r↑", "Spearman ρ↑"]
    rows = []
    for method in ["MoP", "P=1", "BTL"]:
        if method in results:
            m = results[method]
            rows.append([method, _mean_std_str(m, "mse"), _mean_std_str(m, "mae"),
                         _mean_std_str(m, "pearson_r"), _mean_std_str(m, "spearman_rho")])
    if "MoP" in results and "P=1" in results:
        mop, p1 = results["MoP"], results["P=1"]
        rows.append([
            "$\\Delta$ (MoP $-$ P=1)",
            _format_delta(_val(mop, "mse"), _val(p1, "mse"), lower_is_better=True),
            _format_delta(_val(mop, "mae"), _val(p1, "mae"), lower_is_better=True),
            _format_delta(_val(mop, "pearson_r"), _val(p1, "pearson_r")),
            _format_delta(_val(mop, "spearman_rho"), _val(p1, "spearman_rho")),
        ])
    cap = (f"Holdout prediction ($n={n_pairs}$, {n_folds}-fold CV). Mean $\\pm$ std. "
           "Selected by the oracle rule of \\texttt{step2\\_holdout.py}, which reads the "
           "label it predicts: these are not the figures Section~3.2 reports, which come "
           "from \\texttt{step5\\_holdout.py}.")
    return _latex_table(cap, "tab:holdout", header, rows)


def main():
    sim_data = json.load(open(SIM_RESULTS, encoding="utf-8")) if SIM_RESULTS.exists() else {}
    holdout_data = json.load(open(HOLDOUT_RESULTS, encoding="utf-8")) if HOLDOUT_RESULTS.exists() else {}
    print(f"Simulation: {len(sim_data)} scenarios | Holdout: {'yes' if holdout_data else 'no'}")

    report = generate_text_report(sim_data, holdout_data)
    (REPORT_DIR / "validation_report.txt").write_text(report, encoding="utf-8")
    print(f"\n{report}")

    if sim_data:
        (REPORT_DIR / "simulation_tables.tex").write_text(generate_simulation_tables(sim_data), encoding="utf-8")
    if holdout_data:
        (REPORT_DIR / "holdout_table.tex").write_text(generate_holdout_table(holdout_data), encoding="utf-8")
    print("\nLaTeX tables saved to report/")


if __name__ == "__main__":
    main()
