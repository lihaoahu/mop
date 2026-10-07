"""生成论文所需全部 LaTeX 表格"""
import sys, json, math
import numpy as np
from scipy.special import logit
from mop import inference, pairwise_comparison
from mop.inference import comparisons2regression
from _common import ENERGY_PAIRS, TYPE_MAP, T, DIMS, DIM_LABELS

D = 100; EPS = 1e-10; TYPES = T

def load_pairs(dim_name=None, year=None):
    pairs = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if year and r.get("publish_time", "")[:4] != str(year):
                continue
            for p in r.get("comparison_pairs", []):
                if dim_name and p.get("dimension") != dim_name:
                    continue
                ft = TYPE_MAP.get(p.get("former_type", ""), "")
                lt = TYPE_MAP.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": ft, "lt": lt, "score": p.get("score", 0.5)})
    return pairs

def run_one(pairs):
    nJ = len(pairs)
    name_to_id = {t: i+1 for i,t in enumerate(TYPES)}
    pcvector = [pairwise_comparison.PairwiseComparison(
        former_id=name_to_id[p["ft"]], latter_id=name_to_id[p["lt"]],
        agent_id=1, judgment=p["score"]) for p in pairs]
    link_functions = [lambda s, _eps=EPS: logit(np.clip(s, _eps, 1-_eps))]
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA = X.shape[1]; lu = lpi = ls = max(nJ / D, 1.0)
    u_pool = np.linalg.solve(X.T @ X + 1e-10*np.eye(nA), X.T @ y)
    s2_bar = max(np.sum((X @ u_pool - y)**2) / max(nJ - nA + 1, 1), 1e-10)
    prev = None
    best_k, best_bic, best_result = None, np.inf, None
    for k in range(1, 7):
        kw = {}
        if prev is not None:
            pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
            ps = np.array([p.inconsistency_level for p in prev.priorities])
            ok = len(pu); nk = ok+1
            u_new = np.zeros((nk, nA)); u_new[:ok]=pu
            min_rse = np.full(X.shape[0], np.inf)
            for p in range(ok):
                min_rse = np.minimum(min_rse, (X @ pu[p] - y)**2)
            n_edge = min(nA+1, X.shape[0])
            idx = min_rse >= np.sort(min_rse)[-n_edge]
            if np.sum(idx)>0:
                Xe,ye=X[idx],y[idx]
                u_new[ok]=np.linalg.solve(Xe.T@Xe+1e-10*np.eye(nA), Xe.T@ye)
            else:
                u_new[ok]=np.random.randn(nA)*0.1
            s_new=np.ones(nk); s_new[:ok]=ps; s_new[ok]=np.mean(ps)
            pm=np.array(prev.mixture_probabilities)
            m_new=np.zeros(nk); m_new[:ok]=pm*(ok/nk); m_new[ok]=1.0/nk
            floor=1.0/(nk*20); m_new=np.maximum(m_new,floor); m_new/=np.sum(m_new)
            kw={"u_init":u_new,"sigma2_init":s_new,"mp_init":m_new}
        result = inference.mop(pcvector, link_functions, k,
                               lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls, **kw)
        val = inference.pbic(pcvector, result, link_functions,
                             lambda_u=lu, lambda_sigma=ls, lambda_pi=lpi, s2=s2_bar)
        if val < best_bic:
            best_bic=val; best_k=k; best_result=result
        prev=result
    return nJ, best_k, best_bic, best_result

ABBREV = {"纯电":"BEV","插混":"PHEV","增程":"REEV","燃油":"ICEV"}
TYPE_LABELS = [ABBREV[t] for t in TYPES]

def latex_table(result, best_k, caption, label, nJ, best_bic):
    """生成 LaTeX 表格字符串"""
    mp = result.mixture_probabilities
    nT = len(TYPES)
    
    rows = []
    for p in range(best_k):
        w = result[p].priority_vector
        s2 = result[p].inconsistency_level
        rows.append((f"C{p+1}", list(w), mp[p], s2))
    weighted = np.zeros(nT)
    for row in rows:
        weighted += np.array(row[1]) * row[2]
    rows.append(("Weighted", list(weighted), 1.0, None))
    
    cols = "l" + "c" * (nT + 2)
    header = " & ".join(["Component"] + TYPE_LABELS + ["$\\pi_p$", "$\\sigma_p^2$"]) + " \\\\"
    
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\t\\centering")
    lines.append(f"\t\\caption{{{caption}}}")
    lines.append(f"\t\\label{{{label}}}")
    lines.append(f"\t\\begin{{tabular}}{{{cols}}}")
    lines.append("\t\t\\toprule")
    lines.append(f"\t\t{header}")
    lines.append("\t\t\\midrule")
    for name, w, pi, s2 in rows:
        w_str = " & ".join(f"{v:.3f}" for v in w)
        pi_str = f"{pi:.4f}"
        s2_str = f"{s2:.4f}" if s2 is not None else "---"
        style = "\\bfseries " if name == "Weighted" else ""
        lines.append(f"\t\t{style}{name} & {w_str} & {pi_str} & {s2_str} \\\\")
    lines.append("\t\t\\bottomrule")
    lines.append("\t\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

if __name__ == "__main__":
    # 整体
    pairs = load_pairs()
    nJ, best_k, best_bic, result = run_one(pairs)
    print(latex_table(result, best_k,
        f"Mixture of preferences for the overall dataset ($n={nJ}$, $\\mathcal{{P}}={best_k}$, pBIC$={best_bic:.1f}$).",
        "tab:mop_overall", nJ, best_bic))
    print()
    
    # 分维度
    for dim in DIMS:
        pairs = load_pairs(dim_name=dim)
        if len(pairs) < 30: continue
        nJ, best_k, best_bic, result = run_one(pairs)
        dim_en = dim.replace("_"," ").title()
        print(latex_table(result, best_k,
            f"MoP for {dim_en} dimension ($n={nJ}$, $\\mathcal{{P}}={best_k}$, pBIC$={best_bic:.1f}$).",
            f"tab:dim_{dim}", nJ, best_bic))
        print()

    # 分年份整体
    for year in [2025, 2026]:
        pairs = load_pairs(year=year)
        nJ, best_k, best_bic, result = run_one(pairs)
        print(latex_table(result, best_k,
            f"MoP for {year} ($n={nJ}$, $\\mathcal{{P}}={best_k}$, pBIC$={best_bic:.1f}$).",
            f"tab:year_{year}", nJ, best_bic))
        print()
    
    # 分年份×维度
    for year in [2025, 2026]:
        for dim in DIMS:
            pairs = load_pairs(dim_name=dim, year=year)
            if len(pairs) < 30: continue
            nJ, best_k, best_bic, result = run_one(pairs)
            dim_en = dim.replace("_"," ").title()
            print(latex_table(result, best_k,
                f"MoP for {dim_en} dimension, {year} ($n={nJ}$, $\\mathcal{{P}}={best_k}$, pBIC$={best_bic:.1f}$).",
                f"tab:year{year}_{dim}", nJ, best_bic))
            print()
