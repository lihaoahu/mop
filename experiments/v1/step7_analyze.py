"""Step 7: MoP 分析（整体 + 七维度 + 年份对比）。

用法:
    uv run python experiments/v1/step7_analyze.py              # 全部分析
    uv run python experiments/v1/step7_analyze.py --dim cost   # 单维度
"""

import argparse, json
import numpy as np
from scipy.special import logit
from mop import inference, pairwise_comparison
from mop.inference import comparisons2regression

from _common import CFG, ENERGY_PAIRS, MOP_RESULTS, TYPE_MAP, T, DIMS, DIM_LABELS




def load_pairs(dim_name=None, year=None):
    pairs = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if year and r.get("publish_time", "")[:4] != year:
                continue
            for p in r.get("comparison_pairs", []):
                if dim_name and p.get("dimension") != dim_name:
                    continue
                ft = TYPE_MAP.get(p.get("former_type", ""), "")
                lt = TYPE_MAP.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": ft, "lt": lt,
                                  "score": p.get("score", 0.5)})
    return pairs


def warm_init(prev, X, y, nA):
    pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
    ps = np.array([p.inconsistency_level for p in prev.priorities])
    ok = len(pu); nk = ok + 1
    u_new = np.zeros((nk, nA)); u_new[:ok] = pu
    min_rse = np.full(X.shape[0], np.inf)
    for p in range(ok):
        min_rse = np.minimum(min_rse, (X @ pu[p] - y) ** 2)
    n_edge = min(nA + 1, X.shape[0])
    idx = min_rse >= np.sort(min_rse)[-n_edge]
    if np.sum(idx) > 0:
        Xe, ye = X[idx], y[idx]
        u_new[ok] = np.linalg.solve(
            Xe.T @ Xe + 1e-10 * np.eye(nA), Xe.T @ ye)
    else:
        u_new[ok] = np.random.randn(nA) * 0.1
    s_new = np.ones(nk); s_new[:ok] = ps; s_new[ok] = np.mean(ps)
    pm = np.array(prev.mixture_probabilities)
    for p in range(ok):
        if pm[p] < 0.01:
            u_new[p] = np.random.randn(nA) * 0.1
            s_new[p] = np.mean(ps)
    m_new = np.zeros(nk)
    m_new[:ok] = pm * (ok / nk); m_new[ok] = 1.0 / nk
    floor = 1.0 / (nk * 20)
    m_new = np.maximum(m_new, floor); m_new /= np.sum(m_new)
    return u_new, s_new, m_new


def run_mop(pairs):
    nJ = len(pairs)
    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    pcvector = [pairwise_comparison.PairwiseComparison(
        former_id=name_to_id[p["ft"]], latter_id=name_to_id[p["lt"]],
        agent_id=1, judgment=p["score"]) for p in pairs]
    eps = 1e-10
    link_functions = [lambda s, _eps=eps: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA = X.shape[1]; lu = 1e-6; lpi = 1; ls = nJ / 50
    # 先验方差模式 s̄² (Eq. general gci)
    u_pool = np.linalg.solve(X.T @ X + 1e-10 * np.eye(nA), X.T @ y)
    s2_bar = max(np.sum((X @ u_pool - y) ** 2) / (nJ - nA + 1), 1e-10)

    prev = None
    models, bic_vals = {}, {}
    for k in range(1, 10):
        kw = {}
        if prev is not None:
            ui, si, mi = warm_init(prev, X, y, nA)
            kw = {"u_init": ui, "sigma2_init": si, "mp_init": mi}
        result = inference.mop(pcvector, link_functions, k,
                               lambda_u=lu, lambda_pi=lpi,
                               lambda_sigma=ls, **kw)
        bic_val = inference.pbic(pcvector, result, link_functions,
                                 lambda_u=lu, lambda_sigma=ls,
                                 lambda_pi=lpi, s2=s2_bar)
        models[str(k)] = {**result.to_dict(), "bic": bic_val}
        bic_vals[k] = bic_val
        prev = result

    best_k = min(bic_vals, key=bic_vals.get)
    best = models[str(best_k)]
    weighted = np.zeros(4)
    for pi, pri in zip(best["mixture"]["pis"],
                       best["mixture"]["priorities"]):
        weighted += np.array(pri) * pi
    return best_k, best, weighted, nJ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", help="单维度分析")
    args = parser.parse_args()

    if args.dim:
        _analyze_one(args.dim, args.dim)
        return

    # 1. 整体
    bk, best, w, nJ = run_mop(load_pairs())
    ranked = sorted(zip(T, w), key=lambda x: -x[1])
    print(f"[整体] n={nJ}  K={bk}  pBIC={best['bic']:.1f}")
    print(f"  加权: {' > '.join(f'{t}({v:.3f})' for t, v in ranked)}")
    for pi, pri in zip(best["mixture"]["pis"],
                       best["mixture"]["priorities"]):
        r = sorted(zip(T, pri), key=lambda x: -x[1])
        print(f"    C(pi={pi:.4f}): {' > '.join(f'{t}({v:.3f})' for t, v in r)}")

    # 2. 维度分解
    print(f"\n{'='*60}\n维度分解\n{'='*60}")
    for dim in DIMS:
        _analyze_one(dim, f"  {DIM_LABELS[dim]} ({dim})")

    # 3. 年份对比
    print(f"\n{'='*60}\n年份对比\n{'='*60}")
    for year in ["2025", "2026"]:
        _analyze_one(f"{year}", f"  {year}年", year=year)


def _analyze_one(label, display, year=None):
    pairs = load_pairs(dim_name=(None if label in ("2025", "2026")
                                 else label), year=year)
    if len(pairs) < 30:
        print(f"[{display}] 数据不足 (n={len(pairs)})")
        return
    bk, best, w, nJ_ = run_mop(pairs)
    ranked = sorted(zip(T, w), key=lambda x: -x[1])
    r_str = " > ".join(f"{t}({v:.3f})" for t, v in ranked)
    print(f"[{display}] n={nJ_}  K={bk}  pBIC={best['bic']:.1f}  {r_str}")


if __name__ == "__main__":
    main()
