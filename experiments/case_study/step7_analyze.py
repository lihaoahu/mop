"""Step 7: MoP 分析（整体 + 七维度 + 年份对比）。

用法:
    uv run python experiments/case_study/step7_analyze.py              # 全部分析
    uv run python experiments/case_study/step7_analyze.py --dim cost   # 单维度
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.special import logit
from mop import inference, pairwise_comparison
from mop.initialization import kridge
from mop.inference import comparisons2regression

# The MoP fitting parameters live in one file, experiments/mop_params.json, read
# through experiments/_mop_params.py. They used to be hard-coded here (and in four
# other scripts), while config.json documented them without any script reading it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mop_params import (max_components_for, lambda_u, lambda_pi, lambda_sigma_for,   # noqa: E402
                         max_components, summary)

from _common import (CFG, ENERGY_PAIRS, MOP_RESULTS, TYPE_MAP, T, DIMS,
                     DIM_LABELS, expected_utility)




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


def _fit_kridge(pcvector, link_functions, k, lu, lpi, ls, X, y):
    """Fit ``k`` components from the K-Ridge initialiser, from scratch.

    The initialiser seeds the components on the judgments the current mixture explains least,
    refines them by alternating assignment and ridge re-estimation, and hands the result to the
    EM algorithm. It is deterministic: the same data give the same starting point and the same
    fit, which is what makes the selected count reproducible. The grid does not carry a fit
    forward from a smaller count; a chained start inherits the local optimum the smaller fit
    reached, and on this case study that changed the selected count on two dimensions.
    """
    u0, phi0, mp0 = kridge(X, y, k)
    result = inference.mop(pcvector, link_functions, k, lambda_u=lu, lambda_pi=lpi,
                           lambda_sigma=ls, u_init=u0, sigma2_init=phi0, mp_init=mp0)
    bic_val = float(inference.pbic(pcvector, result, link_functions,
                                   lambda_u=lu, lambda_sigma=ls, lambda_pi=lpi))
    return result, bic_val


def run_mop(pairs, trace=None):
    """MoP 拟合，K 从 1 搜到 mop_params.json 的 max_components。

    trace: 可选 dict；给了就写入 trace["bic_vals"]（每个 K 的 pBIC）和
    trace["models"]（每个 K 的完整结果），供画图和审计使用。
    """
    nJ = len(pairs)
    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    pcvector = [pairwise_comparison.PairwiseComparison(
        former_id=name_to_id[p["ft"]], latter_id=name_to_id[p["lt"]],
        agent_id=1, judgment=p["score"]) for p in pairs]
    eps = 1e-10
    link_functions = [lambda s, _eps=eps: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA = X.shape[1]
    lu, lpi, ls = lambda_u, lambda_pi, lambda_sigma_for(nJ)
    # φ₀ (Eq. general gci) 由 inference.prior_mode_phi0 定义一次，拟合与 pBIC 共用

    prev = None
    models, bic_vals = {}, {}
    for k in range(1, max_components_for(nA) + 1):
        result, bic_val = _fit_kridge(pcvector, link_functions, k, lu, lpi, ls, X, y)
        models[str(k)] = {**result.to_dict(), "bic": bic_val}
        bic_vals[k] = bic_val

        # If minprior is turned on and a candidate loses a component to it,
        # raising the candidate further can only reproduce that mixture, so the
        # search stops there. With the default of no removal the test never
        # fires and the whole grid is searched.
        if len(result) < k:
            break

    best_k = min(bic_vals, key=bic_vals.get)
    best = models[str(best_k)]
    weighted = expected_utility(best["mixture"]["priorities"],
                                best["mixture"]["pis"])
    if trace is not None:
        trace["bic_vals"] = dict(bic_vals)
        trace["models"] = models
    return best_k, best, weighted, nJ


def _report(display, n_j, best_k, best):
    """Print one fit in the form the paper's tables are set from.

    Tables 5, 6 and 8 carry the mixture weight, the inconsistency level and the
    priority vector of every component, and this script is their only source, so
    it prints all three rather than the aggregate alone. The component lines are
    in the canonical type order, which is the column order of those tables.
    """
    pis = best["mixture"]["pis"]
    phis = best["mixture"]["inconsistencies"]
    ws = best["mixture"]["priorities"]
    u_bar = expected_utility(ws, pis)
    ranked = sorted(zip(T, u_bar), key=lambda x: -x[1])

    print(f"[{display}] n={n_j}  K={best_k}  pBIC={best['bic']:.1f}")
    for i, (pi, phi, w) in enumerate(zip(pis, phis, ws), start=1):
        cells = "  ".join(f"{t}={v:.3f}" for t, v in zip(T, w))
        print(f"    C{i}  pi={pi:.4f}  phi={phi:.4f}  {cells}")
    print(f"    期望效用: {' > '.join(f'{t}({v:.3f})' for t, v in ranked)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", help="单维度分析")
    args = parser.parse_args()

    if args.dim:
        _analyze_one(args.dim, args.dim)
        return

    # 1. 整体
    bk, best, _, n_j = run_mop(load_pairs())
    _report("整体", n_j, bk, best)

    # 2. 维度分解
    print(f"\n{'='*60}\n维度分解\n{'='*60}")
    for dim in DIMS:
        _analyze_one(dim, f"{DIM_LABELS[dim]} ({dim})")

    # 3. 年份对比
    print(f"\n{'='*60}\n年份对比\n{'='*60}")
    for year in ["2025", "2026"]:
        _analyze_one(f"{year}", f"{year}年", year=year)


def _analyze_one(label, display, year=None):
    pairs = load_pairs(dim_name=(None if label in ("2025", "2026")
                                 else label), year=year)
    if len(pairs) < 30:
        print(f"[{display}] 数据不足 (n={len(pairs)})")
        return
    bk, best, _, n_j = run_mop(pairs)
    _report(display, n_j, bk, best)


if __name__ == "__main__":
    main()
