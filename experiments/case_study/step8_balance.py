"""Step 8b: 维度平衡后的整体 MoP —— bootstrap 平衡各维度样本再合并。

原始整体 MoP 中 Cost 占 1,570/5,238 ≈ 30%，Policy 仅 281/5,238 ≈ 5%。
这里将各维度（除 Other）bootstrap 到相同样本量后合并，重新拟合 MoP，
比较平衡前后整体偏好结构的差异。
"""

import json, math, argparse
import numpy as np
from scipy.special import logit, erf

from mop import inference, pairwise_comparison
from mop.inference import comparisons2regression

from _common import T, DIMS, ENERGY_PAIRS, TYPE_MAP
from step7_analyze import load_pairs


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def vote_scores(weights_list, pi_list, sigma2_list):
    nA = len(weights_list[0])
    S = [0.0] * nA
    for w, pi, sigma2 in zip(weights_list, pi_list, sigma2_list):
        sigma = math.sqrt(max(sigma2, 1e-10))
        for i in range(nA):
            s_i = 0.0
            for j in range(nA):
                if i == j: continue
                wi, wj = w[i], w[j]
                if wi <= 0 or wj <= 0:
                    if wi > 0 and wj == 0: s_i += 1.0
                    elif wi == 0 and wj > 0: s_i += 0.0
                    else: s_i += 0.5
                else:
                    s_i += phi(math.log(wi / wj) / sigma)
            S[i] += pi * s_i
    norm = 2.0 / (nA * (nA - 1))
    return [norm * s for s in S]


def fit_mop(pairs, fixed_k=None):
    """MoP 拟合。fixed_k=None 时搜索 K=1..9；否则固定 K。"""
    nJ = len(pairs)
    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    pcvector = [pairwise_comparison.PairwiseComparison(
        former_id=name_to_id[p["ft"]], latter_id=name_to_id[p["lt"]],
        agent_id=1, judgment=p["score"]) for p in pairs]
    eps = 1e-10
    link_functions = [lambda s, _eps=eps: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA = X.shape[1]; lu = 1e-6; lpi = 1; ls = nJ / 50
    u_pool = np.linalg.solve(X.T @ X + 1e-10 * np.eye(nA), X.T @ y)
    s2_bar = max(np.sum((X @ u_pool - y) ** 2) / (nJ - nA + 1), 1e-10)

    if fixed_k is not None:
        result = inference.mop(pcvector, link_functions, fixed_k,
                               lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls)
        bic_val = inference.pbic(pcvector, result, link_functions,
                                 lambda_u=lu, lambda_sigma=ls,
                                 lambda_pi=lpi, s2=s2_bar)
        return result, fixed_k, bic_val

    # 搜索 K
    prev = None
    bic_vals = {}
    for k in range(1, 10):
        kw = {}
        if prev is not None:
            pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
            ps = np.array([p.inconsistency_level for p in prev.priorities])
            pm = np.array(prev.mixture_probabilities)
            ok = len(pu); nk = ok + 1
            u_new = np.zeros((nk, nA)); u_new[:ok] = pu
            min_rse = np.full(X.shape[0], np.inf)
            for p in range(ok):
                min_rse = np.minimum(min_rse, (X @ pu[p] - y) ** 2)
            n_edge = min(nA + 1, X.shape[0])
            idx = min_rse >= np.sort(min_rse)[-n_edge]
            if np.sum(idx) > 0:
                u_new[ok] = np.linalg.solve(
                    X[idx].T @ X[idx] + 1e-10 * np.eye(nA), X[idx].T @ y[idx])
            else:
                u_new[ok] = np.random.randn(nA) * 0.1
            s_new = np.ones(nk); s_new[:ok] = ps; s_new[ok] = np.mean(ps)
            m_new = np.zeros(nk); m_new[:ok] = pm * (ok / nk); m_new[ok] = 1.0 / nk
            floor = 1.0 / (nk * 20)
            m_new = np.maximum(m_new, floor); m_new /= np.sum(m_new)
            kw = {"u_init": u_new, "sigma2_init": s_new, "mp_init": m_new}
        result = inference.mop(pcvector, link_functions, k,
                               lambda_u=lu, lambda_pi=lpi,
                               lambda_sigma=ls, **kw)
        bic_val = inference.pbic(pcvector, result, link_functions,
                                 lambda_u=lu, lambda_sigma=ls,
                                 lambda_pi=lpi, s2=s2_bar)
        bic_vals[k] = bic_val
        prev = result

    best_k = min(bic_vals, key=bic_vals.get)
    return prev, best_k, bic_vals[best_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--target-per-dim", type=int, default=None)
    args = parser.parse_args()

    dims = [d for d in DIMS if d != "other"]

    # 加载各维度 pairs
    dim_pairs = {}
    for dim in dims:
        pairs = load_pairs(dim_name=dim)
        dim_pairs[dim] = pairs
        print(f"  {dim}: {len(pairs)} pairs")

    target = args.target_per_dim or max(len(v) for v in dim_pairs.values())
    print(f"\n每维度目标样本量: {target}")
    print(f"平衡后总样本量: {target * len(dims)}")

    # 原始整体 MoP（直接用所有非 Other pairs）
    all_pairs = [p for pairs in dim_pairs.values() for p in pairs]
    print(f"\n原始整体: {len(all_pairs)} pairs (各维度混合，未平衡权重)")

    print("\n拟合原始整体 MoP...")
    mop_orig, k_orig, bic_orig = fit_mop(all_pairs)
    vs_orig = vote_scores(
        [p.priority_vector for p in mop_orig.priorities],
        mop_orig.mixture_probabilities,
        [p.inconsistency_level for p in mop_orig.priorities],
    )
    print(f"  K={k_orig}  pBIC={bic_orig:.1f}")
    for i, t in enumerate(T):
        print(f"    {t}: {vs_orig[i]:.4f}")

    # Bootstrap: 每维度重抽样 → 合并 → MoP
    print(f"\nBootstrap ({args.n_bootstrap} 次)...")
    rng = np.random.RandomState(42)
    vs_all = []
    k_all = []

    for b in range(args.n_bootstrap):
        balanced_pairs = []
        for dim in dims:
            dp = dim_pairs[dim]
            idx = rng.choice(len(dp), size=target, replace=True)
            balanced_pairs.extend([dp[i] for i in idx])
        try:
            mop_b, k_b, bic_b = fit_mop(balanced_pairs, fixed_k=k_orig)
            vs_b = vote_scores(
                [p.priority_vector for p in mop_b.priorities],
                mop_b.mixture_probabilities,
                [p.inconsistency_level for p in mop_b.priorities],
            )
            vs_all.append(vs_b)
            k_all.append(k_b)
        except Exception:
            pass
        if (b + 1) % 10 == 0:
            print(f"  {b+1}/{args.n_bootstrap}...")

    if vs_all:
        vs_arr = np.array(vs_all)
        print(f"\n  成功: {len(vs_all)}/{args.n_bootstrap}")
        k_mode = int(np.argmax(np.bincount(k_all)))
        print(f"  K 众数: {k_mode}")
        for i, t in enumerate(T):
            print(f"  {t}: {vs_arr[:, i].mean():.4f} ± {vs_arr[:, i].std():.4f}")

        # 排名对比
        print("\n  === 排名对比 ===")
        r_orig = sorted(zip(T, vs_orig), key=lambda x: -x[1])
        r_bal = sorted(zip(T, vs_arr.mean(axis=0)), key=lambda x: -x[1])
        print(f"  原始:  {' > '.join(f'{t}({v:.3f})' for t, v in r_orig)}")
        print(f"  平衡:  {' > '.join(f'{t}({v:.3f})' for t, v in r_bal)}")

        # 保存
        out = {
            "target_per_dim": target,
            "n_dims": len(dims),
            "n_bootstrap": args.n_bootstrap,
            "orig": {"k": k_orig, "bic": bic_orig,
                     "vote_scores": [float(x) for x in vs_orig]},
            "balanced": {
                "mean": vs_arr.mean(axis=0).tolist(),
                "sd": vs_arr.std(axis=0).tolist(),
                "k_mode": k_mode,
                "n_success": len(vs_all),
            },
        }
        out_path = ENERGY_PAIRS.parent / "balance_pooled_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
