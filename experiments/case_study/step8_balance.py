"""Step 8b: 维度平衡后的整体 MoP —— bootstrap 平衡各维度样本再合并。

原始整体 MoP 中 Cost 占 1,570/5,238 ≈ 30%，Policy 仅 281/5,238 ≈ 5%。
这里将各维度（除 Other）bootstrap 到相同样本量后合并，重新拟合 MoP，
比较平衡前后整体偏好结构的差异。

聚合排序用期望效用（_common.expected_utility），与 step7_analyze 一致。
"""

import json, argparse
import numpy as np
from scipy.special import logit

from mop import inference, pairwise_comparison
from mop.inference import comparisons2regression
from mop.initialization import warm_init

from _common import T, DIMS, ENERGY_PAIRS, TYPE_MAP, expected_utility
from step7_analyze import load_pairs


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
    # φ₀ 由 inference.prior_mode_phi0 定义一次，拟合与 pBIC 共用

    if fixed_k is not None:
        result = inference.mop(pcvector, link_functions, fixed_k,
                               lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls)
        bic_val = inference.pbic(pcvector, result, link_functions,
                                 lambda_u=lu, lambda_sigma=ls,
                                 lambda_pi=lpi)
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
            u_init, sigma2_init, mp_init = warm_init(
                X, y, pu, ps, pm, min_weight_share=0.05,
                n_components=k)
            kw = {"u_init": u_init, "sigma2_init": sigma2_init, "mp_init": mp_init}
        result = inference.mop(pcvector, link_functions, k,
                               lambda_u=lu, lambda_pi=lpi,
                               lambda_sigma=ls, **kw)
        bic_val = inference.pbic(pcvector, result, link_functions,
                                 lambda_u=lu, lambda_sigma=ls,
                                 lambda_pi=lpi)
        bic_vals[k] = bic_val
        prev = result

        # If minprior is turned on and a candidate loses a component to it,
        # raising the candidate further can only reproduce that mixture, so the
        # search stops there. With the default of no removal the test never
        # fires and the whole grid is searched.
        if len(result) < k:
            break

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
    eu_orig = expected_utility(
        [p.priority_vector for p in mop_orig.priorities],
        mop_orig.mixture_probabilities,
    )
    print(f"  K={k_orig}  pBIC={bic_orig:.1f}")
    for i, t in enumerate(T):
        print(f"    {t}: {eu_orig[i]:.4f}")

    # Bootstrap: 每维度重抽样 → 合并 → MoP
    print(f"\nBootstrap ({args.n_bootstrap} 次)...")
    rng = np.random.RandomState(42)
    eu_all = []
    k_all = []

    for b in range(args.n_bootstrap):
        balanced_pairs = []
        for dim in dims:
            dp = dim_pairs[dim]
            idx = rng.choice(len(dp), size=target, replace=True)
            balanced_pairs.extend([dp[i] for i in idx])
        try:
            mop_b, k_b, bic_b = fit_mop(balanced_pairs, fixed_k=k_orig)
            eu_b = expected_utility(
                [p.priority_vector for p in mop_b.priorities],
                mop_b.mixture_probabilities,
            )
            eu_all.append(eu_b)
            k_all.append(k_b)
        except Exception:
            pass
        if (b + 1) % 10 == 0:
            print(f"  {b+1}/{args.n_bootstrap}...")

    if eu_all:
        eu_arr = np.array(eu_all)
        print(f"\n  成功: {len(eu_all)}/{args.n_bootstrap}")
        k_mode = int(np.argmax(np.bincount(k_all)))
        print(f"  K 众数: {k_mode}")
        for i, t in enumerate(T):
            print(f"  {t}: {eu_arr[:, i].mean():.4f} ± {eu_arr[:, i].std():.4f}")

        # 排名对比
        print("\n  === 排名对比 ===")
        r_orig = sorted(zip(T, eu_orig), key=lambda x: -x[1])
        r_bal = sorted(zip(T, eu_arr.mean(axis=0)), key=lambda x: -x[1])
        print(f"  原始:  {' > '.join(f'{t}({v:.3f})' for t, v in r_orig)}")
        print(f"  平衡:  {' > '.join(f'{t}({v:.3f})' for t, v in r_bal)}")

        # 保存
        out = {
            "target_per_dim": target,
            "n_dims": len(dims),
            "n_bootstrap": args.n_bootstrap,
            "orig": {"k": k_orig, "bic": bic_orig,
                     "expected_utility": [float(x) for x in eu_orig]},
            "balanced": {
                "mean": eu_arr.mean(axis=0).tolist(),
                "sd": eu_arr.std(axis=0).tolist(),
                "k_mode": k_mode,
                "n_success": len(eu_all),
            },
        }
        out_path = ENERGY_PAIRS.parent / "balance_pooled_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
