"""
统计推断 - EM算法
"""

import numpy as np
from scipy.stats import norm

# 尝试导入Cython版本
USE_CYTHON = False
try:
    from cython_inference import e_gamma_cython, m_step_cython
    USE_CYTHON = True
except ImportError:
    pass

from .pairwise_comparison import comparisons2regression
from .mixture_of_priorities import MixtureOfPriorities, Priority
from .initialization import kridge


def e_gamma(X, y, u, sigma2, mp):
    """
    E步：计算隐变量 γ_kc 的期望（向量化优化版）
    """
    if USE_CYTHON:
        return e_gamma_cython(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(u, dtype=np.float64),
            np.asarray(sigma2, dtype=np.float64),
            np.asarray(mp, dtype=np.float64)
        )

    nJ, nP = X.shape[0], len(sigma2)
    
    # 向量化计算均值: (nJ, nP)
    means = X @ u.T  # (nJ, nP)
    stds = np.sqrt(sigma2)  # (nP,)
    
    # 向量化计算对数概率
    log_probs = np.log(mp) + norm.logpdf(y[:, np.newaxis], loc=means, scale=stds)
    
    # 数值稳定的指数化和归一化
    max_vals = np.max(log_probs, axis=1, keepdims=True)
    gkc = np.exp(log_probs - max_vals)
    row_sums = np.maximum(np.sum(gkc, axis=1, keepdims=True), 1e-300)
    gkc = gkc / row_sums
    
    return gkc


def e_gamma_from_pc(pcvector, mop, link_functions):
    """E步：从成对比较数据计算"""
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nP = len(mop.mixture_probabilities)
    nA = X.shape[1] + 1

    u = np.zeros((nP, nA - 1))
    sigma2 = np.array([mop[p].inconsistency_level for p in range(nP)])
    mp = mop.mixture_probabilities

    for p in range(nP):
        w = mop[p].priority_vector
        u[p, :] = np.log(w[:nA - 1] / w[nA - 1])

    return e_gamma(X, y, u, sigma2, mp)


def log_likelihood(X, y, u, sigma2, mp):
    """对数似然函数（向量化优化版）"""
    nJ, nC = X.shape[0], len(sigma2)
    
    # 向量化计算
    means = X @ u.T  # (nJ, nC)
    stds = np.sqrt(sigma2)  # (nC,)
    
    # 计算每个组件的似然
    probs = mp * norm.pdf(y[:, np.newaxis], loc=means, scale=stds)
    Lk = np.sum(probs, axis=1)
    Lk = np.maximum(Lk, 1e-300)
    
    return np.sum(np.log(Lk))


def log_likelihood_from_pc(pcvector, mop, link_functions):
    """从成对比较数据计算对数似然"""
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nP = len(mop.mixture_probabilities)
    nA = X.shape[1] + 1

    u = np.zeros((nP, nA - 1))
    sigma2 = np.array([mop[p].inconsistency_level for p in range(nP)])
    mp = mop.mixture_probabilities

    for p in range(nP):
        w = mop[p].priority_vector
        u[p, :] = np.log(w[:nA - 1] / w[nA - 1])

    return log_likelihood(X, y, u, sigma2, mp)


def penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar=None):
    """
    惩罚对数似然函数 (Eq.10 的对数)，与 manuscript 完全一致

    先验 (Eq.5-7):
      σ²_p ~ InvGamma((λ_σ-2)/2, λ_σ s̄²/2)
      u_p | σ²_p ~ N(ū, λ_u⁻¹ σ²_p I)
      π ~ Dirichlet(λ_π, ..., λ_π)

    Args:
        X: 设计矩阵
        y: 响应变量
        u: 对数优先级比矩阵 (nP x nA-1)
        sigma2: 不一致性水平 σ²
        mp: 混合概率 π
        lambda_u: u 先验的精度系数
        lambda_sigma: σ² 先验的影响参数 (manuscript 中的 λ_σ)
        lambda_pi: π 先验的 Dirichlet 浓度参数
        s2: 先验方差模式 s̄²
        u_bar: u 的先验均值向量 ū (nA-1,)，默认为零向量
    """
    from scipy.special import gammaln
    nP = len(sigma2)
    nA_minus_1 = u.shape[1]

    if u_bar is None:
        u_bar = np.zeros(nA_minus_1)

    PLL = log_likelihood(X, y, u, sigma2, mp)

    # 统一为 per-component 数组
    if np.isscalar(lambda_u):
        lambda_u = np.full(nP, lambda_u, dtype=float)
    if np.isscalar(lambda_sigma):
        lambda_sigma = np.full(nP, lambda_sigma, dtype=float)
    if u_bar.ndim == 1:
        u_bar = np.tile(u_bar, (nP, 1))

    # 1. u 的先验: u_p | σ²_p ~ N(ū_p, λ_{u,p}⁻¹ σ²_p I) — Eq.(6)
    for p in range(nP):
        sp2 = max(sigma2[p], 1e-300)
        diff = u[p] - u_bar[p]
        PLL -= (lambda_u[p] / (2.0 * sp2)) * np.sum(diff ** 2)
        PLL -= (nA_minus_1 / 2.0) * np.log(2.0 * np.pi * sp2 / lambda_u[p])

    # 2. σ² 的先验: σ²_p ~ InvGamma((λ_{σ,p}-2)/2, λ_{σ,p} s̄²/2) — Eq.(5)
    for p, sp2 in enumerate(sigma2):
        if sp2 < 1e-300:
            continue
        alpha = (lambda_sigma[p] - 2.0) / 2.0
        beta = lambda_sigma[p] * s2 / 2.0
        PLL += alpha * np.log(beta) - gammaln(alpha)
        PLL -= (alpha + 1.0) * np.log(sp2)
        PLL -= beta / sp2

    # 3. π 的先验: π ~ Dirichlet(λ_π, ..., λ_π) — Eq.(7)
    valid_mp = mp > 1e-300
    if valid_mp.any():
        PLL += (lambda_pi - 1.0) * np.sum(np.log(mp[valid_mp]))
    PLL += gammaln(nP * lambda_pi) - nP * gammaln(lambda_pi)

    return PLL


def bic(pcvector, mop, link_functions):
    """贝叶斯信息准则"""
    nJ = len(pcvector)
    nP = len(mop.mixture_probabilities)

    all_ids = set()
    for pr in pcvector:
        all_ids.add(pr.former_id)
        all_ids.add(pr.latter_id)
    nA = len(all_ids)

    LL = log_likelihood_from_pc(pcvector, mop, link_functions)
    return -2 * LL + np.log(nJ) * ((nA - 1) * nP + 2 * nP - 1)


def pbic(pcvector, mop, link_functions, lambda_u, lambda_sigma, lambda_pi, s2, u_bar=None):
    """惩罚贝叶斯信息准则 (Eq.19)"""
    nJ = len(pcvector)
    nP = len(mop.mixture_probabilities)

    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA = X.shape[1] + 1
    df = (nA - 1) * nP + 2 * nP - 1

    u = np.zeros((nP, nA - 1))
    sigma2 = np.array([mop[p].inconsistency_level for p in range(nP)])
    mp = mop.mixture_probabilities

    for p in range(nP):
        w = mop[p].priority_vector
        u[p, :] = np.log(w[:nA - 1] / w[nA - 1])

    PLL = penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar)
    return -2 * PLL + np.log(nJ) * df


def mop(pcvector, link_functions, nP, lambda_u=None, lambda_sigma=None,
        lambda_pi=None, s2=None, u_bar=None,
        u_init=None, sigma2_init=None, mp_init=None):
    """
    EM算法主函数 - 估计混合优先级模型

    Args:
        pcvector: 成对比较对象列表
        link_functions: 链接函数列表
        nP: 优先级组件数量
        lambda_u: u的L2正则系数
        lambda_sigma: sigma2 的 inverse-Gamma 先验系数
        lambda_pi: mp的Dirichlet先验系数
        s2: 先验方差模式
        u_bar: u的先验均值向量 (nA-1,)，默认为零向量
        u_init, sigma2_init, mp_init: 可选自定义初始值

    Returns:
        MixtureOfPriorities: 估计的混合优先级模型
    """
    nJ = len(pcvector)
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA_minus_1 = X.shape[1]

    # 默认先验均值为零向量 — 支持 per-component
    if u_bar is None:
        u_bar = np.zeros(nA_minus_1)
    if np.ndim(u_bar) == 1:
        u_bar = np.tile(u_bar, (nP, 1))

    # 统一 λ 为 per-component 数组
    if np.isscalar(lambda_u):
        lambda_u = np.full(nP, lambda_u, dtype=float)
    if np.isscalar(lambda_sigma):
        lambda_sigma = np.full(nP, lambda_sigma, dtype=float)

    # 清理NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)

    # 默认参数 — 仅对未指定的参数设默认值
    if s2 is None:
        try:
            u_init_tmp = np.linalg.solve(X.T @ X, X.T @ y)
            s2 = max(np.sum((X @ u_init_tmp - y) ** 2) / nJ, 1e-10)
        except Exception:
            s2 = 1e-10
    if lambda_u is None:
        lambda_u = 1e-10
    if lambda_sigma is None:
        lambda_sigma = 0.1
    if lambda_pi is None:
        lambda_pi = 1

    # 初始化
    if u_init is not None and sigma2_init is not None and mp_init is not None:
        u, sigma2, mp = u_init, sigma2_init, mp_init
    else:
        u, sigma2, mp = kridge(X, y, nP)

    # 检查初始化结果
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        u = np.random.randn(nP, nA_minus_1) * 0.1
        sigma2 = np.ones(nP) * s2
        mp = np.ones(nP) / nP

    # 预计算常量
    eye_matrix = np.eye(nA_minus_1)

    # EM迭代
    for t in range(1, 10002):
        # E步
        gkc = e_gamma(X, y, u, sigma2, mp)

        # 检查NaN
        if np.any(np.isnan(gkc)):
            break

        # M步
        if USE_CYTHON:
            u_new, sigma2_new, mp_new = m_step_cython(
                np.asarray(X, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                np.asarray(gkc, dtype=np.float64),
                np.asarray(u_bar[0] if u_bar.ndim > 1 else u_bar, dtype=np.float64),
                float(lambda_u[0]) if np.ndim(lambda_u) > 0 else float(lambda_u),
                float(lambda_sigma[0]) if np.ndim(lambda_sigma) > 0 else float(lambda_sigma),
                lambda_pi, s2, nP,
            )
        else:
            # 向量化M步
            sum_gkc = np.sum(gkc, axis=0)  # (nP,)

            # 更新mp
            mp_new = (sum_gkc + lambda_pi - 1) / (nJ + lambda_pi * nP - nP)
            mp_new = np.maximum(mp_new, 1e-10)
            mp_new = mp_new / np.sum(mp_new)

            # 更新u和sigma2
            u_new = np.zeros_like(u)
            sigma2_new = np.zeros(nP)

            for p in range(nP):
                gkc_p = gkc[:, p]  # (nJ,)

                # 加权矩阵（修正：添加 lambda_u[p] * u_bar[p] 项）
                XtWX = X.T @ (X * gkc_p[:, np.newaxis]) + lambda_u[p] * eye_matrix
                Xty_w = X.T @ (y * gkc_p) + lambda_u[p] * u_bar[p]

                try:
                    u_new[p, :] = np.linalg.solve(XtWX, Xty_w)
                except np.linalg.LinAlgError:
                    u_new[p, :] = np.linalg.lstsq(XtWX, Xty_w, rcond=None)[0]

                # M-step for the inconsistency variance: the exact maximiser of
                # the penalized log-posterior that penalized_log_likelihood
                # evaluates, so that the iteration stays monotone. The
                # denominator collects one contribution per factor carrying
                # phi_p:
                #   sum_gkc[p]        the Gaussian likelihood, n_p observations
                #   nA_minus_1        the normalising constant of the utility
                #                     prior, whose density carries
                #                     phi_p^{-(nA-1)/2}
                #   lambda_sigma      = 2(alpha+1) with alpha=(lambda_sigma-2)/2
                #                     from the inverse-Gamma density on phi_p
                # The numerator carries the utility prior's quadratic term.
                # An earlier version omitted nA_minus_1, which put this update
                # off the stationary point; see PARAMETERISATION.md section 3.
                residuals = y - X @ u_new[p, :]
                ss_resid = float(np.sum(gkc_p * residuals ** 2))
                ss_prior_u = lambda_u[p] * float(np.sum((u_new[p] - u_bar[p]) ** 2))
                sigma2_new[p] = max(
                    (lambda_sigma[p] * s2 + ss_resid + ss_prior_u)
                    / (lambda_sigma[p] + sum_gkc[p] + nA_minus_1),
                    1e-10
                )

        # 检查NaN
        if np.any(np.isnan(u_new)) or np.any(np.isnan(sigma2_new)) or np.any(np.isnan(mp_new)):
            break

        # 检查收敛
        pll_t = penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar)
        pll_tp1 = penalized_log_likelihood(X, y, u_new, sigma2_new, mp_new, lambda_u, lambda_sigma, lambda_pi, s2, u_bar)

        if np.isnan(pll_tp1) or np.isinf(pll_tp1):
            break

        # Relative change in the penalized log-posterior. 1e-10 is far below
        # any quantity that gets reported, whereas 1e-20 lies under double
        # precision for magnitudes of this size, so the loop ran to the
        # iteration cap instead of stopping.
        if abs(pll_tp1 - pll_t) / (abs(pll_tp1) + 1e-300) < 1e-10:
            break

        u, sigma2, mp = u_new, sigma2_new, mp_new

    # 转换为优先级向量
    # 限制对数优先级比范围，避免 exp 溢出
    # exp(50) ≈ 5.2e21 已足够覆盖实际优先级比范围
    u_clipped = np.clip(u, -50, 50)
    v = np.hstack([np.exp(u_clipped), np.ones((nP, 1))])
    w = v / np.sum(v, axis=1, keepdims=True)

    priorities = [Priority(w[i, :], sigma2[i]) for i in range(nP)]
    return MixtureOfPriorities(priorities, mp)
