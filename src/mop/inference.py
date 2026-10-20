"""
统计推断 - EM算法
"""

import numpy as np
from scipy.stats import norm

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
    
    means = X @ u.T
    stds = np.sqrt(sigma2)
    
    log_probs = np.log(mp) + norm.logpdf(y[:, np.newaxis], loc=means, scale=stds)
    
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


def log_likelihood(X, y, u, sigma2, mp, omega=None):
    """对数似然函数（向量化优化版）"""
    nJ, nC = X.shape[0], len(sigma2)

    means = X @ u.T
    stds = np.sqrt(sigma2)

    probs = mp * norm.pdf(y[:, np.newaxis], loc=means, scale=stds)
    Lk = np.sum(probs, axis=1)
    Lk = np.maximum(Lk, 1e-300)

    log_Lk = np.log(Lk)
    if omega is None:
        return np.sum(log_Lk)
    return float(np.sum(np.asarray(omega, dtype=float) * log_Lk))


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


def penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar=None, omega=None):
    """惩罚对数似然函数 (Eq.10 的对数)，与 manuscript 完全一致

    Args:
        X: 设计矩阵
        y: 响应变量
        u: 对数优先级比矩阵 (nP x nA-1)
        sigma2: 不一致性水平 σ²
        mp: 混合概率 π
        lambda_u: u 先验的精度系数
        lambda_sigma: σ² 先验的影响参数 (manuscript 中的 λ_σ)
        lambda_pi: π 先验的 Dirichlet 浓度参数
        s2: 先验方差模式 φ₀ (manuscript 中的 φ_0)
        u_bar: u 的先验均值向量 u₀ (nA-1,)，默认为零向量
    """
    from scipy.special import gammaln
    nP = len(sigma2)
    nA_minus_1 = u.shape[1]

    if u_bar is None:
        u_bar = np.zeros(nA_minus_1)

    PLL = log_likelihood(X, y, u, sigma2, mp, omega)

    if np.isscalar(lambda_u):
        lambda_u = np.full(nP, lambda_u, dtype=float)
    if np.isscalar(lambda_sigma):
        lambda_sigma = np.full(nP, lambda_sigma, dtype=float)
    if u_bar.ndim == 1:
        u_bar = np.tile(u_bar, (nP, 1))

    for p in range(nP):
        sp2 = max(sigma2[p], 1e-300)
        diff = u[p] - u_bar[p]
        PLL -= (lambda_u[p] / (2.0 * sp2)) * np.sum(diff ** 2)
        PLL -= (nA_minus_1 / 2.0) * np.log(2.0 * np.pi * sp2 / lambda_u[p])

    for p, sp2 in enumerate(sigma2):
        if sp2 < 1e-300:
            continue
        alpha = (lambda_sigma[p] - 2.0) / 2.0
        beta = lambda_sigma[p] * s2 / 2.0
        PLL += alpha * np.log(beta) - gammaln(alpha)
        PLL -= (alpha + 1.0) * np.log(sp2)
        PLL -= beta / sp2

    valid_mp = mp > 1e-300
    if valid_mp.any():
        PLL += (lambda_pi - 1.0) * np.sum(np.log(mp[valid_mp]))
    PLL += gammaln(nP * lambda_pi) - nP * gammaln(lambda_pi)

    return PLL


def bic(pcvector, mop, link_functions):
    """贝叶斯信息准则 — 未被使用，也不等价于论文的 pBIC。"""
    nJ = len(pcvector)
    nP = len(mop.mixture_probabilities)

    all_ids = set()
    for pr in pcvector:
        all_ids.add(pr.former_id)
        all_ids.add(pr.latter_id)
    nA = len(all_ids)

    LL = log_likelihood_from_pc(pcvector, mop, link_functions)
    return -2 * LL + np.log(nJ) * ((nA - 1) * nP + 2 * nP - 1)


def prior_mode_phi0(X, y):
    """先验众数 φ₀ (manuscript eq:general gci)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_j, p = X.shape
    XtX = X.T @ X + 1e-10 * np.eye(p)
    try:
        u_pool = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        u_pool = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    ss = float(np.sum((X @ u_pool - y) ** 2))
    return max(ss / max(n_j - p, 1), 1e-10)


def pbic(pcvector, mop, link_functions, lambda_u, lambda_sigma, lambda_pi, s2=None, u_bar=None, omega=None):
    """惩罚贝叶斯信息准则 (Eq.19)"""
    nJ = len(pcvector)
    nP = len(mop.mixture_probabilities)

    X, y, _ = comparisons2regression(pcvector, link_functions)
    if s2 is None:
        s2 = prior_mode_phi0(X, y)
    nA = X.shape[1] + 1
    df = (nA - 1) * nP + 2 * nP - 1

    u = np.zeros((nP, nA - 1))
    sigma2 = np.array([mop[p].inconsistency_level for p in range(nP)])
    mp = mop.mixture_probabilities

    for p in range(nP):
        w = mop[p].priority_vector
        u[p, :] = np.log(w[:nA - 1] / w[nA - 1])

    if omega is None:
        omega_eff = None
        n_eff = float(nJ)
    else:
        omega_eff = np.asarray(omega, dtype=float).ravel()
        if omega_eff.size != nJ:
            raise ValueError(
                f"omega 长度({omega_eff.size})与判断数({nJ})不一致")
        if np.any(omega_eff <= 0):
            raise ValueError("omega 必须为正")
        omega_eff = omega_eff * (nJ / float(np.sum(omega_eff)))
        n_eff = float(np.sum(omega_eff))

    PLL = penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar, omega_eff)
    return -2 * PLL + np.log(n_eff) * df


def mop(pcvector, link_functions, nP, lambda_u=None, lambda_sigma=None,
        lambda_pi=None, s2=None, u_bar=None,
        u_init=None, sigma2_init=None, mp_init=None, minprior=0.0, omega=None):
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
        u_bar: u 的先验均值向量 u₀ (nA-1,)，默认为零向量
        u_init, sigma2_init, mp_init: 可选自定义初始值
        minprior: 权重低于此值的组件在拟合中被移除；默认 0.0 即不移除。
            flexmix 的这一阈值是 0.05。论文报告候选实际拟合出的组件，
            包括权重很小的那些，因为组件数正是待推断的量。
        omega: 每个判断的观测权重 ω_k（长度 nJ，可空）。与稿件一致：
            ratio 与主观概率判断取 1，重复选择取试验次数 n_k。**必须由协议
            显式给定，不自动推断**。内部归一化到均值 1，此时加权与未加权
            目标只差一个常数 −Σω_k·ln(Σω_k)，M 步与 pBIC 因此不受影响。

    Returns:
        MixtureOfPriorities: 估计的混合优先级模型
    """
    nJ = len(pcvector)
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA_minus_1 = X.shape[1]

    if omega is None:
        omega_eff = None
    else:
        omega_eff = np.asarray(omega, dtype=float).ravel()
        if omega_eff.size != nJ:
            raise ValueError(
                f"omega 长度({omega_eff.size})与判断数({nJ})不一致")
        if np.any(omega_eff <= 0):
            raise ValueError("omega 必须为正")
        omega_eff = omega_eff * (nJ / float(np.sum(omega_eff)))
    use_weighted = omega_eff is not None and not np.allclose(omega_eff, 1.0)

    if u_init is not None and sigma2_init is not None and mp_init is not None:
        nP = len(mp_init)

    if u_bar is None:
        u_bar = np.zeros(nA_minus_1)
    if np.ndim(u_bar) == 1:
        u_bar = np.tile(u_bar, (nP, 1))

    if np.isscalar(lambda_u):
        lambda_u = np.full(nP, lambda_u, dtype=float)
    if np.isscalar(lambda_sigma):
        lambda_sigma = np.full(nP, lambda_sigma, dtype=float)

    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)

    if s2 is None:
        s2 = prior_mode_phi0(X, y)
    if lambda_u is None:
        lambda_u = 1e-6
    if lambda_sigma is None:
        lambda_sigma = X.shape[0] / 50.0
    if lambda_pi is None:
        lambda_pi = 1

    if u_init is not None and sigma2_init is not None and mp_init is not None:
        u, sigma2, mp = u_init, sigma2_init, mp_init
    else:
        u, sigma2, mp = kridge(X, y, nP)

    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        u = np.random.randn(nP, nA_minus_1) * 0.1
        sigma2 = np.ones(nP) * s2
        mp = np.ones(nP) / nP

    eye_matrix = np.eye(nA_minus_1)

    for t in range(1, 10002):
        gkc = e_gamma(X, y, u, sigma2, mp)

        if np.any(np.isnan(gkc)):
            break

        if USE_CYTHON and not use_weighted:
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
            sum_gkc = np.sum(gkc, axis=0)

            if use_weighted:
                sum_wgkc = omega_eff @ gkc
                sum_omega = float(np.sum(omega_eff))
                mp_new = (sum_wgkc + lambda_pi - 1) / (sum_omega + lambda_pi * nP - nP)
            else:
                mp_new = (sum_gkc + lambda_pi - 1) / (nJ + lambda_pi * nP - nP)
            mp_new = np.maximum(mp_new, 1e-10)
            mp_new = mp_new / np.sum(mp_new)

            u_new = np.zeros_like(u)
            sigma2_new = np.zeros(nP)

            for p in range(nP):
                if use_weighted:
                    gkc_w = omega_eff * gkc[:, p]
                    sum_gkc_p = float(np.sum(gkc_w))
                else:
                    gkc_w = gkc[:, p]
                    sum_gkc_p = sum_gkc[p]

                XtWX = X.T @ (X * gkc_w[:, np.newaxis]) + lambda_u[p] * eye_matrix
                Xty_w = X.T @ (y * gkc_w) + lambda_u[p] * u_bar[p]

                try:
                    u_new[p, :] = np.linalg.solve(XtWX, Xty_w)
                except np.linalg.LinAlgError:
                    u_new[p, :] = np.linalg.lstsq(XtWX, Xty_w, rcond=None)[0]

                residuals = y - X @ u_new[p, :]
                ss_resid = float(np.sum(gkc_w * residuals ** 2))
                ss_prior_u = lambda_u[p] * float(np.sum((u_new[p] - u_bar[p]) ** 2))
                sigma2_new[p] = max(
                    (lambda_sigma[p] * s2 + ss_resid + ss_prior_u)
                    / (lambda_sigma[p] + sum_gkc_p + nA_minus_1),
                    1e-10
                )

        if np.any(np.isnan(u_new)) or np.any(np.isnan(sigma2_new)) or np.any(np.isnan(mp_new)):
            break

        pll_t = penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar, omega_eff)
        pll_tp1 = penalized_log_likelihood(X, y, u_new, sigma2_new, mp_new, lambda_u, lambda_sigma, lambda_pi, s2, u_bar, omega_eff)

        if np.isnan(pll_tp1) or np.isinf(pll_tp1):
            break

        keep = None
        if minprior > 0:
            keep = np.flatnonzero(mp_new >= minprior)
            if keep.size == 0:
                keep = np.array([int(np.argmax(mp_new))])
            if keep.size == nP:
                keep = None
            else:
                u_new = u_new[keep]
                sigma2_new = sigma2_new[keep]
                mp_new = mp_new[keep] / np.sum(mp_new[keep])
                if np.ndim(lambda_u) > 0:
                    lambda_u = np.asarray(lambda_u)[keep]
                if np.ndim(lambda_sigma) > 0:
                    lambda_sigma = np.asarray(lambda_sigma)[keep]
                if np.ndim(u_bar) == 2 and np.shape(u_bar)[0] == nP:
                    u_bar = np.asarray(u_bar)[keep]
                nP = int(keep.size)

        if abs(pll_tp1 - pll_t) / (abs(pll_tp1) + 0.1) < 1e-6:
            break

        u, sigma2, mp = u_new, sigma2_new, mp_new

    u_clipped = np.clip(u, -50, 50)
    v = np.hstack([np.exp(u_clipped), np.ones((nP, 1))])
    w = v / np.sum(v, axis=1, keepdims=True)

    priorities = [Priority(w[i, :], sigma2[i]) for i in range(nP)]
    return MixtureOfPriorities(priorities, mp)
