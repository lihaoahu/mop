"""Closed-form EM for the Gaussian variants of the mixture.

Used where the M step is available in closed form: normal utilities, inverse-Gamma
dispersions and a Dirichlet on the weights. The general families are fitted by
``mop.general_fit``.
"""


import numpy as np
from scipy.stats import norm

from . import criteria as _criteria
from .pairwise_comparison import comparisons2regression
from .mixture_of_priorities import MixtureOfPriorities, Priority
from .initialization import kridge


def e_gamma(X, y, u, sigma2, mp):
    """Posterior allocation of each judgment to each component."""
    means = X @ u.T
    stds = np.sqrt(sigma2)
    
    log_probs = np.log(mp) + norm.logpdf(y[:, np.newaxis], loc=means, scale=stds)
    
    max_vals = np.max(log_probs, axis=1, keepdims=True)
    gkc = np.exp(log_probs - max_vals)
    row_sums = np.maximum(np.sum(gkc, axis=1, keepdims=True), 1e-300)
    gkc = gkc / row_sums
    
    return gkc


def e_gamma_from_pc(pcvector, mop, link_functions):
    """Same, from pairwise comparisons and their link functions."""
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
    """Observed-data log-likelihood, weighted by ``omega`` when given."""
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
    """Same, from pairwise comparisons."""
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
    """Penalised log-posterior: likelihood plus the priors on ``u``, ``phi`` and ``pi``."""
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
        alpha = lambda_sigma[p] / 2.0
        beta = lambda_sigma[p] * s2 / 2.0
        PLL += alpha * np.log(beta) - gammaln(alpha)
        PLL -= (alpha + 1.0) * np.log(sp2)
        PLL -= beta / sp2

    valid_mp = mp > 1e-300
    if valid_mp.any():
        PLL += (lambda_pi - 1.0) * np.sum(np.log(mp[valid_mp]))
    PLL += gammaln(nP * lambda_pi) - nP * gammaln(lambda_pi)

    return PLL


def prior_mode_phi0(X, y):
    """Prior mode ``phi_0`` from a pooled least-squares fit."""
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


def pbic(pcvector, mop, link_functions, lambda_u, lambda_sigma, lambda_pi, s2=None,
         u_bar=None, omega=None, profile_utilities=True):
    """``pbic`` with the utility prior profiled out of the criterion."""
    nJ = len(pcvector)
    nP = len(mop.mixture_probabilities)

    X, y, _ = comparisons2regression(pcvector, link_functions)
    if s2 is None:
        s2 = prior_mode_phi0(X, y)
    nA = X.shape[1] + 1
    df = _criteria.degrees_of_freedom(nA, nP, phi_free=True)

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
                f"omega has {omega_eff.size} entries, expected one per judgment ({nJ})")
        if np.any(omega_eff <= 0):
            raise ValueError("omega must be positive")
        omega_eff = omega_eff * (nJ / float(np.sum(omega_eff)))
        n_eff = float(np.sum(omega_eff))

    PLL = penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar, omega_eff)
    u_log_prior = (_criteria.log_prior_utilities(u, sigma2, lambda_u, u_bar)
                   if profile_utilities else 0.0)
    return _criteria.pbic(PLL, n_eff, df, u_log_prior)


def _polish(X, y, u, sigma2, mp, lambda_u, lambda_sigma, lambda_pi, s2, u_bar,
            omega, max_steps=60, grad_tol=1e-9):
    """Finish the mode with Newton steps; returns ``(u, sigma2, mp, info)``. EM stops on the change in the objective, which is not enough near a boundary weight."""
    from .general_fit import GaussianResponse
    from .laplace import polish_mode

    def log_posterior(uu, ph, pp):
        return penalized_log_likelihood(X, y, uu, ph, pp, lambda_u, lambda_sigma,
                                        lambda_pi, s2, u_bar, omega)

    return polish_mode(log_posterior, u, sigma2, mp, X, y, GaussianResponse(),
                       omega=omega, u_bar=u_bar, lambda_u=lambda_u,
                       lambda_pi=lambda_pi, lambda_sigma=lambda_sigma, s2=s2,
                       max_steps=max_steps, grad_tol=grad_tol)


def _fitted_arrays(pcvector, mop, link_functions, lambda_u, lambda_sigma,
                   lambda_pi, s2, u_bar, omega):
    """The fitted mode as arrays, its log-posterior, the parameter count and the judgment count."""
    nJ = len(pcvector)
    nP = len(mop.mixture_probabilities)

    X, y, _ = comparisons2regression(pcvector, link_functions)
    if s2 is None:
        s2 = prior_mode_phi0(X, y)
    nA = X.shape[1] + 1
    df = _criteria.degrees_of_freedom(nA, nP, phi_free=True)

    u = np.zeros((nP, nA - 1))
    sigma2 = np.array([mop[p].inconsistency_level for p in range(nP)])
    mp = mop.mixture_probabilities

    for p in range(nP):
        w = mop[p].priority_vector
        u[p, :] = np.log(w[:nA - 1] / w[nA - 1])

    if omega is None:
        omega_eff = None
    else:
        omega_eff = np.asarray(omega, dtype=float).ravel()
        if omega_eff.size != nJ:
            raise ValueError(
                f"omega has {omega_eff.size} entries, expected one per judgment ({nJ})")
        if np.any(omega_eff <= 0):
            raise ValueError("omega must be positive")
        omega_eff = omega_eff * (nJ / float(np.sum(omega_eff)))

    PLL = penalized_log_likelihood(X, y, u, sigma2, mp, lambda_u, lambda_sigma,
                                   lambda_pi, s2, u_bar, omega_eff)
    return X, y, u, sigma2, mp, PLL, df, s2, omega_eff, nJ


def bic(pcvector, mop, link_functions, lambda_u, lambda_sigma, lambda_pi, s2=None,
        u_bar=None, omega=None):
    """``-2 pll + d ln n`` at the fitted mode."""
    X, y, u, sigma2, mp, PLL, df, s2, omega_eff, nJ = _fitted_arrays(
        pcvector, mop, link_functions, lambda_u, lambda_sigma, lambda_pi,
        s2, u_bar, omega)
    return _criteria.bic(PLL, nJ, df)


def lbic(pcvector, mop, link_functions, lambda_u, lambda_sigma, lambda_pi, s2=None,
         u_bar=None, omega=None):
    """Laplace approximation at the fitted mode; ``inf`` when the fit has no interior mode."""
    from .general_fit import GaussianResponse
    from .laplace import ln_det_hessian

    info = getattr(mop, "_polish", None)
    if info is not None and not info.get("converged", True):
        return float("inf")

    X, y, u, sigma2, mp, PLL, df, s2, omega_eff, _ = _fitted_arrays(
        pcvector, mop, link_functions, lambda_u, lambda_sigma, lambda_pi,
        s2, u_bar, omega)
    lnH = ln_det_hessian(X, y, u, sigma2, mp, GaussianResponse(),
                         omega=omega_eff, u_bar=u_bar, lambda_u=lambda_u,
                         lambda_pi=lambda_pi, lambda_sigma=lambda_sigma, s2=s2)
    return _criteria.lbic(PLL, lnH, df)


def fit_mixture(pcvector, link_functions, nP, lambda_u=None, lambda_sigma=None,
        lambda_pi=None, s2=None, u_bar=None,
        u_init=None, sigma2_init=None, mp_init=None, minprior=0.0, omega=None,
        em_tol=1e-6, em_max_iter=10002, polish=True):
    """Fit the mixture by EM and return the estimated model.

    ``omega`` holds the observation weight of each judgment: 1 for ratio and probability
    judgments, the trial count for repeated choices. It is required by the protocol, not
    inferred, and is scaled to mean 1 internally, which leaves the M step and the
    criterion unchanged. ``minprior`` drops components below that weight, as flexmix
    does at 0.05; the paper keeps them, since the count is the quantity being inferred.
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
                f"omega has {omega_eff.size} entries, expected one per judgment ({nJ})")
        if np.any(omega_eff <= 0):
            raise ValueError("omega must be positive")
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

    for t in range(1, int(em_max_iter) + 2):
        gkc = e_gamma(X, y, u, sigma2, mp)

        if np.any(np.isnan(gkc)):
            break

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
                / (lambda_sigma[p] + 2.0 + sum_gkc_p + nA_minus_1),
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

        if abs(pll_tp1 - pll_t) / (abs(pll_tp1) + 0.1) < em_tol:
            break

        u, sigma2, mp = u_new, sigma2_new, mp_new

    polish_info = None
    if polish:
        u, sigma2, mp, polish_info = _polish(
            X, y, np.asarray(u, dtype=float), np.asarray(sigma2, dtype=float),
            np.asarray(mp, dtype=float), lambda_u, lambda_sigma, lambda_pi, s2,
            u_bar, omega_eff)
        nP = int(len(mp))

    u_clipped = np.clip(u, -50, 50)
    v = np.hstack([np.exp(u_clipped), np.ones((nP, 1))])
    w = v / np.sum(v, axis=1, keepdims=True)

    priorities = [Priority(w[i, :], sigma2[i]) for i in range(nP)]
    result = MixtureOfPriorities(priorities, mp)
    # reported by ``to_dict``; the M step does not use it
    result._log_likelihood = float(log_likelihood(X, y, u, sigma2, mp, omega_eff))
    result._polish = polish_info
    return result
