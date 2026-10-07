"""Beta-binomial and binomial likelihoods for the mixture of preferences.

This module implements the GLM subfamily of the generalized likelihood
framework (manuscript Section 2.2) for judgments recorded as repeated binary
comparisons. A judgment is summarized by a count, and conditional on the
allocation to component ``p``,

    m_k | c_k = p  ~  BetaBinomial(n_k, mu_kp, rho_p),
    mu_kp = logistic(x_k' u_p),

so that

    E[m_k | c_k = p]    = n_k mu_kp
    Var[m_k | c_k = p]  = n_k mu_kp (1 - mu_kp) [1 + (n_k - 1) rho_p].

Two modes are available:

    phi_mode="binomial"        rho_p is fixed at 0 (phi_p = 1). The variance
                               function mu(1 - mu) is locked, so dispersion
                               beyond it can only be explained by adding
                               components: intrinsic variability and
                               structural inconsistency are confounded.

    phi_mode="beta-binomial"   rho_p in [0, 1) is free and carries structural
                               inconsistency. The mixture carries intrinsic
                               variability. The two mechanisms stay
                               separable, and because the model has a genuine
                               likelihood the pBIC of Section 2.4 applies.

The normal-likelihood implementation in :mod:`mop.inference` is untouched; this
module is purely additive.

Note on perfectly separated units
---------------------------------
When every comparison of a unit is decided unanimously (all counts at 0 or at
``n``), the log-likelihood is monotone in ``rho`` and is maximised at the upper
bound, because a Beta with shapes approaching zero concentrates on 0 and 1.
``rho`` is then not identified for that unit: the same fitted ranking and the
same component count are obtained at the boundary or under
``phi_mode="binomial"``. Report ``rho`` only for units with at least one split
decision, or fit those units in binomial mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, logsumexp

try:
    from .initialization import (
        binomial_deviance_residuals,
        binomial_family,
        kridge,
    )
except ImportError:  # pragma: no cover
    from initialization import (
        binomial_deviance_residuals,
        binomial_family,
        kridge,
    )

__all__ = [
    "BinomialMoP",
    "beta_binomial_logpmf",
    "binomial_logpmf",
    "design_matrix",
    "fit",
    "select_components",
]

# Numerical guards. rho below _BINOMIAL_SWITCH is evaluated with the binomial
# formula, both for speed and to avoid the cancellation that arises when the
# Beta concentration 1/rho grows large. _RHO_MAX is kept below 1 because the
# Beta shapes a, b -> 0 as rho -> 1; on perfectly separated data (every pair
# decided unanimously) the likelihood is then maximised at the boundary, which
# is a degenerate limit rather than a usable dispersion estimate.
_RHO_MIN = 1.0e-5
_RHO_MAX = 0.9
_BINOMIAL_SWITCH = 1.0e-6
_EPS_SHAPE = 1.0e-10


def _digamma_safe(x):
    """digamma with a floor on the argument.

    As rho approaches 1 the Beta shapes a = mu*c and b = (1-mu)*c approach
    zero, and digamma diverges. Flooring the argument keeps the gradient
    finite instead of propagating inf/nan into the optimizer.
    """
    return digamma(np.maximum(x, _EPS_SHAPE))


# ----------------------------------------------------------------------------
# Likelihoods
# ----------------------------------------------------------------------------

def binomial_logpmf(m, n, mu):
    """log P(M = m) for M ~ Binomial(n, mu)."""
    m = np.asarray(m, dtype=float)
    n = np.asarray(n, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, 1 - 1e-12)
    return (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
            + m * np.log(mu) + (n - m) * np.log1p(-mu))


def _beta_binomial_logpmf_raw(m, n, mu, rho):
    """Beta-binomial log-pmf under the mean-dispersion parameterisation.

    ``rho`` is the intra-pair correlation: Var = n mu (1 - mu) [1 + (n-1) rho].
    It relates to the Beta(a, b) shape parameters through
    c = (1 - rho) / rho, a = mu * c, b = (1 - mu) * c.
    """
    c = (1.0 - rho) / rho
    a = mu * c
    b = (1.0 - mu) * c
    return (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
            + gammaln(m + a) + gammaln(n - m + b) - gammaln(n + a + b)
            - gammaln(a) - gammaln(b) + gammaln(a + b))


def beta_binomial_logpmf(m, n, mu, rho):
    """log P(M = m) for M ~ BetaBinomial(n, mu, rho).

    Vectorised: ``m``, ``n``, ``mu`` and ``rho`` broadcast against each other.
    Values with ``rho`` at or below ``_BINOMIAL_SWITCH`` fall back to the
    binomial log-pmf, which is the correct rho -> 0 limit.
    """
    m = np.asarray(m, dtype=float)
    n = np.asarray(n, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, 1 - 1e-12)
    rho = np.asarray(rho, dtype=float)
    small = rho <= _BINOMIAL_SWITCH
    if np.all(small):
        return binomial_logpmf(m, n, mu)
    rho_c = np.clip(rho, _RHO_MIN, _RHO_MAX)
    out = _beta_binomial_logpmf_raw(m, n, mu, rho_c)
    if not np.any(small):
        return out
    return np.where(small, binomial_logpmf(m, n, mu), out)


# ----------------------------------------------------------------------------
# Design matrix
# ----------------------------------------------------------------------------

def design_matrix(counts, n_alternatives):
    """Transfer matrix X with ``X[k] @ u = u[former] - u[latter]``.

    ``counts`` is a sequence of mappings with 0-based ``former``/``latter``
    indices. The last alternative serves as the reference (u = 0), so X has
    ``n_alternatives - 1`` columns.
    """
    X = np.zeros((len(counts), n_alternatives - 1), dtype=float)
    for k, c in enumerate(counts):
        i, j = int(c["former"]), int(c["latter"])
        if not (0 <= i < n_alternatives and 0 <= j < n_alternatives):
            raise ValueError(f"alternative index out of range in comparison {k}")
        if i == j:
            raise ValueError(f"comparison {k} pairs an alternative with itself")
        if i < n_alternatives - 1:
            X[k, i] += 1.0
        if j < n_alternatives - 1:
            X[k, j] -= 1.0
    return X


def _extract(counts):
    m = np.array([float(c["successes"]) for c in counts])
    n = np.array([float(c["trials"]) for c in counts])
    if np.any(n <= 0):
        raise ValueError("all comparisons need trials > 0")
    if np.any(m < 0) or np.any(m > n):
        raise ValueError("successes must lie in [0, trials]")
    return m, n


# ----------------------------------------------------------------------------
# Component-wise M step
# ----------------------------------------------------------------------------

def _component_grad(X, m, n, w, u, rho, lambda_u, u_bar, phi_mode):
    """Gradient of  -(sum_k w_k log p(m_k)) + lambda_u/2 ||u - u_bar||^2  wrt (u, rho)."""
    mu = expit(X @ u)
    if phi_mode == "binomial" or rho <= _BINOMIAL_SWITCH:
        d_eta = m - n * mu
        grad_u = -(X.T @ (w * d_eta)) + lambda_u * (u - u_bar)
        return grad_u, 0.0
    c = (1.0 - rho) / rho
    a = mu * c
    b = (1.0 - mu) * c
    d_a = (_digamma_safe(m + a) - _digamma_safe(n + a + b)
           - _digamma_safe(a) + _digamma_safe(a + b))
    d_b = (_digamma_safe(n - m + b) - _digamma_safe(n + a + b)
           - _digamma_safe(b) + _digamma_safe(a + b))
    d_mu = c * (d_a - d_b)
    d_eta = d_mu * mu * (1.0 - mu)
    d_rho = -(mu * d_a + (1.0 - mu) * d_b) / rho ** 2
    grad_u = -(X.T @ (w * d_eta)) + lambda_u * (u - u_bar)
    return grad_u, -float(np.sum(w * d_rho))


def _component_objective(X, m, n, w, u, rho, lambda_u, u_bar, phi_mode):
    mu = expit(X @ u)
    if phi_mode == "binomial":
        ll = float(np.sum(w * binomial_logpmf(m, n, mu)))
    else:
        ll = float(np.sum(w * beta_binomial_logpmf(m, n, mu, rho)))
    return -ll + 0.5 * lambda_u * float(np.sum((u - u_bar) ** 2))


def _m_step_component(X, m, n, w, u0, rho0, lambda_u, u_bar, phi_mode, maxiter=100):
    """Maximise the weighted (penalised) log-likelihood of one component."""
    if phi_mode == "binomial":
        res = minimize(
            lambda u: _component_objective(X, m, n, w, u, 0.0, lambda_u, u_bar, phi_mode),
            np.asarray(u0, dtype=float),
            jac=lambda u: _component_grad(X, m, n, w, u, 0.0, lambda_u, u_bar, phi_mode)[0],
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        return res.x, 0.0

    def fun(theta):
        return _component_objective(X, m, n, w, theta[:-1], float(theta[-1]),
                                    lambda_u, u_bar, phi_mode)

    def jac(theta):
        gu, gr = _component_grad(X, m, n, w, theta[:-1], float(theta[-1]),
                                 lambda_u, u_bar, phi_mode)
        return np.concatenate([gu, [gr]])

    x0 = np.concatenate([np.asarray(u0, dtype=float), [float(rho0)]])
    bounds = [(None, None)] * X.shape[1] + [(_RHO_MIN, _RHO_MAX)]
    res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter})
    return res.x[:-1], float(res.x[-1])


# ----------------------------------------------------------------------------
# Fitted model
# ----------------------------------------------------------------------------

@dataclass
class BinomialMoP:
    """A fitted mixture for count judgments.

    Attributes
    ----------
    utilities : (P, A-1) ndarray
        Utility vectors on the logistic scale, reference alternative at zero.
    rho : (P,) ndarray
        Overdispersion parameters; all zero when ``phi_mode="binomial"``.
    pi : (P,) ndarray
        Mixture probabilities.
    loglik : float
        Observed-data log-likelihood at the solution.
    """

    utilities: np.ndarray
    rho: np.ndarray
    pi: np.ndarray
    loglik: float
    phi_mode: str
    n_alternatives: int
    n_observations: int

    @property
    def n_components(self):
        return int(self.pi.shape[0])

    def priority_vectors(self):
        """Normalised priority vectors w_a = exp(u_a) / sum_a' exp(u_a')."""
        u = np.hstack([self.utilities, np.zeros((self.n_components, 1))])
        e = np.exp(u - u.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def rankings(self, labels=None):
        """Component orderings, best first."""
        w = self.priority_vectors()
        labels = labels if labels is not None else list(range(self.n_alternatives))
        return [" > ".join(str(labels[i]) for i in np.argsort(-row)) for row in w]


# ----------------------------------------------------------------------------
# EM
# ----------------------------------------------------------------------------

def _em_once(X, m, n, n_components, phi_mode, lambda_u, lambda_pi, rho_init,
             max_iter, tol, rng, u_bar):
    K, A1 = X.shape

    # K-Ridge on the count scale: the same seeding the transformed-Gaussian
    # member uses, with the binomial variance function, trial counts as weights,
    # and deviance residuals in place of squared ones. Passing the proportion
    # back together with the weights recovers the counts the deviance needs.
    proportion = m / np.maximum(n, 1.0)
    U, _, _ = kridge(
        X, proportion, n_components, lambda_reg=1e-10,
        family=binomial_family(), weights=n,
        deviance=lambda r, mu, w: binomial_deviance_residuals(r * w, w, mu),
    )
    # the outer multi-start still perturbs the seed it is given
    U = U + rng.normal(0.0, 0.05, U.shape)
    rho = np.full(n_components, rho_init if phi_mode != "binomial" else 0.0)
    pi = np.full(n_components, 1.0 / n_components)

    ll_old = -np.inf
    ll = -np.inf
    for _ in range(max_iter):
        # E step
        mu = expit(X @ U.T)
        if phi_mode == "binomial":
            ll_kp = binomial_logpmf(m[:, None], n[:, None], mu)
        else:
            ll_kp = beta_binomial_logpmf(m[:, None], n[:, None], mu, rho[None, :])
        log_g = np.log(np.maximum(pi, 1e-12))[None, :] + ll_kp
        mx = log_g.max(axis=1, keepdims=True)
        g = np.exp(log_g - mx)
        ll = float(np.sum(np.log(g.sum(axis=1)) + mx[:, 0]))
        g = g / g.sum(axis=1, keepdims=True)

        # M step
        for p in range(n_components):
            U[p], rho[p] = _m_step_component(X, m, n, g[:, p], U[p], rho[p],
                                             lambda_u, u_bar, phi_mode)
        pi = np.maximum((g.sum(axis=0) + lambda_pi - 1.0)
                        / (K + lambda_pi * n_components - n_components), 1e-12)
        pi = pi / pi.sum()

        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

    return U, rho, pi, ll


def fit(counts, n_alternatives, n_components, *, phi_mode="beta-binomial",
        lambda_u=1e-3, lambda_pi=1.0, rho_init=0.05, max_iter=200, tol=1e-7,
        n_starts=4, seed=0):
    """Fit a mixture of preferences to repeated binary comparisons by EM.

    Parameters
    ----------
    counts : sequence of mappings
        Each item needs 0-based ``former`` and ``latter`` indices plus integer
        ``successes`` and ``trials``.
    n_alternatives : int
        Size of the alternative set A.
    n_components : int
        Number of mixture components P.
    phi_mode : {"beta-binomial", "binomial"}
        Whether the dispersion parameter rho_p is estimated or fixed at zero.
    n_starts : int
        Restarts with different random initialisations; the highest observed
        log-likelihood is returned.
    """
    if phi_mode not in ("binomial", "beta-binomial"):
        raise ValueError("phi_mode must be 'binomial' or 'beta-binomial'")
    X = design_matrix(counts, n_alternatives)
    m, n = _extract(counts)
    u_bar = np.zeros(X.shape[1])

    best = None
    for s in range(max(1, n_starts)):
        rng = np.random.default_rng(seed + s)
        U, rho, pi, ll = _em_once(X, m, n, n_components, phi_mode, lambda_u,
                                  lambda_pi, rho_init, max_iter, tol, rng, u_bar)
        if best is None or ll > best[3]:
            best = (U, rho, pi, ll)

    U, rho, pi, ll = best
    return BinomialMoP(utilities=U, rho=rho, pi=pi, loglik=ll,
                       phi_mode=phi_mode, n_alternatives=n_alternatives,
                       n_observations=len(counts))


# ----------------------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------------------

def bic(model, *, n_observations=None, prior_terms=0.0):
    """BIC for a fitted model.

    Free parameters per component: A - 1 utilities, plus one rho in
    beta-binomial mode; plus P - 1 mixture weights. ``prior_terms`` adds the
    -2 log prior contribution used by the manuscript's pBIC, so the default 0.0
    reproduces the plain BIC.
    """
    df = model.n_components * (model.n_alternatives - 1) + (model.n_components - 1)
    if model.phi_mode == "beta-binomial":
        df += model.n_components
    n_obs = model.n_observations if n_observations is None else n_observations
    return -2.0 * model.loglik + np.log(n_obs) * df, df


def select_components(counts, n_alternatives, p_max=None, *, n_observations=None,
                      **fit_kwargs):
    """Fit P = 1..p_max and return the BIC-selected model.

    Returns ``(best_p, results)`` where ``results[p]`` holds the fitted model,
    its BIC and its degrees of freedom.
    """
    if p_max is None:
        p_max = n_alternatives
    results = {}
    for p in range(1, p_max + 1):
        model = fit(counts, n_alternatives, p, **fit_kwargs)
        bic_val, df = bic(model, n_observations=n_observations)
        results[p] = {"model": model, "bic": bic_val, "df": df}
    best_p = min(results, key=lambda k: results[k]["bic"])
    return best_p, results
