"""
Initialisation for the EM algorithm.

``kridge`` seeds the mixture components. Two regimes are supported:

* ``family=None`` -- the response ``y`` is already on the transformed scale
  (``y = g(r)``). This is the transformed-Gaussian member, and the routine
  behaves exactly as it always has.

* ``family=Family(...)`` -- ``y`` is the response ``r`` on its own scale, and
  the same construction is carried out by iteratively reweighted least squares
  on that scale, with deviance residuals standing in for squared residuals.

The Gaussian family makes the second regime collapse onto the first: its
variance function is identically one, its response function is the identity, and
a deviance residual is a squared residual. The two paths are therefore the same
algorithm, not two algorithms sharing a name.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional accelerator
    from cython_initialization import kridge_cython
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

try:
    from .pairwise_comparison import comparisons2regression
except ImportError:  # pragma: no cover
    from pairwise_comparison import comparisons2regression


@dataclass
class Family:
    """Everything the initialiser needs to know about the observation model.

    h        response function, ``h(eta) = E[r | eta]``
    h_prime  its derivative with respect to eta
    V        variance function, ``Var[r | eta] = phi * V(h(eta))``
    loglik   ``loglik(y, mu)``, the log-density of a single response at mean mu,
             used for the deviance residuals and for the Pearson estimate
    """

    h: Callable[[np.ndarray], np.ndarray]
    h_prime: Callable[[np.ndarray], np.ndarray]
    V: Callable[[np.ndarray], np.ndarray]
    loglik: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class GaussianFamily(Family):
    """The transformed-Gaussian member, written in the same language."""

    h: Callable = staticmethod(lambda eta: eta)
    h_prime: Callable = staticmethod(lambda eta: np.ones_like(eta))
    V: Callable = staticmethod(lambda mu: np.ones_like(mu))
    loglik: Callable = staticmethod(
        lambda y, mu: -0.5 * (y - mu) ** 2 - 0.5 * np.log(2.0 * np.pi)
    )


def _irls(X, r, family, weights, lambda_reg, max_iter=25, tol=1e-10):
    """Weighted ridge fit of u by iteratively reweighted least squares.

    ``weights`` are the observation weights (responsibilities, or binomial
    trial counts); the IRLS weights from the variance function multiply them.
    """
    _, d = X.shape
    u = np.zeros(d)
    prior_w = np.asarray(weights, dtype=np.float64)

    for _ in range(max_iter):
        eta = X @ u
        mu = family.h(eta)
        irw = 1.0 / np.maximum(family.V(mu), 1e-12)
        w = prior_w * irw
        z = eta + (r - mu) / np.maximum(np.abs(family.h_prime(eta)), 1e-12) * np.sign(
            family.h_prime(eta)
        )
        XtWX = X.T @ (X * w[:, None]) + lambda_reg * np.eye(d)
        XtWz = X.T @ (w * z)
        try:
            u_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            u_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]
        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u = u_new

    return u


def _deviance_residuals(r, mu, family):
    """Signed deviance residuals, the GLM analogue of |r - mu|."""
    # saturated log-likelihood minus fitted log-likelihood, doubled
    dev = 2.0 * (family.loglik(r, r) - family.loglik(r, mu))
    dev = np.maximum(dev, 0.0)
    return np.sign(r - mu) * np.sqrt(dev)


def _pearson_dispersion(r, mu, family, weights, n_params):
    """Pearson chi-square divided by the residual degrees of freedom."""
    irw = 1.0 / np.maximum(family.V(mu), 1e-12)
    w = np.asarray(weights, dtype=np.float64)
    num = float(np.sum(w * irw * (r - mu) ** 2))
    den = max(float(np.sum(w)) - n_params, 1.0)
    return num / den


def kridge(
    X: np.ndarray,
    y: np.ndarray,
    nP: int,
    lambda_reg: float = 1e-10,
    family: Optional[Family] = None,
    weights: Optional[np.ndarray] = None,
):
    """Seed ``nP`` mixture components.

    Args:
        X: design matrix, ``(nJ, nA-1)``
        y: response. ``g(r_k)`` when ``family`` is None, ``r_k`` otherwise.
        nP: number of components
        lambda_reg: ridge penalty
        family: an observation model; None means the response is already
            transformed and the routine behaves as it always has.
        weights: per-observation weights (e.g. binomial trial counts). Ones if
            not given.

    Returns:
        u: ``(nP, nA-1)`` utility vectors
        phi: ``(nP,)`` dispersion levels, on the transformed scale when
            ``family`` is None and on the response scale otherwise
        mp: ``(nP,)`` mixture weights
    """
    if family is None and USE_CYTHON:
        X_float = np.asarray(X, dtype=np.float64)
        y_float = np.asarray(y, dtype=np.float64)
        return kridge_cython(X_float, y_float, nP, lambda_reg)

    nJ, nA_minus_1 = X.shape
    nA = nA_minus_1 + 1
    r = np.asarray(y, dtype=np.float64)
    if weights is None:
        weights = np.ones(nJ)
    w_all = np.asarray(weights, dtype=np.float64)

    def fit(idx, w):
        """One component: IRLS when a family is given, ridge regression otherwise."""
        if family is None:
            XtX = X[idx].T @ (X[idx] * w[:, None]) + lambda_reg * np.eye(nA_minus_1)
            Xty = X[idx].T @ (w * r[idx])
        else:
            return _irls(X[idx], r[idx], family, w, lambda_reg)
        try:
            return np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    def residual(idx, u_p):
        """What 'large residual' means on this scale."""
        mu = X[idx] @ u_p if family is None else family.h(X[idx] @ u_p)
        if family is None:
            return (r[idx] - mu) ** 2
        return _deviance_residuals(r[idx], mu, family) ** 2

    u = np.zeros((nP, nA_minus_1))
    all_idx = np.arange(nJ)

    # first component: everything
    u[0, :] = fit(all_idx, w_all)

    # later components: the observations least explained so far
    for p in range(nP - 1):
        res = np.column_stack([residual(all_idx, u[q, :]) for q in range(p + 1)])
        worst = np.max(res, axis=1)
        n_edge = min(nA, nJ)
        threshold = np.sort(worst)[-n_edge] if n_edge > 0 else np.max(worst)
        idx = worst >= threshold
        if np.sum(idx) > 0:
            u[p + 1, :] = fit(np.where(idx)[0], w_all[idx])

    # alternate between assigning and refitting
    for _ in range(100):
        res = np.column_stack([residual(all_idx, u[p, :]) for p in range(nP)])
        cls = np.argmin(res, axis=1)
        for p in range(nP):
            mask = cls == p
            if np.sum(mask) > 0:
                u[p, :] = fit(all_idx[mask], w_all[mask])

    # dispersion: residual mean square on the transformed scale, Pearson on the response scale
    if family is None:
        phi0 = float(np.sum(w_all * (X @ u[0, :] - r) ** 2) / max(np.sum(w_all), 1.0))
    else:
        mu0 = family.h(X @ u[0, :])
        phi0 = _pearson_dispersion(r, mu0, family, w_all, nA_minus_1)

    return u, np.ones(nP) * max(phi0, 1e-10), np.ones(nP) / nP
