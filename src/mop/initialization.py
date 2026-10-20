"""Initialisation for the EM algorithm."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from .pairwise_comparison import comparisons2regression
except ImportError:  # pragma: no cover
    from pairwise_comparison import comparisons2regression


@dataclass
class Family:
    """Everything the initialiser needs to know about the observation model."""

    h: Callable[[np.ndarray], np.ndarray]
    h_prime: Callable[[np.ndarray], np.ndarray]
    V: Callable[[np.ndarray], np.ndarray]
    loglik: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class GaussianFamily(Family):
    """The Gaussian family for an already-transformed response."""

    h: Callable = staticmethod(lambda eta: eta)
    h_prime: Callable = staticmethod(lambda eta: np.ones_like(eta))
    V: Callable = staticmethod(lambda mu: np.ones_like(mu))
    loglik: Callable = staticmethod(
        lambda y, mu: -0.5 * (y - mu) ** 2 - 0.5 * np.log(2.0 * np.pi)
    )


def _irls(X, r, family, weights, lambda_reg, max_iter=25, tol=1e-10):
    """Weighted ridge fit of u by iteratively reweighted least squares."""
    _, d = X.shape
    u = np.zeros(d)
    prior_w = np.asarray(weights, dtype=np.float64)

    for _ in range(max_iter):
        eta = X @ u
        mu = family.h(eta)
        irw = family.h_prime(eta) ** 2 / np.maximum(family.V(mu), 1e-12)
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


def binomial_deviance_residuals(m, n, mu):
    """Signed deviance residuals for counts."""
    m = np.asarray(m, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    nm = n - m
    t1 = np.where(m > 0, m * np.log(np.maximum(m, 1e-300) / (n * mu)), 0.0)
    t2 = np.where(nm > 0, nm * np.log(np.maximum(nm, 1e-300) / (n * (1.0 - mu))), 0.0)
    dev = 2.0 * (t1 + t2)
    return np.sign(m - n * mu) * np.sqrt(np.maximum(dev, 0.0))


def _logistic(eta: np.ndarray) -> np.ndarray:
    """Numerically stable logistic; exp(-eta) overflows for large eta."""
    eta = np.asarray(eta, dtype=np.float64)
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    ex = np.exp(eta[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def binomial_family() -> Family:
    """The logistic model for counts, in the language the initialiser expects."""
    return Family(
        h=_logistic,
        h_prime=lambda eta: _logistic(eta) * (1.0 - _logistic(eta)),
        V=lambda mu: mu * (1.0 - mu),
        loglik=lambda y, mu: np.zeros_like(np.asarray(y, dtype=np.float64)),
    )


def _component_fit(X, r, weights, family, lambda_reg):
    """Return ``fit(idx, w)``: the weighted fit of a single component."""
    nA_minus_1 = X.shape[1]

    def fit(idx, w):
        if family is None:
            XtX = X[idx].T @ (X[idx] * w[:, None]) + lambda_reg * np.eye(nA_minus_1)
            Xty = X[idx].T @ (w * r[idx])
            try:
                return np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        return _irls(X[idx], r[idx], family, w, lambda_reg)

    return fit


def _component_residual(X, r, weights, family, deviance):
    """Return ``residual(idx, u)``: what 'large residual' means on this scale."""
    def residual(idx, u_p):
        eta = X[idx] @ u_p
        if family is None:
            return (r[idx] - eta) ** 2
        if deviance is not None:
            return deviance(r[idx], family.h(eta), weights[idx]) ** 2
        return _deviance_residuals(r[idx], family.h(eta), family) ** 2

    return residual


def ridge_init(X, y, family=None, weights=None, lambda_reg=1e-10):
    """The ``P = 1`` initialisation: one weighted ridge regression, all judgments."""
    nJ = X.shape[0]
    r = np.asarray(y, dtype=np.float64)
    w_all = np.ones(nJ) if weights is None else np.asarray(weights, dtype=np.float64)
    return _component_fit(X, r, w_all, family, lambda_reg)(np.arange(nJ), w_all)


def seed_component(X, y, u_prev, family=None, weights=None, lambda_reg=1e-10,
                   deviance=None):
    """The K-Ridge step: one further component, where the fit is worst."""
    nJ, nA_minus_1 = X.shape
    r = np.asarray(y, dtype=np.float64)
    w_all = np.ones(nJ) if weights is None else np.asarray(weights, dtype=np.float64)
    residual = _component_residual(X, r, w_all, family, deviance)
    all_idx = np.arange(nJ)

    prev = np.atleast_2d(np.asarray(u_prev, dtype=np.float64))
    worst = np.min(np.column_stack([residual(all_idx, u) for u in prev]), axis=1)
    n_edge = min(nA_minus_1 + 1, nJ)
    idx = np.where(worst >= np.sort(worst)[-n_edge])[0]
    if idx.size == 0:
        return np.zeros(nA_minus_1)
    return _component_fit(X, r, w_all, family, lambda_reg)(idx, w_all[idx])


def kridge(
    X: np.ndarray,
    y: np.ndarray,
    nP: int,
    lambda_reg: float = 1e-10,
    family: Optional[Family] = None,
    weights: Optional[np.ndarray] = None,
    deviance: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
):
    """Seed ``nP`` components from scratch, then refine them by hard assignment.

    Args:
        X: design matrix, ``(nJ, nA-1)``
        y: response. ``g(r_k)`` when ``family`` is None, ``r_k`` otherwise.
        nP: number of components
        lambda_reg: ridge penalty
        family: an observation model; None means the response is already
            transformed.
        weights: per-observation weights (e.g. binomial trial counts). Ones if
            not given.
        deviance: ``deviance(y, mu, weights)`` returning signed deviance
            residuals, needed when the response is a count.

    Returns:
        u: ``(nP, nA-1)`` utility vectors
        phi: ``(nP,)`` dispersion levels
        mp: ``(nP,)`` mixture weights
    """
    nJ, nA_minus_1 = X.shape
    r = np.asarray(y, dtype=np.float64)
    w_all = np.ones(nJ) if weights is None else np.asarray(weights, dtype=np.float64)
    fit = _component_fit(X, r, w_all, family, lambda_reg)
    residual = _component_residual(X, r, w_all, family, deviance)
    all_idx = np.arange(nJ)

    u = np.zeros((nP, nA_minus_1))
    u[0, :] = fit(all_idx, w_all)

    for p in range(nP - 1):
        res = np.column_stack([residual(all_idx, u[q, :]) for q in range(p + 1)])
        worst = np.min(res, axis=1)
        n_edge = min(nA_minus_1 + 1, nJ)
        threshold = np.sort(worst)[-n_edge] if n_edge > 0 else np.max(worst)
        idx = np.where(worst >= threshold)[0]
        if idx.size > 0:
            u[p + 1, :] = fit(idx, w_all[idx])

    prev_cls = None
    for _ in range(100):
        res = np.column_stack([residual(all_idx, u[p, :]) for p in range(nP)])
        cls = np.argmin(res, axis=1)
        if prev_cls is not None and np.array_equal(cls, prev_cls):
            break
        prev_cls = cls
        for p in range(nP):
            mask = cls == p
            if np.sum(mask) > 0:
                u[p, :] = fit(all_idx[mask], w_all[mask])

    if family is None:
        phi0 = float(np.sum(w_all * (X @ u[0, :] - r) ** 2) / max(np.sum(w_all), 1.0))
    else:
        mu0 = family.h(X @ u[0, :])
        phi0 = _pearson_dispersion(r, mu0, family, w_all, nA_minus_1)

    return u, np.ones(nP) * max(phi0, 1e-10), np.ones(nP) / nP
