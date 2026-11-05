"""A second, hand-written EM, used to cross-check the library estimator.

The formulas of Section 2.3 are re-derived here independently, so an error in
one implementation shows up as a disagreement rather than as a wrong number.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logit

from mop.criteria import degrees_of_freedom as _df      # noqa: F401
from mop.criteria import log_prior_utilities as _lpu
from mop.criteria import pbic as _pbic
from mop.initialization import kridge
from mop.pairwise_comparison import comparisons2regression

from .params import LAMBDA_SIGMA_OVER_N
from .params import lambda_pi as LAMBDA_PI
from .params import lambda_u as LAMBDA_U
from .paths import T

EPS = 1e-10
CLIP = 50.0


def build_design(pairs):
    """Design matrix and logit-transformed responses for the ratio protocol."""
    from mop.pairwise_comparison import PairwiseComparison
    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    pcv = [PairwiseComparison(former_id=name_to_id[p["ft"]],
                              latter_id=name_to_id[p["lt"]],
                              agent_id=1, judgment=p["score"]) for p in pairs]
    link = [lambda s, _eps=EPS: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcv, link)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)



def _e_step(X, y, u, phi, pi):
    """Allocation posterior, `eq:e_gamma`."""
    means = X @ u.T
    logp = np.log(np.maximum(pi, EPS))[None, :] \
        - 0.5 * np.log(2 * np.pi * phi)[None, :] \
        - 0.5 * ((y[:, None] - means) ** 2) / phi[None, :]
    mx = logp.max(axis=1, keepdims=True)
    g = np.exp(logp - mx)
    return g / np.maximum(g.sum(axis=1, keepdims=True), EPS)


def _penalized(X, y, u, phi, pi, lam_u, lam_sigma, lam_pi, s2, u_bar,
               free_phi):
    """Penalized log-posterior, the quantity pBIC penalizes (`eq:pbic`)."""
    from scipy.special import gammaln
    nJ, A1 = X.shape
    nP = u.shape[0]
    means = X @ u.T
    comp = (np.log(np.maximum(pi, EPS))[None, :]
            - 0.5 * np.log(2 * np.pi * phi)[None, :]
            - 0.5 * ((y[:, None] - means) ** 2) / phi[None, :])
    mx = comp.max(axis=1, keepdims=True)
    ll = float(np.sum(mx.ravel() + np.log(np.exp(comp - mx).sum(axis=1))))
    total = ll
    a = lam_sigma / 2.0
    b = lam_sigma * s2 / 2.0
    # The inverse-Gamma term is the prior density of an estimated dispersion.
    # When the dispersion is pinned there is no such parameter, so including it
    # would charge the pinned specification a constant it does not have -- about
    # 13 nats per component here, which is enough to change the selected K.
    use_phi_prior = free_phi and lam_sigma > 0.0
    lg_b = np.log(b) if use_phi_prior else 0.0
    for p in range(nP):
        sp2 = max(phi[p], 1e-300)
        diff = u[p] - u_bar[p]
        total -= (lam_u / (2.0 * sp2)) * float(np.sum(diff ** 2))
        total -= (A1 / 2.0) * np.log(2.0 * np.pi * sp2 / lam_u)
        if use_phi_prior:
            total += a * lg_b - gammaln(a) - (a + 1.0) * np.log(sp2) - b / sp2
    total += (lam_pi - 1.0) * float(np.sum(np.log(np.maximum(pi, EPS))))
    total += gammaln(nP * lam_pi) - nP * gammaln(lam_pi)
    return total, ll


def fit(X, y, nP, free_phi, lam_u=LAMBDA_U, lam_pi=LAMBDA_PI, lam_sigma=None,
        u_init=None, phi_init=None, pi_init=None, s2=None, u_bar=None,
        max_iter=5000, tol=1e-6, phi_fixed=None):
    """EM for the Gaussian (ratio) variant, with phi_p estimated or fixed."""
    nJ, A1 = X.shape
    if s2 is None:
        s2 = 1.0
    if lam_sigma is None:
        lam_sigma = LAMBDA_SIGMA_OVER_N * float(nJ)


    if u_init is None:
        u0, phi0, pi0 = kridge(X, y, nP)
        u = np.atleast_2d(u0)
        phi = np.full(nP, float(s2)) if phi_init is None else np.asarray(phi_init, float)
        pi = np.full(nP, 1.0 / nP) if pi_init is None else np.asarray(pi_init, float)
    else:
        u = np.atleast_2d(np.asarray(u_init, dtype=float)).copy()
        nP = u.shape[0]
        phi = (np.full(nP, float(s2)) if phi_init is None
               else np.asarray(phi_init, dtype=float).copy())
        pi = (np.full(nP, 1.0 / nP) if pi_init is None
              else np.asarray(pi_init, dtype=float).copy())

    if u_bar is None:
        u_bar = np.zeros((nP, A1))
    else:
        u_bar = np.asarray(u_bar, dtype=float)
        if u_bar.ndim == 1:
            u_bar = np.tile(u_bar, (nP, 1))
        if u_bar.shape[0] != nP:
            raise ValueError(
                f"u_bar has {u_bar.shape[0]} rows but the fit has {nP} components; "
                f"u_init changed the component count after u_bar was supplied")

    if not free_phi:
        phi = (np.ones(nP) if phi_fixed is None
               else np.full(nP, float(phi_fixed)))

    prev, n_iter, converged = None, 0, False
    for n_iter in range(1, max_iter + 1):
        g = _e_step(X, y, u, phi, pi)
        sum_g = g.sum(axis=0)

        pi = (sum_g + lam_pi - 1.0) / (nJ + lam_pi * nP - nP)
        pi = np.maximum(pi, 1e-10)
        pi = pi / pi.sum()

        for p in range(nP):
            gp = g[:, p]
            XtWX = X.T @ (X * gp[:, None]) + lam_u * np.eye(A1)
            Xty = X.T @ (y * gp) + lam_u * u_bar[p]
            try:
                u[p] = np.linalg.solve(XtWX, Xty)
            except np.linalg.LinAlgError:
                u[p] = np.linalg.lstsq(XtWX, Xty, rcond=None)[0]

            if free_phi:
                ss = float(np.sum(gp * (y - X @ u[p]) ** 2))
                ss_prior = lam_u * float(np.sum((u[p] - u_bar[p]) ** 2))
                phi[p] = max((lam_sigma * s2 + ss + ss_prior)
                             / (lam_sigma + 2.0 + float(np.sum(gp)) + A1), 1e-10)

        pll, ll = _penalized(X, y, u, phi, pi, lam_u, lam_sigma, lam_pi, s2,
                             u_bar, free_phi)
        if not np.isfinite(pll):
            break
        if prev is not None and abs(pll - prev) / (abs(pll) + 0.1) < tol:
            converged = True
            break
        prev = pll

    pll, ll = _penalized(X, y, u, phi, pi, lam_u, lam_sigma, lam_pi, s2, u_bar,
                         free_phi)

    # Finish the mode.  EM stops when the objective stops moving, which near a
    # weight of 1e-4 happens well before the gradient vanishes; the criterion
    # reads the curvature, so it has to be read where the gradient is zero.
    from mop.general_fit import GaussianResponse
    from mop.laplace import polish_mode

    def log_posterior(uu, ph, pp):
        return _penalized(X, y, uu, ph, pp, lam_u, lam_sigma, lam_pi, s2, u_bar,
                          free_phi)[0]

    u, phi, pi, info = polish_mode(
        log_posterior, u, phi, pi, X, y,
        GaussianResponse() if free_phi else GaussianResponse(phi_free=False),
        u_bar=u_bar, lambda_u=lam_u, lambda_pi=lam_pi,
        lambda_sigma=(lam_sigma if free_phi else None),
        s2=(s2 if free_phi else None))
    pll, ll = _penalized(X, y, u, phi, pi, lam_u, lam_sigma, lam_pi, s2, u_bar,
                         free_phi)
    return {"u": u, "phi": phi, "pi": pi, "pll": pll, "ll": ll,
            "n_iter": n_iter, "converged": converged, "s2": s2,
            "polish": info}


def degrees_of_freedom(A, nP, free_phi):
    """Free parameters: utilities, weights, and dispersions only when estimated."""
    return _df(A, nP, free_phi)


def criteria(fitres, X, y, A, nP, free_phi):
    nJ = X.shape[0]
    df = degrees_of_freedom(A, nP, free_phi)
    u_prior = _lpu(fitres["u"], fitres["phi"], LAMBDA_U)
    return {"BIC": -2.0 * fitres["pll"] + np.log(nJ) * df,
            "BIC_profiled": -2.0 * fitres["ll"] + np.log(nJ) * df,
            "pBIC": _pbic(fitres["pll"], nJ, df, u_prior),
            "pBIC_unprofiled": _pbic(fitres["pll"], nJ, df),
            "df": df,
            "loglik": fitres["ll"], "pll": fitres["pll"],
            "u_prior": u_prior}
