"""Gaussian log-ratio mixtures with the dispersion free, shared or pinned.

The three restrictions differ only in how many dispersions are estimated; the
likelihood, the prior and the criterion are the same, which is what makes the
comparison in Section 3.2 meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from mop import criteria as _criteria
from mop.initialization import kridge

EPS = 1e-300
PHI_FLOOR = 1e-10
PHI_MODES = ("free", "common", "one")


@dataclass
class MixtureFit:
    """One fitted mixture, with the quantities the criterion needs."""

    u: np.ndarray          # (K, A-1) log-ratios
    phi: np.ndarray        # (K,) dispersions, possibly all equal
    pi: np.ndarray         # (K,) mixture weights
    pll: float
    ll: float
    df: int
    n_iter: int
    converged: bool
    phi_mode: str
    lambda_u: float = 1e-6
    polish: dict | None = None

    @property
    def n_components(self) -> int:
        return self.u.shape[0]

    def priorities(self) -> np.ndarray:
        v = np.hstack([np.exp(np.clip(self.u, -50.0, 50.0)),
                       np.ones((self.n_components, 1))])
        return v / v.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {
            "k": int(self.n_components),
            "u": self.u.tolist(),
            "phi": self.phi.tolist(),
            "pi": self.pi.tolist(),
            "priorities": self.priorities().tolist(),
            "pll": float(self.pll), "ll": float(self.ll), "df": int(self.df),
            "n_iter": int(self.n_iter), "converged": bool(self.converged),
            "phi_mode": self.phi_mode,
        }


def _e_step(X, y, u, phi, pi):
    means = X @ u.T
    logp = (np.log(np.maximum(pi, EPS))[None, :]
            - 0.5 * np.log(2.0 * np.pi * phi)[None, :]
            - 0.5 * (y[:, None] - means) ** 2 / phi[None, :])
    mx = logp.max(axis=1, keepdims=True)
    g = np.exp(logp - mx)
    return g / np.maximum(g.sum(axis=1, keepdims=True), EPS)


def _penalised(X, y, u, phi, pi, lambda_u, lambda_pi, lambda_sigma, s2, phi_mode):
    """Penalised log-posterior matching ``mop.inference.penalised_log_likelihood``."""
    nJ, A1 = X.shape
    K = u.shape[0]
    comp = (np.log(np.maximum(pi, EPS))[None, :]
            - 0.5 * np.log(2.0 * np.pi * phi)[None, :]
            - 0.5 * (y[:, None] - X @ u.T) ** 2 / phi[None, :])
    mx = comp.max(axis=1, keepdims=True)
    ll = float(np.sum(mx.ravel() + np.log(np.exp(comp - mx).sum(axis=1))))
    total = ll

    for p in range(K):
        sp2 = max(phi[p], PHI_FLOOR)
        total -= (lambda_u / (2.0 * sp2)) * float(np.sum(u[p] ** 2))
        total -= 0.5 * A1 * np.log(2.0 * np.pi * sp2 / lambda_u)

    # The inverse-Gamma prior is written once for a shared dispersion and once
    # per component otherwise; the two must not be counted differently by
    # accident, since the criterion would then compare priors rather than likelihoods.
    if phi_mode in ("free", "common") and lambda_sigma:
        alpha = lambda_sigma / 2.0
        beta = lambda_sigma * s2 / 2.0
        for p in range(K if phi_mode == "free" else 1):
            sp2 = max(phi[p], PHI_FLOOR)
            total += alpha * np.log(beta) - gammaln(alpha)
            total -= (alpha + 1.0) * np.log(sp2)
            total -= beta / sp2

    total += (lambda_pi - 1.0) * float(np.sum(np.log(np.maximum(pi, EPS))))
    total += gammaln(K * lambda_pi) - K * gammaln(lambda_pi)
    return total, ll


def fit_gaussian_mixture(X, y, K, phi_mode="free", lambda_u=1e-6, lambda_pi=1.0,
                         lambda_sigma=None, s2=None, max_iter=5000, tol=1e-6,
                         u_init=None, phi_init=None, pi_init=None, polish=True):
    """EM for the Gaussian log-ratio mixture under one of three restrictions.

    Args:
        phi_mode: ``"free"`` (per component), ``"common"`` (shared), or
            ``"one"`` (fixed at one).
        lambda_sigma, s2: required unless ``phi_mode="one"``; the Inverse-Gamma
            prior that makes the likelihood proper (Section 2.4).
    """
    if phi_mode not in PHI_MODES:
        raise ValueError(f"phi_mode must be one of {PHI_MODES}, got {phi_mode!r}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    nJ, A1 = X.shape
    K = int(K)

    if phi_mode != "one":
        if lambda_sigma is None or s2 is None:
            raise ValueError("a free or shared dispersion requires lambda_sigma and s2")
        lambda_sigma, s2 = float(lambda_sigma), float(s2)

    if u_init is None:
        u0, _, _ = kridge(X, y, K)
        u = np.atleast_2d(np.asarray(u0, dtype=float)).copy()
    else:
        u = np.atleast_2d(np.asarray(u_init, dtype=float)).copy()

    phi = (np.ones(K) if phi_mode == "one"
           else (np.full(K, float(s2)) if phi_init is None
                 else np.asarray(phi_init, dtype=float).copy()))
    pi = (np.full(K, 1.0 / K) if pi_init is None
          else np.asarray(pi_init, dtype=float).copy())

    prev, n_iter, converged = None, 0, False
    for n_iter in range(1, max_iter + 1):
        g = _e_step(X, y, u, phi, pi)
        sum_g = g.sum(axis=0)

        pi = np.maximum((sum_g + lambda_pi - 1.0) / (nJ + lambda_pi * K - K), 1e-10)
        pi = pi / pi.sum()

        ss_total, n_total, prior_total = 0.0, 0.0, 0.0
        for p in range(K):
            gp = g[:, p]
            XtWX = X.T @ (X * gp[:, None]) + lambda_u * np.eye(A1)
            Xty = X.T @ (y * gp)
            try:
                u[p] = np.linalg.solve(XtWX, Xty)
            except np.linalg.LinAlgError:
                u[p] = np.linalg.lstsq(XtWX, Xty, rcond=None)[0]

            if phi_mode != "one":
                ss = float(np.sum(gp * (y - X @ u[p]) ** 2))
                ss_prior = lambda_u * float(np.sum(u[p] ** 2))
                if phi_mode == "free":
                    phi[p] = max((lambda_sigma * s2 + ss + ss_prior)
                                 / (lambda_sigma + 2.0 + float(np.sum(gp)) + A1), PHI_FLOOR)
                else:
                    ss_total += ss + ss_prior
                    n_total += float(np.sum(gp))
                    prior_total = lambda_sigma

        if phi_mode == "common":
            phi = np.full(K, max((prior_total * s2 + ss_total)
                                 / (prior_total + n_total + K * A1), PHI_FLOOR))

        pll, _ = _penalised(X, y, u, phi, pi, lambda_u, lambda_pi, lambda_sigma, s2,
                            phi_mode)
        if not np.isfinite(pll):
            break
        if prev is not None and abs(pll - prev) / (abs(pll) + 0.1) < tol:
            converged = True
            break
        prev = pll

    # Finish the mode: EM stops when the objective stops moving, which near a
    # weight of 1e-4 happens well before the gradient vanishes, and the criterion
    # reads the curvature at the mode.  ``common`` is left unpolished; it is a
    # diagnostic restriction that no table reports.
    info = None
    if polish and phi_mode in ("free", "one"):
        from mop.general_fit import GaussianResponse
        from mop.laplace import polish_mode

        def log_posterior(uu, ph, pp):
            return _penalised(X, y, uu, ph, pp, lambda_u, lambda_pi, lambda_sigma,
                              s2, phi_mode)[0]

        u, phi, pi, info = polish_mode(
            log_posterior, u, phi, pi, X, y,
            GaussianResponse() if phi_mode == "free" else GaussianResponse(phi_free=False),
            lambda_u=lambda_u, lambda_pi=lambda_pi,
            lambda_sigma=(lambda_sigma if phi_mode == "free" else None),
            s2=(s2 if phi_mode == "free" else None))

    pll, ll = _penalised(X, y, u, phi, pi, lambda_u, lambda_pi, lambda_sigma, s2,
                         phi_mode)
    df = A1 * K + (K - 1) + (K if phi_mode == "free" else (1 if phi_mode == "common" else 0))
    return MixtureFit(u=u, phi=phi, pi=pi, pll=pll, ll=ll, df=int(df),
                      n_iter=n_iter, converged=converged, phi_mode=phi_mode,
                      polish=info,
                      lambda_u=float(np.ravel(lambda_u)[0]
                                     if not np.isscalar(lambda_u) else lambda_u))


def bic(fit: MixtureFit, n_eff: float) -> float:
    """``-2 pll + d ln n`` for one fitted mixture: the criterion of the paper.

    No mode is required, so a count whose fit puts a weight on the boundary is
    still ranked; :func:`lbic` is the version that declines to rank it.
    """
    return _criteria.bic(fit.pll, n_eff, fit.df)


def pbic(fit: MixtureFit, n_eff: float, profile_utilities: bool = True) -> float:
    """``bic`` with the utility prior profiled out, for the one comparison it is for.

    ``profile_utilities`` drops the normalising constant of the utility prior;
    see ``mop.criteria``.  The fit still carries ``pll`` with that term, so it is
    removed here rather than in the fit.
    """
    pll = fit.pll
    if profile_utilities:
        pll -= _criteria.log_prior_utilities(fit.u, fit.phi, fit.lambda_u)
    return _criteria.pbic(pll, n_eff, fit.df)


def lbic(fit: MixtureFit, X, y, lambda_pi: float = 1.0,
         lambda_sigma: float | None = None, s2: float | None = None) -> float:
    """Laplace approximation of the marginal likelihood for one fitted mixture.

    ``inf`` when the fit has no interior mode or the curvature there is singular:/neither way there is nothing to take the approximation at.
    """
    from mop.general_fit import GaussianResponse
    from mop.laplace import ln_det_hessian

    if fit.polish is not None and not fit.polish.get("converged", True):
        return float("inf")

    if fit.phi_mode == "common":
        response, shared = GaussianResponse(), True
    elif fit.phi_mode == "one":
        response, shared = GaussianResponse(phi_free=False), False
    else:
        response, shared = GaussianResponse(), False
    ln_h = ln_det_hessian(X, y, fit.u, fit.phi, fit.pi, response,
                          lambda_u=fit.lambda_u, lambda_pi=lambda_pi,
                          lambda_sigma=lambda_sigma, s2=s2,
                          phi_shared=shared)
    return _criteria.lbic(fit.pll, ln_h, fit.df)


def predictive_ll(fit: MixtureFit, X, y) -> float:
    """Held-out Gaussian mixture log-likelihood, summed over judgments."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    comp = (np.log(np.maximum(fit.pi, EPS))[None, :]
            - 0.5 * np.log(2.0 * np.pi * fit.phi)[None, :]
            - 0.5 * (y[:, None] - X @ fit.u.T) ** 2 / fit.phi[None, :])
    mx = comp.max(axis=1, keepdims=True)
    return float(np.sum(mx.ravel() + np.log(np.exp(comp - mx).sum(axis=1))))


def select(X, y, grid, phi_mode="free", **kwargs):
    """Fit each candidate count and return the BIC-minimising fit and the curve."""
    fits, curve = {}, {}
    for K in grid:
        f = fit_gaussian_mixture(X, y, K, phi_mode=phi_mode, **kwargs)
        fits[K] = f
        curve[K] = bic(f, len(y))
    best_k = min(curve, key=curve.get)
    return fits[best_k], fits, curve
