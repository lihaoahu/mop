"""Direct MAP fitting of the mixture, for any observation family.

One optimiser over ``(u, phi, pi)`` with analytic derivatives, used where no
closed-form M step exists.
"""



from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp

from . import criteria as _criteria
from .mixture_of_priorities import MixtureOfPriorities, Priority
from .pairwise_comparison import comparisons2regression

EPS = 1e-300
_ETA_CLIP = 30.0



@dataclass
class GaussianResponse:
    """Gaussian judgments on the transformed scale ``y = h(r)``."""

    name: str = "gaussian"
    phi_free: bool = True

    @staticmethod
    def fitted_mean(eta):
        return eta

    @staticmethod
    def loglik(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return -0.5 * np.log(2.0 * np.pi * phi / omega) - 0.5 * omega * (y - eta) ** 2 / phi

    @staticmethod
    def dloglik_deta(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return omega * (y - eta) / phi

    @staticmethod
    def dloglik_dphi(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return -0.5 / phi + 0.5 * omega * (y - eta) ** 2 / phi ** 2

    @staticmethod
    def d2loglik_deta2(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return -omega / phi

    @staticmethod
    def d2loglik_dphi2(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return 0.5 / phi ** 2 - omega * (y - eta) ** 2 / phi ** 3

    @staticmethod
    def d2loglik_deta_dphi(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return -omega * (y - eta) / phi ** 2


@dataclass
class BinomialResponse:
    """Repeated binomial choices: ``y = m/n`` is the proportion, ``omega = n`` the trials."""

    name: str = "binomial"
    phi_free: bool = False

    @staticmethod
    def fitted_mean(eta):
        return expit(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))

    @staticmethod
    def loglik(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return omega * (y * eta - np.logaddexp(0.0, eta))

    @staticmethod
    def dloglik_deta(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        return omega * (y - expit(eta))

    @staticmethod
    def dloglik_dphi(y, eta, omega, phi):
        return np.zeros_like(np.asarray(eta, dtype=float))

    @staticmethod
    def d2loglik_deta2(y, eta, omega, phi):
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        mu = expit(eta)
        return -omega * mu * (1.0 - mu)

    @staticmethod
    def d2loglik_dphi2(y, eta, omega, phi):
        return np.zeros_like(np.asarray(eta, dtype=float))

    @staticmethod
    def d2loglik_deta_dphi(y, eta, omega, phi):
        return np.zeros_like(np.asarray(eta, dtype=float))



def _priors(u, pi, phi, u_bar, lambda_u, lambda_pi, lambda_sigma, s2, phi_free):
    """Log-prior and its derivatives."""
    nP, A1 = u.shape
    diff = u - u_bar
    total = 0.0
    grad_u = np.zeros_like(u)
    grad_phi = np.zeros(nP)

    for p in range(nP):
        total -= 0.5 * lambda_u * float(np.sum(diff[p] ** 2)) / phi[p]
        grad_u[p] += -lambda_u * diff[p] / phi[p]
        if phi_free:
            total -= 0.5 * A1 * np.log(2.0 * np.pi * phi[p] / lambda_u)
            grad_phi[p] += 0.5 * lambda_u * float(np.sum(diff[p] ** 2)) / phi[p] ** 2
            grad_phi[p] += -0.5 * A1 / phi[p]
        else:
            total -= 0.5 * A1 * np.log(2.0 * np.pi / lambda_u)

    pi_safe = np.maximum(pi, EPS)
    total += (lambda_pi - 1.0) * float(np.sum(np.log(pi_safe)))
    total += gammaln(nP * lambda_pi) - nP * gammaln(lambda_pi)

    if phi_free:
        # InvGamma(lambda_sigma/2, lambda_sigma * phi0/2): lambda_sigma is the
        # strength, phi0 the scale, the mode is lambda_sigma*phi0/(lambda_sigma+2)
        # and the density tends to phi^-1 as lambda_sigma -> 0.
        alpha = lambda_sigma / 2.0
        beta = lambda_sigma * s2 / 2.0
        for p in range(nP):
            total += alpha * np.log(beta) - gammaln(alpha)
            total -= (alpha + 1.0) * np.log(phi[p])
            total -= beta / phi[p]
            grad_phi[p] += -(alpha + 1.0) / phi[p] + beta / phi[p] ** 2

    return total, grad_u, grad_phi


def _objective(theta, X, y, omega, nP, response, u_bar, lambda_u, lambda_pi,
               lambda_sigma, s2):
    """Penalised log-posterior and its gradient with respect to ``theta``."""
    nJ, A1 = X.shape
    phi_free = response.phi_free
    k = nP * A1
    u = theta[:k].reshape(nP, A1)
    z = theta[k:k + nP - 1]
    if phi_free:
        phi = np.exp(theta[k + nP - 1:])
    else:
        phi = np.ones(nP)

    # shifted softmax: the shift cancels in the normalisation and keeps exp finite
    z_full = np.concatenate([z, [0.0]])
    pi = np.exp(z_full - z_full.max())
    pi /= pi.sum()

    eta = X @ u.T
    ll = response.loglik(y[:, None], eta, omega[:, None], phi[None, :])
    logp = ll + np.log(np.maximum(pi, EPS))[None, :]
    norm = logsumexp(logp, axis=1)
    total = float(norm.sum())

    prior, grad_u_prior, grad_phi_prior = _priors(
        u, pi, phi, u_bar, lambda_u, lambda_pi, lambda_sigma, s2, phi_free)
    total += prior

    gamma = np.exp(logp - norm[:, None])

    deta = response.dloglik_deta(y[:, None], eta, omega[:, None], phi[None, :])
    grad_u = (gamma * deta).T @ X + grad_u_prior

    grad_z = gamma[:, :-1].sum(axis=0) - nJ * pi[:-1]
    grad_z += (lambda_pi - 1.0) * (1.0 - nP * pi[:-1])

    grad = [grad_u.ravel(), grad_z]

    if phi_free:
        dphi = response.dloglik_dphi(y[:, None], eta, omega[:, None], phi[None, :])
        grad_phi = (gamma * dphi).sum(axis=0) + grad_phi_prior
        grad.append(grad_phi * phi)
    return total, np.concatenate([g for g in grad if g.size])



def _pooled_u(X, y, omega, response, lambda_u, u_bar, weight=None):
    """One-component start by weighted ridge (Gaussian) or ridge logistic (binomial)."""
    A1 = X.shape[1]
    if weight is not None:
        omega = omega * np.asarray(weight, dtype=float)
    if isinstance(response, GaussianResponse):
        w = omega
        XtW = X.T * w
        return np.linalg.solve(XtW @ X + lambda_u * np.eye(A1), XtW @ (y - X @ u_bar)) + u_bar

    u = u_bar.copy()

    def neg(u_):
        eta = np.clip(X @ u_, -_ETA_CLIP, _ETA_CLIP)
        return -float(np.sum(omega * (y * eta - np.logaddexp(0.0, eta)))) \
            + 0.5 * lambda_u * float(np.sum((u_ - u_bar) ** 2))

    def grad(u_):
        eta = np.clip(X @ u_, -_ETA_CLIP, _ETA_CLIP)
        return -(X.T @ (omega * (y - expit(eta)))) + lambda_u * (u_ - u_bar)

    return minimize(neg, u, jac=grad, method="L-BFGS-B").x


def _starts(X, y, omega, response, nP, lambda_u, u_bar, phi_init, n_starts):
    """Deterministic starts: pooled fit, residual-sign split, then jittered copies."""
    A1 = X.shape[1]
    phi0 = float(phi_init if phi_init is not None else 1.0)
    u_pool = _pooled_u(X, y, omega, response, lambda_u, u_bar)
    u0 = np.tile(u_pool, (nP, 1))
    out = [(u0.copy(), np.full(nP, 1.0 / nP), np.full(nP, phi0))]

    if nP > 1:
        resid = y - response.fitted_mean(X @ u_pool)
        pos, neg = resid > 0, resid < 0
        if pos.sum() > A1 + 2 and neg.sum() > A1 + 2:
            u_pos = _pooled_u(X[pos], y[pos], omega[pos], response, lambda_u, u_bar)
            u_neg = _pooled_u(X[neg], y[neg], omega[neg], response, lambda_u, u_bar)
            base = [u_pos, u_neg, (u_pos + u_neg) / 2.0]
            rng_split = np.random.default_rng(97 + nP)
            u_split = np.vstack([base[k % len(base)] + (rng_split.normal(0, 0.2, A1)
                                                        if k >= len(base) else 0.0)
                                 for k in range(nP)])
            out.append((u_split, np.full(nP, 1.0 / nP), np.full(nP, phi0)))

    rng = np.random.default_rng(20260119 + nP)
    for s in range(len(out), max(n_starts, len(out))):
        out.append((u0 + rng.normal(0, 0.6, u0.shape), np.full(nP, 1.0 / nP),
                    np.full(nP, phi0 * (1.0 + 0.3 * (s % 2)))))
    return out



def fit_map(pcvector, response, nP, lambda_u=1e-6, lambda_pi=1.0, lambda_sigma=None,
            s2=None, u_bar=None, omega=None, u_init=None, pi_init=None, phi_init=None,
            n_starts=6, max_iter=5000, tol=1e-9):
    """Maximise the penalised log-posterior and return the fitted mixture.

    ``s2`` and ``lambda_sigma`` are required when ``response.phi_free`` and ignored
    otherwise; ``omega`` carries the trial counts for the binomial family.
    """
    X, y, _ = comparisons2regression(pcvector, [float])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nJ, A1 = X.shape

    if omega is None:
        omega = np.ones(nJ)
    omega = np.asarray(omega, dtype=np.float64).ravel()
    if omega.size != nJ:
        raise ValueError(f"omega has {omega.size} entries but there are {nJ} judgments")
    if np.any(omega <= 0):
        raise ValueError("omega must be positive")

    if u_bar is None:
        u_bar = np.zeros(A1)
    u_bar = np.asarray(u_bar, dtype=np.float64).ravel()
    if u_bar.size != A1:
        raise ValueError(f"u_bar has {u_bar.size} entries but the design has {A1} columns")

    if response.phi_free:
        if lambda_sigma is None or s2 is None:
            raise ValueError("a free dispersion requires lambda_sigma and s2")
        lambda_sigma = float(lambda_sigma)
        s2 = float(s2)

    if u_init is not None:
        u0 = np.atleast_2d(np.asarray(u_init, dtype=np.float64))
        if u0.shape != (nP, A1):
            raise ValueError(f"u_init has shape {u0.shape}, expected {(nP, A1)}")
        pi0 = np.full(nP, 1.0 / nP) if pi_init is None else np.asarray(pi_init, dtype=np.float64).ravel()
        if pi0.size != nP:
            raise ValueError(f"pi_init has {pi0.size} entries but nP={nP}")
        if np.any(pi0 < 0) or not np.isclose(pi0.sum(), 1.0, atol=1e-8):
            raise ValueError("pi_init must be non-negative and sum to one")
        if response.phi_free and phi_init is not None:
            if np.ndim(phi_init) != 0:
                raise ValueError("phi_init must be a scalar")
        phi0 = np.full(nP, 1.0 if not response.phi_free
                        else (1.0 if phi_init is None else float(phi_init)))
        starts = [(u0, pi0, phi0)]
    else:
        starts = _starts(X, y, omega, response, nP, lambda_u, u_bar, phi_init, n_starts)

    def pack(u, pi, phi):
        logpi = np.log(np.maximum(pi, EPS))
        z = logpi[:-1] - logpi[-1]
        parts = [u.ravel(), z]
        if response.phi_free:
            parts.append(np.log(np.maximum(phi, 1e-12)))
        return np.concatenate(parts)

    best = None
    for u0, pi0, phi0 in starts:
        theta0 = pack(u0, pi0, phi0)
        res = minimize(
            lambda th: -_objective(th, X, y, omega, nP, response, u_bar, lambda_u,
                                   lambda_pi, lambda_sigma, s2)[0],
            theta0,
            jac=lambda th: -_objective(th, X, y, omega, nP, response, u_bar, lambda_u,
                                       lambda_pi, lambda_sigma, s2)[1],
            method="L-BFGS-B", options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-10},
        )
        pll = -float(res.fun)
        if best is None or pll > best[0]:
            best = (pll, res)

    pll, res = best
    u, pi, phi = _unpack(res.x, nP, A1, response.phi_free)
    v = np.hstack([np.exp(np.clip(u, -50, 50)), np.ones((nP, 1))])
    w = v / v.sum(axis=1, keepdims=True)
    priorities = [Priority(w[p], float(phi[p])) for p in range(nP)]
    model = MixtureOfPriorities(priorities, pi)
    model.converged = bool(res.success)
    model.n_iter = int(res.nit)
    model.penalized_log_posterior = float(pll)
    return model


def _unpack(theta, nP, A1, phi_free):
    k = nP * A1
    u = theta[:k].reshape(nP, A1)
    z = np.concatenate([theta[k:k + nP - 1], [0.0]])
    pi = np.exp(z - z.max())
    pi /= pi.sum()
    phi = np.exp(theta[k + nP - 1:]) if phi_free else np.ones(nP)
    return u, pi, phi


def lbic_map(pcvector, model, response, omega=None, lambda_u=1e-6, lambda_pi=1.0,
             lambda_sigma=None, s2=None, u_bar=None, phi_shared=False):
    """Laplace approximation of the marginal likelihood for a :func:`fit_map` fit."""
    from .laplace import ln_det_hessian

    X, y, _ = comparisons2regression(pcvector, [float])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nJ, A1 = X.shape
    nP = len(model.mixture_probabilities)
    if omega is None:
        omega = np.ones(nJ)
    omega = np.asarray(omega, dtype=np.float64).ravel()

    u = np.array([np.log(np.asarray(model[p].priority_vector[:A1], dtype=np.float64)
                         / np.asarray(model[p].priority_vector[A1], dtype=np.float64))
                  for p in range(nP)])
    pi = np.asarray(model.mixture_probabilities, dtype=np.float64)
    phi = np.array([model[p].inconsistency_level for p in range(nP)], dtype=np.float64)
    logpi = np.log(np.maximum(pi, EPS))
    theta = np.concatenate([
        u.ravel(), logpi[:-1] - logpi[-1],
        (np.log(phi) if response.phi_free else np.array([])),
    ])
    if u_bar is None:
        u_bar = np.zeros(A1)
    pll, _ = _objective(theta, X, y, omega, nP, response,
                        np.asarray(u_bar, dtype=np.float64),
                        lambda_u, lambda_pi, lambda_sigma, s2)
    df = _criteria.degrees_of_freedom(A1 + 1, nP, bool(response.phi_free))
    lnH = ln_det_hessian(X, y, u, phi, pi, response, omega=omega,
                         u_bar=np.asarray(u_bar, dtype=np.float64),
                         lambda_u=lambda_u, lambda_pi=lambda_pi,
                         lambda_sigma=lambda_sigma, s2=s2,
                         phi_shared=phi_shared)
    return _criteria.lbic(pll, lnH, df)


def pbic_map(pcvector, model, response, omega=None, lambda_u=1e-6, lambda_pi=1.0,
             lambda_sigma=None, s2=None, u_bar=None, profile_utilities=True):
    """Penalised BIC of a :func:`fit_map` fit, with the utility prior profiled out."""
    X, y, _ = comparisons2regression(pcvector, [float])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nJ, A1 = X.shape
    nP = len(model.mixture_probabilities)
    n_eff = float(nJ if omega is None else np.sum(np.asarray(omega, dtype=np.float64)))
    if omega is None:
        omega = np.ones(nJ)
    omega = np.asarray(omega, dtype=np.float64).ravel()

    u = np.array([np.log(np.asarray(model[p].priority_vector[:A1], dtype=np.float64)
                         / np.asarray(model[p].priority_vector[A1], dtype=np.float64))
                  for p in range(nP)])
    pi = np.asarray(model.mixture_probabilities, dtype=np.float64)
    phi = np.array([model[p].inconsistency_level for p in range(nP)], dtype=np.float64)
    logpi = np.log(np.maximum(pi, EPS))
    theta = np.concatenate([
        u.ravel(), logpi[:-1] - logpi[-1],
        (np.log(phi) if response.phi_free else np.array([])),
    ])
    if u_bar is None:
        u_bar = np.zeros(A1)
    pll, _ = _objective(theta, X, y, omega, nP, response, np.asarray(u_bar, dtype=np.float64),
                        lambda_u, lambda_pi, lambda_sigma, s2)
    df = _criteria.degrees_of_freedom(A1 + 1, nP, bool(response.phi_free))
    u_log_prior = (_criteria.log_prior_utilities(
        u, phi, lambda_u, np.asarray(u_bar, dtype=np.float64))
        if profile_utilities else 0.0)
    return _criteria.pbic(pll, n_eff, df, u_log_prior)


def bic_map(pcvector, model, response, omega=None, lambda_u=1e-6, lambda_pi=1.0,
            lambda_sigma=None, s2=None, u_bar=None):
    """``-2 pll + d ln(omega.sum())`` for a :func:`fit_map` fit."""
    return pbic_map(pcvector, model, response, omega=omega, lambda_u=lambda_u,
                    lambda_pi=lambda_pi, lambda_sigma=lambda_sigma, s2=s2,
                    u_bar=u_bar, profile_utilities=False)
