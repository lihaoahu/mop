"""MAP fitting of the mixture of preferences by direct numerical optimisation.

Two estimation routes exist in this package and they answer the same question.

* :func:`fit_map` here maximises the penalised log-posterior over every parameter at once --
  utilities, mixture weights and, where the family allows it, the dispersion. It knows nothing
  about conjugacy, so it applies to any observation family. This is the route for the binomial
  family and for any protocol whose likelihood does not factor into closed-form updates.
* :func:`mop.inference.mop` is the Gaussian variant's closed-form iteration: that family is
  conjugate, the E-step and M-step are analytic, and the iteration is exact and cheaper. It is
  the route the case study uses.

The two must agree where both apply. With unit observation weights they share the same
objective, so :func:`fit_map` on Gaussian data is a check on :func:`mop.inference.mop` as much
as a fitting routine, and the tests compare them.

A note on the weight convention. This module writes the Gaussian likelihood as Eq (2) states
it, ``Var[y_k] = phi V(mu_k) / omega_k``, so a judgment resting on ``omega_k`` repetitions
contributes ``-0.5 log(2 pi phi / omega_k) - 0.5 omega_k (y_k - eta_k)^2 / phi``. With
``omega_k = 1`` this is exactly the density :mod:`mop.inference` uses. For ``omega_k > 1`` that
module instead multiplies the unit-weight log-density by ``omega_k`` (a frequency weight), which
is a different model; the two agree only at unit weights, and the difference is stated here
rather than hidden.

Usage::

    from mop.general_fit import BinomialResponse, fit_map, pbic_map
    model = fit_map(pcvector, BinomialResponse(), nP=2, omega=trials)
    criterion = pbic_map(pcvector, model, BinomialResponse(), omega=trials)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp

from .mixture_of_priorities import MixtureOfPriorities, Priority
from .pairwise_comparison import comparisons2regression

EPS = 1e-300
_ETA_CLIP = 30.0


# ── observation families ────────────────────────────────────────────────────────────────────────

@dataclass
class GaussianResponse:
    """Gaussian judgments already on the transformed scale ``y = h(r)`` (Eq 2)."""

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


@dataclass
class BinomialResponse:
    """Repeated binomial choices: ``y = m/n`` is the proportion, ``omega = n`` the trials.

    ``phi`` is fixed at one (Section 2.3), so it is not a parameter, contributes no gradient, and
    the criterion still counts it among the degrees of freedom.
    """

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


# ── the penalised log-posterior, written once ───────────────────────────────────────────────────

def _priors(u, pi, phi, u_bar, lambda_u, lambda_pi, lambda_sigma, s2, phi_free):
    """Log-prior and its derivatives, matching eq:prior in the manuscript."""
    nP, A1 = u.shape
    diff = u - u_bar
    total = 0.0
    grad_u = np.zeros_like(u)
    grad_phi = np.zeros(nP)

    # u_p | phi_p ~ N(u_0, phi_p / lambda_u I), eq:u
    for p in range(nP):
        total -= 0.5 * lambda_u * float(np.sum(diff[p] ** 2)) / phi[p]
        grad_u[p] += -lambda_u * diff[p] / phi[p]
        if phi_free:
            total -= 0.5 * A1 * np.log(2.0 * np.pi * phi[p] / lambda_u)
            grad_phi[p] += 0.5 * lambda_u * float(np.sum(diff[p] ** 2)) / phi[p] ** 2
            grad_phi[p] += -0.5 * A1 / phi[p]
        else:
            total -= 0.5 * A1 * np.log(2.0 * np.pi / lambda_u)

    # pi ~ Dirichlet(lambda_pi), eq:pi
    pi_safe = np.maximum(pi, EPS)
    total += (lambda_pi - 1.0) * float(np.sum(np.log(pi_safe)))
    total += gammaln(nP * lambda_pi) - nP * gammaln(lambda_pi)

    # phi_p ~ InvGamma((lambda_sigma - 2)/2, lambda_sigma phi_0 / 2), eq:phi
    if phi_free:
        alpha = (lambda_sigma - 2.0) / 2.0
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

    pi = np.exp(np.concatenate([z, [0.0]]))
    pi /= pi.sum()

    eta = X @ u.T                                    # (nJ, nP)
    ll = response.loglik(y[:, None], eta, omega[:, None], phi[None, :])
    logp = ll + np.log(np.maximum(pi, EPS))[None, :]
    norm = logsumexp(logp, axis=1)
    total = float(norm.sum())

    prior, grad_u_prior, grad_phi_prior = _priors(
        u, pi, phi, u_bar, lambda_u, lambda_pi, lambda_sigma, s2, phi_free)
    total += prior

    gamma = np.exp(logp - norm[:, None])              # responsibilities (nJ, nP)

    # d/d u_pj
    deta = response.dloglik_deta(y[:, None], eta, omega[:, None], phi[None, :])
    grad_u = (gamma * deta).T @ X + grad_u_prior

    # d/d z_q, reference parameterisation (last logit fixed at zero)
    grad_z = gamma[:, :-1].sum(axis=0) - nJ * pi[:-1]
    grad_z += (lambda_pi - 1.0) * (1.0 - nP * pi[:-1])

    grad = [grad_u.ravel(), grad_z]

    if phi_free:
        dphi = response.dloglik_dphi(y[:, None], eta, omega[:, None], phi[None, :])
        grad_phi = (gamma * dphi).sum(axis=0) + grad_phi_prior
        grad.append(grad_phi * phi)                   # chain rule through log phi
    return total, np.concatenate([g for g in grad if g.size])


# ── starts ─────────────────────────────────────────────────────────────────────────────────────

def _pooled_u(X, y, omega, response, lambda_u, u_bar, weight=None):
    """One-component start: ridge least squares for Gaussian, ridge logistic for binomial."""
    A1 = X.shape[1]
    if weight is not None:
        omega = omega * np.asarray(weight, dtype=float)
    if isinstance(response, GaussianResponse):
        w = omega / np.maximum(omega, 1.0)            # same weights, unit scale for the solve
        XtW = X.T * w
        return np.linalg.solve(XtW @ X + lambda_u * np.eye(A1), XtW @ (y - X @ u_bar)) + u_bar

    # binomial: damped Newton on the pooled logistic
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
    """Deterministic multi-start set: pooled, residual-sign split, then jittered copies.

    The pooled start tiles one component, which is a symmetric point: every component is
    identical and the EM-equivalent descent can sit there. The sign split gives the first extra
    components somewhere to go -- two opposing preference states put the judgments in two tails
    of the residual, and fitting one component per tail starts them apart.
    """
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


# ── the fit ───────────────────────────────────────────────────────────────────────────────────

def fit_map(pcvector, response, nP, lambda_u=1e-6, lambda_pi=1.0, lambda_sigma=None,
            s2=None, u_bar=None, omega=None, u_init=None, pi_init=None, phi_init=None,
            n_starts=6, max_iter=5000, tol=1e-9):
    """Maximise the penalised log-posterior directly, for any observation family.

    Args:
        pcvector: comparisons. For :class:`GaussianResponse` the judgment is already on the
            transformed scale; for :class:`BinomialResponse` it is the outcome or its proportion.
        response: :class:`GaussianResponse` or :class:`BinomialResponse`.
        nP: number of mixture components.
        lambda_sigma, s2: required when ``response.phi_free``, ignored otherwise. ``s2`` is the
            prior mode ``phi_0``; ``lambda_sigma`` the Inverse-Gamma influence parameter.
        omega: observation weights (trials for the binomial family, unit weights by default).
        u_init, pi_init, phi_init: optional start; when omitted a pooled fit plus jittered
            copies are used, and the best of them is kept.
        n_starts: how many starts to try when no start is supplied.

    Returns:
        :class:`~mop.mixture_of_priorities.MixtureOfPriorities` with ``converged``,
        ``n_iter`` and ``penalized_log_posterior`` set.
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
        phi0 = np.full(nP, 1.0 if not response.phi_free
                        else (1.0 if phi_init is None else float(phi_init)))
        starts = [(u0, pi0, phi0)]
    else:
        starts = _starts(X, y, omega, response, nP, lambda_u, u_bar, phi_init, n_starts)

    def pack(u, pi, phi):
        # Reference parameterisation: the last component's logit is fixed at zero, so the free
        # parameters are log(pi_q / pi_last), not log(pi_q). Writing log(pi_q) here does not
        # invert to pi under the softmax the objective applies, and the fit then reports a
        # penalised posterior that does not belong to the parameters it returns.
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


def pbic_map(pcvector, model, response, omega=None, lambda_u=1e-6, lambda_pi=1.0,
             lambda_sigma=None, s2=None, u_bar=None):
    """Penalised BIC of a :func:`fit_map` fit: ``-2 pll + log(n_eff) df``.

    ``df = (A-1) P + 2P - 1`` counts the utilities, the ``P-1`` free mixture weights and the
    ``P`` dispersions. The dispersions are counted even when the family fixes them, because they
    are part of the model rather than removed from it; the binomial family therefore spends ``P``
    degrees of freedom on a parameter it does not estimate, exactly as :func:`mop.winloss.pbic`
    does. ``n_eff`` is the number of Bernoulli trials ``sum_k omega_k``, which for unit weights
    is the number of judgments.
    """
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
    df = (A1) * nP + 2 * nP - 1
    return -2.0 * pll + np.log(n_eff) * df
