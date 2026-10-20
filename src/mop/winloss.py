"""Win/loss observation models for the mixture of preferences."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from .initialization import binomial_deviance_residuals
from .mixture_of_priorities import MixtureOfPriorities, Priority
from .pairwise_comparison import comparisons2regression

EPS = 1e-12

_ETA_CLIP = 30.0



def binom_logpdf(y, eta, omega=1.0):
    """Binomial log-density of a proportion ``y = m / n``, up to an eta-free term."""
    eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
    return omega * (y * eta - np.logaddexp(0.0, eta))


def dlogpdf_deta(y, eta, omega=1.0):
    """``d/d eta`` of :func:`binom_logpdf`, i.e. ``omega * (y - logistic(eta))``."""
    eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
    return omega * (y - expit(eta))



def e_gamma_winloss(X, y, u, mp, omega=None):
    """Posterior allocation of each judgment to each component."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    mp = np.asarray(mp, dtype=np.float64)
    if omega is None:
        omega = np.ones(X.shape[0])
    omega = np.asarray(omega, dtype=np.float64).reshape(-1, 1)

    log_probs = np.log(np.maximum(mp, EPS)) + binom_logpdf(y[:, None], X @ u.T, omega)
    mx = np.max(log_probs, axis=1, keepdims=True)
    g = np.exp(log_probs - mx)
    g /= np.maximum(g.sum(axis=1, keepdims=True), EPS)
    return g



def _fit_utility_logistic(X, y, w, u_bar, lambda_u, u_init=None):
    """Maximise ``sum_k w_k [y_k eta_k - log(1 + exp(eta_k))] - (lambda_u/2)||u - u_bar||^2``."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    u_bar = np.asarray(u_bar, dtype=np.float64)
    u0 = np.zeros(X.shape[1]) if u_init is None else np.asarray(u_init, float).copy()

    def objective(u):
        eta = X @ u
        pen = float(u @ u) - 2.0 * float(u_bar @ u)
        return -(float(np.sum(w * binom_logpdf(y, eta, 1.0)))
                 - 0.5 * lambda_u * pen)

    def gradient(u):
        eta = X @ u
        return -(X.T @ dlogpdf_deta(y, eta, w)) + lambda_u * (u - u_bar)

    res = minimize(objective, u0, jac=gradient, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
    return res.x


def m_step_winloss(X, y, gkc, u_bar, lambda_u, omega=None, u_init=None):
    """One M-step: re-fit every component's utilities under the new allocations."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nJ, nP = gkc.shape
    if omega is None:
        omega = np.ones(nJ)
    omega = np.asarray(omega, dtype=np.float64).ravel()

    lam_u = (np.full(nP, float(lambda_u)) if np.isscalar(lambda_u)
             else np.asarray(lambda_u, dtype=np.float64))
    u_bar = np.asarray(u_bar, dtype=np.float64)
    if u_bar.ndim == 1:
        u_bar = np.tile(u_bar, (nP, 1))

    u_new = np.zeros((nP, X.shape[1]))
    for p in range(nP):
        u_new[p] = _fit_utility_logistic(
            X, y, omega * gkc[:, p], u_bar[p], lam_u[p],
            u_init=None if u_init is None else u_init[p])
    return u_new


def _seed_winloss(X, y, nP, omega, lambda_u, u_bar):
    """Starting values for ``P`` components: one pooled fit plus K-ridge seeding."""
    nJ, nA_minus_1 = X.shape
    omega = np.asarray(omega, dtype=np.float64)

    u = np.zeros((nP, nA_minus_1))
    u[0] = _fit_utility_logistic(X, y, omega, u_bar, lambda_u)
    for p in range(1, nP):
        worst = np.min(np.column_stack([
            np.abs(binomial_deviance_residuals(y * omega, omega, expit(X @ u[q])))
            for q in range(p)]), axis=1)
        n_edge = min(nA_minus_1 + 1, nJ)
        idx = np.where(worst >= np.sort(worst)[-n_edge])[0]
        if idx.size == 0:
            continue
        w = np.full(nJ, 1e-6)
        w[idx] = 1.0
        u[p] = _fit_utility_logistic(X, y, w, u_bar, lambda_u)
    return u, np.full(nP, 1.0 / nP)


def _sign_split_seed(X, y, nP, omega, lambda_u, u_bar):
    """Seed ``nP`` components by the sign of each judgment's residual under the pooled fit."""
    nJ, nA_minus_1 = X.shape
    omega = np.asarray(omega, dtype=np.float64)

    u0 = _fit_utility_logistic(X, y, omega, u_bar, lambda_u)
    resid = y - expit(X @ u0)
    seeds = [u0]
    for mask in (resid > 0, resid < 0):
        w = mask.astype(np.float64)
        if w.sum() > nA_minus_1 + 2:
            seeds.append(_fit_utility_logistic(X, y, w * omega, u_bar, lambda_u))
        else:
            seeds.append(u0 + 0.1)
    u = np.vstack((seeds * ((nP // len(seeds)) + 1))[:nP])
    return u, np.full(nP, 1.0 / nP)


def _mixture_weights(gkc, omega, lambda_pi):
    """The Dirichlet-mode update of ``eq:pi_general``."""
    nP = gkc.shape[1]
    sum_w = float(np.sum(omega))
    mp = (omega @ gkc + lambda_pi - 1.0) / (sum_w + lambda_pi * nP - nP)
    mp = np.maximum(mp, 1e-10)
    return mp / mp.sum()



def penalized_log_posterior(X, y, u, mp, lambda_u, lambda_pi, u_bar=None, omega=None):
    """Penalized log-posterior of the win/loss mixture."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    mp = np.asarray(mp, dtype=np.float64)
    nP = u.shape[0]
    if omega is None:
        omega = np.ones(X.shape[0])
    omega = np.asarray(omega, dtype=np.float64).ravel()
    if u_bar is None:
        u_bar = np.zeros(X.shape[1])
    u_bar = np.asarray(u_bar, dtype=np.float64)
    if u_bar.ndim == 1:
        u_bar = np.tile(u_bar, (nP, 1))
    lam_u = (np.full(nP, float(lambda_u)) if np.isscalar(lambda_u)
             else np.asarray(lambda_u, dtype=np.float64))

    eta = X @ u.T
    comp = np.log(np.maximum(mp, EPS)) + binom_logpdf(y[:, None], eta, omega[:, None])
    mx = np.max(comp, axis=1, keepdims=True)
    total = float(np.sum(mx.ravel() + np.log(np.sum(np.exp(comp - mx), axis=1))))

    for p in range(nP):
        diff = u[p] - u_bar[p]
        total -= 0.5 * lam_u[p] * float(np.sum(diff ** 2))
        total -= 0.5 * X.shape[1] * np.log(2.0 * np.pi / lam_u[p])

    total += (lambda_pi - 1.0) * float(np.sum(np.log(np.maximum(mp, EPS))))
    total += gammaln(nP * lambda_pi) - nP * gammaln(lambda_pi)
    return total



def fit_winloss(pcvector, nP, lambda_u=1e-6, lambda_pi=1.0, u_bar=None,
                omega=None, u_init=None, mp_init=None, max_iter=2000, tol=1e-5,
                verbose=False):
    """EM for the win/loss family, mirroring :func:`mop.inference.mop`.

    Args:
        pcvector: comparisons whose ``judgment`` is the outcome: ``1`` if
            ``former_id`` was preferred, ``0`` otherwise, or the proportion for a
            repeated comparison.
        nP: number of mixture components.
        lambda_u: precision of the Normal prior on ``u_p``.
        lambda_pi: Dirichlet concentration of the prior on ``pi``.
        u_bar: prior mean ``u_0`` of the utilities, shape ``(A-1,)`` or
            ``(nP, A-1)``.
        omega: observation weight ``omega_k`` per comparison, length ``nJ``.
            ``n_k`` for a repeated comparison, ones for binary choices. It is
            used as given and is not rescaled: ``n_k`` is the number of trials
            behind judgment ``k``, and dividing it out would change the model,
            because ``phi`` is fixed at one here and cannot absorb the scale.
        u_init, mp_init: optional starting values.
        max_iter, tol: iteration cap and relative tolerance on the penalized
            log-posterior, using the same normalisation as
            :func:`mop.inference.mop`. The defaults were set from measurement on
            the SUSHI win/loss data rather than by eye: there, ``tol=1e-10``
            needs more than 700 iterations and, at ``P = 3``, thousands without
            reaching it, while the fitted quantities are stable well before that.
            A caller who wants a fully converged fit should lower ``tol`` and
            raise ``max_iter`` and check ``model.converged``.

    Returns:
        :class:`~mop.mixture_of_priorities.MixtureOfPriorities` whose components
        carry ``inconsistency_level = 1``, since ``phi_p`` is fixed at one for
        this family. ``model.converged`` and ``model.n_iter`` report whether the
        iteration actually reached ``tol``; a fit that did not is approximate and
        should be reported as such.
    """
    nJ = len(pcvector)
    X, y, _ = comparisons2regression(pcvector, [float])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nA_minus_1 = X.shape[1]

    if omega is None:
        omega_eff = np.ones(nJ)
    else:
        omega_eff = np.asarray(omega, dtype=np.float64).ravel()
        if omega_eff.size != nJ:
            raise ValueError(
                f"omega 长度({omega_eff.size})与判断数({nJ})不一致")
        if np.any(omega_eff <= 0):
            raise ValueError("omega 必须为正")

    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError(
            "win/loss judgments must lie in [0, 1]: pass the outcome itself, "
            "not a transformed judgment")

    lam_u_arr = (np.full(nP, float(lambda_u)) if np.isscalar(lambda_u)
                 else np.asarray(lambda_u, dtype=np.float64))
    if u_bar is None:
        u_bar_arr = np.zeros((nP, nA_minus_1))
    else:
        u_bar_arr = np.asarray(u_bar, dtype=np.float64)
        if u_bar_arr.ndim == 1:
            u_bar_arr = np.tile(u_bar_arr, (nP, 1))

    def run_em(u_start, mp_start):
        """One EM run from one starting point. Returns (u, mp, pll, n_iter, converged)."""
        u, mp = u_start, mp_start
        prev = None
        pll_last = -np.inf
        n_iter = 0
        converged = False
        for n_iter in range(1, max_iter + 1):
            g = e_gamma_winloss(X, y, u, mp, omega_eff)
            u_new = m_step_winloss(X, y, g, u_bar_arr, lam_u_arr, omega=omega_eff,
                                   u_init=u)
            mp_new = _mixture_weights(g, omega_eff, lambda_pi)
            pll_tp1 = penalized_log_posterior(X, y, u_new, mp_new, lam_u_arr,
                                              lambda_pi, u_bar_arr, omega_eff)
            if not np.isfinite(pll_tp1):
                break
            pll_last = pll_tp1
            if verbose:
                print(f"  iter {n_iter:5d}  penalized log-posterior {pll_tp1:.6f}")
            u, mp = u_new, mp_new
            if prev is not None and abs(pll_tp1 - prev) / (abs(pll_tp1) + 0.1) < tol:
                converged = True
                break
            prev = pll_tp1
        return u, mp, pll_last, n_iter, converged

    if u_init is not None:
        u_start = np.atleast_2d(np.asarray(u_init, dtype=np.float64))
        if u_start.shape[0] != nP:
            raise ValueError(
                f"u_init has {u_start.shape[0]} components but nP={nP}; pass starts for "
                f"{nP} components, or set nP to match")
        if mp_init is None:
            mp_start = np.full(nP, 1.0 / nP)
        else:
            mp_start = np.asarray(mp_init, dtype=np.float64).ravel()
            if mp_start.size != nP:
                raise ValueError(
                    f"mp_init has {mp_start.size} entries but nP={nP}")
            if np.any(mp_start < 0) or not np.isclose(mp_start.sum(), 1.0, atol=1e-8):
                raise ValueError("mp_init must be non-negative and sum to one")
        starts = [(u_start, mp_start)]
    else:
        starts = [_seed_winloss(X, y, nP, omega_eff, lam_u_arr[0], u_bar_arr[0])]
        if mp_init is not None:
            mp_start = np.asarray(mp_init, dtype=np.float64).ravel()
            if mp_start.size != nP:
                raise ValueError(f"mp_init has {mp_start.size} entries but nP={nP}")
            starts = [(starts[0][0], mp_start)]
        if nP > 1:
            starts.append(_sign_split_seed(X, y, nP, omega_eff, lam_u_arr[0], u_bar_arr[0]))

    if np.isscalar(lambda_u):
        lam_u_arr = np.full(nP, float(lambda_u))
    else:
        lam_u_arr = np.asarray(lambda_u, dtype=np.float64).ravel()
        if lam_u_arr.size == 1:
            lam_u_arr = np.full(nP, float(lam_u_arr[0]))
        elif lam_u_arr.size != nP:
            raise ValueError(
                f"lambda_u has {lam_u_arr.size} entries but the fit has {nP} components")
    _ub = np.asarray(u_bar if u_bar is not None else np.zeros((1, nA_minus_1)),
                     dtype=np.float64)
    if _ub.ndim == 1:
        _ub = _ub.reshape(1, -1)
    if _ub.shape[0] == 1:
        u_bar_arr = np.tile(_ub, (nP, 1))
    elif _ub.shape[0] == nP:
        u_bar_arr = _ub
    else:
        raise ValueError(
            f"u_bar has {_ub.shape[0]} rows but the fit has {nP} components")

    results = [run_em(u_s, mp_s) for u_s, mp_s in starts]
    u, mp, prev, n_iter, converged = max(results, key=lambda r: r[2])

    u_clipped = np.clip(u, -50.0, 50.0)
    v = np.hstack([np.exp(u_clipped), np.ones((nP, 1))])
    w = v / np.sum(v, axis=1, keepdims=True)
    priorities = [Priority(w[p], 1.0) for p in range(nP)]

    model = MixtureOfPriorities(priorities, mp)
    model.n_iter = n_iter
    model.converged = converged
    model.penalized_log_posterior = prev
    return model


def log_likelihood(X, y, u, mp, omega=None):
    """Observed-data log-likelihood of the fitted win/loss mixture."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    mp = np.asarray(mp, dtype=np.float64)
    if omega is None:
        omega = np.ones(X.shape[0])
    omega = np.asarray(omega, dtype=np.float64).ravel()

    comp = np.log(np.maximum(mp, EPS)) + binom_logpdf(y[:, None], X @ u.T, 1.0)
    mx = np.max(comp, axis=1, keepdims=True)
    ll = mx.ravel() + np.log(np.sum(np.exp(comp - mx), axis=1))
    return float(np.sum(omega * ll))


def log_likelihood_from_pc(pcvector, model, omega=None):
    """Observed-data log-likelihood of a fitted model, from the comparisons."""
    X, y, _ = comparisons2regression(pcvector, [float])
    nA = X.shape[1] + 1
    u = np.array([np.log(model[p].priority_vector[:nA - 1]
                         / model[p].priority_vector[nA - 1])
                  for p in range(len(model.mixture_probabilities))])
    return log_likelihood(X, y, u, model.mixture_probabilities, omega)


def pbic(pcvector, model, lambda_u=1e-6, lambda_pi=1.0, u_bar=None, omega=None):
    """Penalized BIC of ``eq:pbic``, for the win/loss family."""
    X, y, _ = comparisons2regression(pcvector, [float])
    nA = X.shape[1] + 1
    nP = len(model.mixture_probabilities)

    if omega is None:
        omega_eff = np.ones(X.shape[0])
    else:
        omega_eff = np.asarray(omega, dtype=np.float64).ravel()
        if omega_eff.size != X.shape[0]:
            raise ValueError(
                f"omega 长度({omega_eff.size})与判断数({X.shape[0]})不一致")
        if np.any(omega_eff <= 0):
            raise ValueError("omega 必须为正")

    u = np.array([np.log(model[p].priority_vector[:nA - 1]
                         / model[p].priority_vector[nA - 1])
                  for p in range(nP)])
    pll = penalized_log_posterior(X, y, u, model.mixture_probabilities,
                                  lambda_u, lambda_pi, u_bar, omega_eff)
    df = (nA - 1) * nP + 2 * nP - 1
    return -2.0 * pll + np.log(float(np.sum(omega_eff))) * df


def select_n_components(pcvector, candidates=(1, 2, 3, 4, 5), omega=None,
                        lambda_u=1e-6, lambda_pi=1.0, **kwargs):
    """Fit each candidate ``P`` and return the one with the lowest pBIC."""
    models, criteria = {}, {}
    for nP in candidates:
        try:
            model = fit_winloss(pcvector, nP, lambda_u=lambda_u,
                                lambda_pi=lambda_pi, omega=omega, **kwargs)
            models[nP] = model
            criteria[nP] = pbic(pcvector, model, lambda_u, lambda_pi,
                                kwargs.get("u_bar"), omega)
        except Exception as exc:                                  # noqa: BLE001
            models[nP] = None
            criteria[nP] = None
            print(f"  win/loss P={nP} failed: {type(exc).__name__}: {exc}")
    return models, criteria
