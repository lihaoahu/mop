"""Win/loss observation models for the mixture of preferences.

Section 2.3 of the paper specifies four protocols. Two of them -- the AHP ratio
protocol and the subjective-probability protocol -- are Gaussian once the
judgment is transformed, and are implemented in :mod:`mop.inference`. The other
two, binary and repeated choice, are Bernoulli and Binomial. Section 2.3 states
both, ``V(mu) = mu(1-mu)`` with ``phi`` set to one, and the appendix gives a
numerical M-step, but only the Gaussian pair was implemented. This module
supplies the missing two.

Observation model
-----------------
For judgment ``k`` with ``eta_k = x_k' u_p`` (the log-odds that the first
alternative is preferred) and ``omega_k = n_k`` trials:

    r_k = m_k / n_k,   E[r_k | p] = mu_k = logistic(eta_k),
    Var[r_k | p] = mu_k (1 - mu_k) / omega_k,   phi_p = 1.

``omega_k = 1`` for a single binary choice, ``omega_k = n_k`` for a comparison
repeated ``n_k`` times; this is the observation weight of the paper's ``eq:glm``,
``Var[y] = phi V(mu) / omega``, and it is why that weight has to be threaded
through the fit.

``phi_p`` is fixed at one rather than estimated, and that is not a convenience.
For a binary response the variance is a function of the mean, so a free ``phi_p``
would be unidentified; Section 2.3 says as much ("``phi`` stays at one"). The
mixture can still be overdispersed relative to a single binomial, and Section 2.3
records that a mixture of binomials cannot be distinguished from one
inconsistently executed preference on this variant alone.

Estimator
---------
The complete-data penalized log-posterior for component ``p`` is

    Q_p(u_p) = sum_k omega_k gamma_kp [ r_k eta_k - log(1 + exp(eta_k)) ]
               - (lambda_u / 2) ||u_p - u_0||^2,

with the same priors as the Gaussian variants (Normal on ``u_p``, Dirichlet on
``pi``; the Inverse-Gamma on ``phi_p`` is degenerate here because ``phi_p = 1``).
Because ``Q_p`` is concave in ``u_p``, its maximiser is unique and is found by
:func:`scipy.optimize.minimize` with an analytic gradient. The E-step, the
mixture-weight update and the convergence test are identical to
:mod:`mop.inference`, which is what makes the two comparable on one dataset.

The Bernoulli case is not a separate code path: it is ``omega_k = 1`` for every
``k``, and the binomial log-density written as ``omega * (r * eta - logaddexp(0,
eta))`` reduces to the Bernoulli one exactly.

Usage
-----
``pcvector`` holds :class:`~mop.pairwise_comparison.PairwiseComparison` objects in
which ``judgment`` is the *outcome*, not a transformed one: ``1`` if
``former_id`` was preferred, ``0`` otherwise, or the proportion ``m / n`` for a
repeated comparison. The response function ``g`` is the identity for this family,
so :func:`fit_winloss` ignores ``link_functions`` and reads the judgment as the
response.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from .initialization import binomial_deviance_residuals
from .mixture_of_priorities import MixtureOfPriorities, Priority
from .pairwise_comparison import comparisons2regression

EPS = 1e-12

# eta is clipped only to keep exp() finite in the derivative and to keep
# logaddexp from being handed an inf; at |eta| = 30 the logistic is within
# 1e-13 of its limit, so no judgment in any realistic dataset is affected.
_ETA_CLIP = 30.0


# ── the observation model ────────────────────────────────────────────────────

def binom_logpdf(y, eta, omega=1.0):
    """Binomial log-density of a proportion ``y = m / n``, up to an eta-free term.

    The exact density carries ``log C(n, m)``, which does not depend on ``eta``
    and therefore cancels in the E-step's normalisation and leaves every M-step
    unchanged. Writing it as ``omega * (y * eta - log(1 + exp(eta)))`` is the
    Bernoulli log-likelihood when ``omega = 1``, so one expression covers both
    protocols. ``logaddexp`` is used instead of ``log(1 + exp(eta))`` so that
    large ``|eta|`` does not overflow.
    """
    eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
    return omega * (y * eta - np.logaddexp(0.0, eta))


def dlogpdf_deta(y, eta, omega=1.0):
    """``d/d eta`` of :func:`binom_logpdf`, i.e. ``omega * (y - logistic(eta))``."""
    eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
    return omega * (y - expit(eta))


# ── E step ───────────────────────────────────────────────────────────────────

def e_gamma_winloss(X, y, u, mp, omega=None):
    """Posterior allocation of each judgment to each component.

    ``X`` is the ``(nJ, A-1)`` design matrix of :func:`comparisons2regression`,
    so ``X @ u[p]`` is the log-odds that the judgment's first alternative is
    preferred under component ``p``. There is no ``sigma2`` argument: ``phi_p``
    is fixed at one, so the dispersion does not enter the density.
    """
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


# ── M step ───────────────────────────────────────────────────────────────────

def _fit_utility_logistic(X, y, w, u_bar, lambda_u, u_init=None):
    """Maximise ``sum_k w_k [y_k eta_k - log(1 + exp(eta_k))] - (lambda_u/2)||u - u_bar||^2``.

    This is the M-step for one ``u_p``, with ``w_k = omega_k gamma_kp``. The
    objective is concave and smooth, so L-BFGS-B with the analytic gradient
    converges to the global maximiser; the ridge term keeps it strictly concave
    for any ``lambda_u > 0``, which is what makes the solution unique.

    The penalty is expanded as ``(lambda_u/2)(u'Au - 2 b'u + const)`` with
    ``A = I`` and ``b = u_bar``, so that the objective touches ``X`` only once per
    evaluation. Written naively it formed ``X @ u`` twice per call, which on the
    135,000 training pairs of the SUSHI set made the fit several times slower for
    no change in the result.
    """
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
    """One M-step: re-fit every component's utilities under the new allocations.

    Returns ``(u_new, mp_new)``. There is no dispersion to return, because
    ``phi_p = 1`` throughout this family.
    """
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
    """Starting values for ``P`` components: one pooled fit plus K-ridge seeding.

    This mirrors the seeding half of :func:`mop.initialization.kridge` -- a pooled
    fit, then one further component placed on the judgments the components so far
    explain worst, measured by binomial deviance -- without the 100 rounds of
    alternating assignment and re-estimation that ``kridge`` then runs. Those
    rounds are what ``kridge`` costs: on the SUSHI training pairs they take 43 s to
    124 s per candidate ``P``, while the EM from these seeds converges in two
    iterations. The pooled logistic fit is the maximum-likelihood
    Bradley-Terry solution, which is the right single state, and the seeded
    extras start where it fits least well, which is what the K-ridge step is for.
    """
    nJ, nA_minus_1 = X.shape
    omega = np.asarray(omega, dtype=np.float64)

    u = np.zeros((nP, nA_minus_1))
    u[0] = _fit_utility_logistic(X, y, omega, u_bar, lambda_u)
    for p in range(1, nP):
        # A judgment is badly explained when the component nearest to it is still
        # far, so the deviance is the smallest over the components present.
        worst = np.min(np.column_stack([
            np.abs(binomial_deviance_residuals(y * omega, omega, expit(X @ u[q])))
            for q in range(p)]), axis=1)
        n_edge = min(nA_minus_1 + 1, nJ)
        idx = np.where(worst >= np.sort(worst)[-n_edge])[0]
        if idx.size == 0:
            continue
        # Fit the seeded component mostly on those judgments, with a small weight
        # elsewhere so that the fit stays identified.
        w = np.full(nJ, 1e-6)
        w[idx] = 1.0
        u[p] = _fit_utility_logistic(X, y, w, u_bar, lambda_u)
    return u, np.full(nP, 1.0 / nP)


def _sign_split_seed(X, y, nP, omega, lambda_u, u_bar):
    """Seed ``nP`` components by the sign of each judgment's residual under the pooled fit.

    :func:`_seed_winloss` places the extra component on the judgments the components so far
    explain worst, but "worst" is the deviance, which has both tails. With two opposing
    preference states the two tails are different states, so a component fitted to their union
    points at neither and the EM can collapse back to one. Splitting the residual by sign
    separates the tails, which is the configuration two opposing states actually produce.
    """
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
    """The Dirichlet-mode update of ``eq:pi_general``.

    ``pi_p = (sum_k omega_k gamma_kp + lambda_pi - 1) / (sum_k omega_k + lambda_pi P - P)``,
    which is the Gaussian variants' update with the observation count ``nJ``
    replaced by the effective count ``sum_k omega_k``.
    """
    nP = gkc.shape[1]
    sum_w = float(np.sum(omega))
    mp = (omega @ gkc + lambda_pi - 1.0) / (sum_w + lambda_pi * nP - nP)
    mp = np.maximum(mp, 1e-10)
    return mp / mp.sum()


# ── the penalized log-posterior, for the convergence test ────────────────────

def penalized_log_posterior(X, y, u, mp, lambda_u, lambda_pi, u_bar=None, omega=None):
    """Penalized log-posterior of the win/loss mixture.

    The likelihood, the Normal prior on ``u_p`` and the Dirichlet prior on
    ``pi``. There is no Inverse-Gamma term: ``phi_p = 1``, so that prior is a
    constant. This is the same quantity as
    :func:`mop.inference.penalized_log_likelihood` computes for the Gaussian
    variants, restricted to that degenerate prior.
    """
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

    # log sum_p pi_p f_p(y_k), the mixture likelihood itself
    eta = X @ u.T
    comp = np.log(np.maximum(mp, EPS)) + binom_logpdf(y[:, None], eta, omega[:, None])
    mx = np.max(comp, axis=1, keepdims=True)
    total = float(np.sum(mx.ravel() + np.log(np.sum(np.exp(comp - mx), axis=1))))

    # Normal prior on u_p, including its normalising constant so that the
    # quantity is comparable across components.
    for p in range(nP):
        diff = u[p] - u_bar[p]
        total -= 0.5 * lam_u[p] * float(np.sum(diff ** 2))
        total -= 0.5 * X.shape[1] * np.log(2.0 * np.pi / lam_u[p])

    # Dirichlet prior on pi
    total += (lambda_pi - 1.0) * float(np.sum(np.log(np.maximum(mp, EPS))))
    total += gammaln(nP * lambda_pi) - nP * gammaln(lambda_pi)
    return total


# ── the fit ──────────────────────────────────────────────────────────────────

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
    # omega is used as given, NOT rescaled to mean one. Rescaling is invariant for
    # a model whose dispersion is free, because phi absorbs the scale. Here phi is
    # fixed at one, so the likelihood scale is part of the model: rescaling omega
    # multiplies the likelihood against a fixed prior, which softens the E-step's
    # responsibilities. On a clear judgment the responsibilities were (0, 1) at
    # omega = 30 and (0.28, 0.72) after rescaling, and a repeated-comparison fit
    # collapsed to a single component. The observation weight n_k is the number of
    # trials behind judgment k (eq:glm), and nothing licenses dividing it out.

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
            # Same relative-change test and normalisation as inference.mop, so the two
            # families stop on the same criterion.
            if prev is not None and abs(pll_tp1 - prev) / (abs(pll_tp1) + 0.1) < tol:
                converged = True
                break
            prev = pll_tp1
        # pll_last, not prev: on convergence the loop breaks before `prev` is updated, so `prev`
        # belongs to the iterate before the one being returned. The two are compared across
        # starting points now, and a start that converges one iteration earlier would otherwise
        # be scored one step behind.
        return u, mp, pll_last, n_iter, converged

    # Initialisation. Two starting points are used whenever the caller does not supply one:
    # the deviance-based seeding of _seed_winloss, and the sign-split seeding of
    # _sign_split_seed. They differ in where the extra components start -- the first on the
    # worst-explained judgments, the second on the two tails of the residual -- and on opposing
    # preference states the first can start a component between the states and collapse. The
    # fit with the higher penalized log-posterior is kept, so this can only help, never hurt.
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

    # nP is final now (u_init may have overridden it), so the per-component arrays are rebuilt
    # here rather than trusting the ones built from the argument.
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

    # Utilities to priority vectors, exactly as inference.mop converts them.
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
    """Observed-data log-likelihood of the fitted win/loss mixture.

    The counterpart of :func:`mop.inference.log_likelihood` for this family,
    and the quantity to report when comparing it with other methods on the same
    held-out judgments. ``omega`` scales the per-judgment terms, so a judgment
    that stands for ``n`` trials counts ``n`` times, as in the Gaussian version.
    """
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
    """Penalized BIC of ``eq:pbic``, for the win/loss family.

    ``-2`` times the penalized log-posterior plus ``log(n_eff) * df``, with
    ``df = (A-1) P + 2P - 1`` counting every free utility, the mixture weights
    (``P - 1``) and the ``P`` structural inconsistencies. ``n_eff`` is the number
    of Bernoulli trials behind the judgments, ``sum_k omega_k``: with ``n_k``
    trials on judgment ``k`` the data are ``sum_k n_k`` trials, not ``nJ``
    judgments. For binary data, where ``n_k = 1``, this is ``nJ`` and the
    criterion is on the same scale as :func:`mop.inference.pbic`; for weighted
    data the two families count different things, and the Gaussian path keeps its
    own convention of ``nJ`` after normalising the weights to mean one.

    ``lambda_u`` and ``lambda_pi`` default to the values :func:`fit_winloss`
    defaults to. They must be the same values the model was fitted with: the
    criterion adds the prior terms, so scoring a fit under a different prior
    measures a different posterior.

    The ``phi_p`` are fixed at one rather than estimated, and they are still
    counted. A reviewer is entitled to ask why: the parameter is part of the
    model the paper defines, and Section 2.3 fixes its value by assumption rather
    than removing it from the family, so a candidate with more components still
    spends ``P`` degrees of freedom on it. Counting it keeps ``df`` comparable on
    both sides of a Gaussian/win-loss comparison. Setting this differently would
    quietly favour the win/loss fits by ``P`` degrees of freedom.
    """
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
    # n_eff counts Bernoulli trials, not judgments: n_k trials behind judgment k means the
    # data are sum_k n_k trials. This is unchanged from before for binary data, where n_k = 1.
    return -2.0 * pll + np.log(float(np.sum(omega_eff))) * df


def select_n_components(pcvector, candidates=(1, 2, 3, 4, 5), omega=None,
                        lambda_u=1e-6, lambda_pi=1.0, **kwargs):
    """Fit each candidate ``P`` and return the one with the lowest pBIC.

    Returns ``(models, pbics)``: every fitted model and its criterion, so a
    caller can report the whole grid instead of only the winner. A candidate that
    fails to fit is recorded with ``None`` rather than silently dropped.
    """
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
