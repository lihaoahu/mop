"""Analytic curvature of the posterior at the mode.

A numerical Hessian cannot resolve the utility block at ``lambda_u = 1e-6``, so the
gradient, the observed information and the Newton polish are written out by hand in
natural coordinates ``(u, phi, pi)``.
"""


from __future__ import annotations

import numpy as np

EPS = 1e-300


def _responsibilities(y, eta, phi, pi, omega, response):
    """``gamma`` and the component log-densities ``b``, computed log-sum-exp stable."""
    log_w = np.log(np.maximum(np.asarray(pi, dtype=float), EPS))
    b = response.loglik(y[:, None], eta, omega[:, None], np.asarray(phi)[None, :])
    z = log_w[None, :] + b
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z)
    tot = np.maximum(w.sum(axis=1, keepdims=True), EPS)
    return w / tot, b


def curvature_terms(y, eta, phi, omega, response):
    """First and second derivatives of ``ln f_kp`` with respect to ``eta``, ``ln phi`` and across the two."""
    y2 = y[:, None]
    om = omega[:, None]
    phi2 = np.asarray(phi, dtype=float)[None, :]
    n, k = eta.shape
    phi_b = np.broadcast_to(phi2, (n, k))

    s = response.dloglik_deta(y2, eta, om, phi_b)
    g = response.dloglik_dphi(y2, eta, om, phi_b)
    h = response.d2loglik_deta2(y2, eta, om, phi_b)
    q = response.d2loglik_dphi2(y2, eta, om, phi_b)
    m = response.d2loglik_deta_dphi(y2, eta, om, phi_b)
    return s, h, g, q, m


def _weights_and_dispersion(response, k, phi, lambda_u, u_bar, lambda_sigma, s2, a1):
    """Broadcast the priors to one scalar per component, keeping the dispersions positive."""
    lam_u = np.broadcast_to(np.asarray(lambda_u, dtype=float), (k,))
    if not response.phi_free:
        phi = np.full(k, 1.0)
    phi = np.asarray(phi, dtype=float).ravel()
    u_bar = np.zeros(a1) if u_bar is None else np.asarray(u_bar, dtype=float).ravel()
    u_bar = (u_bar.reshape(k, a1) if u_bar.size == k * a1
             else np.broadcast_to(u_bar, (k, a1)))
    return phi, u_bar, lam_u


def penalised_gradient(X, y, u, phi, pi, response, omega=None, u_bar=None,
                       lambda_u=1e-6, lambda_pi=1.0, lambda_sigma=None, s2=None):
    """Gradient of the penalised log-posterior, in the coordinates of the Hessian."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.atleast_2d(np.asarray(u, dtype=float))
    n, a1 = X.shape
    k = u.shape[0]
    pi = np.asarray(pi, dtype=float).ravel()
    phi, u_bar, lam_u = _weights_and_dispersion(response, k, phi, lambda_u,
                                                u_bar, lambda_sigma, s2, a1)
    if omega is None:
        omega = np.ones(n)
    omega = np.asarray(omega, dtype=float).ravel()

    eta = X @ u.T
    gam, _ = _responsibilities(y, eta, phi, pi, omega, response)
    s, _, g, _, _ = curvature_terms(y, eta, phi, omega, response)

    grad = np.zeros(a1 * k + (k if response.phi_free else 0) + (k - 1))
    for p in range(k):
        grad[p * a1:(p + 1) * a1] = (X.T @ (gam[:, p] * s[:, p])
                                     - lam_u[p] * (u[p] - u_bar[p]) / phi[p])
    if response.phi_free:
        for p in range(k):
            d = u[p] - u_bar[p]
            val = float(np.sum(gam[:, p] * g[:, p]))
            val += lam_u[p] * float(np.sum(d ** 2)) / (2.0 * phi[p] ** 2)
            val -= 0.5 * a1 / phi[p]
            if lambda_sigma is not None:
                alpha = 0.5 * float(np.broadcast_to(
                    np.asarray(lambda_sigma, dtype=float), (k,))[p])
                beta = alpha * float(s2)
                val -= (alpha + 1.0) / phi[p]
                val += beta / phi[p] ** 2
            grad[a1 * k + p] = val
    if k > 1:
        # last weight: ``1 - sum``, so the same derivative enters with a minus
        tail = (float(np.sum(gam[:, k - 1])) / pi[k - 1]
                + (lambda_pi - 1.0) / pi[k - 1])
        for mm in range(k - 1):
            grad[a1 * k + (k if response.phi_free else 0) + mm] = (
                float(np.sum(gam[:, mm])) / pi[mm] + (lambda_pi - 1.0) / pi[mm]
                - tail)
    return grad


def polish_mode(log_posterior, u, phi, pi, X, y, response, omega=None,
                u_bar=None, lambda_u=1e-6, lambda_pi=1.0, lambda_sigma=None,
                s2=None, max_steps=60, grad_tol=1e-9, grad_accept=None):
    """Newton steps on the penalised log-posterior; returns ``(u, phi, pi, info)``. Reports whether the decrement reached its tolerance, which is what decides whether a mode was found."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    n, a1 = X.shape
    k = u.shape[0]
    phi = (np.asarray(phi, dtype=float).ravel() if response.phi_free
           else np.ones(k))
    pi = np.asarray(pi, dtype=float).ravel()
    n_phi = k if response.phi_free else 0
    grad_accept = (1e-4 * n) if grad_accept is None else float(grad_accept)

    def unpack(t):
        uu = t[:k * a1].reshape(k, a1)
        ph = t[k * a1:k * a1 + n_phi] if n_phi else np.ones(k)
        pp = np.concatenate([t[k * a1 + n_phi:], [1.0 - np.sum(t[k * a1 + n_phi:])]])
        return uu, ph, pp

    def f(t):
        # reject overshooting steps here: the coordinates are free, the derived
        # dispersion and weight are not
        uu, ph, pp = unpack(t)
        if np.any(ph <= 0) or np.any(pp <= 0):
            return -np.inf
        return float(log_posterior(uu, ph, pp))

    theta = np.concatenate([u.ravel(), phi if n_phi else np.zeros(0), pi[:-1]])
    f0 = f(theta)
    steps, why = 0, "step cap"
    while steps < max_steps:
        uu, ph, pp = unpack(theta)
        g = penalised_gradient(X, y, uu, ph, pp, response, omega=omega,
                               u_bar=u_bar, lambda_u=lambda_u,
                               lambda_pi=lambda_pi, lambda_sigma=lambda_sigma,
                               s2=s2)
        if not np.all(np.isfinite(g)):
            why = "non-finite gradient"
            break
        if np.linalg.norm(g) <= grad_tol:
            why = "gradient"
            break
        H = ln_det_hessian(X, y, uu, ph, pp, response, omega=omega, u_bar=u_bar,
                           lambda_u=lambda_u, lambda_pi=lambda_pi,
                           lambda_sigma=lambda_sigma, s2=s2, return_matrix=True)
        H = 0.5 * (H + H.T)
        w = np.linalg.eigvalsh(H)
        floor = 1e-8 * max(1.0, float(np.abs(w).max()))
        mu = max(0.0, floor - float(w.min()))
        step = np.linalg.solve(H + mu * np.eye(len(w)), g)
        slope = float(g @ step)
        if not np.isfinite(slope) or slope <= 0:
            why = "no ascent direction"
            break
        t = 1.0
        while t > 1e-12:
            trial = theta + t * step
            ft = f(trial)
            if np.isfinite(ft) and ft >= f0 + 1e-4 * t * slope:
                break
            t *= 0.5
        if t <= 1e-12:
            why = "boundary"
            break
        theta, f0 = trial, ft
        steps += 1
        if t * slope <= grad_tol * max(1.0, abs(f0)):
            why = "decrement"
            break
    uu, ph, pp = unpack(theta)
    g = penalised_gradient(X, y, uu, ph, pp, response, omega=omega, u_bar=u_bar,
                           lambda_u=lambda_u, lambda_pi=lambda_pi,
                           lambda_sigma=lambda_sigma, s2=s2)
    grad_inf = float(np.max(np.abs(g))) if np.all(np.isfinite(g)) else float("inf")
    info = {"steps": steps, "pll": float(f0), "stopped": why,
            "converged": bool(grad_inf <= grad_accept),
            "grad_inf": grad_inf, "min_pi": float(np.min(pp))}
    return uu, ph, pp, info


def ln_det_hessian(X, y, u, phi, pi, response, omega=None, u_bar=None,
                   lambda_u=1e-6, lambda_pi=1.0, lambda_sigma=None, s2=None,
                   phi_shared=False, return_matrix=False):
    """``ln|H|`` with ``H = -grad^2 ln[L f]``, assembled block by block."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.atleast_2d(np.asarray(u, dtype=float))
    n, a1 = X.shape
    k = u.shape[0]
    pi = np.asarray(pi, dtype=float).ravel()
    phi = np.full(k, 1.0) if not response.phi_free else np.asarray(phi, dtype=float).ravel()
    if omega is None:
        omega = np.ones(n)
    omega = np.asarray(omega, dtype=float).ravel()
    if u_bar is None:
        u_bar = np.zeros(a1)
    u_bar = np.asarray(u_bar, dtype=float).ravel()
    u_bar = (u_bar.reshape(k, a1) if u_bar.size == k * a1
             else np.broadcast_to(u_bar, (k, a1)))
    lam_u = np.broadcast_to(np.asarray(lambda_u, dtype=float), (k,))

    eta = X @ u.T
    gam, _ = _responsibilities(y, eta, phi, pi, omega, response)
    s, h, g, q, m = curvature_terms(y, eta, phi, omega, response)

    phi_free = bool(response.phi_free)
    n_phi = k if phi_free else 0
    n_pi = k - 1
    dim = a1 * k + n_phi + n_pi
    H = np.zeros((dim, dim))
    iu = lambda p: slice(p * a1, (p + 1) * a1)
    iphi = lambda p: a1 * k + p
    ipi = lambda mm: a1 * k + n_phi + mm

    # utilities: gamma_p (h_p + s_p^2) on the diagonal, minus gamma_p gamma_r
    # s_p s_r between components, plus the ridge from the prior
    for p in range(k):
        for r in range(p, k):
            if p == r:
                w = gam[:, p] * (h[:, p] + s[:, p] ** 2) - (gam[:, p] * s[:, p]) ** 2
            else:
                w = -(gam[:, p] * s[:, p]) * (gam[:, r] * s[:, r])
            block = -(X.T @ (X * w[:, None]))
            if p == r:
                block = block + (lam_u[p] / phi[p]) * np.eye(a1)
            H[iu(p), iu(r)] = block
            if p != r:
                H[iu(r), iu(p)] = block.T

    # utilities against dispersions
    if phi_free:
        for p in range(k):
            for r in range(k):
                if p == r:
                    w = gam[:, p] * (m[:, p] + s[:, p] * g[:, p]) \
                        - (gam[:, p] * s[:, p]) * (gam[:, p] * g[:, p])
                else:
                    w = -(gam[:, p] * s[:, p]) * (gam[:, r] * g[:, r])
                col = -(X.T @ w)
                if p == r:
                    # the utility prior couples u_p to phi_p through its variance
                    col = col - lam_u[p] * (u[p] - u_bar[p]) / phi[p] ** 2
                H[iu(p), iphi(r)] = col
                H[iphi(r), iu(p)] = col

    # dispersions
    if phi_free:
        for p in range(k):
            for r in range(p, k):
                if p == r:
                    w = gam[:, p] * (q[:, p] + g[:, p] ** 2) - (gam[:, p] * g[:, p]) ** 2
                else:
                    w = -(gam[:, p] * g[:, p]) * (gam[:, r] * g[:, r])
                val = -float(np.sum(w))
                if p == r:
                    d = u[p] - u_bar[p]
                    val += float(lam_u[p] * np.sum(d ** 2) / phi[p] ** 3)
                    val -= 0.5 * a1 / phi[p] ** 2
                    if lambda_sigma is not None:
                        ls_p = float(np.broadcast_to(
                            np.asarray(lambda_sigma, dtype=float), (k,))[p])
                        alpha = 0.5 * ls_p
                        beta = 0.5 * ls_p * float(s2)
                        val -= (alpha + 1.0) / phi[p] ** 2
                        val += 2.0 * beta / phi[p] ** 3
                H[iphi(p), iphi(r)] = val
                if p != r:
                    H[iphi(r), iphi(p)] = val

    # weights
    if n_pi:
        d = gam / np.maximum(pi, EPS)[None, :]                  # gamma_kp / pi_p
        c = d[:, :n_pi] - d[:, [k - 1]]                          # (n, P-1)
        block = c.T @ c
        block += (lambda_pi - 1.0) * (
            np.diag(1.0 / np.maximum(pi[:n_pi], EPS) ** 2)
            + 1.0 / max(pi[k - 1], EPS) ** 2)
        H[ipi(0):ipi(n_pi), ipi(0):ipi(n_pi)] = block

        last = k - 1
        for mm in range(n_pi):
            for p in range(k):
                # d2 ln f_k / d pi_m d u_pi, term by term:
                # + d_km s_km  if m is this component,  - d_kP s_kP  if P is,
                # and -c_km times the responsibility-weighted score.
                own = d[:, mm] * s[:, mm] if p == mm else 0.0
                base = d[:, last] * s[:, last] if p == last else 0.0
                row_u = own - base - c[:, mm] * gam[:, p] * s[:, p]
                col = -(X.T @ np.broadcast_to(np.asarray(row_u, dtype=float), (n,)))
                H[ipi(mm), iu(p)] = col
                H[iu(p), ipi(mm)] = col
                if phi_free:
                    own_g = d[:, mm] * g[:, mm] if p == mm else 0.0
                    base_g = d[:, last] * g[:, last] if p == last else 0.0
                    val = -float(np.sum(own_g - base_g
                                        - c[:, mm] * gam[:, p] * g[:, p]))
                    H[ipi(mm), iphi(p)] = val
                    H[iphi(p), ipi(mm)] = val

    # shared dispersion: the image of the diagonal direction of the K x K block,
    # with the inverse-Gamma prior counted once instead of once per component
    if n_phi and phi_shared:
        dim_shared = a1 * k + 1 + n_pi
        T = np.zeros((dim, dim_shared))
        for p in range(k):
            T[iu(p), iu(p)] = np.eye(a1)
        T[a1 * k: a1 * k + k, a1 * k] = 1.0
        for mm in range(n_pi):
            T[a1 * k + n_phi + mm, a1 * k + 1 + mm] = 1.0
        H = T.T @ H @ T
        if lambda_sigma is not None:
            alpha = 0.5 * float(lambda_sigma)
            beta = 0.5 * float(lambda_sigma) * float(s2)
            prior_curv = (-(alpha + 1.0) / float(phi[0]) ** 2
                          + 2.0 * beta / float(phi[0]) ** 3)
            H[a1 * k, a1 * k] -= (k - 1) * prior_curv

    if return_matrix:
        return H
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0 or not np.isfinite(logdet):
        return float("-inf")
    return float(logdet)
