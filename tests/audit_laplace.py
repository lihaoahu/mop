"""Analytic curvature against finite differences."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from mop.general_fit import BinomialResponse, GaussianResponse
from mop.inference import penalized_log_likelihood
from mop.laplace import ln_det_hessian, penalised_gradient


def gaussian_objective(X, y, k, a1, lam_u, lam_sigma, lam_pi, s2):
    def objective(theta):
        u = theta[: k * a1].reshape(k, a1)
        phi = theta[k * a1: k * a1 + k]
        pi = np.concatenate([theta[k * a1 + k:], [1.0 - np.sum(theta[k * a1 + k:])]])
        if np.any(phi <= 0) or np.any(pi <= 0):
            return -np.inf
        return penalized_log_likelihood(X, y, u, phi, pi, lam_u, lam_sigma,
                                        lam_pi, s2, None, None)

    return objective


def central_hessian(f, theta, step=1e-4):
    d = theta.size
    H = np.zeros((d, d))
    f0 = f(theta)
    h = step * np.maximum(1.0, np.abs(theta))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h[j]
        H[j, j] = (f(theta + e) - 2.0 * f0 + f(theta - e)) / h[j] ** 2
    for j in range(d):
        for i in range(j + 1, d):
            ei = np.zeros(d)
            ei[i] = h[i]
            ej = np.zeros(d)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(theta + ei + ej) - f(theta + ei - ej)
                                 - f(theta - ei + ej) + f(theta - ei - ej)) \
                / (4.0 * h[i] * h[j])
    return H


def synthetic(n=400, a1=3, k=3, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, a1))
    u = np.array([[0.0, 0.0, 0.0], [2.0, -1.5, 0.5], [-1.0, 2.0, 1.0]])
    phi = np.array([0.4, 0.9, 1.4])
    pi = np.array([0.45, 0.35, 0.20])
    comp = rng.choice(k, size=n, p=pi)
    y = X @ u[comp].T
    y = y[np.arange(n), comp] + np.sqrt(phi[comp]) * rng.normal(size=n)
    return X, y


def check_matrix():
    X, y = synthetic()
    n, a1 = X.shape
    k = 3
    lam_u, lam_sigma, lam_pi, s2 = 0.5, 40.0, 1.0, 0.8
    u = np.array([[0.1, -0.2, 0.3], [1.6, -1.1, 0.4], [-0.8, 1.5, 0.9]])
    phi = np.array([0.45, 0.85, 1.30])
    pi = np.array([0.42, 0.36, 0.22])

    analytic = ln_det_hessian(X, y, u, phi, pi, GaussianResponse(), u_bar=None,
                              lambda_u=lam_u, lambda_pi=lam_pi,
                              lambda_sigma=lam_sigma, s2=s2)
    theta = np.concatenate([u.ravel(), phi, pi[:-1]])
    H = central_hessian(gaussian_objective(X, y, k, a1, lam_u, lam_sigma, lam_pi, s2),
                        theta)
    sign, logdet = np.linalg.slogdet(-H)
    assert sign > 0, "the finite-difference matrix is not negative definite"
    exact = ln_det_hessian(X, y, u, phi, pi, GaussianResponse(),
                           lambda_u=lam_u, lambda_pi=lam_pi,
                           lambda_sigma=lam_sigma, s2=s2, return_matrix=True)
    entry = np.abs(exact - (-H)).max() / np.abs(exact).max()
    print(f"  gaussian: analytic {analytic:12.6f}   finite {logdet:12.6f}"
          f"   gap {abs(analytic - logdet):.2e}   worst entry {entry:.1e}")
    assert entry < 1e-6, "the analytic blocks do not reproduce the log-posterior"
    assert abs(analytic - logdet) < 1e-4 * max(1.0, abs(logdet)), "blocks disagree"
    return analytic


def check_simplex_coordinates():
    """ln|H_pi| - ln|H_z| = -2 sum_p ln pi_p, the softmax Jacobian.

    The rule that turns one Hessian into another by the Jacobian holds only at a
    stationary point: away from it the map contributes a term in the gradient,
    which is large enough here to swamp the identity. So the comparison is made
    at the mode, which is where the criterion evaluates the determinant anyway.
    """
    from scipy.optimize import minimize

    X, y = synthetic()
    k, a1 = 3, X.shape[1]
    lam_u, lam_sigma, lam_pi, s2 = 0.5, 40.0, 1.0, 0.8

    def objective_z(theta):
        uu = theta[: k * a1].reshape(k, a1)
        pp = theta[k * a1: k * a1 + k]
        z = np.concatenate([theta[k * a1 + k:], [0.0]])
        w = np.exp(z - z.max())
        w = w / w.sum()
        return penalized_log_likelihood(X, y, uu, pp, w, lam_u, lam_sigma,
                                        lam_pi, s2, None, None)

    start = np.concatenate([
        np.array([[0.1, -0.2, 0.3], [1.6, -1.1, 0.4], [-0.8, 1.5, 0.9]]).ravel(),
        [0.45, 0.85, 1.30], np.log(np.array([0.42, 0.36]) / 0.22)])
    fit = minimize(lambda t: -objective_z(t), start, method="L-BFGS-B",
                   options={"maxiter": 20000, "ftol": 1e-16, "gtol": 1e-10})
    theta_z = fit.x
    u = theta_z[: k * a1].reshape(k, a1)
    phi = theta_z[k * a1: k * a1 + k]
    zz = np.concatenate([theta_z[k * a1 + k:], [0.0]])
    w = np.exp(zz - zz.max())
    pi = w / w.sum()

    natural = ln_det_hessian(X, y, u, phi, pi, GaussianResponse(),
                             lambda_u=lam_u, lambda_pi=lam_pi,
                             lambda_sigma=lam_sigma, s2=s2)
    Hz = central_hessian(objective_z, theta_z)
    sign, logdet_z = np.linalg.slogdet(-Hz)
    assert sign > 0
    jacobian = -2.0 * float(np.sum(np.log(pi)))
    print(f"  simplex:  natural {natural:12.6f}   log-ratio {logdet_z:12.6f}"
          f"   gap {natural - logdet_z:12.6f}   Jacobian {jacobian:12.6f}"
          f"   |grad| {np.abs(fit.jac).max():.1e}")
    assert np.abs(fit.jac).max() < 1e-4, "the comparison point is not the mode"
    assert abs((natural - logdet_z) - jacobian) < 1e-4 * max(1.0, abs(jacobian)), \
        "the determinant is not in the coordinates the integral uses"


def check_binomial():
    """The protocol with the dispersion pinned: the phi block must vanish."""
    rng = np.random.default_rng(3)
    n, a1, k = 500, 3, 2
    X = rng.normal(size=(n, a1))
    trials = rng.integers(5, 20, size=n)
    u = np.array([[1.2, -0.6, 0.3], [-0.9, 1.4, -0.2]])
    pi = np.array([0.6, 0.4])
    comp = rng.choice(k, size=n, p=pi)
    eta = np.einsum("ij,ij->i", X, u[comp])
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(trials, prob) / trials

    lam_u, lam_pi = 0.5, 1.0
    val = ln_det_hessian(X, y, u, np.ones(k), pi, BinomialResponse(),
                         omega=trials, lambda_u=lam_u, lambda_pi=lam_pi,
                         lambda_sigma=None, s2=None)

    resp = BinomialResponse()

    def objective(theta):
        uu = theta[: k * a1].reshape(k, a1)
        w = np.concatenate([theta[k * a1:], [1.0 - np.sum(theta[k * a1:])]])
        b = resp.loglik(y[:, None], X @ uu.T, trials[:, None], np.ones((n, k)))
        z = np.log(w)[None, :] + b
        mx = z.max(axis=1, keepdims=True)
        ll = float(np.sum(mx.ravel() + np.log(np.exp(z - mx).sum(axis=1))))
        ll -= 0.5 * lam_u * float(np.sum(uu ** 2))
        ll -= 0.5 * a1 * np.log(2.0 * np.pi / lam_u) * k
        ll += (lam_pi - 1.0) * float(np.sum(np.log(w)))
        return ll

    theta = np.concatenate([u.ravel(), pi[:-1]])
    H = central_hessian(objective, theta)
    sign, logdet = np.linalg.slogdet(-H)
    assert sign > 0
    exact = ln_det_hessian(X, y, u, np.ones(k), pi, BinomialResponse(),
                           omega=trials, lambda_u=lam_u, lambda_pi=lam_pi,
                           lambda_sigma=None, s2=None, return_matrix=True)
    entry = np.abs(exact - (-H)).max() / np.abs(exact).max()
    print(f"  binomial: analytic {val:12.6f}   finite {logdet:12.6f}"
          f"   gap {abs(val - logdet):.2e}   worst entry {entry:.1e}")
    assert entry < 1e-6, "the analytic blocks do not reproduce the log-posterior"
    assert abs(val - logdet) < 1e-4 * max(1.0, abs(logdet)), "blocks disagree"


def check_criterion_is_the_restored_column():
    """Recompute both criteria from the stored fit and compare with the record."""
    import json

    path = ROOT / "experiments" / "results" / "laplace.json"
    if not path.exists():
        print("  criterion: skipped, laplace.json absent")
        return
    blob = json.loads(path.read_text(encoding="utf-8"))
    n_eff = blob["n_eff"]
    for row in blob["rows"]:
        d = row["df"]
        restored_bic = -2.0 * row["pll"] + d * np.log(n_eff)
        assert abs(restored_bic - row["bic"]) < 1e-6, (
            f"K={row['k']}: BIC {restored_bic} vs {row['bic']}")
        restored = (-2.0 * row["pll"] + row["lnH_analytic"]
                    - d * np.log(2.0 * np.pi))
        assert abs(restored - row["lbic"]) < 1e-6, (
            f"K={row['k']}: {restored} vs {row['lbic']}")
        assert abs(row["lnH_analytic"] - row["lnH_h"]) < 0.1, (
            f"K={row['k']}: analytic ln|H| {row['lnH_analytic']} vs "
            f"central differences {row['lnH_h']}")
    print(f"  criterion: {len(blob['rows'])} rows reproduce BIC from the "
          f"penalised likelihood and LBIC from the analytic determinant, "
          f"and the central difference agrees")


def check_gradient():
    """The analytic gradient against a central difference of the objective.

    The Newton polish is its only consumer, and a sign or factor error there would
    leave the fit short of the mode without showing up in the criterion.
    """
    X, y = synthetic()
    k, a1 = 3, X.shape[1]
    lam_u, lam_sigma, lam_pi, s2 = 0.5, 40.0, 1.0, 0.8
    u = np.array([[0.1, -0.2, 0.3], [1.6, -1.1, 0.4], [-0.8, 1.5, 0.9]])
    phi = np.array([0.45, 0.85, 1.30])
    pi = np.array([0.42, 0.36, 0.22])

    g = penalised_gradient(X, y, u, phi, pi, GaussianResponse(), lambda_u=lam_u,
                           lambda_pi=lam_pi, lambda_sigma=lam_sigma, s2=s2)
    f = gaussian_objective(X, y, k, a1, lam_u, lam_sigma, lam_pi, s2)
    theta = np.concatenate([u.ravel(), phi, pi[:-1]])
    num = np.zeros_like(theta)
    for j in range(theta.size):
        h = 1e-5 * max(1.0, abs(theta[j]))
        e = np.zeros_like(theta)
        e[j] = h
        num[j] = (f(theta + e) - f(theta - e)) / (2.0 * h)
    err = np.abs(g - num) / np.maximum(1.0, np.abs(num))
    print(f"  gradient: worst relative error {err.max():.1e}")
    assert err.max() < 1e-6, "the analytic gradient is not the derivative of pll"

    # the protocol with the dispersion pinned: no phi coordinates at all
    rng = np.random.default_rng(5)
    Xb = rng.normal(size=(300, 2))
    trials = rng.integers(4, 12, size=300)
    ub = np.array([[0.8, -0.4], [-0.5, 1.1]])
    pb = np.array([0.55, 0.45])
    comp = rng.choice(2, size=300, p=pb)
    etab = np.einsum("ij,ij->i", Xb, ub[comp])
    yb = rng.binomial(trials, 1.0 / (1.0 + np.exp(-etab))) / trials
    resp = BinomialResponse()

    def fb(t):
        uu = t[: 2 * 2].reshape(2, 2)
        w = np.concatenate([t[4:], [1.0 - np.sum(t[4:])]])
        b = resp.loglik(yb[:, None], Xb @ uu.T, trials[:, None], np.ones((300, 2)))
        z = np.log(w)[None, :] + b
        mx = z.max(axis=1, keepdims=True)
        ll = float(np.sum(mx.ravel() + np.log(np.exp(z - mx).sum(axis=1))))
        ll -= 0.5 * lam_u * float(np.sum(uu ** 2))
        ll += (lam_pi - 1.0) * float(np.sum(np.log(w)))
        return ll

    gb = penalised_gradient(Xb, yb, ub, np.ones(2), pb, resp, omega=trials,
                            lambda_u=lam_u, lambda_pi=lam_pi, lambda_sigma=None)
    tb = np.concatenate([ub.ravel(), pb[:-1]])
    numb = np.zeros_like(tb)
    for j in range(tb.size):
        h = 1e-5 * max(1.0, abs(tb[j]))
        e = np.zeros_like(tb)
        e[j] = h
        numb[j] = (fb(tb + e) - fb(tb - e)) / (2.0 * h)
    errb = np.abs(gb - numb) / np.maximum(1.0, np.abs(numb))
    print(f"  gradient: binomial, worst relative error {errb.max():.1e}")
    assert errb.max() < 1e-6, "the pinned-dispersion gradient is wrong"


def main():
    print("audit_laplace")
    check_matrix()
    check_simplex_coordinates()
    check_binomial()
    check_gradient()
    check_criterion_is_the_restored_column()
    print("all assertions pass")


if __name__ == "__main__":
    main()
