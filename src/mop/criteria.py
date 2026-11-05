"""Selection criterion, defined once for every path that reports a BIC.

BIC = -2 lnL(theta_hat) - 2 ln f_0(theta_hat) + d ln n, with f_0 the joint prior of
the model. Keeping it in one module is what makes counts comparable across the
different observation families.
"""


from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def degrees_of_freedom(n_alternatives: int, k: int, phi_free: bool) -> int:
    """Free parameters: (A - 1) log-ratios, P - 1 weights, and P dispersions when they are estimated."""
    return (int(n_alternatives) - 1) * int(k) + (int(k) - 1) + (int(k) if phi_free else 0)


def df_from_design(n_columns: int, k: int, phi_free: bool) -> int:
    """Same count, with A - 1 taken from the width of the design matrix."""
    return degrees_of_freedom(int(n_columns) + 1, k, phi_free)


def log_prior_utilities(u, phi, lambda_u, u_bar=None) -> float:
    """Log density of ``u_p ~ N(u_bar, phi_p / lambda_u I)``, summed over components."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    n_p, a1 = u.shape
    phi = np.asarray(phi, dtype=float)
    phi = np.full(n_p, float(phi)) if phi.ndim == 0 else phi.ravel()
    lam = np.full(n_p, float(lambda_u)) if np.isscalar(lambda_u) else np.asarray(
        lambda_u, dtype=float).ravel()
    bar = np.zeros(a1) if u_bar is None else np.asarray(u_bar, dtype=float).ravel()

    total = 0.0
    for p in range(n_p):
        diff = u[p] - bar
        total -= 0.5 * lam[p] * float(np.sum(diff ** 2)) / phi[p]
        total -= 0.5 * a1 * np.log(TWO_PI * phi[p] / lam[p])
    return float(total)


def bic(penalised_ll: float, n_eff: float, n_free: int) -> float:
    """``-2 pll + d ln n``. Needs no mode, so a boundary weight can still be ranked."""
    n = float(n_eff)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError(f"BIC needs a positive sample size, got {n_eff!r}")
    return -2.0 * float(penalised_ll) + int(n_free) * np.log(n)


def lbic(penalised_ll: float, ln_det_h: float, n_free: int) -> float:
    """``-2 pll + ln|H| - d ln(2 pi)``, the Laplace approximation at the mode. ``inf`` when ``ln|H|`` is not finite, which happens when no interior mode exists."""
    ll = float(ln_det_h)
    if not np.isfinite(ll):
        return float("inf")
    return -2.0 * float(penalised_ll) + ll - int(n_free) * np.log(TWO_PI)


def pbic(penalised_ll: float, n_eff: float, n_free: int,
         utility_log_prior: float = 0.0) -> float:
    """``bic`` with the utility prior profiled out; reported for comparison, never used to select."""
    return (-2.0 * (float(penalised_ll) - float(utility_log_prior))
            + int(n_free) * np.log(float(n_eff)))
