"""The second derivative of the proposition against a finite difference."""


from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import invgamma, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import mop.inference as inference                                   # noqa: E402
from mop.initialization import kridge                               # noqa: E402

from common.params import lambda_pi, lambda_sigma_for, lambda_u     # noqa: E402

from case_study.estimator import EPS, LINK, as_pcvector, design, load_pairs  # noqa: E402

K = 4


def main():
    pcv = as_pcvector(load_pairs())
    n_j = len(pcv)
    X, y = design(pcv)
    A = X.shape[1] + 1
    lu, lpi, ls = lambda_u, lambda_pi, lambda_sigma_for(n_j)

    u0, phi_init, mp_init = kridge(X, y, K)
    res = inference.fit_mixture(pcv, LINK, K, lambda_u=lu, lambda_pi=lpi,
                        lambda_sigma=ls, u_init=u0, sigma2_init=phi_init,
                        mp_init=mp_init)
    d = res.to_dict()
    pis = np.asarray(d["mixture"]["pis"], dtype=float)
    phis = np.asarray(d["mixture"]["inconsistencies"], dtype=float)
    ws = np.asarray(d["mixture"]["priorities"], dtype=float)
    u = np.log(np.clip(ws[:, :-1], EPS, None)) - np.log(np.clip(ws[:, -1:], EPS, None))
    # The prior scale the fit itself uses, not k-ridge's starting dispersion: the
    # two differ (3.73 against 1.88 here) and using the wrong one moves the mode,
    # so the finite difference would be taken at a point that is not the optimum.
    phi0 = float(inference.prior_mode_phi0(X, y))

    eta = X @ u.T

    def pen(varphi, p):
        phi = phis.copy()
        phi[p] = varphi
        dens = norm.pdf(y[:, None], loc=eta, scale=np.sqrt(phi)[None, :])
        mix = dens * pis[None, :]
        tot = mix.sum(axis=1)
        ll = float(np.sum(np.log(tot)))
        ll += float(np.sum(norm.logpdf(u[p], loc=0.0, scale=np.sqrt(phi[p] / lu))))
        ll += float(invgamma.logpdf(phi[p], ls / 2.0, scale=ls * phi0 / 2.0))
        return ll

    print(f"{'comp':>5} {'phi':>9} {'fd':>12} {'closed':>12} {'closed+2':>12}")
    for p in range(K):
        h = 1e-4 * phis[p]
        fd = (pen(phis[p] + h, p) - 2 * pen(phis[p], p) + pen(phis[p] - h, p)) / h ** 2
        dens = norm.pdf(y[:, None], loc=eta, scale=np.sqrt(phis)[None, :])
        num = dens * pis[None, :]
        gamma = num / num.sum(axis=1, keepdims=True)
        g = gamma[:, p]
        a = ((y - eta[:, p]) ** 2 / phis[p] - 1.0) / (2.0 * phis[p])
        n_eff = float(g.sum())
        second = float(np.sum(g * (1 - g) * a ** 2))
        closed = -((A - 1) + ls + n_eff) / (2.0 * phis[p] ** 2) + second
        closed2 = -((A - 1) + ls + n_eff + 2.0) / (2.0 * phis[p] ** 2) + second
        print(f"{p + 1:>5} {phis[p]:9.4f} {fd:12.2f} {closed:12.2f} {closed2:12.2f}")


if __name__ == "__main__":
    main()
