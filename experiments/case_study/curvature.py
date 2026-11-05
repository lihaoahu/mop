"""The curvature that identifies the dispersion, evaluated at the reported mode."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
from scipy.stats import norm

import mop.inference as inference
from mop.initialization import kridge
from mop.mixture_of_priorities import expected_utility

from common.params import lambda_pi, lambda_sigma_for, lambda_u
from common.paths import ABBREV, T, result

from case_study.estimator import EPS, LINK, as_pcvector, design, load_pairs

OUT = result("curvature.json")
K = 4


def main():
    pcv = as_pcvector(load_pairs())
    n_j = len(pcv)
    X, y = design(pcv)
    n_a = X.shape[1] + 1
    lu, lpi, ls = lambda_u, lambda_pi, lambda_sigma_for(n_j)

    u0, phi_init, mp_init = kridge(X, y, K)
    res = inference.fit_mixture(pcv, LINK, K, lambda_u=lu, lambda_pi=lpi,
                        lambda_sigma=ls, u_init=u0, sigma2_init=phi_init,
                        mp_init=mp_init)
    d = res.to_dict()
    pis = np.asarray(d["mixture"]["pis"], dtype=float)
    phis = np.asarray(d["mixture"]["inconsistencies"], dtype=float)
    ws = np.asarray(d["mixture"]["priorities"], dtype=float)

    # Utilities on the log scale against the benchmark alternative, which is the
    # parameterisation of the fit.
    u = np.log(np.clip(ws[:, :-1], EPS, None)) - np.log(np.clip(ws[:, -1:], EPS, None))

    eta = X @ u.T                                     # K x P
    dens = norm.pdf(y[:, None], loc=eta, scale=np.sqrt(phis)[None, :])
    num = dens * pis[None, :]
    gamma = num / num.sum(axis=1, keepdims=True)
    n_eff = gamma.sum(axis=0)

    rows = []
    for p in range(K):
        g = gamma[:, p]
        a = ((y - eta[:, p]) ** 2 / phis[p] - 1.0) / (2.0 * phis[p])
        first = -((n_a - 1) + ls + 2.0 + n_eff[p]) / (2.0 * phis[p] ** 2)
        second = float(np.sum(g * (1.0 - g) * a ** 2))
        rows.append({
            "component": p + 1,
            "pi": float(pis[p]),
            "phi": float(phis[p]),
            "n_eff": float(n_eff[p]),
            "complete_data_term": float(first),
            "missing_info_term": second,
            "second_derivative": float(first + second),
            "concave": bool(first + second < 0.0),
        })
        print(f"C{p + 1}  pi={pis[p]:.4f} phi={phis[p]:.4f} "
              f"n_eff={n_eff[p]:8.2f}  d2={first + second:12.2f}  "
              f"({first:10.2f} + {second:8.2f})")

    out = {"n": n_j, "A": n_a, "K": K, "lambda_sigma": float(ls),
           "components": rows,
           "min": min(r["second_derivative"] for r in rows),
           "max": max(r["second_derivative"] for r in rows),
           "all_concave": all(r["concave"] for r in rows)}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"all concave: {out['all_concave']}  range "
          f"{out['min']:.1f} .. {out['max']:.1f}")

    eu = expected_utility(ws, pis)
    print("expected utility:", {ABBREV[t]: round(float(v), 3) for t, v in zip(T, eu)})


if __name__ == "__main__":
    main()
