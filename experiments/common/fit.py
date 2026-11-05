"""Count selection over a grid, for index-coded judgments.

Shared by the validation section and the holdout evaluation: same starts, same
criterion, same grid.
"""


from __future__ import annotations

import numpy as np
from scipy.special import logit

from mop import inference
from mop.initialization import kridge
from mop.pairwise_comparison import PairwiseComparison, comparisons2regression

from .params import lambda_pi, lambda_sigma_for, lambda_u, max_components_for

EPS = 1e-10


def fit_mop(pairs, n_alternatives, max_K=None, s2=None):
    """Fit every count up to ``max_K`` and return the BIC-selected mixture."""
    if max_K is None:
        max_K = max_components_for(n_alternatives)

    name_to_id = {i: i + 1 for i in range(n_alternatives)}
    pcv = [PairwiseComparison(
        former_id=name_to_id[int(p["ft"])],
        latter_id=name_to_id[int(p["lt"])],
        agent_id=1, judgment=p["score"],
    ) for p in pairs]

    link_fns = [lambda s, _eps=EPS: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcv, link_fns)
    nJ = len(pcv)
    lu, lpi, ls = lambda_u, lambda_pi, lambda_sigma_for(nJ)

    best_k, best_bic, best_result = None, np.inf, None

    for k in range(1, max_K + 1):
        u0, phi0, mp0 = kridge(X, y, k)
        result = inference.fit_mixture(pcv, link_fns, k, lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls,
                               s2=s2, u_init=u0, sigma2_init=phi0, mp_init=mp0)
        bic_val = inference.bic(pcv, result, link_fns, lambda_u=lu, lambda_sigma=ls,
                                lambda_pi=lpi, s2=s2)
        if bic_val < best_bic:
            best_bic, best_k, best_result = bic_val, k, result

    w_avg = np.zeros(n_alternatives)
    for pi, pr in zip(best_result.mixture_probabilities, best_result.priorities):
        w_avg += np.array(pr.priority_vector) * pi

    return {
        "k": best_k,
        "priorities": np.array([pr.priority_vector for pr in best_result.priorities]),
        "pis": best_result.mixture_probabilities.copy(),
        "sigma2": np.array([pr.inconsistency_level for pr in best_result.priorities]),
        "weighted_priority": w_avg,
        "bic": best_bic,
    }
