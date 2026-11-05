"""The one-component reference model: the same estimator with a single component and the same prior."""


import numpy as np
from scipy.special import logit

from mop import inference
from mop.pairwise_comparison import PairwiseComparison, comparisons2regression

from .params import lambda_pi, lambda_sigma_for, lambda_u

EPS = 1e-10


def fit_single_k1(pairs, n_alternatives, s2=None):
    pcv, _, _ = _pairs_to_pcvector(pairs, link="logit")
    nJ = len(pcv)
    lu, lpi, ls = lambda_u, lambda_pi, lambda_sigma_for(nJ)
    result = inference.fit_mixture(pcv, [lambda s: logit(np.clip(s, EPS, 1 - EPS))], 1,
                           lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls, s2=s2)
    return {"mop_result": result}


def predict_single_k1(model_state, pairs):
    w = model_state["mop_result"][0].priority_vector
    preds = []
    for p in pairs:
        log_odds = np.log(w[int(p["ft"])] / (w[int(p["lt"])] + EPS))
        preds.append(1.0 / (1.0 + np.exp(-log_odds)))
    return np.array(preds)


def get_priorities_single_k1(model_state):
    return model_state["mop_result"][0].priority_vector.reshape(1, -1)


def _pairs_to_pcvector(pairs, link="logit"):
    link_fn = (lambda s: logit(np.clip(s, EPS, 1 - EPS))) if link == "logit" else None
    pcv = [PairwiseComparison(former_id=int(p["ft"]) + 1, latter_id=int(p["lt"]) + 1,
                               agent_id=1, judgment=p["score"]) for p in pairs]
    X, y, _ = comparisons2regression(pcv, [link_fn])
    return pcv, X, y
