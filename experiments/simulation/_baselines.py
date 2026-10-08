"""Baseline: P=1 — MoP with K=1, identical prior specification."""

import numpy as np
from scipy.special import logit

from mop import inference
from mop.pairwise_comparison import PairwiseComparison, comparisons2regression

EPS = 1e-10


def fit_single_k1(pairs, n_alternatives):
    pcv, _, _ = _pairs_to_pcvector(pairs, link="logit")
    nJ = len(pcv)
    lu = 1e-6; lpi = 1; ls = nJ / 50
    # φ₀ comes from inference.prior_mode_phi0, which the fit applies itself;
    # the copy that used to be computed here was never passed anywhere.
    result = inference.mop(pcv, [lambda s: logit(np.clip(s, EPS, 1 - EPS))], 1,
                           lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls)
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
