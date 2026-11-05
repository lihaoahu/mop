"""Fitting and count selection for the corpus.

Every case-study number in the paper goes through this module, which fixes the
link, the clipping of the scores, the prior and the grid, and nothing else.
"""


from __future__ import annotations

import json

import numpy as np
from scipy.special import logit

from mop import inference, pairwise_comparison
from mop.inference import comparisons2regression
from mop.initialization import kridge
from mop.mixture_of_priorities import expected_utility

from common.params import (
    LAMBDA_SIGMA_OVER_N,
    max_components_for,
    lambda_pi,
    lambda_sigma_for,
    lambda_u,
)
from common.paths import DIMS, ENERGY_PAIRS, T, TYPE_MAP  # noqa: F401

EPS = 1e-10
LAMBDA_U = lambda_u
LAMBDA_PI = lambda_pi

LINK = [lambda s, _eps=EPS: logit(np.clip(s, _eps, 1 - _eps))]
NAME_TO_ID = {t: i + 1 for i, t in enumerate(T)}


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def load_pairs(dim_name=None, year=None):
    """Cross-energy-type judgments, optionally within one dimension or year.

    Multi-label judgments appear once per label, so a dimension's ``n`` is the
    number of judgments carrying that label and the seven dimensions do not sum
    to the pooled count.
    """
    pairs = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if year and r.get("publish_time", "")[:4] != year:
                continue
            for p in r.get("comparison_pairs", []):
                if dim_name and p.get("dimension") != dim_name:
                    continue
                ft = TYPE_MAP.get(p.get("former_type", ""), "")
                lt = TYPE_MAP.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": ft, "lt": lt,
                                  "score": p.get("score", 0.5)})
    return pairs


def load_articles():
    """The same judgments, kept as one cluster per article.

    Only the article-level bootstrap needs the clustering; everything else wants
    the flat list.  An article with no usable judgment is dropped rather than
    kept empty, so a draw never contains an empty cluster.
    """
    articles = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pairs = []
            for p in r.get("comparison_pairs", []):
                ft = TYPE_MAP.get(p.get("former_type", ""), "")
                lt = TYPE_MAP.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": ft, "lt": lt,
                                  "score": float(p.get("score", 0.5))})
            if pairs:
                articles.append(pairs)
    return articles


def as_pcvector(pairs):
    """The library's comparison objects for an already-mapped list."""
    return [pairwise_comparison.PairwiseComparison(
        NAME_TO_ID[p["ft"]], NAME_TO_ID[p["lt"]], 1, p["score"]) for p in pairs]


def design(pcv):
    """The regression design and the logit responses for the ratio protocol."""
    X, y, _ = comparisons2regression(pcv, LINK)
    return np.asarray(X, float), np.asarray(y, float)


# --------------------------------------------------------------------------
# one count
# --------------------------------------------------------------------------

def mode_arrays(result, X):
    """The fitted utilities, dispersions and weights of a result, as arrays."""
    k = len(result.priorities)
    u = np.zeros((k, X.shape[1]))
    for i, pr in enumerate(result.priorities):
        w = np.asarray(pr.priority_vector, float)
        u[i] = np.log(w[:-1] / w[-1])
    phi = np.array([float(pr.inconsistency_level) for pr in result.priorities])
    return u, phi, np.asarray(result.mixture_probabilities, float)


def _posterior(result, X, y):
    """The penalised log-posterior at the returned mode, the quantity a fit maximizes."""
    u, phi, pi = mode_arrays(result, X)
    return float(inference.penalized_log_likelihood(
        X, y, u, phi, pi, LAMBDA_U, lambda_sigma_for(len(y)), LAMBDA_PI,
        float(inference.prior_mode_phi0(X, y))))


def _warm_init(warm, X, y, k):
    """The previous count's mode plus one K-Ridge seed for the component added.

    The seed is the last row of the K-Ridge solution at ``k``, which is the
    component that start would have placed where the mixture is weakest, so the
    warm start adds a component in the same place as the cold one and differs
    from it only in the components already fitted.
    """
    u_prev, phi_prev, pi_prev = warm
    u0, phi0, mp0 = kridge(X, y, k)
    u = np.vstack([u_prev, u0[-1:]])
    phi = np.concatenate([phi_prev, phi0[-1:]])
    pi = np.concatenate([pi_prev, mp0[-1:]])
    return u, phi, pi / pi.sum()


def fit_at(pcv, k, X=None, y=None, warm=None):
    """One count, fitted from a fresh K-Ridge start.

    Returns the mode together with its BIC and pBIC. ``warm``, when given, starts
    from the mode at ``k - 1`` plus one K-Ridge seed; it is a rescue for a cold start
    that lands on the boundary, not a default, because a chained start inherits the
    local optimum of the smaller fit -- fifty nats worse at four components here.
    """
    if X is None or y is None:
        X, y = design(pcv)
    lu, lpi, ls = lambda_u, lambda_pi, lambda_sigma_for(len(pcv))
    if warm is None:
        u0, phi0, mp0 = kridge(X, y, k)
    else:
        u0, phi0, mp0 = _warm_init(warm, X, y, k)
    result = inference.fit_mixture(pcv, LINK, k, lambda_u=lu, lambda_pi=lpi,
                           lambda_sigma=ls, u_init=u0, sigma2_init=phi0,
                           mp_init=mp0)
    bi = float(inference.bic(pcv, result, LINK, lambda_u=lu,
                             lambda_sigma=ls, lambda_pi=lpi))
    lb = float(inference.lbic(pcv, result, LINK, lambda_u=lu,
                              lambda_sigma=ls, lambda_pi=lpi))
    pb = float(inference.pbic(pcv, result, LINK, lambda_u=lu,
                              lambda_sigma=ls, lambda_pi=lpi))
    return result, bi, lb, pb


def fit_arrays(pcv, k, X=None, y=None):
    """One count as arrays: the utilities, the dispersions, the weights, the pll.

    ``pll`` is the penalized log-posterior evaluated at the returned mode, so the
    caller can rebuild pBIC component by component instead of reading the
    library's cached value.
    """
    if X is None or y is None:
        X, y = design(pcv)
    n_j = len(pcv)
    lam_sigma = LAMBDA_SIGMA_OVER_N * float(n_j)
    s2 = float(inference.prior_mode_phi0(X, y))
    u0, phi0, mp0 = kridge(X, y, k)
    res = inference.fit_mixture(pcv, LINK, k, lambda_u=LAMBDA_U, lambda_pi=LAMBDA_PI,
                        lambda_sigma=lam_sigma, s2=s2,
                        u_init=u0, sigma2_init=phi0, mp_init=mp0)
    k_fit = len(res.mixture_probabilities)
    u = np.zeros((k_fit, X.shape[1]))
    for i, pr in enumerate(res.priorities):
        w = np.asarray(pr.priority_vector, float)
        u[i] = np.log(w[:-1] / w[-1])
    sigma2 = np.array([float(pr.inconsistency_level) for pr in res.priorities])
    pi = np.asarray(res.mixture_probabilities, float)
    pll = float(inference.penalized_log_likelihood(
        X, y, u, sigma2, pi, LAMBDA_U, lam_sigma, LAMBDA_PI, s2))
    ll = float(inference.log_likelihood(X, y, u, sigma2, pi))
    return {"k": int(k_fit), "u": u, "sigma2": sigma2, "pi": pi,
            "pll": pll, "ll": ll, "s2": s2, "n": n_j,
            "lambda_sigma": lam_sigma, "X": X, "y": y, "pcv": pcv,
            "polish": res._polish}


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def run_mop(pairs, trace=None):
    """Fit the grid and return the BIC-selected fit.

    Returns ``(k, fit, expected_utility_vector, n_judgments)``, where ``fit`` is the
    ``to_dict`` of the selected model that the tables read. ``trace``, when given,
    receives the whole curve, which is what Figure 3 plots.
    """
    n_j = len(pairs)
    pcv = as_pcvector(pairs)
    X, y = design(pcv)
    n_alt = X.shape[1] + 1

    models, bic_vals, lbic_vals, pbic_vals = {}, {}, {}, {}
    warm = None
    for k in range(1, max_components_for(n_alt) + 1):
        result, bi, lb, pb = fit_at(pcv, k, X, y)
        if not np.isfinite(lb) and warm is not None:
            # The from-scratch fit put a weight on the boundary, where the
            # Laplace version is undefined.  That is a property of the start, not
            # of the count: refit from the previous count's mode plus one seed,
            # and keep whichever mode has the higher posterior.  The BIC does not
            # need this -- it is defined either way -- but the supplement reports
            # the determinant, and a bound is worth one refit.
            alt, bi_alt, lb_alt, pb_alt = fit_at(pcv, k, X, y, warm=warm)
            if np.isfinite(lb_alt) and _posterior(alt, X, y) > _posterior(result, X, y):
                result, bi, lb, pb = alt, bi_alt, lb_alt, pb_alt
        steps = (result._polish or {}).get("steps") if result._polish is not None else None
        models[str(k)] = {**result.to_dict(), "bic": bi, "lbic": lb, "pbic": pb,
                          "polish_steps": steps}
        bic_vals[k], lbic_vals[k], pbic_vals[k] = bi, lb, pb
        warm = mode_arrays(result, X)
        if len(result) < k:
            break

    defined = {k: v for k, v in bic_vals.items() if np.isfinite(v)}
    if not defined:
        raise RuntimeError("no count has a finite BIC")
    best_k = min(defined, key=defined.get)
    best = models[str(best_k)]
    weighted = expected_utility(best["mixture"]["priorities"],
                                best["mixture"]["pis"])
    if trace is not None:
        trace["bic_vals"] = dict(bic_vals)
        trace["lbic_vals"] = dict(lbic_vals)
        trace["pbic_vals"] = dict(pbic_vals)
        trace["undefined"] = sorted(k for k in lbic_vals
                                    if not np.isfinite(lbic_vals[k]))
        trace["models"] = models
    return best_k, best, weighted, n_j
