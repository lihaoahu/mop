"""Comparing a fitted mixture with the truth that generated it."""


from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau


def priority_vector_from_logratios(u):
    """Convert log-ratio matrix (nP × nA-1) to normalized priority vectors (nP × nA)."""
    u = np.clip(np.asarray(u), -50, 50)
    nP = u.shape[0]
    v = np.hstack([np.exp(u), np.ones((nP, 1))])
    return v / np.sum(v, axis=1, keepdims=True)


def logratio_from_priority(w):
    """Convert normalized priority vectors (nP x nA) to log-ratios against the last alternative.

    Utilities are reported on this scale throughout, so that the recovered structure is
    compared with the truth in the space the model estimates rather than in shares.
    """
    w = np.maximum(np.asarray(w, dtype=float), 1e-300)
    return np.log(w[..., :-1] / w[..., -1:])


def match_components(u_true, u_est):
    """Pair each fitted component with a true one by utility-vector similarity."""
    w_true = np.atleast_2d(np.asarray(u_true, dtype=float))
    w_est = np.atleast_2d(np.asarray(u_est, dtype=float))
    nt, ne = w_true.shape[0], w_est.shape[0]
    sim = np.zeros((nt, ne))
    for i in range(nt):
        for j in range(ne):
            denom = np.linalg.norm(w_true[i]) * np.linalg.norm(w_est[j])
            sim[i, j] = float(np.dot(w_true[i], w_est[j]) / denom) if denom > 0 else 0.0
    ti, ei = linear_sum_assignment(-sim)
    return ti, ei, sim


def cosine_similarity(u_true, u_est):
    """Mean cosine similarity over the matched component pairs, computed on utilities."""
    _, _, sim = match_components(u_true, u_est)
    if sim.size == 0:
        return 0.0
    ti, ei = linear_sum_assignment(-sim)
    return float(np.mean(sim[ti, ei]))


def kl_divergence(u_true, pi_true, u_est, pi_est):
    """KL divergence between the true mixture weights and a fitted permutation."""
    pi_true = np.asarray(pi_true, dtype=float)
    pi_est = np.asarray(pi_est, dtype=float)
    if len(pi_true) != len(pi_est) or len(pi_true) < 2:
        return float("nan")
    ti, ei, _ = match_components(u_true, u_est)
    if len(ti) != len(pi_true):
        return float("nan")
    pi_est_matched = np.empty_like(pi_est)
    pi_est_matched[ti] = pi_est[ei]
    p = np.maximum(pi_true, 0.0)
    q = np.maximum(pi_est_matched, 0.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(np.where(p > 0, p * np.log(p / np.maximum(q, 1e-300)), 0.0)))


def ranking_from_priority(w):
    """Convert priority vector to ranking (highest=1)."""
    return np.argsort(np.argsort(-w)) + 1


def kendall_tau(w_true, w_est):
    """Kendall's τ between true and estimated weighted-average rankings."""
    r_true = ranking_from_priority(w_true)
    r_est = ranking_from_priority(w_est)
    tau, _ = kendalltau(r_true, r_est)
    return float(tau) if not np.isnan(tau) else 0.0
