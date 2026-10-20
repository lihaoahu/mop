"""experiments/simulation shared utilities — metrics, data generation, result helpers."""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from scipy.stats import kendalltau, spearmanr, pearsonr

SIM_DIR = Path(__file__).resolve().parent
PROJECT = SIM_DIR.parent.parent

with open(SIM_DIR / "config.json", encoding="utf-8") as f:
    CFG = json.load(f)

ENERGY_PAIRS = PROJECT / "experiments" / "case_study" / "energy_pairs.jsonl"
SIM_RESULTS = SIM_DIR / "simulation_results.json"
HOLDOUT_RESULTS = SIM_DIR / "holdout_results.json"
REPORT_DIR = SIM_DIR / "report"
FIGURES_DIR = PROJECT / "manuscript" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_TYPES = ["纯电", "插混", "增程", "燃油"]
TYPE_MAP_FULL = None

def _get_type_map():
    global TYPE_MAP_FULL
    if TYPE_MAP_FULL is None:
        TYPE_MAP_FULL = {**TYPE_MAP, "燃油": "燃油", "纯电": "纯电", "插混": "插混", "增程": "增程"}
    return TYPE_MAP_FULL


def priority_vector_from_logratios(u):
    """Convert log-ratio matrix (nP × nA-1) to normalized priority vectors (nP × nA)."""
    u = np.clip(np.asarray(u), -50, 50)
    nP = u.shape[0]
    nA = u.shape[1] + 1
    v = np.hstack([np.exp(u), np.ones((nP, 1))])
    return v / np.sum(v, axis=1, keepdims=True)


def match_components(w_true, w_est):
    """Pair each fitted component with a true one, by priority-vector similarity."""
    w_true = np.atleast_2d(np.asarray(w_true, dtype=float))
    w_est = np.atleast_2d(np.asarray(w_est, dtype=float))
    nt, ne = w_true.shape[0], w_est.shape[0]
    sim = np.zeros((nt, ne))
    for i in range(nt):
        for j in range(ne):
            denom = np.linalg.norm(w_true[i]) * np.linalg.norm(w_est[j])
            sim[i, j] = float(np.dot(w_true[i], w_est[j]) / denom) if denom > 0 else 0.0
    ti, ei = linear_sum_assignment(-sim)
    return ti, ei, sim


def cosine_similarity(w_true, w_est):
    """Mean cosine similarity over the matched component pairs."""
    _, _, sim = match_components(w_true, w_est)
    if sim.size == 0:
        return 0.0
    ti, ei = linear_sum_assignment(-sim)
    return float(np.mean(sim[ti, ei]))


def kl_divergence(w_true, pi_true, w_est, pi_est):
    """KL divergence between the true mixture weights and a fitted permutation."""
    pi_true = np.asarray(pi_true, dtype=float)
    pi_est = np.asarray(pi_est, dtype=float)
    if len(pi_true) != len(pi_est) or len(pi_true) < 2:
        return float("nan")
    ti, ei, _ = match_components(w_true, w_est)
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



def generate_synthetic_pairs(u_logratios, pi, sigma2, n_comparisons, rng=None):
    """
    Generate synthetic pairwise comparisons from a known mixture model.

    Args:
        u_logratios: (nP × nA-1) log-ratio matrix for each component
        pi: (nP,) mixture probabilities
        sigma2: (nP,) inconsistency variances
        n_comparisons: total number of comparisons to generate
        rng: numpy random generator

    Returns:
        list of dicts: [{"ft": int, "lt": int, "score": float}, ...]
    """
    if rng is None:
        rng = np.random.default_rng()
    nP, nA_minus_1 = u_logratios.shape
    nA = nA_minus_1 + 1
    w = priority_vector_from_logratios(u_logratios)

    comp_idx = rng.choice(nP, size=n_comparisons, p=pi)

    all_pairs = [(i, j) for i in range(nA) for j in range(i + 1, nA)]
    pair_idx = rng.choice(len(all_pairs), size=n_comparisons, replace=True)

    pairs = []
    eps = 1e-6
    for k in range(n_comparisons):
        p = comp_idx[k]
        i, j = all_pairs[pair_idx[k]]
        log_odds = np.log(w[p, i] / (w[p, j] + eps))
        noisy = log_odds + rng.normal(0, np.sqrt(sigma2[p]))
        score = 1.0 / (1.0 + np.exp(-noisy))
        score = float(np.clip(score, 0.001, 0.999))
        pairs.append({"ft": i, "lt": j, "score": score})

    return pairs, w, comp_idx


def generate_cancellation():
    """Cancellation — opposite preferences log-ratios, balanced → near-uniform."""
    u = np.array([
        [2.0, 1.0, 0.5],
        [-2.0, -1.5, -1.0],
    ])
    pi = np.array([0.5, 0.5])
    return u, pi


def generate_random_mixture(K, nA, rng=None):
    """Generate a random mixture model with K components and nA alternatives."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.normal(0, 1, (K, nA - 1))
    pi = rng.dirichlet(np.full(K, 2.0))
    return u, pi


def generate_noise_ground_truth():
    """Noise scenario — true K=1 but a fraction of comparisons are random.

    Returns:
        u_logratios: (1 x nA-1) the single true preference
        pi_true: just [1.0] (ground truth is K=1)
    """
    u = np.array([[1.5, 1.0, 0.5]])
    pi = np.array([1.0])
    return u, pi


def generate_noise_pairs(u_logratios, sigma2, n_comparisons, noise_fraction, rng=None):
    """Generate pairs for noise scenario: majority follow true preference,
    a noise_fraction are completely random (score ~ Uniform(0,1))."""
    if rng is None:
        rng = np.random.default_rng()
    nA = u_logratios.shape[1] + 1
    w = priority_vector_from_logratios(u_logratios)[0]

    all_pairs = [(i, j) for i in range(nA) for j in range(i + 1, nA)]
    n_signal = int(n_comparisons * (1 - noise_fraction))
    n_noise = n_comparisons - n_signal

    pairs = []
    pair_idx = rng.choice(len(all_pairs), size=n_signal, replace=True)
    eps = 1e-6
    for k in range(n_signal):
        i, j = all_pairs[pair_idx[k]]
        log_odds = np.log(w[i] / (w[j] + eps))
        noisy = log_odds + rng.normal(0, np.sqrt(sigma2))
        score = 1.0 / (1.0 + np.exp(-noisy))
        score = float(np.clip(score, 0.001, 0.999))
        pairs.append({"ft": i, "lt": j, "score": score})

    pair_idx = rng.choice(len(all_pairs), size=n_noise, replace=True)
    for k in range(n_noise):
        i, j = all_pairs[pair_idx[k]]
        score = float(rng.uniform(0, 1))
        pairs.append({"ft": i, "lt": j, "score": score})

    rng.shuffle(pairs)
    w_gt = w.reshape(1, -1)
    return pairs, w_gt



_CASE_CFG_PATH = SIM_DIR.parent / "case_study" / "config.json"
with open(_CASE_CFG_PATH, encoding="utf-8") as f:
    _CASE_CFG = json.load(f)
TYPE_MAP = _CASE_CFG["energy_type_mapping"]
