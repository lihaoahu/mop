"""experiments/simulation shared utilities — metrics, data generation, result helpers."""

import json
import numpy as np
from pathlib import Path
from scipy.special import logit
from scipy.stats import kendalltau, spearmanr, pearsonr

SIM_DIR = Path(__file__).resolve().parent
PROJECT = SIM_DIR.parent.parent

with open(SIM_DIR / "config.json", encoding="utf-8") as f:
    CFG = json.load(f)

# ── Paths ──────────────────────────────────────────────────────────
ENERGY_PAIRS = PROJECT / "experiments" / "case_study" / "energy_pairs.jsonl"
SIM_RESULTS = SIM_DIR / "simulation_results.json"
HOLDOUT_RESULTS = SIM_DIR / "holdout_results.json"
REPORT_DIR = SIM_DIR / "report"
FIGURES_DIR = PROJECT / "manuscript" / "figures"

# Only the report directory is created here. TABLES_DIR used to be made as well,
# which meant importing any simulation script silently added an empty
# manuscript/tables/ to the manuscript tree; nothing ever wrote there.
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_TYPES = ["纯电", "插混", "增程", "燃油"]
TYPE_MAP_FULL = None  # lazy-loaded below

def _get_type_map():
    global TYPE_MAP_FULL
    if TYPE_MAP_FULL is None:
        TYPE_MAP_FULL = {**TYPE_MAP, "燃油": "燃油", "纯电": "纯电", "插混": "插混", "增程": "增程"}
    return TYPE_MAP_FULL

# ── Metrics ────────────────────────────────────────────────────────

def priority_vector_from_logratios(u):
    """Convert log-ratio matrix (nP × nA-1) to normalized priority vectors (nP × nA)."""
    u = np.clip(np.asarray(u), -50, 50)
    nP = u.shape[0]
    nA = u.shape[1] + 1
    v = np.hstack([np.exp(u), np.ones((nP, 1))])
    return v / np.sum(v, axis=1, keepdims=True)


def cosine_similarity(w_true, w_est):
    """Mean cosine similarity between best-matched component pairs."""
    nP_true = w_true.shape[0]
    nP_est = w_est.shape[0]
    # Greedy matching: for each true component, find best estimated match
    similarities = []
    used = set()
    for i in range(nP_true):
        best_sim = -1.0
        best_j = -1
        for j in range(nP_est):
            if j in used:
                continue
            sim = np.dot(w_true[i], w_est[j]) / (
                np.linalg.norm(w_true[i]) * np.linalg.norm(w_est[j]) + 1e-300
            )
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            similarities.append(best_sim)
    return float(np.mean(similarities)) if similarities else 0.0


def kl_divergence(pi_true, pi_est):
    """KL divergence D_KL(pi_true || pi_est)."""
    pi_true = np.asarray(pi_true, dtype=float)
    pi_est = np.asarray(pi_est, dtype=float)
    # Align sizes: pad smaller with zeros
    n = max(len(pi_true), len(pi_est))
    p = np.zeros(n)
    q = np.zeros(n)
    p[:len(pi_true)] = pi_true
    q[:len(pi_est)] = pi_est
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def ranking_from_priority(w):
    """Convert priority vector to ranking (highest=1)."""
    return np.argsort(np.argsort(-w)) + 1


def kendall_tau(w_true, w_est):
    """Kendall's τ between true and estimated weighted-average rankings."""
    r_true = ranking_from_priority(w_true)
    r_est = ranking_from_priority(w_est)
    tau, _ = kendalltau(r_true, r_est)
    return float(tau) if not np.isnan(tau) else 0.0


# ── Synthetic data generation ──────────────────────────────────────

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
    w = priority_vector_from_logratios(u_logratios)  # (nP × nA)

    # Sample component memberships
    comp_idx = rng.choice(nP, size=n_comparisons, p=pi)

    # Sample pairs uniformly from all (i,j) with i<j
    all_pairs = [(i, j) for i in range(nA) for j in range(i + 1, nA)]
    pair_idx = rng.choice(len(all_pairs), size=n_comparisons, replace=True)

    pairs = []
    eps = 1e-6
    for k in range(n_comparisons):
        p = comp_idx[k]
        i, j = all_pairs[pair_idx[k]]
        # Log-odds of i > j under component p
        log_odds = np.log(w[p, i] / (w[p, j] + eps))
        # Add Gaussian noise scaled by sigma
        noisy = log_odds + rng.normal(0, np.sqrt(sigma2[p]))
        # Convert to probability via sigmoid
        score = 1.0 / (1.0 + np.exp(-noisy))
        score = float(np.clip(score, 0.001, 0.999))
        pairs.append({"ft": i, "lt": j, "score": score})

    return pairs, w, comp_idx


def generate_cancellation():
    """Cancellation — opposite preferences log-ratios, balanced → near-uniform."""
    # u1 = [log(w_A/w_D), log(w_B/w_D), log(w_C/w_D)], D is the reference (u=0)
    # Opposite: u2 = -u1
    u = np.array([
        [2.0, 1.0, 0.5],     # A > B > C > D (strict ordering, no ties)
        [-2.0, -1.5, -1.0],  # D > C > B > A (exact reverse)
    ])
    pi = np.array([0.5, 0.5])
    return u, pi


def generate_random_mixture(K, nA, rng=None):
    """Generate a random mixture model with K components and nA alternatives."""
    if rng is None:
        rng = np.random.default_rng()
    # Random log-ratios from N(0, 1)
    u = rng.normal(0, 1, (K, nA - 1))
    # Random mixture probabilities from Dirichlet(2, ..., 2)
    pi = rng.dirichlet(np.full(K, 2.0))
    return u, pi


def generate_noise_ground_truth(noise_fraction=0.3):
    """Noise scenario — true K=1 but fraction of comparisons are random.

    The population has ONE true preference, but a fraction of raters give
    random (uniform) responses. MoP should discover K=2: one clean component
    capturing the real preference, and one noise component with high σ²
    and near-uniform priority vector to absorb the random responses.

    Returns:
        u_logratios: (1 × nA-1) the single true preference
        pi_true: just [1.0] (ground truth is K=1)
        noise_fraction: the fraction of random comparisons to inject
    """
    # Single true preference: A > B > C > D
    u = np.array([[1.5, 1.0, 0.5]])
    pi = np.array([1.0])
    return u, pi, noise_fraction


def generate_noise_pairs(u_logratios, sigma2, n_comparisons, noise_fraction, rng=None):
    """Generate pairs for noise scenario: majority follow true preference,
    a noise_fraction are completely random (score ~ Uniform(0,1))."""
    if rng is None:
        rng = np.random.default_rng()
    nA = u_logratios.shape[1] + 1
    w = priority_vector_from_logratios(u_logratios)[0]  # (nA,)

    all_pairs = [(i, j) for i in range(nA) for j in range(i + 1, nA)]
    n_signal = int(n_comparisons * (1 - noise_fraction))
    n_noise = n_comparisons - n_signal

    pairs = []
    # Signal comparisons
    pair_idx = rng.choice(len(all_pairs), size=n_signal, replace=True)
    eps = 1e-6
    for k in range(n_signal):
        i, j = all_pairs[pair_idx[k]]
        log_odds = np.log(w[i] / (w[j] + eps))
        noisy = log_odds + rng.normal(0, np.sqrt(sigma2))
        score = 1.0 / (1.0 + np.exp(-noisy))
        score = float(np.clip(score, 0.001, 0.999))
        pairs.append({"ft": i, "lt": j, "score": score})

    # Noise comparisons — random scores
    pair_idx = rng.choice(len(all_pairs), size=n_noise, replace=True)
    for k in range(n_noise):
        i, j = all_pairs[pair_idx[k]]
        score = float(rng.uniform(0, 1))
        pairs.append({"ft": i, "lt": j, "score": score})

    rng.shuffle(pairs)
    # Ground truth: single component priority
    w_gt = w.reshape(1, -1)
    return pairs, w_gt


# ── case-study TYPE_MAP for holdout ───────────────────────────────

# Import TYPE_MAP from the case-study config
_CASE_CFG_PATH = SIM_DIR.parent / "case_study" / "config.json"
with open(_CASE_CFG_PATH, encoding="utf-8") as f:
    _CASE_CFG = json.load(f)
TYPE_MAP = _CASE_CFG["energy_type_mapping"]
