"""Synthetic judgments for the validation section."""


from __future__ import annotations

import numpy as np
from scipy.special import expit

from .metrics import priority_vector_from_logratios

EPS = 1e-12


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------

def pairs(n_alt: int):
    """All unordered pairs, ``(i, j)`` with ``i < j``, 0-indexed."""
    return [(i, j) for i in range(n_alt) for j in range(i + 1, n_alt)]


def design(n_alt: int):
    """Design matrix for every pair, in :func:`pairs` order."""
    rows = []
    for i, j in pairs(n_alt):
        row = np.zeros(n_alt - 1)
        if i < n_alt - 1:
            row[i] += 1.0
        if j < n_alt - 1:
            row[j] -= 1.0
        rows.append(row)
    return np.array(rows)


def priorities(u):
    """Normalised priority vectors from log-ratios."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    v = np.hstack([np.exp(np.clip(u, -50, 50)), np.ones((u.shape[0], 1))])
    return v / v.sum(axis=1, keepdims=True)


def _draw(u, pi, n_obs, rng):
    u = np.atleast_2d(np.asarray(u, dtype=float))
    pi = np.asarray(pi, dtype=float)
    X_all = design(u.shape[1] + 1)
    idx = rng.integers(0, X_all.shape[0], size=n_obs)
    comp = rng.choice(len(pi), size=n_obs, p=pi)
    return X_all[idx], comp, idx


# --------------------------------------------------------------------------
# the three protocols of Section 2
# --------------------------------------------------------------------------

def simulate_gaussian(u, pi, phi, n_obs, rng):
    """Log-ratio judgments ``y = x'u_c + N(0, phi_c)`` on the logit scale."""
    X, comp, idx = _draw(u, pi, n_obs, rng)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    u = np.atleast_2d(np.asarray(u, dtype=float))
    eta = np.einsum("ij,ij->i", X, u[comp])
    y = eta + rng.normal(0.0, np.sqrt(phi[comp]))
    return {"X": X, "y": y, "comp": comp, "eta": eta, "idx": idx}


def simulate_ratio(u, pi, phi, n_obs, rng, lo=1.0 / 9.0, hi=9.0):
    """Saaty ratio judgments ``r = exp(x'u_c + N(0, phi_c))``, clipped to 1/9..9.

    Returns both the ratio ``r`` and its logarithm ``y``; the paper's ratio
    protocol applies ``g(r) = ln r``, so ``y`` is the response the model sees.
    """
    d = simulate_gaussian(u, pi, phi, n_obs, rng)
    r = np.clip(np.exp(d["y"]), lo, hi)
    d["r"] = r
    d["y"] = np.log(r)
    return d


def simulate_binomial(u, pi, trials, n_obs, rng):
    """Repeated choices: ``m ~ Binomial(trials, expit(x'u_c))``, response ``m/trials``."""
    X, comp, idx = _draw(u, pi, n_obs, rng)
    u = np.atleast_2d(np.asarray(u, dtype=float))
    p = expit(np.einsum("ij,ij->i", X, u[comp]))
    m = rng.binomial(trials, p)
    return {"X": X, "y": m / trials, "m": m, "n": np.full(n_obs, float(trials)),
            "comp": comp, "p": p, "idx": idx}


def separated_pair(n_alt, delta, rng, base=None):
    """Two components whose log-ratio vectors differ by ``delta`` in norm."""
    if base is None:
        base = rng.normal(0.0, 0.6, n_alt - 1)
    direction = rng.normal(0.0, 1.0, n_alt - 1)
    direction /= max(np.linalg.norm(direction), 1e-12)
    return np.vstack([base, base + delta * direction])


# --------------------------------------------------------------------------
# the five conditions of Table 1
# --------------------------------------------------------------------------

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

    all_pairs = pairs(nA)
    n_signal = int(n_comparisons * (1 - noise_fraction))
    n_noise = n_comparisons - n_signal

    out = []
    pair_idx = rng.choice(len(all_pairs), size=n_signal, replace=True)
    eps = 1e-6
    for k in range(n_signal):
        i, j = all_pairs[pair_idx[k]]
        log_odds = np.log(w[i] / (w[j] + eps))
        noisy = log_odds + rng.normal(0, np.sqrt(sigma2))
        score = 1.0 / (1.0 + np.exp(-noisy))
        score = float(np.clip(score, 0.001, 0.999))
        out.append({"ft": i, "lt": j, "score": score})

    pair_idx = rng.choice(len(all_pairs), size=n_noise, replace=True)
    for k in range(n_noise):
        i, j = all_pairs[pair_idx[k]]
        score = float(rng.uniform(0, 1))
        out.append({"ft": i, "lt": j, "score": score})

    rng.shuffle(out)
    w_gt = w.reshape(1, -1)
    return out, w_gt


def generate_synthetic_pairs(u_logratios, pi, sigma2, n_comparisons, rng=None):
    """Draw comparisons from a known mixture.

    ``u_logratios`` is (nP, nA-1), ``pi`` the weights and ``sigma2`` the dispersions.
    Returns a list of ``{"ft", "lt", "score"}`` dicts with zero-based indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    nP, nA_minus_1 = u_logratios.shape
    nA = nA_minus_1 + 1
    w = priority_vector_from_logratios(u_logratios)

    comp_idx = rng.choice(nP, size=n_comparisons, p=pi)

    all_pairs = [(i, j) for i in range(nA) for j in range(i + 1, nA)]
    pair_idx = rng.choice(len(all_pairs), size=n_comparisons, replace=True)

    out = []
    eps = 1e-6
    for k in range(n_comparisons):
        p = comp_idx[k]
        i, j = all_pairs[pair_idx[k]]
        log_odds = np.log(w[p, i] / (w[p, j] + eps))
        noisy = log_odds + rng.normal(0, np.sqrt(sigma2[p]))
        score = 1.0 / (1.0 + np.exp(-noisy))
        score = float(np.clip(score, 0.001, 0.999))
        out.append({"ft": i, "lt": j, "score": score})

    return out, w, comp_idx
