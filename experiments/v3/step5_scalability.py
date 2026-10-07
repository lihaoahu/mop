"""Scalability: a distribution over orders versus a parametric mixture.

The behavioural-social-choice framework places a free probability on each
transitive order, so its parameter space is the simplex over A! orders. The
mixture of preferences placed on the same footing uses P utility vectors of
dimension A-1 each. Both are fitted here to the *same* pairwise comparison
data, so the comparison is like for like:

  orders mixture          P(o) >= 0, sum_o P(o) = 1 over the A! linear orders,
                          fitted by EM on the marginal pairwise likelihood,
                          so that sum_{o : i > j} P(o) = P(i > j);
  mixture of preferences  P components, each with a utility vector, fitted by
                          the same EM as the rest of the paper.

Compared are the things that decide feasibility: the number of free
parameters, the wall-clock cost, and how each extrapolates as A grows. That is
not a claim about which model is more correct -- they encode different
assumptions -- but about which one can be estimated at a given A.

Usage:
    uv run python experiments/v3/step5_scalability.py
"""

import argparse
import json
import math
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mop import inference_binomial as ib  # noqa: E402

HERE = Path(__file__).resolve().parent

N_TRIALS = 45
P_TRUE = 3
P_MOP = 4


def order_mask(n_alt):
    """mask[k, o] = 1 when order o ranks the former alternative of pair k higher."""
    orders = np.array(list(permutations(range(n_alt))))          # (n_orders, A)
    rank = np.argsort(np.argsort(orders, axis=1), axis=1)        # rank of each alternative
    pairs = [(i, j) for i in range(n_alt) for j in range(i + 1, n_alt)]
    mask = np.zeros((len(pairs), len(orders)), dtype=bool)
    for k, (i, j) in enumerate(pairs):
        mask[k] = rank[:, i] < rank[:, j]
    return pairs, mask


def simulate(n_alt, n_trials, rng):
    """Draw counts from a P_TRUE-component mixture of preferences."""
    pairs = [(i, j) for i in range(n_alt) for j in range(i + 1, n_alt)]
    u_true = rng.normal(0.0, 1.0, (P_TRUE, n_alt - 1))
    u_full = np.hstack([u_true, np.zeros((P_TRUE, 1))])
    pi_true = rng.dirichlet(np.full(P_TRUE, 2.0))
    counts = []
    for i, j in pairs:
        c = int(rng.choice(P_TRUE, p=pi_true))
        mu = float(expit(u_full[c, i] - u_full[c, j]))
        counts.append({"former": i, "latter": j,
                       "successes": float(rng.binomial(n_trials, mu)),
                       "trials": n_trials})
    return counts


def fit_orders_mixture(counts, mask, alpha=1.0, max_iter=3000, tol=1e-13):
    """EM for the marginal pairwise likelihood of the orders mixture.

    Complete-data augmentation: each of the t_k binary judgments in pair k is
    assigned to one of the orders consistent with the side it favoured. Both
    branches contribute to the expected counts, which is what the M-step needs.
    """
    n_orders = mask.shape[1]
    succ = np.array([c["successes"] for c in counts])
    tri = np.array([c["trials"] for c in counts])
    p = np.full(n_orders, 1.0 / n_orders)
    not_mask = ~mask

    for _ in range(max_iter):
        marg = np.clip(mask @ p, 1e-15, 1 - 1e-15)
        w_win = succ / marg
        w_lose = (tri - succ) / (1.0 - marg)
        n_o = p * (mask.T @ w_win + not_mask.T @ w_lose)
        p_new = np.maximum(n_o + alpha - 1.0, 1e-15)
        p_new /= p_new.sum()
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    return p


def marginal_loglik(counts, probs):
    succ = np.array([c["successes"] for c in counts])
    tri = np.array([c["trials"] for c in counts])
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return float(np.sum(succ * np.log(probs) + (tri - succ) * np.log(1.0 - probs)))


def mop_marginal_probs(model, counts):
    """sum_p pi_p * logistic(log(w_pi / w_pj)) for every pair."""
    w = model.priority_vectors()
    out = np.zeros(len(counts))
    for k, c in enumerate(counts):
        i, j = c["former"], c["latter"]
        for p in range(model.n_components):
            log_odds = np.log((w[p, i] + 1e-12) / (w[p, j] + 1e-12))
            out[k] += model.pi[p] * expit(log_odds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260303)
    ap.add_argument("--a-min", type=int, default=4)
    ap.add_argument("--a-max", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "scalability_results.json")
    args = ap.parse_args()

    rows = []
    header = (f"{'A':>3} {'A!':>10} {'orders par':>11} {'MoP par':>8} "
              f"{'t MoP s':>9} {'t orders s':>11} {'P_hat':>6} {'LL MoP':>9} {'LL orders':>10}")
    print(header)
    print("-" * len(header))

    for n_alt in range(args.a_min, args.a_max + 1):
        rng = np.random.default_rng(args.seed + 97 * n_alt)
        counts = simulate(n_alt, N_TRIALS, rng)
        n_orders = math.factorial(n_alt)

        t0 = time.perf_counter()
        best_p, table = ib.select_components(counts, n_alt, p_max=min(P_MOP, n_alt),
                                             phi_mode="binomial", n_starts=1)
        t_mop = time.perf_counter() - t0
        model = table[best_p]["model"]

        _, mask = order_mask(n_alt)
        t0 = time.perf_counter()
        p_orders = fit_orders_mixture(counts, mask)
        t_orders = time.perf_counter() - t0

        ll_mop = marginal_loglik(counts, mop_marginal_probs(model, counts))
        ll_orders = marginal_loglik(counts, mask @ p_orders)

        row = {
            "n_alternatives": n_alt,
            "n_orders": n_orders,
            "orders_parameters": n_orders - 1,
            "mop_parameters": best_p * (n_alt - 1) + (best_p - 1),
            "mop_selected_P": int(best_p),
            "t_mop_s": t_mop,
            "t_orders_s": t_orders,
            "ll_mop": ll_mop,
            "ll_orders": ll_orders,
        }
        rows.append(row)
        print(f"{n_alt:>3} {n_orders:>10,} {n_orders-1:>11,} {row['mop_parameters']:>8} "
              f"{t_mop:>9.3f} {t_orders:>11.3f} {best_p:>6} {ll_mop:>9.1f} {ll_orders:>10.1f}")

    print()
    print("=" * 78)
    print("beyond the fitted range")
    print("=" * 78)
    print(f"{'A':>3} {'A!':>16} {'orders par':>18} {'MoP par (P=4)':>15}")
    for n_alt in [9, 10, 12, 15]:
        n_orders = math.factorial(n_alt)
        print(f"{n_alt:>3} {n_orders:>16,} {n_orders-1:>18,} "
              f"{P_MOP*(n_alt-1) + P_MOP - 1:>15,}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"n_trials": N_TRIALS, "p_true": P_TRUE, "rows": rows}, fh, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
