"""Step 6: the binomial protocol -- mixture of preferences versus one inconsistent preference.

Why this experiment exists
--------------------------
The manuscript claims that the two sources of variability occupy separate parameters, but that the
separation is only recoverable where the likelihood leaves the dispersion free. On binary and
repeated choices it does not: the variance is fixed by the response scale, a mixture of binomials
is itself overdispersed, and "several preferences executed consistently" and "one preference
executed inconsistently" therefore imply the same distribution of judgments.

Until now that claim rested on the analytic second derivative in phi and on the Gaussian-side
recovery study, and the third contribution -- one framework across three protocols -- had no
empirical support outside the Gaussian variants. This experiment supplies it.

  1. Recovery. Repeated binomial choices generated from a two-state mixture. A binomial mixture is
     fitted for P = 1, 2, 3 and selected by pBIC: does the mixture recover the states and the
     weights from binary data at all, given that phi is fixed?

  2. Non-identification. The same data fitted by a beta-binomial single-preference model, which is
     the competing explanation: one preference, executed with inconsistent precision. If the two
     models predict held-out judgments equally well, the protocol cannot separate them, and the
     manuscript's claim is measured rather than asserted.

Why the fitter is local
-----------------------
``mop.winloss.fit_winloss`` rescales ``omega`` to mean one, documenting the fit as invariant to
that scale. For a single component the invariance holds, but inside the EM it does not: the
responsibilities are a function of the likelihood, and rescaling ``omega`` from 30 to 1 softens
them from (0, 1) to (0.28, 0.72) on a clear judgment, so a repeated-comparison fit collapses to a
single state. The rescaling is harmless for binary data, where ``omega`` is one already. Rather
than change a library the rest of the paper depends on, the binomial mixture is fitted here with
its own EM, on the same model, priors and criterion that ``winloss.pbic`` implements.

Usage:
    uv run python experiments/simulation/step6_binomial.py
    uv run python experiments/simulation/step6_binomial.py --reps 5      # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln, logsumexp

from mop.pairwise_comparison import PairwiseComparison

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mop_params import lambda_pi, lambda_u, max_components          # noqa: E402
from _common import cosine_similarity, priority_vector_from_logratios  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "binomial_results.json"
A = 4                    # alternatives
N_TRIALS = 30            # trials per repeated comparison
N_COMPARISONS = 240      # judgments per replicate
TEST_FRACTION = 0.30

# Two preference states with opposite orderings, the binomial analogue of the cancellation
# scenario: the aggregate ranks the alternatives in the middle and looks like indifference.
# The separation is set so the two states are distinguishable at N_TRIALS = 30: the strongest
# pair separates at p = 0.82 against 0.18, about six binomial standard errors.
U_TRUE = np.array([
    [1.50, 0.75, -0.75],     # state 1: a > b > c > d
    [-1.50, -0.75, 0.75],    # state 2: d > c > b > a
])
PI_TRUE = np.array([0.5, 0.5])
RHO_BETA_BINOM = 0.30     # intra-class correlation of the reverse condition


def all_pairs(n_alt):
    return [(i, j) for i in range(n_alt) for j in range(i + 1, n_alt)]


def design_row(i, j):
    """The row comparisons2regression builds for the pair (i, j), 0-indexed."""
    row = np.zeros(A - 1)
    if i < A - 1:
        row[i] += 1.0
    if j < A - 1:
        row[j] -= 1.0
    return row


def generate(pairs, rng, condition="mixture"):
    """Either a two-state mixture, or one preference executed with inconsistent precision.

    The second condition is the competing explanation and is what makes the comparison a test
    rather than an illustration: if the beta-binomial wins on data it generated, the two models
    are distinguishable, and the mixture's count on such data is an upper bound.
    """
    pc, rows, judgments = [], [], []
    per_pair = N_COMPARISONS // len(pairs)
    u_single = np.array([1.50, 0.75, -0.75])
    conc = 1.0 / RHO_BETA_BINOM - 1.0
    for pair in pairs:
        row = design_row(*pair)
        for _ in range(per_pair):
            if condition == "mixture":
                state = int(rng.choice(len(PI_TRUE), p=PI_TRUE))
                p = expit(float(row @ U_TRUE[state]))
            elif condition == "betabinom":
                p_mean = expit(float(row @ u_single))
                a, b = p_mean * conc, (1.0 - p_mean) * conc
                p = float(rng.beta(a, b))
            else:
                raise ValueError(condition)
            m = int(rng.binomial(N_TRIALS, p))
            pc.append(PairwiseComparison(former_id=pair[0] + 1, latter_id=pair[1] + 1,
                                         agent_id=1, judgment=m / N_TRIALS))
            rows.append(row)
            judgments.append(m / N_TRIALS)
    return pc, np.array(rows), np.array(judgments)


# ── the binomial mixture, fitted here ────────────────────────────────────────────────────────────

def _eta(X, u):
    return np.clip(X @ u, -30.0, 30.0)


def _loglik_rows(X, y, n, u):
    """Per-judgment binomial log-density, log C(n, m) included so cross-model comparison is valid."""
    eta = _eta(X, u)
    m = y * n
    return (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
            + n * (y * eta - np.logaddexp(0.0, eta)))


def _penalized(X, y, n, u, pi, lam_u, lam_pi):
    """Penalized log-posterior: likelihood + Normal prior on u + Dirichlet prior on pi."""
    ll = logsumexp(np.vstack([_loglik_rows(X, y, n, u[p]) for p in range(len(pi))])
                   + np.log(np.maximum(pi, 1e-300))[:, None], axis=0).sum()
    prior_u = -0.5 * lam_u * float(np.sum(u ** 2))
    prior_pi = float((lam_pi - 1.0) * np.log(np.maximum(pi, 1e-300)).sum())
    return float(ll + prior_u + prior_pi)


def _mstep(X, y, n, resp, lam_u, u_init):
    """Weighted, ridge-penalized logistic regression for one component.

    Maximises sum_k resp_k * n_k * (y_k eta - log(1+e^eta)) - lambda_u/2 ||u||^2, whose gradient
    is X^T [resp * n * (y - sigmoid(eta))] - lambda_u u.
    """
    def neg_obj(u):
        eta = _eta(X, u)
        ll = float(np.sum(resp * n * (y * eta - np.logaddexp(0.0, eta))))
        return -ll + 0.5 * lam_u * float(u @ u)

    def grad(u):
        return -(X.T @ (resp * n * (y - expit(_eta(X, u))))) + lam_u * u

    res = minimize(neg_obj, x0=u_init, jac=grad, method="L-BFGS-B",
                   bounds=[(-10, 10)] * X.shape[1])
    return res.x


def _pi_update(resp, n, lam_pi):
    num = resp @ n + lam_pi - 1.0
    num = np.maximum(num, 1e-10)
    return num / num.sum()


def fit_binomial_mixture(X, y, n, nP, u_init, mp_init,
                         lam_u=lambda_u, lam_pi=lambda_pi, max_iter=500, tol=1e-9):
    u = np.array(u_init, dtype=float).copy()
    pi = np.array(mp_init, dtype=float).copy()
    pi = pi / pi.sum()
    prev = None
    for _ in range(max_iter):
        logs = np.vstack([_loglik_rows(X, y, n, u[p]) for p in range(nP)])
        logresp = logs + np.log(np.maximum(pi, 1e-300))[:, None]
        logresp -= logsumexp(logresp, axis=0, keepdims=True)
        resp = np.exp(logresp)
        u = np.array([_mstep(X, y, n, resp[p], lam_u, u[p]) for p in range(nP)])
        pi = _pi_update(resp, n, lam_pi)
        pll = _penalized(X, y, n, u, pi, lam_u, lam_pi)
        if prev is not None and abs(pll - prev) / (abs(pll) + 0.1) < tol:
            break
        prev = pll
    return {"u": u, "pi": pi, "pll": _penalized(X, y, n, u, pi, lam_u, lam_pi)}


def pbic_binomial(model, n, lam_u=lambda_u, lam_pi=lambda_pi):
    """-2 * penalized log-posterior + log(n_eff) * df, on winloss.pbic's scale.

    ``df = (A-1)P + 2P - 1``: every utility, the P-1 free mixture weights, and the P dispersion
    parameters, which are fixed at one here but still belong to the model and are still counted.
    """
    u, pi = model["u"], model["pi"]
    nP = len(pi)
    n_eff = float(np.sum(n))
    df = (A - 1) * nP + 2 * nP - 1
    return -2.0 * model["pll"] + np.log(n_eff) * df


def seed_components(X, y, n, nP):
    """Pooled fit, then the two residual tails as candidates for opposing states."""
    def fit_logistic(w):
        def neg(u):
            eta = _eta(X, u)
            return -float(np.sum(w * n * (y * eta - np.logaddexp(0.0, eta)))) + 0.5 * lambda_u * float(u @ u)

        def grad(u):
            return -(X.T @ (w * n * (y - expit(_eta(X, u))))) + lambda_u * u

        return minimize(neg, x0=np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x

    u0 = fit_logistic(np.ones(X.shape[0]))
    resid = y - expit(_eta(X, u0))
    tails = [u0]
    for mask in (resid > 0, resid < 0):
        w = mask.astype(float)
        tails.append(fit_logistic(w) if w.sum() > X.shape[1] + 2 else u0 + 0.1)
    u_init = np.vstack((tails * ((nP // len(tails)) + 1))[:nP])
    return u_init, np.full(nP, 1.0 / nP)


def fit_best(X, y, n, nP):
    """Several restarts, keep the best penalized log-posterior.

    Reporting the worse of two reachable fits would measure the initialiser rather than the model.
    """
    starts = [seed_components(X, y, n, nP)]
    rng = np.random.default_rng(1234 + nP)
    for _ in range(3):
        u_init, mp = seed_components(X, y, n, nP)
        starts.append((u_init + rng.normal(0, 0.5, u_init.shape), mp))
    fits = [fit_binomial_mixture(X, y, n, nP, ui, mp) for ui, mp in starts]
    return max(fits, key=lambda m: m["pll"])


# ── the beta-binomial single-preference baseline ────────────────────────────────────────────────

def _beta_binom_ll(X, y, n, u, rho):
    p = np.clip(expit(_eta(X, u)), 1e-9, 1 - 1e-9)
    conc = 1.0 / rho - 1.0
    a, b = p * conc, (1.0 - p) * conc
    m = y * n
    return (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
            + betaln(m + a, n - m + b) - betaln(a, b))


def fit_beta_binomial(X, y, n, u_init):
    def neg(params):
        u, rho_logit = params[:A - 1], params[-1]
        rho = expit(rho_logit) * 0.98 + 1e-4
        return -float(_beta_binom_ll(X, y, n, u, rho).sum()) + 0.5 * lambda_u * float(u @ u)

    res = minimize(neg, x0=np.concatenate([u_init, [-4.0]]), method="L-BFGS-B",
                   bounds=[(-10, 10)] * (A - 1) + [(-9, 4)])
    u, rho = res.x[:A - 1], expit(res.x[-1]) * 0.98 + 1e-4
    return u, rho


def beta_binomial_predictive_ll(X, y, n, u, rho):
    return float(_beta_binom_ll(X, y, n, u, rho).sum())


def mixture_predictive_ll(X, y, n, model):
    u, pi = model["u"], model["pi"]
    logs = np.vstack([_loglik_rows(X, y, n, u[p]) for p in range(len(pi))])
    return float(logsumexp(logs + np.log(np.maximum(pi, 1e-300))[:, None], axis=0).sum())


# ── one replicate ───────────────────────────────────────────────────────────────────────────────

def one_replicate(rng, condition="mixture"):
    pairs = all_pairs(A)
    pc, X, y = generate(pairs, rng, condition)
    n = np.full(len(y), float(N_TRIALS))
    n_eval = int(round(TEST_FRACTION * len(y)))
    idx = rng.permutation(len(y))
    te, tr = idx[:n_eval], idx[n_eval:]

    models = {}
    for nP in range(1, min(max_components, A) + 1):
        models[nP] = fit_best(X[tr], y[tr], n[tr], nP)
    criteria = {nP: pbic_binomial(models[nP], n[tr]) for nP in models}
    p_hat = min(criteria, key=criteria.get)

    cos = None
    if p_hat > 1:
        est = priority_vector_from_logratios(models[p_hat]["u"])
        truth = priority_vector_from_logratios(U_TRUE)
        cos = cosine_similarity(truth, est)

    ll_mop = mixture_predictive_ll(X[te], y[te], n[te], models[p_hat])
    ll_single = mixture_predictive_ll(X[te], y[te], n[te], models[1])

    u0 = models[1]["u"][0]
    u_bb, rho_bb = fit_beta_binomial(X[tr], y[tr], n[tr], u0)
    ll_bb = beta_binomial_predictive_ll(X[te], y[te], n[te], u_bb, rho_bb)

    return {
        "p_hat": int(p_hat),
        "pbic": {str(k): float(v) for k, v in criteria.items()},
        "cosine": cos,
        "ll_mop": ll_mop,
        "ll_single": ll_single,
        "ll_beta_binomial": ll_bb,
        "rho_beta_binomial": float(rho_bb),
        "n_test": int(n_eval),
    }


def summarise(rows, reps):
    p_hats = [x["p_hat"] for x in rows]
    cos = [x["cosine"] for x in rows if x["cosine"] is not None]
    d_ll = [x["ll_mop"] - x["ll_beta_binomial"] for x in rows]
    d_single = [x["ll_single"] - x["ll_beta_binomial"] for x in rows]
    n_test = rows[0]["n_test"]
    return {
        "reps": reps,
        "p_hat_counts": {str(k): int(p_hats.count(k)) for k in sorted(set(p_hats))},
        "shared_state_selected_share": float(np.mean([p == 2 for p in p_hats])),
        "cosine_mean": float(np.mean(cos)) if cos else None,
        "cosine_min": float(np.min(cos)) if cos else None,
        "delta_ll_mixture_vs_betabinom": {
            "mean": float(np.mean(d_ll)),
            "median": float(np.median(d_ll)),
            "share_mixture_better": float(np.mean([d > 0 for d in d_ll])),
            "mean_per_test_judgment": float(np.mean(d_ll) / n_test),
        },
        "delta_ll_single_vs_betabinom": {
            "mean": float(np.mean(d_single)),
            "mean_per_test_judgment": float(np.mean(d_single) / n_test),
        },
        "per_replicate": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260119)
    ap.add_argument("--condition", choices=["both", "mixture", "betabinom"], default="both")
    args = ap.parse_args()

    conditions = ["mixture", "betabinom"] if args.condition == "both" else [args.condition]
    out = {
        "design": {"A": A, "trials_per_comparison": N_TRIALS,
                   "comparisons_per_replicate": N_COMPARISONS,
                   "test_fraction": TEST_FRACTION,
                   "rho_beta_binom_generator": RHO_BETA_BINOM},
    }
    for cond in conditions:
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        rows = []
        for r in range(args.reps):
            rows.append(one_replicate(rng, cond))
            if (r + 1) % 10 == 0:
                print(f"  {cond} {r + 1}/{args.reps}  {time.time() - t0:.1f}s", flush=True)
        out[cond] = summarise(rows, args.reps)

    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    short = {k: v for k, v in out.items() if k != "design"}
    for cond, s in short.items():
        s = {k: v for k, v in s.items() if k != "per_replicate"}
        print(f"\n== data generated by: {cond}")
        print(json.dumps(s, indent=2))
    print(f"\nwritten: {RESULTS}")


if __name__ == "__main__":
    main()
