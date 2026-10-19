"""Step 1: Simulation — MoP vs P=1 on three targeted scenarios.

  cancellation — opposite preferences cancel, K1 sees indifference
  general     — random ground truth, MoP never worse
  noise       — random raters, MoP isolates noise component

Usage:
    uv run python experiments/simulation/step1_simulation.py
    uv run python experiments/simulation/step1_simulation.py --scenario noise
    uv run python experiments/simulation/step1_simulation.py --trials 500
"""

import argparse, json, time
import sys
from pathlib import Path
import numpy as np
from scipy.special import logit

from mop import inference
from mop.pairwise_comparison import PairwiseComparison, comparisons2regression
from mop.initialization import kridge

from _common import (
    CFG, SIM_RESULTS,
    priority_vector_from_logratios,
    cosine_similarity, kl_divergence, kendall_tau,
    generate_synthetic_pairs,
    generate_cancellation,
    generate_noise_ground_truth, generate_noise_pairs,
    generate_random_mixture,
)
from _baselines import fit_single_k1, get_priorities_single_k1

# Fitting parameters come from experiments/mop_params.json, the single source of
# truth. This script used to read max_K from simulation/config.json and hard-code the
# three lambdas and the weight floor.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mop_params import (max_components_for, lambda_u, lambda_pi, lambda_sigma_for,   # noqa: E402
                         min_weight_share, max_components)

EPS = 1e-10
METHODS = ["MoP", "P=1"]


# ═══════════════════════════════════════════════════════════════════
# MoP fit
# ═══════════════════════════════════════════════════════════════════

def fit_mop(pairs, n_alternatives, max_K=None, s2=None):
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
    # s2 is the prior mode on φ: None for the data-based value, a number to pin it.

    best_k, best_bic, best_result = None, np.inf, None

    for k in range(1, max_K + 1):
        # K-Ridge from scratch at every candidate count; deterministic, no warm chain.
        u0, phi0, mp0 = kridge(X, y, k)
        result = inference.mop(pcv, link_fns, k, lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls,
                               s2=s2, u_init=u0, sigma2_init=phi0, mp_init=mp0)
        bic_val = inference.pbic(pcv, result, link_fns, lambda_u=lu, lambda_sigma=ls,
                                 lambda_pi=lpi, s2=s2)
        if bic_val < best_bic:
            best_bic = bic_val; best_k = k; best_result = result

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



# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_one(method, pairs, n_alternatives, ground_truth, s2=None):
    t0 = time.perf_counter()
    if method == "MoP":
        state = fit_mop(pairs, n_alternatives, s2=s2)
        w_est = state["priorities"]
        pi_est = state["pis"]
        w_avg_est = state["weighted_priority"]
        k_est = state["k"]
    else:  # P=1
        state = fit_single_k1(pairs, n_alternatives, s2=s2)
        w_est = get_priorities_single_k1(state)
        pi_est = np.array([1.0])
        w_avg_est = w_est[0]
        k_est = 1
    t_elapsed = time.perf_counter() - t0
    w_true = priority_vector_from_logratios(ground_truth["u_logratios"])
    if method == "MoP":
        s2 = [float(x) for x in state["sigma2"]]
    else:
        s2 = [float(p.inconsistency_level) for p in state["mop_result"]]
    return {
        "k_est": k_est,
        "k_true": ground_truth.get("k_true", len(ground_truth["pi_true"])),
        # the fitted state of this replicate, so the illustrative row of the
        # paper's recovery table can be regenerated rather than transcribed
        "fit": {
            "k": int(k_est),
            "weights": [[float(x) for x in row] for row in w_est],
            "pis": [float(x) for x in pi_est],
            "sigma2": s2,
        },
        "cosine_sim": cosine_similarity(w_true, w_est),
        # NaN where the comparison is undefined, i.e. where the two models have
        # different numbers of components and one of them has no mixture weights
        # to compare. The aggregate drops NaN, so the key is simply absent there.
        "kl_pi": kl_divergence(w_true, ground_truth["pi_true"], w_est, pi_est),
        "kendall_tau": kendall_tau(ground_truth["w_true_avg"], w_avg_est),
        "time_s": t_elapsed,
    }


# ═══════════════════════════════════════════════════════════════════
# Scenario: Cancellation
# ═══════════════════════════════════════════════════════════════════

def run_cancellation(trials):
    print("=" * 60)
    print("Cancellation — opposite preferences cancel")
    print("=" * 60)
    u, pi = generate_cancellation()
    nA = u.shape[1] + 1
    sigma2 = CFG["simulation"]["sigma2"]
    n_comp = CFG["simulation"]["n_comparisons"]
    key = f"sigma2={sigma2}_n={n_comp}"
    trial_results = {m: [] for m in METHODS}
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        pairs, w_true_mat, _ = generate_synthetic_pairs(
            u, pi, np.full(len(pi), sigma2), n_comp, rng)
        gt = {
            "u_logratios": u, "pi_true": pi,
            "k_true": len(pi),
            "w_true_avg": np.average(w_true_mat, axis=0, weights=pi),
        }
        for method in METHODS:
            trial_results[method].append(evaluate_one(method, pairs, nA, gt, s2=sigma2))
    agg = _aggregate(trial_results, replicate_1_seed=1000)
    agg.update(n_trials=trials, sigma2=sigma2, n_comparisons=n_comp)
    _print_agg(key, agg)
    return {key: agg}


# ═══════════════════════════════════════════════════════════════════
# Scenario: General recovery
# ═══════════════════════════════════════════════════════════════════

def run_general(trials):
    print("\n" + "=" * 60)
    print("General recovery — random ground truth")
    print("=" * 60)
    K_vals = CFG["simulation"]["scenarios"]["general"]["K_values"]
    sigma2 = CFG["simulation"]["sigma2"]
    n_comp = CFG["simulation"]["n_comparisons"]
    nA = 4
    all_results = {}
    for K in K_vals:
        key = f"K={K}_sigma2={sigma2}_n={n_comp}"
        trial_results = {m: [] for m in METHODS}
        for trial in range(trials):
            rng = np.random.default_rng(3000 + trial + K * 1000)
            u, pi = generate_random_mixture(K, nA, rng)
            pairs, w_true_mat, _ = generate_synthetic_pairs(
                u, pi, np.full(K, sigma2), n_comp, rng)
            gt = {
                "u_logratios": u, "pi_true": pi, "k_true": K,
                "w_true_avg": np.average(w_true_mat, axis=0, weights=pi),
            }
            for method in METHODS:
                trial_results[method].append(evaluate_one(method, pairs, nA, gt, s2=sigma2))
        agg = _aggregate(trial_results, replicate_1_seed=3000 + K * 1000)
        agg.update(n_trials=trials, sigma2=sigma2, n_comparisons=n_comp, K_true=K)
        all_results[key] = agg
        _print_agg(key, agg)
    return all_results


# ═══════════════════════════════════════════════════════════════════
# Scenario: Noise
# ═══════════════════════════════════════════════════════════════════

def run_noise(trials):
    print("\n" + "=" * 60)
    print("Noise — random raters, MoP isolates noise")
    print("=" * 60)
    u_true, pi_true = generate_noise_ground_truth()
    nA = u_true.shape[1] + 1
    sigma2 = CFG["simulation"]["sigma2"]
    noise_f = CFG["simulation"]["scenarios"]["noise"]["noise_fraction"]
    n_comp = CFG["simulation"]["n_comparisons"]

    key = f"sigma2={sigma2}_noise={noise_f}_n={n_comp}"
    trial_results = {m: [] for m in METHODS}
    mop_k_vals = []
    for trial in range(trials):
        rng = np.random.default_rng(4000 + trial)
        pairs, w_true_mat = generate_noise_pairs(
            u_true, sigma2, n_comp, noise_f, rng)
        gt = {
            "u_logratios": u_true, "pi_true": pi_true, "k_true": 1,
            "w_true_avg": w_true_mat[0],
        }
        for method in METHODS:
            result = evaluate_one(method, pairs, nA, gt, s2=sigma2)
            trial_results[method].append(result)
            if method == "MoP":
                mop_k_vals.append(result["k_est"])
    agg = _aggregate(trial_results, replicate_1_seed=4000)
    for m in METHODS:
        agg[m].pop("k_accuracy", None)
    agg.update(n_trials=trials, sigma2=sigma2, noise_fraction=noise_f, n_comparisons=n_comp)
    agg["mop_k_mean"] = float(np.mean(mop_k_vals))
    agg["mop_k_gt1_pct"] = float(np.mean([1.0 if k > 1 else 0.0 for k in mop_k_vals]))
    _print_agg(key, agg)
    print(f"    MoP K>1 in {agg['mop_k_gt1_pct']:.0%} of trials")
    return {key: agg}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _aggregate(trial_results, replicate_1_seed=None):
    metrics = ["cosine_sim", "kl_pi", "kendall_tau", "time_s"]
    agg = {}
    for method in METHODS:
        agg[method] = {}
        for metric in metrics:
            vals = [tr[metric] for tr in trial_results[method] if metric in tr and not np.isnan(tr[metric])]
            if vals:
                agg[method][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        k_vals = [tr["k_est"] for tr in trial_results[method]]
        k_true = trial_results[method][0]["k_true"]
        agg[method]["k_accuracy"] = float(np.mean([1.0 if k == k_true else 0.0 for k in k_vals]))
        agg[method]["k_mean"] = float(np.mean(k_vals))
    # the first replicate of each method, the row the paper's recovery table
    # illustrates; kept so the table can be regenerated rather than transcribed
    agg["replicate_1"] = {m: trial_results[m][0]["fit"] for m in METHODS}
    agg["replicate_1_seed"] = replicate_1_seed
    return agg


def _print_agg(key, agg):
    print(f"\n  [{key}]")
    hdr = f"  {'Method':<14} {'CosSim↑':>9} {'KL(pi)↓':>9} {'Kendall↑':>9} {'K_acc':>7} {'k_mean':>7} {'Time(s)':>8}"
    print("  " + "-" * len(hdr))
    for method in METHODS:
        if method in agg and "cosine_sim" in agg[method]:
            m = agg[method]
            ka = f"{m['k_accuracy']:>7.3f}" if "k_accuracy" in m else "    ---"
            # absent where the comparison is undefined, e.g. the mixture weights
            # of a model that has no mixture
            kl = f"{m['kl_pi']['mean']:>9.4f}" if "kl_pi" in m else "      ---"
            print(f"  {method:<14} {m['cosine_sim']['mean']:>9.4f} {kl} {m['kendall_tau']['mean']:>9.4f} {ka} {m.get('k_mean',0):>7.2f} {m['time_s']['mean']:>8.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["cancellation", "general", "noise"])
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="write results here instead of the default path; used to run "
                             "the scenarios in parallel")
    args = parser.parse_args()
    results_path = Path(args.out) if args.out else SIM_RESULTS

    n_trials = args.trials or CFG["simulation"]["n_trials"]
    print(f"Simulation — {n_trials} trials | MoP vs P=1\n")

    # Load existing results to preserve scenarios not being re-run
    all_data = {}
    if results_path.exists():
        try:
            all_data = json.load(open(results_path, encoding="utf-8"))
        except Exception:
            pass

    runners = {"cancellation": run_cancellation, "general": run_general, "noise": run_noise}
    for key, runner in runners.items():
        if args.scenario is None or args.scenario == key:
            all_data[key] = runner(n_trials)

    # Wall-clock is printed above but not stored: it differs on every run, and
    # storing it made this the one result file a re-run could not reproduce.
    # Everything else here is deterministic and comes back byte for byte.
    def without_timings(obj):
        if isinstance(obj, dict):
            return {k: without_timings(v) for k, v in obj.items() if k != "time_s"}
        if isinstance(obj, list):
            return [without_timings(v) for v in obj]
        return obj

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(without_timings(all_data), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
