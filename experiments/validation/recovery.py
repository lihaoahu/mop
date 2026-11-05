"""Table 1: recovery over 500 replicates per condition."""


import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.fit import fit_mop                                     # noqa: E402
from common.generate import (                                      # noqa: E402
    generate_cancellation,
    generate_noise_ground_truth,
    generate_noise_pairs,
    generate_random_mixture,
    generate_synthetic_pairs,
)
from common.metrics import (                                       # noqa: E402
    cosine_similarity,
    kl_divergence,
    kendall_tau,
    logratio_from_priority,
)
from common.parallel import run as run_parallel                     # noqa: E402
from common.paths import SIM, result                                # noqa: E402
from common.single import fit_single_k1, get_priorities_single_k1   # noqa: E402

CFG = SIM
SIM_RESULTS = result("recovery.json")

EPS = 1e-10
METHODS = ["MoP", "P=1"]


def evaluate_one(method, pairs, n_alternatives, ground_truth, s2=None):
    t0 = time.perf_counter()
    if method == "MoP":
        state = fit_mop(pairs, n_alternatives, s2=s2)
        w_est = state["priorities"]
        pi_est = state["pis"]
        w_avg_est = state["weighted_priority"]
        k_est = state["k"]
    else:
        state = fit_single_k1(pairs, n_alternatives, s2=s2)
        w_est = get_priorities_single_k1(state)
        pi_est = np.array([1.0])
        w_avg_est = w_est[0]
        k_est = 1
    t_elapsed = time.perf_counter() - t0
    u_true = np.asarray(ground_truth["u_logratios"], dtype=float)
    if method == "MoP":
        s2 = [float(x) for x in state["sigma2"]]
    else:
        s2 = [float(p.inconsistency_level) for p in state["mop_result"]]
    return {
        "k_est": k_est,
        "k_true": ground_truth.get("k_true", len(ground_truth["pi_true"])),
        "fit": {
            "k": int(k_est),
            "weights": [[float(x) for x in row] for row in w_est],
            "utilities": [[float(x) for x in row]
                          for row in logratio_from_priority(w_est)],
            "pis": [float(x) for x in pi_est],
            "sigma2": s2,
        },
        "cosine_sim": cosine_similarity(u_true, logratio_from_priority(w_est)),
        "kl_pi": kl_divergence(u_true, ground_truth["pi_true"],
                               logratio_from_priority(w_est), pi_est),
        "kendall_tau": kendall_tau(ground_truth["w_true_avg"], w_avg_est),
        "time_s": t_elapsed,
    }



def _run_trial(task):
    """One replicate.  The seed is fixed by the trial index, so the draw, and the
    result, do not depend on how many workers ran or in what order."""
    kind, trial, ex = task
    sigma2, n_comp = ex["sigma2"], ex["n_comp"]
    if kind == "cancellation":
        u, pi = generate_cancellation()
        nA = u.shape[1] + 1
        rng = np.random.default_rng(1000 + trial)
        pairs, w_true_mat, _ = generate_synthetic_pairs(
            u, pi, np.full(len(pi), sigma2), n_comp, rng)
        gt = {
            "u_logratios": u, "pi_true": pi, "k_true": len(pi),
            "w_true_avg": np.average(w_true_mat, axis=0, weights=pi),
        }
    elif kind == "general":
        K = ex["K"]
        nA = ex["n_alt"]
        rng = np.random.default_rng(3000 + trial + K * 1000)
        u, pi = generate_random_mixture(K, nA, rng)
        pairs, w_true_mat, _ = generate_synthetic_pairs(
            u, pi, np.full(K, sigma2), n_comp, rng)
        gt = {
            "u_logratios": u, "pi_true": pi, "k_true": K,
            "w_true_avg": np.average(w_true_mat, axis=0, weights=pi),
        }
    else:
        u, pi = generate_noise_ground_truth()
        nA = u.shape[1] + 1
        rng = np.random.default_rng(4000 + trial)
        pairs, w_true_mat = generate_noise_pairs(
            u, sigma2, n_comp, ex["noise_f"], rng)
        gt = {
            "u_logratios": u, "pi_true": pi, "k_true": 1,
            "w_true_avg": w_true_mat[0],
        }
    return {m: evaluate_one(m, pairs, nA, gt, s2=sigma2) for m in METHODS}


def _trial_results(kind, trials, jobs=None, **extra):
    base = {"sigma2": CFG["simulation"]["sigma2"],
            "n_alt": CFG["simulation"]["n_alternatives"],
            "n_comp": CFG["simulation"]["n_comparisons"]}
    base.update(extra)
    tasks = [(kind, t, base) for t in range(trials)]
    out = run_parallel(_run_trial, tasks, jobs)
    return {m: [row[m] for row in out] for m in METHODS}


def run_cancellation(trials, jobs=None):
    print("=" * 60)
    print("Cancellation — opposite preferences cancel")
    print("=" * 60)
    sigma2 = CFG["simulation"]["sigma2"]
    n_comp = CFG["simulation"]["n_comparisons"]
    key = f"sigma2={sigma2}_n={n_comp}"
    trial_results = _trial_results("cancellation", trials, jobs)
    agg = _aggregate(trial_results, replicate_1_seed=1000)
    agg.update(n_trials=trials, sigma2=sigma2, n_comparisons=n_comp)
    _print_agg(key, agg)
    return {key: agg}



def run_general(trials, jobs=None):
    print("\n" + "=" * 60)
    print("General recovery — random ground truth")
    print("=" * 60)
    K_vals = CFG["simulation"]["scenarios"]["general"]["K_values"]
    sigma2 = CFG["simulation"]["sigma2"]
    n_comp = CFG["simulation"]["n_comparisons"]
    all_results = {}
    for K in K_vals:
        key = f"K={K}_sigma2={sigma2}_n={n_comp}"
        trial_results = _trial_results("general", trials, jobs, K=K)
        agg = _aggregate(trial_results, replicate_1_seed=3000 + K * 1000)
        agg.update(n_trials=trials, sigma2=sigma2, n_comparisons=n_comp, K_true=K)
        all_results[key] = agg
        _print_agg(key, agg)
    return all_results



def run_noise(trials, jobs=None):
    print("\n" + "=" * 60)
    print("Noise — random raters, MoP isolates noise")
    print("=" * 60)
    sigma2 = CFG["simulation"]["sigma2"]
    noise_f = CFG["simulation"]["scenarios"]["noise"]["noise_fraction"]
    n_comp = CFG["simulation"]["n_comparisons"]

    key = f"sigma2={sigma2}_noise={noise_f}_n={n_comp}"
    trial_results = _trial_results("noise", trials, jobs, noise_f=noise_f)
    mop_k_vals = [r["k_est"] for r in trial_results["MoP"]]
    agg = _aggregate(trial_results, replicate_1_seed=4000)
    for m in METHODS:
        agg[m].pop("k_accuracy", None)
    agg.update(n_trials=trials, sigma2=sigma2, noise_fraction=noise_f, n_comparisons=n_comp)
    agg["mop_k_mean"] = float(np.mean(mop_k_vals))
    agg["mop_k_gt1_pct"] = float(np.mean([1.0 if k > 1 else 0.0 for k in mop_k_vals]))
    _print_agg(key, agg)
    print(f"    MoP K>1 in {agg['mop_k_gt1_pct']:.0%} of trials")
    return {key: agg}



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
            kl = f"{m['kl_pi']['mean']:>9.4f}" if "kl_pi" in m else "      ---"
            print(f"  {method:<14} {m['cosine_sim']['mean']:>9.4f} {kl} {m['kendall_tau']['mean']:>9.4f} {ka} {m.get('k_mean',0):>7.2f} {m['time_s']['mean']:>8.4f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["cancellation", "general", "noise"])
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None,
                        help="worker processes; the result is identical for any value")
    parser.add_argument("--out", default=None,
                        help="write results here instead of the default path; used to run "
                             "the scenarios in parallel")
    args = parser.parse_args()
    results_path = Path(args.out) if args.out else SIM_RESULTS

    n_trials = args.trials or CFG["simulation"]["n_trials"]
    print(f"Simulation — {n_trials} trials | MoP vs P=1\n")

    all_data = {}
    if results_path.exists():
        try:
            all_data = json.load(open(results_path, encoding="utf-8"))
        except Exception:
            pass

    runners = {"cancellation": run_cancellation, "general": run_general, "noise": run_noise}
    for key, runner in runners.items():
        if args.scenario is None or args.scenario == key:
            all_data[key] = runner(n_trials, jobs=args.jobs)

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
