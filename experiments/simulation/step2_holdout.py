"""Step 2: Holdout prediction — per-dimension CV, best-match for MoP.

MoP predicts K scores (one per component); error is computed against the
component whose prediction is closest to the true score. P=1 predicts a
single score.

Usage:
    uv run python experiments/simulation/step2_holdout.py
    uv run python experiments/simulation/step2_holdout.py --min-pairs 200
"""

import argparse, json, time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold

from _common import CFG, V2_DIR, HOLDOUT_RESULTS, ENERGY_PAIRS, CANONICAL_TYPES, TYPE_MAP
from _baselines import fit_single_k1, predict_single_k1
from step1_simulation import fit_mop

EPS = 1e-10
TYPE_MAP_FULL = {**TYPE_MAP, "燃油": "燃油", "纯电": "纯电", "插混": "插混", "增程": "增程"}
T = CANONICAL_TYPES
METHODS = ["MoP", "P=1"]

DIM_LABELS = {
    "cost": "Cost",
    "driving_experience": "Driving",
    "range": "Range",
    "refueling_convenience": "Refueling",
    "reliability": "Reliability",
    "policy": "Policy",
    "other": "Other",
}


def load_real_pairs():
    pairs = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for p in r.get("comparison_pairs", []):
                ft = TYPE_MAP_FULL.get(p.get("former_type", ""), "")
                lt = TYPE_MAP_FULL.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({
                        "ft": T.index(ft),
                        "lt": T.index(lt),
                        "score": p.get("score", 0.5),
                        "dimension": p.get("dimension", ""),
                    })
    return pairs


def predict_mop_per_component(model_state, pairs):
    """Return (K, n) matrix: each component's prediction for each pair."""
    priorities = model_state["priorities"]
    K = len(priorities)
    n = len(pairs)
    preds = np.zeros((K, n))
    for k in range(K):
        w = priorities[k]
        for j, p in enumerate(pairs):
            i, l = int(p["ft"]), int(p["lt"])
            log_odds = np.log(w[i] / (w[l] + EPS))
            preds[k, j] = 1.0 / (1.0 + np.exp(-log_odds))
    return preds


def compute_best_match_metrics(y_true, preds_matrix):
    """preds_matrix: (K, n) or (1, n). For MoP (K>1), pick best-match component per pair."""
    yt = np.asarray(y_true, dtype=float)
    if preds_matrix.ndim == 1:
        preds_matrix = preds_matrix.reshape(1, -1)
    K, n = preds_matrix.shape
    if K == 1:
        yp = preds_matrix[0]
    else:
        abs_err = np.abs(preds_matrix - yt[np.newaxis, :])
        best_idx = np.argmin(abs_err, axis=0)
        yp = preds_matrix[best_idx, np.arange(n)]

    valid = ~(np.isnan(yp) | np.isinf(yp))
    yt_v, yp_v = yt[valid], yp[valid]
    if len(yt_v) < 3:
        return {}

    mse = float(np.mean((yt_v - yp_v) ** 2))
    mae = float(np.mean(np.abs(yt_v - yp_v)))
    pr, _ = pearsonr(yt_v, yp_v)
    sr, _ = spearmanr(yt_v, yp_v)
    yt_bin = (yt_v > 0.5).astype(int)
    yp_bin = (yp_v > 0.5).astype(int)
    acc = float(np.mean(yt_bin == yp_bin))

    return {"mse": mse, "mae": mae, "pearson_r": float(pr),
            "spearman_rho": float(sr), "accuracy": acc}


def run_cv(pairs, n_folds=5, random_seed=42):
    nA = len(T)
    pairs_arr = np.array(pairs, dtype=object)
    dims = np.array([p["dimension"] for p in pairs])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    all_fold_results = {m: {"mse": [], "mae": [], "pearson_r": [],
                             "spearman_rho": [], "accuracy": [], "time_s": []}
                         for m in METHODS}

    for train_idx, test_idx in skf.split(pairs_arr, dims):
        train_pairs = pairs_arr[train_idx].tolist()
        test_pairs = pairs_arr[test_idx].tolist()
        y_true = np.array([p["score"] for p in test_pairs])

        for method in METHODS:
            try:
                t0 = time.perf_counter()

                if method == "MoP":
                    state = fit_mop(train_pairs, nA, max_K=4)
                    preds_mat = predict_mop_per_component(state, test_pairs)
                else:  # P=1
                    state = fit_single_k1(train_pairs, nA)
                    preds_mat = predict_single_k1(state, test_pairs).reshape(1, -1)

                t_elapsed = time.perf_counter() - t0
                metrics = compute_best_match_metrics(y_true, preds_mat)
                if metrics:
                    for k, v in metrics.items():
                        all_fold_results[method][k].append(v)
                    all_fold_results[method]["time_s"].append(t_elapsed)

            except Exception as e:
                print(f"      {method}: ERROR — {e}")

    final = {}
    for method in METHODS:
        final[method] = {}
        for metric in ["mse", "mae", "pearson_r", "spearman_rho", "accuracy", "time_s"]:
            vals = all_fold_results[method][metric]
            if vals:
                final[method][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-pairs", type=int, default=100)
    args = parser.parse_args()

    all_pairs = load_real_pairs()
    print(f"Loaded {len(all_pairs)} comparison pairs ({len(T)} types)")

    labeled_pairs = [p for p in all_pairs if p["dimension"]]
    dim_pairs = {}
    for p in labeled_pairs:
        dim_pairs.setdefault(p["dimension"], []).append(p)

    print(f"\n{'='*65}")
    print("Overall")
    print("=" * 65)
    overall = run_cv(all_pairs, n_folds=args.folds)

    dim_results = {}
    for dim in sorted(dim_pairs.keys(), key=lambda d: -len(dim_pairs[d])):
        pairs_d = dim_pairs[dim]
        n = len(pairs_d)
        if n < args.min_pairs:
            print(f"\n  {DIM_LABELS.get(dim, dim)} ({n} pairs) — SKIP")
            continue
        label = DIM_LABELS.get(dim, dim)
        print(f"\n{'='*65}")
        print(f"{label} ({n} pairs)")
        print("=" * 65)
        dim_results[dim] = {"n_pairs": n, "results": run_cv(pairs_d, n_folds=min(args.folds, n // 30))}

    def print_section(name, results, n_pairs=None):
        hdr = f"  {'Method':<8} {'MSE↓':>9} {'MAE↓':>8} {'r↑':>7} {'ρ↑':>7} {'Acc↑':>7}"
        print(f"\n  {name}" + (f" (n={n_pairs})" if n_pairs else ""))
        print("  " + "-" * len(hdr)); print(hdr); print("  " + "-" * len(hdr))
        for method in METHODS:
            if method in results:
                m = results[method]
                def s(k): return f"{m[k]['mean']:.4f}" if k in m else "---"
                print(f"  {method:<8} {s('mse'):>9} {s('mae'):>8} {s('pearson_r'):>7} "
                      f"{s('spearman_rho'):>7} {s('accuracy'):>7}")

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print_section("Overall", overall, len(all_pairs))
    for dim in sorted(dim_results.keys(), key=lambda d: -dim_results[d]["n_pairs"]):
        dr = dim_results[dim]
        print_section(DIM_LABELS.get(dim, dim), dr["results"], dr["n_pairs"])

    output = {
        "n_folds": args.folds,
        "n_pairs_total": len(all_pairs),
        "overall": overall,
        "per_dimension": {dim: {"n_pairs": dr["n_pairs"], "results": dr["results"]}
                          for dim, dr in dim_results.items()},
    }
    with open(HOLDOUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {HOLDOUT_RESULTS}")


if __name__ == "__main__":
    main()
