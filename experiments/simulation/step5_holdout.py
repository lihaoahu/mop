"""Step 5: Holdout evaluation with the mixture-marginal predictor.

This supersedes the best-match rule of `step2_holdout.py`. That rule scores a
held-out comparison with the component whose prediction is closest to the
*observed* value,

    p* = argmin_p | s_j - s_hat_{j,p} |,

which reads the label it is supposed to predict. The MoP column in the original
holdout table was therefore an oracle selection over P candidate predictions,
not a forecast, and the improvement it showed does not survive.

Two honest quantities are reported here, both computed from the trained model
alone:

  log-likelihood   per held-out comparison, using the marginal mixture density
                   sum_p pi_p * Normal(logit(s_j); x_j' u_p, sigma_p^2). This is
                   the quantity the model actually defines a likelihood for, so
                   it is the primary evidence.

  MAE              of the mixture-marginal point prediction sum_p pi_p * s_hat_jp
                   on the probability scale. Reported for completeness: under a
                   squared-error criterion the optimal point forecast is the
                   conditional mean, and a mixture does not reveal which
                   component generated a held-out comparison, so this quantity
                   is expected to be near-identical across models.

Usage:
    uv run python experiments/simulation/step5_holdout.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logit, logsumexp
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import CANONICAL_TYPES, SIM_DIR  # noqa: E402
from _baselines import fit_single_k1, predict_single_k1  # noqa: E402
from step1_simulation import fit_mop  # noqa: E402
from step2_holdout import DIM_LABELS, load_real_pairs, predict_mop_per_component  # noqa: E402

EPS = 1e-10
OUT = SIM_DIR / "holdout_marginal_results.json"


def design(pairs, n_a):
    X = np.zeros((len(pairs), n_a - 1))
    for k, p in enumerate(pairs):
        i, j = int(p["ft"]), int(p["lt"])
        if i < n_a - 1:
            X[k, i] += 1.0
        if j < n_a - 1:
            X[k, j] -= 1.0
    y = logit(np.clip(np.array([p["score"] for p in pairs], dtype=float), EPS, 1 - EPS))
    return X, y


def mixture_loglik(pairs, w_mat, sigma2, pis, n_a):
    """log p(logit(s_j) | mixture) for every held-out comparison."""
    X, y = design(pairs, n_a)
    w = np.asarray(w_mat, dtype=float)
    u = np.log(w[:, :n_a - 1] / w[:, [n_a - 1]])
    s2 = np.asarray(sigma2, dtype=float).reshape(-1)
    resid = y[:, None] - (X @ u.T)
    log_comp = -0.5 * np.log(2 * np.pi * s2)[None, :] - 0.5 * resid ** 2 / s2[None, :]
    return logsumexp(np.log(np.asarray(pis, dtype=float))[None, :] + log_comp, axis=1)


def single_loglik(pairs, result, n_a):
    w = np.asarray(result[0].priority_vector, dtype=float).reshape(1, -1)
    s2 = np.array([result[0].inconsistency_level], dtype=float)
    return mixture_loglik(pairs, w, s2, [1.0], n_a)


def run_cv(pairs, n_folds=5, random_seed=42):
    n_a = len(CANONICAL_TYPES)
    arr = np.array(pairs, dtype=object)
    dims = np.array([p["dimension"] for p in pairs])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    per_fold = []
    for train_idx, test_idx in skf.split(arr, dims):
        train, test = arr[train_idx].tolist(), arr[test_idx].tolist()
        y_true = np.array([p["score"] for p in test])

        t0 = time.perf_counter()
        state = fit_mop(train, n_a, max_K=4)
        preds = predict_mop_per_component(state, test)
        pis = np.asarray(state["pis"], dtype=float)
        marginal = pis @ preds
        ll_mop = mixture_loglik(test, state["priorities"], state["sigma2"], pis, n_a)
        t_mop = time.perf_counter() - t0

        state1 = fit_single_k1(train, n_a)
        pred1 = predict_single_k1(state1, test)
        ll_p1 = single_loglik(test, state1["mop_result"], n_a)

        per_fold.append({
            "n_test": len(test),
            "k_selected": int(state["k"]),
            "ll_mop": float(ll_mop.mean()),
            "ll_p1": float(ll_p1.mean()),
            "mae_mop": float(np.mean(np.abs(y_true - marginal))),
            "mae_p1": float(np.mean(np.abs(y_true - pred1))),
            "time_mop_s": t_mop,
        })
    return per_fold


def summarise(folds):
    keys = ["ll_mop", "ll_p1", "mae_mop", "mae_p1"]
    out = {k: {"mean": float(np.mean([f[k] for f in folds])),
               "std": float(np.std([f[k] for f in folds]))} for k in keys}
    out["ll_gain"] = {"mean": out["ll_mop"]["mean"] - out["ll_p1"]["mean"]}
    out["ll_gain"]["folds_better"] = int(sum(1 for f in folds if f["ll_mop"] > f["ll_p1"]))
    out["n_folds"] = len(folds)
    out["k_selected"] = [f["k_selected"] for f in folds]
    out["n_test"] = int(sum(f["n_test"] for f in folds))
    return out


def main():
    all_pairs = load_real_pairs()
    print(f"loaded {len(all_pairs)} comparison pairs ({len(CANONICAL_TYPES)} types)")

    sections = {}
    print("running: Overall ...")
    sections["overall"] = summarise(run_cv(all_pairs))

    dim_pairs = {}
    for p in [q for q in all_pairs if q["dimension"]]:
        dim_pairs.setdefault(p["dimension"], []).append(p)
    for dim in sorted(dim_pairs, key=lambda d: -len(dim_pairs[d])):
        label = DIM_LABELS.get(dim, dim)
        print(f"running: {label} ...")
        sections[dim] = summarise(run_cv(dim_pairs[dim]))

    print()
    print("=" * 84)
    print("held-out log-likelihood per comparison (higher is better) and marginal MAE")
    print("=" * 84)
    print(f"{'section':<22} {'n':>6} {'K':>12} {'LL MoP':>9} {'LL P=1':>9} "
          f"{'gain':>8} {'MAE MoP':>9} {'MAE P=1':>9}")
    print("-" * 84)
    for name, s in sections.items():
        label = DIM_LABELS.get(name, name.title() if name != "overall" else "Overall")
        print(f"{label:<22} {s['n_test']:>6} {str(s['k_selected']):>12} "
              f"{s['ll_mop']['mean']:>9.4f} {s['ll_p1']['mean']:>9.4f} "
              f"{s['ll_gain']['mean']:>+8.4f} "
              f"{s['mae_mop']['mean']:>9.4f} {s['mae_p1']['mean']:>9.4f}")

    gains = [s["ll_gain"]["mean"] for k, s in sections.items() if k != "overall"]
    print()
    print(f"dimensions with a positive log-likelihood gain: "
          f"{sum(1 for g in gains if g > 0)}/{len(gains)}")
    print(f"dimensions where the equally good MAE differs by more than 0.005: "
          f"{sum(1 for k, s in sections.items() if k != 'overall' and abs(s['mae_mop']['mean'] - s['mae_p1']['mean']) > 0.005)}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sections": sections}, fh, indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
