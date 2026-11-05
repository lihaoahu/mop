"""Five-fold holdout evaluation and the retained-gain column.

Folds are stratified within each dimension. Both models are scored on the held-out
judgments by the predictive density they imply for them, so the comparison is
between distributions rather than between point forecasts.
"""


import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import expit, logit
from scipy.stats import kstest, norm, spearmanr
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.fit import fit_mop                                        # noqa: E402
from common.paths import ENERGY_PAIRS, T, TYPE_MAP, result            # noqa: E402
from common.single import fit_single_k1                               # noqa: E402

EPS = 1e-10
HOLDOUT_MAX_K = 4          # the search stops at A, not at A!
NA = len(T)
OUT = result("holdout.json")
CURVES = result("dim_curves.json")
TYPE_MAP_FULL = {**TYPE_MAP, "燃油": "燃油", "纯电": "纯电", "插混": "插混", "增程": "增程"}


def load_real_pairs():
    """The corpus with its dimension label, indices instead of energy-type names."""
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


# Gauss-Hermite nodes for E[sigmoid(mu + sigma Z)] over Z ~ N(0, 1).
GH_X, GH_W = np.polynomial.hermite.hermgauss(120)
GH_W = GH_W / np.sqrt(np.pi)
# Grid on the logit scale for the predictive CDF, its median and the CRPS integral.
GRID = np.linspace(-12.0, 12.0, 4001)


def design(pairs, n_a=NA):
    X = np.zeros((len(pairs), n_a - 1))
    for k, p in enumerate(pairs):
        i, j = int(p["ft"]), int(p["lt"])
        if i < n_a - 1:
            X[k, i] += 1.0
        if j < n_a - 1:
            X[k, j] -= 1.0
    y = logit(np.clip(np.array([p["score"] for p in pairs], dtype=float), EPS, 1 - EPS))
    return X, y


def component_means(priorities, n_a=NA):
    """x'u_p is recovered from the priority vector, as in step2 and step5."""
    w = np.asarray(priorities, dtype=float)
    return np.log(w[:, :n_a - 1] / w[:, [n_a - 1]])


def predictive(mu, s2, pis, X):
    """Moments, median and CDF of the mixture predictive distribution on the p scale."""
    pis = np.asarray(pis, dtype=float).reshape(-1)
    sd = np.sqrt(np.asarray(s2, dtype=float).reshape(-1))
    M = X @ mu.T                                   # (n, K) component locations

    nodes = (sd[:, None] * np.sqrt(2.0) * GH_X[None, :])[None, :, :]
    P = expit(M[:, :, None] + nodes)               # (n, K, Q)
    qw = GH_W[None, :] * pis[:, None]              # (K, Q)
    m1 = np.einsum("nkq,kq->n", P, qw)
    m2 = np.einsum("nkq,kq->n", P ** 2, qw)
    mean = m1
    var = np.maximum(m2 - m1 ** 2, 0.0)

    Fm = np.zeros((M.shape[0], GRID.size))
    for j in range(len(pis)):
        Fm += pis[j] * norm.cdf((GRID[None, :] - M[:, [j]]) / sd[j])

    # Median by inverting the CDF on the grid, with linear interpolation.
    med = np.empty(M.shape[0])
    for i in range(M.shape[0]):
        k = int(np.searchsorted(Fm[i], 0.5))
        k = min(max(k, 1), GRID.size - 1)
        lo, hi = Fm[i, k - 1], Fm[i, k]
        w = 0.0 if hi <= lo else (0.5 - lo) / (hi - lo)
        med[i] = GRID[k - 1] + w * (GRID[k] - GRID[k - 1])
    return mean, np.sqrt(var), med, Fm


def crps(Fm, y):
    """CRPS in probability space, integrating ``(F - 1{x >= y})^2`` over p.

    The integral runs on the logit scale with the Jacobian ``p(1-p)`` and is split at
    the observation, the split point being added to both pieces; without it the
    omitted strip is one grid wide and the rule degrades to first order.
    """
    grid = expit(GRID)
    jac = grid * (1.0 - grid)
    yp = expit(y)
    out = np.empty(Fm.shape[0])
    for i in range(Fm.shape[0]):
        k = int(np.searchsorted(GRID, y[i]))
        k = min(max(k, 1), GRID.size - 1)
        Fy = Fm[i, k - 1] + ((y[i] - GRID[k - 1]) / (GRID[k] - GRID[k - 1])
                             * (Fm[i, k] - Fm[i, k - 1]))
        jy = yp[i] * (1.0 - yp[i])
        left = GRID[:k] < y[i]
        c = trapezoid(np.r_[Fm[i, :k][left] ** 2 * jac[:k][left], Fy ** 2 * jy],
                      np.r_[GRID[:k][left], y[i]])
        right = GRID[k:] >= y[i]
        c += trapezoid(np.r_[(1.0 - Fy) ** 2 * jy,
                             (1.0 - Fm[i, k:][right]) ** 2 * jac[k:][right]],
                       np.r_[y[i], GRID[k:][right]])
        out[i] = c
    return out


def pit(Fm, y):
    """F_j(y_j): uniform under a correctly specified predictive distribution."""
    k = np.clip(np.searchsorted(GRID, y), 1, GRID.size - 1)
    i = np.arange(y.size)
    lo = Fm[i, k - 1]
    w = (y - GRID[k - 1]) / (GRID[k] - GRID[k - 1])
    return lo + w * (Fm[i, k] - lo)


def log_score(mu, s2, pis, X, y):
    pis = np.asarray(pis, dtype=float).reshape(-1)
    sd = np.sqrt(np.asarray(s2, dtype=float).reshape(-1))
    M = X @ mu.T
    lp = (-0.5 * np.log(2 * np.pi * sd ** 2)[None, :]
          - 0.5 * ((y[:, None] - M) / sd[None, :]) ** 2
          + np.log(pis)[None, :])
    mx = lp.max(axis=1)
    return mx + np.log(np.exp(lp - mx[:, None]).sum(axis=1))


def score_one(mu, s2, pis, X, y):
    mean, sd, med, Fm = predictive(mu, s2, pis, X)
    yp = expit(y)
    p = pit(Fm, y)
    return {
        "log_score": log_score(mu, s2, pis, X, y),
        "crps": crps(Fm, y),
        "abs_err_med": np.abs(yp - expit(med)),
        "abs_err_mean": np.abs(yp - mean),
        "sq_err_mean": (yp - mean) ** 2,
        "pred_sd": sd,
        "pred_med": med,
        "pit": p,
    }


def run_cv(pairs, n_folds=5, random_seed=42):
    arr = np.array(pairs, dtype=object)
    dims = np.array([p["dimension"] for p in pairs])
    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=random_seed).split(arr, dims))

    per_fold, pooled = [], {m: {"log_score": [], "crps": [], "abs_err_med": [],
                                "abs_err_mean": [], "sq_err_mean": [],
                                "pred_sd": [], "pred_med": [], "pit": []}
                            for m in ("MoP", "P=1")}

    for train_idx, test_idx in folds:
        train, test = arr[train_idx].tolist(), arr[test_idx].tolist()
        Xtr, _ = design(train)
        Xte, yte = design(test)

        st = fit_mop(train, NA, max_K=HOLDOUT_MAX_K)
        mu = component_means(st["priorities"])

        st1 = fit_single_k1(train, NA)
        w1 = np.asarray(st1["mop_result"][0].priority_vector, float).reshape(1, -1)
        mu1 = np.log(w1[:, :NA - 1] / w1[:, [NA - 1]])

        scores = {
            "MoP": score_one(mu, st["sigma2"], st["pis"], Xte, yte),
            "P=1": score_one(mu1, [st1["mop_result"][0].inconsistency_level],
                             [1.0], Xte, yte),
        }
        for m, s in scores.items():
            for k, v in s.items():
                pooled[m][k].append(v)

        row = {"k_mop": int(st["k"]), "n_test": int(len(yte))}
        for m, s in scores.items():
            row[m] = {
                "log_score": float(s["log_score"].mean()),
                "crps": float(s["crps"].mean()),
                "mae_median": float(s["abs_err_med"].mean()),
                "mae_mean": float(s["abs_err_mean"].mean()),
                "mse_mean": float(s["sq_err_mean"].mean()),
                "cov50": float(((s["pit"] >= .25) & (s["pit"] <= .75)).mean()),
                "cov90": float(((s["pit"] >= .05) & (s["pit"] <= .95)).mean()),
                "pit_ks": float(kstest(s["pit"], "uniform").statistic),
                "sd_rank_corr": float(spearmanr(s["pred_sd"],
                                                np.abs(yte - s["pred_med"])).statistic),
            }
        per_fold.append(row)

    cat = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in pooled.items()}
    summary = {}
    for m in ("MoP", "P=1"):
        s = cat[m]
        summary[m] = {
            "log_score": float(s["log_score"].mean()),
            "crps": float(s["crps"].mean()),
            "mae_median": float(s["abs_err_med"].mean()),
            "mae_mean": float(s["abs_err_mean"].mean()),
            "mse_mean": float(s["sq_err_mean"].mean()),
            "cov50": float(((s["pit"] >= .25) & (s["pit"] <= .75)).mean()),
            "cov90": float(((s["pit"] >= .05) & (s["pit"] <= .95)).mean()),
            "pit_ks": float(kstest(s["pit"], "uniform").statistic),
            "n": int(s["log_score"].size),
        }
    summary["k_selected"] = [int(r["k_mop"]) for r in per_fold]
    summary["sd_rank_corr"] = {
        m: {"mean": float(np.mean([r[m]["sd_rank_corr"] for r in per_fold])),
            "per_fold": [float(r[m]["sd_rank_corr"]) for r in per_fold]}
        for m in ("MoP", "P=1")}
    return summary, per_fold, cat


def paired_ci(a, b, lower_is_better=True, n_boot=10000, seed=0):
    """Bootstrap CI for mean(a) - mean(b) over paired per-judgment scores.

    ``share_mop_better`` is oriented by the metric: for a score where lower is
    better the mixture wins on negative differences, for the log score on positive
    ones.
    """
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return {"diff": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "share_mop_better": float((d < 0).mean() if lower_is_better else (d > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pairs", type=int, default=100)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()

    curves = json.loads(CURVES.read_text(encoding="utf-8")) if CURVES.exists() else {}
    pairs = load_real_pairs()
    labeled = [p for p in pairs if p["dimension"]]
    dim_pairs = {}
    for p in labeled:
        dim_pairs.setdefault(p["dimension"], []).append(p)
    print(f"{len(pairs)} pairs; K grid <= {HOLDOUT_MAX_K}; folds {args.folds}")

    out = {"predictor": "model predictive distribution",
           "k_grid_max": HOLDOUT_MAX_K, "n_folds": args.folds,
           "n_pairs_total": len(pairs), "sections": {}}

    sections = [("overall", pairs)] + [
        (d, dim_pairs[d]) for d in sorted(dim_pairs, key=lambda d: -len(dim_pairs[d]))
        if len(dim_pairs[d]) >= args.min_pairs]

    for name, ps in sections:
        t0 = time.perf_counter()
        summ, per_fold, cat = run_cv(ps, n_folds=min(args.folds, max(2, len(ps) // 30)))
        # The ratio of held-out to in-sample gain, against the Delta-ell of Table 3.
        if name in curves:
            d_ll = float(curves[name]["delta_ll"])
            gain = float(summ["MoP"]["log_score"] - summ["P=1"]["log_score"])
            # A dimension whose count the criterion leaves at one has no
            # in-sample gain to divide by; the ratio is then undefined rather
            # than zero, and the table prints a dash.
            summ["in_sample"] = {
                "delta_ll": d_ll,
                "delta_ll_per_judgment": d_ll / len(ps),
                "holdout_gain_per_judgment": gain,
                "retained": (gain * len(ps) / d_ll) if d_ll > 1e-9 else None,
            }
        entry = {"n_pairs": len(ps), "summary": summ, "per_fold": per_fold}
        if not args.no_bootstrap:
            entry["paired"] = {
                "crps": paired_ci(cat["MoP"]["crps"], cat["P=1"]["crps"]),
                "log_score": paired_ci(cat["MoP"]["log_score"],
                                       cat["P=1"]["log_score"], lower_is_better=False),
                "mae_median": paired_ci(cat["MoP"]["abs_err_med"],
                                        cat["P=1"]["abs_err_med"]),
                "mae_mean": paired_ci(cat["MoP"]["abs_err_mean"],
                                      cat["P=1"]["abs_err_mean"]),
            }
        out["sections"][name] = entry
        s = summ
        print(f"\n{name:>20s} (n={len(ps)}, K={s['k_selected']}, {time.perf_counter()-t0:.0f}s)")
        print(f"{'':>6s}{'logscore':>10s}{'CRPS':>9s}{'MAEm':>8s}{'MAEmu':>8s}"
              f"{'cov90':>7s}{'rho_sd':>8s}")
        for m in ("MoP", "P=1"):
            d = s[m]
            rho = s["sd_rank_corr"][m]["mean"]
            print(f"{m:>6s}{d['log_score']:>10.4f}{d['crps']:>9.5f}{d['mae_median']:>8.4f}"
                  f"{d['mae_mean']:>8.4f}{d['cov90']:>7.3f}{rho:>8.3f}")
        if "in_sample" in s:
            ins = s["in_sample"]
            if ins["retained"] is not None:
                print(f"       retained {ins['retained']:.1%} of {ins['delta_ll']:.1f} nats "
                      f"(in {ins['delta_ll_per_judgment']:.4f}, out "
                      f"{ins['holdout_gain_per_judgment']:.4f} per judgment)")
            else:
                print("       no in-sample gain to divide (one component selected)")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
