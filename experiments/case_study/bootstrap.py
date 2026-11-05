"""Article-level bootstrap of the selected count.

Judgments are not independent: they come from 2,328 articles, each contributing
several comparisons. Resampling judgments would understate the spread, so the
article is the unit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.parallel import run, seeds                              # noqa: E402
from common.paths import ENERGY_PAIRS, T, TYPE_MAP, result          # noqa: E402
from mop import inference                                           # noqa: E402
from mop.initialization import kridge                               # noqa: E402
from mop.pairwise_comparison import (                               # noqa: E402
    PairwiseComparison,
    comparisons2regression,
)

RESULTS = result("bootstrap.json")
GRID = (1, 2, 3, 4, 5)
LAMBDA_U, LAMBDA_PI = 1e-6, 1.0
LAMBDA_SIGMA_OVER_N = 0.02
DEGENERATE_PI = 0.01


def load_articles():
    """Every article's usable cross-type comparisons, kept as one cluster."""
    articles = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pairs = []
            for p in r.get("comparison_pairs", []):
                ft = TYPE_MAP.get(p.get("former_type", ""), "")
                lt = TYPE_MAP.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": ft, "lt": lt,
                                  "score": float(p.get("score", 0.5))})
            if pairs:
                articles.append(pairs)
    return articles


def fit_one(pairs, k, name_to_id):
    pcv = [PairwiseComparison(name_to_id[p["ft"]], name_to_id[p["lt"]], 1,
                              p["score"]) for p in pairs]
    link = [lambda s, _e=1e-10: logit(np.clip(s, _e, 1 - _e))]
    X, y, _ = comparisons2regression(pcv, link)
    nJ = len(pcv)
    ls = LAMBDA_SIGMA_OVER_N * float(nJ)
    u0, phi0, mp0 = kridge(np.asarray(X, float), np.asarray(y, float), k)
    r = inference.fit_mixture(pcv, link, k, lambda_u=LAMBDA_U, lambda_pi=LAMBDA_PI,
                      lambda_sigma=ls, u_init=u0, sigma2_init=phi0, mp_init=mp0)
    bi = float(inference.bic(pcv, r, link, lambda_u=LAMBDA_U,
                             lambda_sigma=ls, lambda_pi=LAMBDA_PI))
    W = np.array([np.asarray(p.priority_vector, float) for p in r.priorities])
    return {"k": k, "bic": bi, "pi": np.asarray(r.mixture_probabilities, float),
            "phi": np.array([float(p.inconsistency_level) for p in r.priorities]),
            "w": W, "n": nJ}


def select_k(pairs, name_to_id, grid=GRID):
    """BIC-minimising component count, as in the paper.

    Counts whose criterion is undefined are excluded rather than scored as if it
    were infinite: a draw whose fit puts a weight on the boundary has no mode, so
    the criterion says nothing about that count.
    """
    fits = [fit_one(pairs, k, name_to_id) for k in grid]
    curve = {f["k"]: f["bic"] for f in fits}
    defined = {k: v for k, v in curve.items() if np.isfinite(v)}
    if not defined:
        return None, curve
    best = min(defined, key=defined.get)
    return fits[[f["k"] for f in fits].index(best)], curve


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def match(reference, candidate):
    """Best assignment of candidate rows to reference rows by cosine.

    Returns one index per reference row, or -1 when the candidate is smaller.
    """
    nr, nc = len(reference), len(candidate)
    if nc < nr:
        return [-1] * nr
    S = np.array([[_cos(reference[i], candidate[j]) for j in range(nc)]
                  for i in range(nr)])
    best, best_score = None, -np.inf
    for perm in permutations(range(nc), nr):
        score = sum(S[i, perm[i]] for i in range(nr))
        if score > best_score:
            best, best_score = perm, score
    return list(best)


def _bootstrap_rep(job):
    article_ids, name_to_id, n_articles, seed = job
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_articles, size=n_articles)
    pairs = [p for i in pick for p in article_ids[i]]
    if len(pairs) < 50:
        return {"rep": -1, "skipped": True}
    best, curve = select_k(pairs, name_to_id)
    if best is None:
        return {"rep": -1, "skipped": True, "why": "no interior mode at any count"}
    return {"k": int(best["k"]),
            "curve": {str(k): (v if np.isfinite(v) else None)
                      for k, v in curve.items()},
            "pi": best["pi"].tolist(), "phi": best["phi"].tolist(),
            "w": best["w"].tolist(), "n": int(best["n"]),
            "degenerate": int((best["pi"] < DEGENERATE_PI).sum())}


def _percentile(xs, q):
    return float(np.percentile(np.asarray(xs, float), q)) if len(xs) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    articles = load_articles()
    flat = [p for a in articles for p in a]
    print(f"{len(articles)} articles, {len(flat)} judgments")

    t0 = time.perf_counter()
    point, curve = select_k(flat, name_to_id)
    print(f"point estimate: K={point['k']}  BIC={point['bic']:.1f}  "
          f"({time.perf_counter() - t0:.1f}s)")
    for k, v in sorted(curve.items()):
        print(f"  K={k}  BIC {v:11.2f}")
    print(f"  phi {np.round(point['phi'], 3)}   pi {np.round(point['pi'], 4)}")
    ref = point["w"]

    print(f"\ncluster bootstrap over articles, {args.reps} draws")
    t0 = time.perf_counter()
    rows = run(_bootstrap_rep,
               [(articles, name_to_id, len(articles), s)
                for s in seeds(args.seed, args.reps)],
               jobs=args.jobs)
    rows = [r for r in rows if not r.get("skipped")]
    print(f"  {len(rows)} usable draws in {time.perf_counter() - t0:.1f}s")

    ks = [r["k"] for r in rows]
    k_counts = {str(k): int(ks.count(k)) for k in sorted(set(ks))}
    print(f"  selected K: {k_counts}")
    print(f"  share at the point estimate ({point['k']}): "
          f"{np.mean([k == point['k'] for k in ks]):.3f}")

    # component recurrence, measured only on draws that returned the same count
    same = [r for r in rows if r["k"] == point["k"]]
    rec = []
    if same:
        for i in range(len(ref)):
            hits = 0
            for r in same:
                idx = match(ref, np.asarray(r["w"], float))
                if idx[i] >= 0 and _cos(ref[i], np.asarray(r["w"], float)[idx[i]]) > 0.95:
                    hits += 1
            rec.append(hits / len(same))
    print("\n  component recurrence among draws with the same count")
    for i, v in enumerate(rec):
        print(f"    C{i + 1}: pi={point['pi'][i]:.4f}  recurs in {v:.2f} of them")

    print("\n  weight and dispersion, 5-95% across all draws")
    for i in range(len(ref)):
        pis = [r["pi"][i] for r in rows if len(r["pi"]) > i]
        phis = [r["phi"][i] for r in rows if len(r["phi"]) > i]
        print(f"    C{i + 1}: pi [{_percentile(pis, 5):.3f}, {_percentile(pis, 95):.3f}]"
              f"   phi [{_percentile(phis, 5):.2f}, {_percentile(phis, 95):.2f}]")

    out = {
        "design": {"source": "case study, energy types", "n_articles": len(articles),
                   "n_judgments": len(flat), "grid": list(GRID), "reps": args.reps,
                   "seed": args.seed, "seed_stride": 1000003,
                   "resampling_unit": "article", "degenerate_pi": DEGENERATE_PI},
        "point": {"k": int(point["k"]), "bic": point["bic"],
                  "curve": {str(k): (v if np.isfinite(v) else None)
                            for k, v in curve.items()},
                  "pi": point["pi"].tolist(), "phi": point["phi"].tolist(),
                  "w": point["w"].tolist()},
        "bootstrap": {"usable": len(rows), "k_counts": k_counts,
                      "share_at_point": float(np.mean([k == point["k"] for k in ks])),
                      "component_recurrence": rec,
                      "pi_interval": [[_percentile([r["pi"][i] for r in rows
                                                    if len(r["pi"]) > i], 5),
                                       _percentile([r["pi"][i] for r in rows
                                                    if len(r["pi"]) > i], 95)]
                                      for i in range(len(ref))],
                      "phi_interval": [[_percentile([r["phi"][i] for r in rows
                                                     if len(r["phi"]) > i], 5),
                                        _percentile([r["phi"][i] for r in rows
                                                     if len(r["phi"]) > i], 95)]
                                       for i in range(len(ref))],
                      "degenerate_share": [float(np.mean([r["degenerate"] >= j + 1
                                                          for r in rows]))
                                           for j in range(len(ref))]},
        "per_draw": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
