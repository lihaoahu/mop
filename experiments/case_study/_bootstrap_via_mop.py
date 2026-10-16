"""Bootstrap the component count through the PAPER's own fitting function.

Two implementations of the same model can stop at different optima, and the component count
is sensitive to that. Before any resampling result is reported for the case study it has to
be established through `inference.mop` -- the function step7_analyze uses and the only one
whose output the manuscript reports -- rather than through a reimplementation. An earlier
resampling through `step9_phi_identification.fit` gave P=3 in 50%, P=4 in 40% and P=5 in 10%
of 20 replicates; if that is a property of my implementation and not of the data, the paper
must not carry it.

Method: resample the pooled judgments with replacement, refit with `inference.mop` over the
same candidate set the paper uses, warm-starting low to high exactly as step7_analyze does,
and count the pBIC minimum each time. The prior mode is recomputed on each resample from
`prior_mode_phi0`.

Run:
    uv run python experiments/case_study/_bootstrap_via_mop.py --reps 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import _mop_params as MP                                            # noqa: E402
from mop import inference                                           # noqa: E402
from mop.inference import comparisons2regression, prior_mode_phi0   # noqa: E402
from mop.initialization import warm_init                            # noqa: E402
from mop.pairwise_comparison import PairwiseComparison              # noqa: E402

from _common import T                                                # noqa: E402
from step7_analyze import load_pairs                                 # noqa: E402

EPS = 1e-10


def fit_chain(X, y, pcv, link, grid, s2):
    """The paper's search: warm start low to high, pBIC at each count."""
    lam = MP.lambda_sigma_for(len(pcv))
    prev, out = None, {}
    for k in grid:
        kw = {}
        if prev is not None:
            pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
            ps = np.array([p.inconsistency_level for p in prev.priorities])
            pm = np.array(prev.mixture_probabilities)
            try:
                ui, si, mi = warm_init(X, y, pu, ps, pm,
                                       min_weight_share=MP.min_weight_share,
                                       n_components=k)
                kw = {"u_init": ui, "sigma2_init": si, "mp_init": mi}
            except Exception:
                kw = {}
        r = inference.mop(pcv, link, k, lambda_u=MP.lambda_u, lambda_pi=MP.lambda_pi,
                          lambda_sigma=lam, s2=s2, **kw)
        out[k] = inference.pbic(pcv, r, link, lambda_u=MP.lambda_u,
                                lambda_sigma=lam, lambda_pi=MP.lambda_pi)
        prev = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = load_pairs()
    X, y = comparisons2regression(
        [PairwiseComparison(former_id={t: i + 1 for i, t in enumerate(T)}[p["ft"]],
                            latter_id={t: i + 1 for i, t in enumerate(T)}[p["lt"]],
                            agent_id=1, judgment=p["score"]) for p in pairs],
        [lambda s, _e=EPS: logit(np.clip(s, _e, 1 - _e))])[:2]
    nJ = X.shape[0]
    n2i = {t: i + 1 for i, t in enumerate(T)}
    link = [lambda s, _e=EPS: logit(np.clip(s, _e, 1 - _e))]

    def pcv_of(rows):
        return [PairwiseComparison(former_id=n2i[p["ft"]], latter_id=n2i[p["lt"]],
                                   agent_id=1, judgment=p["score"]) for p in rows]

    print(f"pooled judgments: {nJ}; fitting through inference.mop, the paper's function")
    print(f"candidate set: {args.grid}; {args.reps} replicates\n")

    # Point estimate on the full sample, the paper's way.
    s2_full = float(prior_mode_phi0(X, y))
    full = fit_chain(X, y, pcv_of(pairs), link, args.grid, s2_full)
    best_full = min(full, key=full.get)
    print(f"  full sample: " + "  ".join(f"P={k}:{full[k]:,.1f}" for k in sorted(full)))
    print(f"               selects P={best_full}\n")

    rng = np.random.default_rng(args.seed)
    counts = Counter()
    for rep in range(args.reps):
        idx = rng.integers(0, nJ, size=nJ)
        rows = [pairs[i] for i in idx]
        Xb, yb = X[idx], y[idx]
        s2b = float(prior_mode_phi0(Xb, yb))
        cur = fit_chain(Xb, yb, pcv_of(rows), link, args.grid, s2b)
        b = min(cur, key=cur.get)
        counts[b] += 1
        print(f"  rep {rep + 1:>3}/{args.reps}  P={b}  "
              + " ".join(f"{k}:{v:,.0f}" for k, v in sorted(cur.items())), flush=True)

    print()
    pretty = ", ".join(f"P={k}: {100 * c / args.reps:.0f}%"
                       for k, c in sorted(counts.items()))
    print(f"  selection frequencies through inference.mop: {pretty}")
    out = Path(args.out) if args.out else (HERE / "bootstrap_via_mop.json")
    out.write_text(json.dumps(
        {"n": int(nJ), "reps": args.reps, "seed": args.seed, "grid": args.grid,
         "full_sample": {str(k): float(v) for k, v in full.items()},
         "full_sample_selected": int(best_full),
         "frequencies": {str(k): c / args.reps for k, c in sorted(counts.items())}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
