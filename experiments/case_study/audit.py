"""What the low-weight component is made of.

A component can be idle or it can carry a handful of judgments. This reports the
mass it absorbs and the likelihood it accounts for, so the two can be told apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.em import _e_step, bic, fit_gaussian_mixture          # noqa: E402
from common.paths import result                                    # noqa: E402
from mop.inference import prior_mode_phi0                           # noqa: E402

from case_study.estimator import as_pcvector, design, load_pairs   # noqa: E402

GRID = (1, 2, 3, 4, 5, 6, 7, 8)
LOW = 0.01          # a component below this weight is never called a type


def _kw(y, s2):
    return {"lambda_u": 1e-6, "lambda_pi": 1.0,
            "lambda_sigma": 0.02 * float(len(y)), "s2": float(s2)}


def audit_fit(fit, X, y, s2=None):
    """Weights, effective counts, dispersion, and the likelihood each carries."""
    r = _e_step(X, y, fit.u, fit.phi, fit.pi)
    n_eff = r.sum(axis=0)
    logp = (np.log(np.maximum(fit.pi, 1e-300))[None, :]
            - 0.5 * np.log(2.0 * np.pi * fit.phi)[None, :]
            - 0.5 * (y[:, None] - X @ fit.u.T) ** 2 / fit.phi[None, :])
    carry = (r * logp).sum(axis=0)
    return {
        "pi": fit.pi.tolist(), "phi": fit.phi.tolist(),
        "n_eff": n_eff.tolist(), "ll_carried": carry.tolist(),
        "n_low_weight": int((fit.pi < LOW).sum()),
        "pll": float(fit.pll), "df": int(fit.df),
        "BIC": float(bic(fit, len(y))), "converged": bool(fit.converged),
        "responsibilities": r,
    }


def curve(X, y, mode="free", grid=GRID):
    s2 = float(prior_mode_phi0(X, y))
    rows = {}
    for K in grid:
        if mode == "one":
            f = fit_gaussian_mixture(X, y, K, phi_mode="one",
                                     lambda_u=1e-6, lambda_pi=1.0)
        else:
            f = fit_gaussian_mixture(X, y, K, phi_mode=mode, **_kw(y, s2))
        a = audit_fit(f, X, y, s2)
        a.pop("responsibilities")
        rows[str(K)] = a
    best = min(rows, key=lambda k: rows[k]["BIC"])
    return {"phi_mode": mode, "phi0": s2, "curve": rows, "best_K": int(best)}


def restarts(X, y, K, n_restart, seed, s2):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_restart):
        u0 = rng.normal(0.0, 1.0, size=(K, X.shape[1]))
        pi0 = rng.dirichlet(np.ones(K))
        phi0 = s2 * np.exp(rng.normal(0.0, 0.7, size=K))
        f = fit_gaussian_mixture(X, y, K, phi_mode="free", u_init=u0,
                                 pi_init=pi0, phi_init=phi0, **_kw(y, s2))
        rows.append({"pll": float(f.pll), "pi": f.pi.tolist(),
                     "phi": f.phi.tolist(),
                     "n_low_weight": int((f.pi < LOW).sum()),
                     "min_pi": float(f.pi.min())})
    rows.sort(key=lambda r: -r["pll"])
    return {
        "K": K, "n_restart": n_restart,
        "n_low_weight_counts": np.bincount(
            [r["n_low_weight"] for r in rows],
            minlength=K + 1).tolist(),
        "min_pi_min": float(min(r["min_pi"] for r in rows)),
        "min_pi_median": float(np.median([r["min_pi"] for r in rows])),
        "best": rows[0],
    }


def carried_judgments(fit, X, y, pairs):
    """The judgments the lowest-weight component carries, with their raw scores."""
    r = _e_step(X, y, fit.u, fit.phi, fit.pi)
    j = int(np.argmin(fit.pi))
    order = np.argsort(-r[:, j])[:8]
    return {
        "component": j + 1, "pi": float(fit.pi[j]), "phi": float(fit.phi[j]),
        "n_eff": float(r[:, j].sum()),
        "top": [{"i": int(i), "responsibility": float(r[i, j]),
                 "y": float(y[i]), "score": float(pairs[i]["score"]),
                 "former": pairs[i]["ft"], "latter": pairs[i]["lt"]}
                for i in order],
        "n_scored_zero_or_one": int(sum(1 for p in pairs
                                        if p["score"] in (0.0, 1.0))),
        "n_abs_y_gt_5": int((np.abs(y) > 5).sum()),
        "max_abs_y": float(np.abs(y).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--out", default=str(result("audit.json")))
    args = ap.parse_args()

    pairs = load_pairs()
    X, y = design(as_pcvector(pairs))
    s2 = float(prior_mode_phi0(X, y))
    out = {"data": {"n_judgments": int(len(y)), "A": int(X.shape[1] + 1),
                    "phi0": s2, "low_weight_threshold": LOW},
           "grid": list(GRID)}

    print(f"n {len(y)}, A {X.shape[1] + 1}, phi0 {s2:.4f}")
    print("\n== free dispersion ==")
    out["free"] = curve(X, y, "free")
    for K, a in out["free"]["curve"].items():
        print(f"  K={K} BIC  {a['BIC']:11.2f} pi {np.round(a['pi'], 5).tolist()}"
              f" n_eff {np.round(a['n_eff'], 2).tolist()}"
              f" low {a['n_low_weight']}")
    print(f"  -> best K {out['free']['best_K']}")

    print("\n== pinned dispersion (phi == 1) ==")
    out["pinned"] = curve(X, y, "one")
    for K, a in out["pinned"]["curve"].items():
        print(f"  K={K} BIC  {a['BIC']:11.2f} pi {np.round(a['pi'], 5).tolist()}"
              f" low {a['n_low_weight']}")
    print(f"  -> best K {out['pinned']['best_K']}")

    print("\n== random starts ==")
    out["restarts"] = [restarts(X, y, K, args.restarts, 20261025 + K, s2)
                       for K in (4, 5)]
    for r in out["restarts"]:
        print(f"  K={r['K']}: #low-weight counts {r['n_low_weight_counts']},"
              f" min pi median {r['min_pi_median']:.6f}")

    print("\n== what the low-weight component carries ==")
    f4 = fit_gaussian_mixture(X, y, 4, phi_mode="free", **_kw(y, s2))
    out["carried"] = carried_judgments(f4, X, y, pairs)
    c = out["carried"]
    print(f"  C{c['component']}: pi {c['pi']:.5f} phi {c['phi']:.3f}"
          f" n_eff {c['n_eff']:.2f}")
    for t in c["top"][:4]:
        print(f"    i={t['i']:5d} r={t['responsibility']:.3f}"
              f" y={t['y']:8.2f} score={t['score']}")
    print(f"  judgments scored exactly 0 or 1: {c['n_scored_zero_or_one']};"
          f" |y|>5: {c['n_abs_y_gt_5']}")

    print("\n== dropping the saturating judgment ==")
    keep = np.abs(y) < 5.0
    out["trimmed"] = {"n_dropped": int((~keep).sum()),
                      "dropped_y": y[~keep].tolist(),
                      "audit": curve(X[keep], y[keep], "free")}
    for K, a in out["trimmed"]["audit"]["curve"].items():
        print(f"  K={K} BIC  {a['BIC']:11.2f} pi {np.round(a['pi'], 4).tolist()}"
              f" low {a['n_low_weight']}")
    print(f"  -> best K {out['trimmed']['audit']['best_K']}")

    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
