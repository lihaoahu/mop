"""Per-dimension BIC curves, the gain over one component, and the margins around each count."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np

from common.paths import DIM_LABELS, DIMS, result

from case_study.estimator import load_pairs, run_mop

OUT = result("dim_curves.json")


def curve(pairs):
    trace = {}
    best, _, _, n_j = run_mop(pairs, trace=trace)
    vals = {int(k): float(v) for k, v in trace["bic_vals"].items()}
    lap = {int(k): float(v) for k, v in trace["lbic_vals"].items()}
    ll = {int(k): float(m["log_likelihood"]) for k, m in trace["models"].items()}
    order = [k for k in sorted(vals) if np.isfinite(vals[k])]
    i = order.index(best)
    left = vals[order[i]] - vals[order[i - 1]] if i > 0 else None
    right = (vals[order[i + 1]] - vals[order[i]] if i + 1 < len(order) else None)
    return {"n": int(n_j), "k": int(best),
            "bic": {k: (v if np.isfinite(v) else None) for k, v in vals.items()},
            "lbic": {k: (v if np.isfinite(v) else None) for k, v in lap.items()},
            "undefined": [k for k in sorted(lap) if not np.isfinite(lap[k])],
            "ll": ll, "delta_ll": float(ll[best] - ll[1]),
            "margin_left": left, "margin_right": right}


def main():
    out = {"overall": curve(load_pairs())}
    r = out["overall"]
    print(f"{'overall':<12} n={r['n']:<5} K={r['k']}  "
          f"left={r['margin_left']}  right={r['margin_right']}")

    for dim in DIMS:
        sub = load_pairs(dim_name=dim)
        if len(sub) < 30:
            continue
        r = curve(sub)
        out[dim] = r
        top = ", ".join(f"{k}:{v:.0f}" for k, v in sorted(r["bic"].items())[:4]
                        if v is not None)
        print(f"{DIM_LABELS.get(dim, dim):<12} n={r['n']:<5} K={r['k']}  "
              f"left={r['margin_left']}  right={r['margin_right']}  [{top}]")

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
