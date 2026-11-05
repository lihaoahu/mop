"""Component-level fits behind Tables 2 and 4 and the yearly table."""


from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np

from common.paths import ABBREV, DIMS, T, result

from case_study.estimator import load_pairs, run_mop

YEARS = ("2025", "2026")


def _logratio(w):
    """Utilities relative to the last alternative, the benchmark of the tables."""
    w = np.maximum(np.asarray(w, dtype=float), 1e-300)
    return np.log(w[:-1] / w[-1])


def fit_block(pairs):
    """Fit one dataset and return everything the tables need."""
    best_k, best, u_bar, nJ = run_mop(pairs)
    mix = best["mixture"]
    return {
        "n": int(nJ),
        "k": int(best_k),
        "bic": float(best["bic"]),
        "lbic": float(best["lbic"]),
        "pbic": float(best["pbic"]),
        "components": [
            {"weights": [float(x) for x in pri],
             "utilities": [float(v) for v in _logratio(pri)],
             "pi": float(pi),
             "phi": float(phi)}
            for pri, pi, phi in zip(mix["priorities"], mix["pis"],
                                    mix["inconsistencies"])
        ],
        "expected_utility": [float(x) for x in u_bar],
    }


def _print_block(name, blk):
    head = "  ".join(f"{ABBREV[t]:>6}" for t in T)
    print(f"\n[{name}]  n={blk['n']:,}  K={blk['k']}  BIC={blk['bic']:,.1f}")
    print(f"  {'':6} {head} {'pi':>8} {'phi':>7}")
    for i, c in enumerate(blk["components"], start=1):
        u = list(c["utilities"]) + [0.0]
        cells = "  ".join(f"{v:6.3f}" for v in u)
        print(f"  C{i:<5} {cells} {c['pi']:8.4f} {c['phi']:7.4f}")
    cells = "  ".join(f"{v:6.3f}" for v in blk["expected_utility"])
    print(f"  {'E[U]':5} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(result("tables.json")),
                    help="where the raw numbers go")
    args = ap.parse_args()

    out = {}

    overall = fit_block(load_pairs())
    out["overall"] = overall
    _print_block("Overall", overall)

    out["dimensions"] = {}
    for dim in DIMS:
        pairs = load_pairs(dim_name=dim)
        if len(pairs) < 30:
            continue
        blk = fit_block(pairs)
        out["dimensions"][dim] = blk
        _print_block(dim, blk)

    out["years"] = {}
    for year in YEARS:
        blk = fit_block(load_pairs(year=year))
        out["years"][year] = blk
        _print_block(year, blk)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
