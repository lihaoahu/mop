"""Refit on the six named dimensions, leaving out the residual category."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from common.paths import ABBREV, ENERGY_PAIRS, T, TYPE_MAP, result

from case_study.estimator import run_mop

OUT = result("claims.json")


def load_pairs_with_dimension():
    """The case study's judgments with their dimension label kept."""
    pairs = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for p in r.get("comparison_pairs", []):
                ft = TYPE_MAP.get(p.get("former_type", ""), "")
                lt = TYPE_MAP.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": ft, "lt": lt, "score": p.get("score", 0.5),
                                  "dimension": p.get("dimension")})
    return pairs


def six_named_dimensions():
    pairs = [p for p in load_pairs_with_dimension()
             if p["dimension"] is not None and p["dimension"] != "other"]
    best_k, best, weighted, n_j = run_mop(pairs)
    out = {"n": n_j, "selected_k": best_k, "bic": best["bic"],
           "lbic": best["lbic"], "pbic": best["pbic"],
           "expected_utility": {ABBREV[t]: float(v) for t, v in zip(T, weighted)}}
    print(f"six named dimensions: n={n_j}, P={best_k}, BIC={best['bic']:.1f}")
    for t, v in zip(T, weighted):
        print(f"  EU {ABBREV[t]:5s} = {v: .3f}")
    return out


def main():
    out = {"six_named_dimensions": six_named_dimensions()}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
