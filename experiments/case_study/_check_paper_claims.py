"""The two case-study numbers Section 3 states from a fit of its own, with a path.

They were quoted in the manuscript without a script that produced them, which is
why an audit could not trace them. Each function below reproduces one of them
under the paper's specification (K-Ridge from scratch at every candidate,
`prior_mode_phi0`, `lambda_sigma = nJ/50`), prints the value the text quotes, and
returns it for `_paper_claims_check.json`.

  1. Six named dimensions: the 5,238 dimension-labelled judgments less the 798 in
     the catch-all Other, i.e. 4,440. Section 3.3 reports BEV's expected utility
     on that subset as -0.050 against the ICEV benchmark.

  2. Fixed versus free dispersion: Section 3.6 reports that with the dispersion
     fixed at one the same criterion selects three components rather than four.
     Both parameterisations receive the same prior mode, since fixing phi at one
     is itself a claim about phi_0 (see PHI_IDENTIFICATION.md).

Run:
    uv run python experiments/case_study/_check_paper_claims.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mop.inference import prior_mode_phi0                        # noqa: E402
from mop.initialization import kridge                            # noqa: E402

import step7_analyze as S7                                       # noqa: E402
import step9_phi_identification as S9                            # noqa: E402
from _common import ENERGY_PAIRS, TYPE_MAP, T                    # noqa: E402

OUT = HERE / "_paper_claims_check.json"


def load_pairs_with_dimension():
    """The case study's judgments with their dimension label kept.

    step7_analyze.load_pairs drops the label, which is what the six-dimension
    subset needs, so the same filter is applied here and the label retained.
    """
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
    best_k, best, weighted, n_j = S7.run_mop(pairs)
    out = {"n": n_j, "selected_k": best_k, "pbic": best["bic"],
           "expected_utility": {t: float(v) for t, v in zip(T, weighted)}}
    print(f"  six named dimensions: n={n_j}, P={best_k}, pBIC={best['bic']:.1f}")
    for t, v in zip(T, weighted):
        print(f"    EU {t:5s} = {v: .3f}")
    return out


def fixed_vs_free_dispersion(grid=(1, 2, 3, 4, 5)):
    pairs = S9.load_pairs()
    X, y = S9.build_design(pairs)
    A = X.shape[1] + 1
    s2 = prior_mode_phi0(X, y)
    out = {"n": int(X.shape[0]), "s2": float(s2), "grid": list(grid)}
    print(f"  fixed vs free: n={X.shape[0]}, phi0={s2:.4f}")
    for tag, free in (("fixed_phi", False), ("free_phi", True)):
        vals = {}
        for k in grid:
            u0, ph0, pi0 = kridge(X, y, k)
            r = S9.fit(X, y, k, free, s2=s2, u_init=u0, phi_init=ph0, pi_init=pi0)
            c = S9.criteria(r, X, y, A, k, free)
            vals[k] = float(c["pBIC"])
        best = min(vals, key=vals.get)
        out[tag] = {"pbic": vals, "selected_k": int(best)}
        print(f"    {tag}: selected P={best}  "
              + "  ".join(f"P{k}={v:,.1f}" for k, v in vals.items()))
    return out


def main():
    print("checking the two claims Section 3 states:")
    out = {"six_named_dimensions": six_named_dimensions(),
           "fixed_vs_free_dispersion": fixed_vs_free_dispersion()}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
