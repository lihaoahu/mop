"""The two case-study numbers Section 3 states from a fit of its own, with a path."""
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


def degenerate_component():
    """What the near-degenerate component of the four-component fit is made of."""
    from scipy.special import logit
    from mop import inference
    from mop.pairwise_comparison import PairwiseComparison, comparisons2regression

    pairs = S7.load_pairs()
    name_to_id = {t: i + 1 for i, t in enumerate(S7.T)}
    pcv = [PairwiseComparison(former_id=name_to_id[p["ft"]],
                              latter_id=name_to_id[p["lt"]],
                              agent_id=1, judgment=p["score"]) for p in pairs]
    link = [lambda s, _eps=1e-10: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcv, link)
    nA = X.shape[1]
    best_k, best, _, n_j = S7.run_mop(pairs)
    mix = best["mixture"]
    u = np.array([np.log(np.asarray(w, dtype=float)[:nA]
                         / np.asarray(w, dtype=float)[nA])
                  for w in mix["priorities"]])
    g = inference.e_gamma(X, y, u, np.asarray(mix["inconsistencies"], dtype=float),
                          np.asarray(mix["pis"], dtype=float))
    hard = np.argmax(g, axis=1)
    out = {"n": n_j, "selected_k": best_k, "components": []}
    for c in range(best_k):
        idx = np.where(hard == c)[0]
        scores = [float(pcv[k].judgment) for k in idx]
        entry = {"component": c + 1, "pi": float(mix["pis"][c]),
                 "n_hard": int(len(idx)),
                 "score_min": float(min(scores)) if scores else None,
                 "score_max": float(max(scores)) if scores else None}
        out["components"].append(entry)
        print(f"  C{c + 1}: pi={entry['pi']:.5f}  hard assignments={len(idx)}  "
              f"score range {entry['score_min']}..{entry['score_max']}")
    n_zero = int(sum(1 for p in pairs if p["score"] == 0.0))
    out["n_scores_exactly_zero"] = n_zero
    print(f"  judgments scored exactly zero: {n_zero}")
    return out


def main():
    print("checking the two claims Section 3 states:")
    out = {"six_named_dimensions": six_named_dimensions(),
           "fixed_vs_free_dispersion": fixed_vs_free_dispersion(),
           "degenerate_component": degenerate_component()}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
