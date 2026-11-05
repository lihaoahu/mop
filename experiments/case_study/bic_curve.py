"""The BIC curve behind the selected count (Figure 3), with the two margins the text quotes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np

from common.paths import result

from case_study.estimator import load_pairs, run_mop


def _numbered(vals):
    return {str(k): (float(v) if np.isfinite(v) else None)
            for k, v in sorted(vals.items())}


def main():
    trace = {}
    best, _, _, n_j = run_mop(load_pairs(), trace=trace)
    bic = {k: float(v) for k, v in trace["bic_vals"].items()}
    lbic = {k: float(v) for k, v in trace["lbic_vals"].items()}
    defined = [k for k in sorted(bic) if np.isfinite(bic[k])]
    i = defined.index(best)
    payload = {
        "n_judgments": int(n_j),
        "selected": int(best),
        "BIC": _numbered(bic),
        "LBIC": _numbered(lbic),
        "undefined": [k for k in sorted(lbic) if not np.isfinite(lbic[k])],
        "gaps": {
            "minus_left": (bic[best] - bic[defined[i - 1]] if i > 0 else None),
            "plus_right": (bic[defined[i + 1]] - bic[best]
                           if i + 1 < len(defined) else None),
        },
    }
    out = result("bic_curve.json")
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
