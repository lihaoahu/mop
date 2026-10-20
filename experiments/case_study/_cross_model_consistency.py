"""Cross-model consistency of the pairwise extraction (reviewer 2, R-06-02-02).

Compares two extraction files in the schema of `energy_pairs.jsonl` and reports
how far the second extraction agrees with the first:

  * pair F1: over (former_type, latter_type, dimension) keys, with the scores of
    repeated keys averaged, so a repeated judgment is one weighted entry rather
    than an overwritten one;
  * type-pair F1 and reversed directions: the same match collapsed to
    (former_type, latter_type), plus how many pairs appear in the opposite
    orientation in one of the two files;
  * score agreement: mean absolute difference and Pearson r on the matched keys;
  * dimension agreement: of the pairs that match on (former_type, latter_type),
    the share that carry the same dimension label.

Two uses. With a second extraction produced by another model, this is the
cross-model check. With a re-extraction by the same model, it measures extraction
stability. A file compared with itself is the self-test (`--self-test`).

The second extraction must come from the same pipeline: same prompt and same
json_object schema. The pipeline is not in this repository (only its rendered
prompt is in the supplement), so the second run has to be produced where the
pipeline lives and its output passed here.

Usage:
    python _cross_model_consistency.py --second path/to/second_extraction.jsonl
    python _cross_model_consistency.py --second energy_pairs.jsonl --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "energy_pairs.jsonl"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pair_scores(row):
    """{(former_type, latter_type, dimension): mean score} for one item."""
    acc = defaultdict(list)
    for p in row.get("comparison_pairs", []):
        f = (p.get("former_type") or "").strip().lower()
        l = (p.get("latter_type") or "").strip().lower()
        if not f or not l:
            continue
        d = p.get("dimension")
        try:
            s = float(p.get("score", 0.5))
        except (TypeError, ValueError):
            continue
        acc[(f, l, d)].append(s)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def f1(a, b):
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def compare(a, b):
    pa, pb = pair_scores(a), pair_scores(b)
    ka, kb = set(pa), set(pb)
    ta = {(f, l) for (f, l, _) in ka}
    tb = {(f, l) for (f, l, _) in kb}
    matched = ka & kb
    reversed_dir = sum(1 for (f, l, _) in kb if (l, f) not in ta and (l, f) in ta)
    # pairs whose type-level orientation is found only reversed
    reversed_dir = sum(1 for (l, f) in tb if (f, l) in ta and (l, f) not in ta)
    dim_ok = 0
    dim_checked = 0
    by_pair_a = defaultdict(set)
    by_pair_b = defaultdict(set)
    for (f, l, d) in ka:
        by_pair_a[(f, l)].add(d)
    for (f, l, d) in kb:
        by_pair_b[(f, l)].add(d)
    for key in ta & tb:
        dim_checked += 1
        if by_pair_a[key] & by_pair_b[key]:
            dim_ok += 1
    sa = np.array([pa[k] for k in matched])
    sb = np.array([pb[k] for k in matched])
    return {
        "keys_a": len(ka), "keys_b": len(kb), "keys_matched": len(matched),
        "pair_f1": f1(ka, kb), "type_pair_f1": f1(ta, tb),
        "reversed_type_pairs": reversed_dir,
        "dimension_agreement": (dim_ok / dim_checked) if dim_checked else None,
        "score_mae": float(np.mean(np.abs(sa - sb))) if len(matched) else None,
        "score_pearson_r": (float(np.corrcoef(sa, sb)[0, 1])
                            if len(matched) > 2 and sa.std() > 0 and sb.std() > 0 else None),
        "_sa": sa, "_sb": sb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--second", required=True)
    ap.add_argument("--sample", type=int, default=0, help="compare the first N common items")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load(args.base)
    second = load(args.second)
    by_a = {str(r.get("article_id")): r for r in base}
    by_b = {str(r.get("article_id")): r for r in second}
    common = [i for i in by_a if i in by_b]
    if args.sample:
        common = common[: args.sample]
    print(f"base items {len(base)}, second items {len(second)}, common {len(common)}")
    if not common:
        sys.exit("no common article_id between the two files")

    rows = [compare(by_a[i], by_b[i]) for i in common]
    n_a = sum(r["keys_a"] for r in rows)
    n_b = sum(r["keys_b"] for r in rows)
    matched = sum(r["keys_matched"] for r in rows)
    sa = np.concatenate([r["_sa"] for r in rows]) if matched else np.array([])
    sb = np.concatenate([r["_sb"] for r in rows]) if matched else np.array([])
    dim_checked = sum(1 for r in rows if r["dimension_agreement"] is not None)
    dim_agreed = sum(r["dimension_agreement"] for r in rows
                     if r["dimension_agreement"] is not None)
    out = {
        "base": str(args.base), "second": str(args.second),
        "items_compared": len(common),
        "keys_base": n_a, "keys_second": n_b, "keys_matched": matched,
        "pair_f1_micro": (2 * matched / (n_a + n_b)) if (n_a + n_b) else 1.0,
        "type_pair_f1_macro": float(np.mean([r["type_pair_f1"] for r in rows])),
        "reversed_type_pairs": sum(r["reversed_type_pairs"] for r in rows),
        "dimension_agreement": (dim_agreed / dim_checked) if dim_checked else None,
        "score_mae": float(np.mean(np.abs(sa - sb))) if matched else None,
        "score_pearson_r": (float(np.corrcoef(sa, sb)[0, 1])
                            if matched > 2 and sa.std() > 0 and sb.std() > 0 else None),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.self_test:
        ok = (matched == n_a == n_b and out["reversed_type_pairs"] == 0
              and (out["score_mae"] is None or out["score_mae"] == 0.0))
        print("SELF-TEST:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
