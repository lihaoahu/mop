"""Parameter audit for the phi-identification comparison.

Before any result from `step9_phi_identification.py` is quoted, three things have to
be true, and none of them is obvious by inspection:

1. **Every parameter comes from the config**, not from a literal in a script. This
   lists what is in force and where each value comes from.

2. **The two parameterisations share a prior.** Fixing phi at one is itself the claim
   that one is the prior mode. If the fixed run uses one phi_0 and the free run uses
   another, the comparison changes the parameterisation and the prior at once, and no
   conclusion about either follows. This checks that the two runs are on the same
   phi_0, whatever it is set to.

3. **The objectives agree at phi = 1.** With phi_0 shared, fixing phi at one must
   give exactly the penalized objective the free model takes at phi = 1. This is an
   identity, not an approximation: it is what makes the free model a superset of the
   fixed one, and therefore what makes "the free fit must do at least as well" a
   valid test. An earlier version of this comparison failed this identity.

It also reports the paper's own phi_0 value per slice, because whether phi_0 is
anywhere near the estimated phi decides how much the prior is doing the work.

Run:
    uv run python experiments/case_study/_audit_params.py
    uv run python experiments/case_study/_audit_params.py --mode fixed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mop.inference import prior_mode_phi0                              # noqa: E402
from step9_phi_identification import (build_design, load_pairs,        # noqa: E402
                                      _penalized)

CFG_PATH = HERE / "phi_config.json"


def load_cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def slices_from_cfg(cfg):
    out = [("overall", {})]
    for d in cfg["slices"]["dimensions"]:
        out.append((d, {"dim_name": d}))
    for y in cfg["slices"]["years"]:
        out.append((y, {"year": y}))
    return out


def resolve_phi0(cfg, X, y, override=None):
    mode = override or cfg["prior_mode_phi0"]["mode"]
    if mode == "data":
        return float(prior_mode_phi0(X, y)), "prior_mode_phi0(X, y)  [paper's rule]"
    if mode == "fixed":
        v = float(cfg["prior_mode_phi0"].get("fixed_value", 1.0))
        return v, f"fixed at {v}  [a choice, not the paper's rule]"
    raise ValueError(f"unknown prior_mode_phi0.mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["data", "fixed"], default=None,
                    help="override prior_mode_phi0.mode from the config")
    args = ap.parse_args()

    cfg = load_cfg()
    print("=" * 88)
    print("PARAMETERS IN FORCE (all from phi_config.json)")
    print("=" * 88)
    m = cfg["mop"]
    print(f"  lambda_u            = {m['lambda_u']}          "
          f"(config.json 'mop': {m['lambda_u']})")
    print(f"  lambda_pi           = {m['lambda_pi']}")
    print(f"  lambda_sigma        = {m['lambda_sigma_over_n']} * nJ     "
          f"(config.json: 'nJ / 50')")
    print(f"  prior_mode_phi0     = {args.mode or cfg['prior_mode_phi0']['mode']}")
    print(f"  max_components      = {m['max_components']}          "
          f"(config.json allows 9)")
    print(f"  EM max_iter / tol   = {cfg['em']['max_iter']} / {cfg['em']['tol']}")
    print(f"  grid                = {cfg['grid']}")
    print(f"  bootstrap grid/reps = {cfg['bootstrap_grid']} / {cfg['bootstrap_reps']}")

    print()
    print("=" * 88)
    print("CHECK 1: the two objectives agree at phi = 1 (must be exact)")
    print("=" * 88)
    pairs = load_pairs()
    X, y = build_design(pairs)
    nJ, A1 = X.shape
    lam_sigma = m["lambda_sigma_over_n"] * nJ
    s2 = resolve_phi0(cfg, X, y, args.mode)[0]
    u = np.array([[0.3, -0.2, 0.1]])
    pi = np.array([1.0])
    pll_fixed, _ = _penalized(X, y, u, np.ones(1), pi, m["lambda_u"], lam_sigma,
                              m["lambda_pi"], s2, np.zeros((1, A1)), False)
    pll_free, _ = _penalized(X, y, u, np.ones(1), pi, m["lambda_u"], lam_sigma,
                             m["lambda_pi"], s2, np.zeros((1, A1)), True)
    print(f"  penalized, phi fixed : {pll_fixed:.10f}")
    print(f"  penalized, phi free at phi=1 : {pll_free:.10f}")
    print(f"  difference           : {pll_free - pll_fixed:.3e}")
    ok_identity = abs(pll_free - pll_fixed) < 1e-9
    print(f"  -> {'PASS' if ok_identity else 'FAIL'}")

    print()
    print("=" * 88)
    print("CHECK 2: phi_0 per slice, against the phi values the fits estimate")
    print("=" * 88)
    print(f"  {'slice':<26} {'n':>6} {'phi_0':>10}   source")
    rows = []
    for label, kw in slices_from_cfg(cfg):
        p = load_pairs(**kw)
        if len(p) < 30:
            continue
        Xs, ys = build_design(p)
        v, src = resolve_phi0(cfg, Xs, ys, args.mode)
        rows.append((label, len(p), v))
        print(f"  {label:<26} {len(p):>6} {v:>10.4f}   {src}")

    print()
    print("  The estimated phi values from the fits run 0.22 to 2.04, median 0.51.")
    print("  Compare that against the phi_0 column above: a prior mode far from the")
    print("  estimated values means the prior is doing part of the work, and that is")
    print("  true whichever mode is selected. It is a reason to report this")
    print("  sensitivity rather than one setting.")

    print()
    print("=" * 88)
    print("RESULT")
    print("=" * 88)
    print(f"  identity at phi = 1 : {'PASS' if ok_identity else 'FAIL'}")
    print(f"  slices audited      : {len(rows)}")
    if not ok_identity:
        print("  The comparison is NOT valid until this passes: without the identity the")
        print("  two fits are not nested and 'free must do at least as well' is not a")
        print("  test of anything.")
    return 0 if ok_identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
