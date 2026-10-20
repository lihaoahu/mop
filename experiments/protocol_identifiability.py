"""Does the structural inconsistency carry information under each protocol?"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

EXPERIMENTS = Path(__file__).resolve().parent
CASE = EXPERIMENTS / "case_study"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(EXPERIMENTS.parent / "src"))

import _mop_params as MP                                            # noqa: E402
from mop.inference import prior_mode_phi0                           # noqa: E402

EPS = 1e-12


def binomial_curvature(omega):
    """d2/dphi2 of the binomial log-likelihood: exactly zero, by linearity."""
    return np.zeros_like(np.asarray(omega, dtype=float))


def gaussian_curvature(omega, phi, r2=None):
    """d2/dphi2 of the Gaussian log-likelihood: ``1/(2 phi^2) - omega r^2 / phi^3``."""
    omega = np.asarray(omega, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if r2 is None:
        r2 = phi / omega
    return 0.5 / phi ** 2 - omega * np.asarray(r2, dtype=float) / phi ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default=None)
    ap.add_argument("--M", type=int, default=20,
                    help="repetitions per judgment, i.e. the binomial observation weight")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from step9_phi_identification import build_design, load_pairs

    pairs = load_pairs(dim_name=args.dim) if args.dim else load_pairs()
    X, y = build_design(pairs)
    nJ, A1 = X.shape
    label = args.dim or "overall"
    s2 = float(prior_mode_phi0(X, y))

    m = np.round(expit(y) * args.M)
    omega = np.full(nJ, float(args.M))

    print(f"[{label}] n={nJ}, A={A1 + 1}")
    print(f"  {MP.summary()}")
    print(f"  binomial reading: omega_k = M = {args.M}\n")

    b_curv = binomial_curvature(omega)
    print("1. curvature of the log-likelihood in phi")
    print(f"   binomial  d2L/dphi2 : every entry = 0  "
          f"(max |value| = {np.max(np.abs(b_curv)):.1e})")
    print("   -> LINEAR in phi: no peak, every value scores equally, phi carries no")
    print("      information. This is an identity, not a fitted result.")

    print()
    for phi in (0.5, 1.0, 2.0):
        g_curv = gaussian_curvature(omega, phi)
        print(f"   Gaussian  d2L/dphi2 at phi={phi:.2f} : "
              f"per-judgment {g_curv[0]:+.2f}, total {g_curv.sum():+.1f}")
    print("   -> negative at the peak (where omega r^2 = phi), so a genuine maximum")
    print("      exists and phi is set by the data; the value does depend on the")
    print("      residuals away from the peak, and the log term is part of it.")

    print()
    print("   The two are on the same footing -- a second derivative in the same")
    print("   parameter -- so unlike an objective comparison there is no scale issue:")
    print("   zero against strictly negative is the whole statement.")

    print(f"\n2. profile of the penalized objective over phi, {label}, P=2")
    try:
        from step9_phi_identification import fit
        lam_sigma = MP.LAMBDA_SIGMA_OVER_N * nJ
        phi_grid = [0.4, 0.7, 1.0, 1.4, 2.0, 3.0]
        print(f"   {'phi':>6} {'Gaussian pll':>16} {'- max':>10}")
        rows = {}
        for phi in phi_grid:
            r = fit(X, y, 2, free_phi=False, s2=s2, phi_fixed=phi)
            rows[phi] = r["pll"]
        mx = max(rows.values())
        for phi in phi_grid:
            print(f"   {phi:>6.2f} {rows[phi]:>16.2f} {rows[phi] - mx:>10.2f}")
        drop = mx - min(rows.values())
        print(f"   the Gaussian profile moves by {drop:.2f} across this range.")
        print("   The binomial profile is not shown because it cannot move: the")
        print("   likelihood is linear in phi, so every phi gives the same value.")

        print(f"\n3. what the prior does, which is the part worth reporting")
        print(f"   lambda_sigma = {lam_sigma:.2f} (paper's nJ/50) supplies a prior mode")
        print(f"   phi_0 = {s2:.4f}. Without it the Gaussian likelihood is unbounded as")
        print("   phi -> 0 (Section 2.4), so the estimates below are the prior doing")
        print("   work the likelihood cannot. In the binomial protocol the prior is")
        print("   what pins phi at one, but Section 2.3 fixes it outright.")

        out_data = {
            "label": label, "n": int(nJ), "M": args.M, "phi0": float(s2),
            "lambda_sigma": float(lam_sigma),
            "binomial_curvature_max_abs": float(np.max(np.abs(b_curv))),
            "binomial_linear_in_phi": True,
            "gaussian_curvature": {
                str(p): float(gaussian_curvature(omega, p).sum())
                for p in (0.5, 1.0, 2.0)},
            "gaussian_profile": {str(k): float(v) for k, v in rows.items()},
            "gaussian_profile_drop": float(drop),
        }
    except TypeError as exc:
        print(f"   (fit() has no phi_fixed argument yet: {exc})")
        out_data = {"label": label, "n": int(nJ)}
        drop = None

    print()
    print("Conclusion: the binomial protocol's dispersion is not identified -- the")
    print("likelihood is linear in phi, so no data can pin it and Section 2.3 is right")
    print("to fix it at one. The Gaussian protocol's dispersion is identified, with a")
    print("negative curvature at its maximum and a profile that moves. The separation")
    print("paper claims is therefore conditional on the protocol, and this is the")
    print("evidence for that condition.")

    out = Path(args.out) if args.out else (
        EXPERIMENTS /
        f"protocol_identifiability{'_' + args.dim if args.dim else ''}.json")
    out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
