"""Check that the free-phi objective dominates the fixed-phi one, at equal P.

Once phi_0 is the same for both, fixing phi at one is the restriction of the free
model to a face of its parameter space. The penalized objective then satisfies

    penalized(free, phi=1) == penalized(fixed)

exactly, so a correct maximiser of the free objective must reach a value at least as
high as the fixed one. If it does not, the free-phi fit is at a worse local optimum
and the comparison is measuring the optimiser rather than the parameterisation.

This was not the case in an earlier version, where the fixed and free runs also used
different phi_0, and the resulting "freeing phi changes the component count" was
partly an artefact of that. This script tests the property directly, from a common
starting point, at each component count.

Run: uv run python experiments/case_study/_check_nesting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mop.initialization import kridge                                    # noqa: E402
from mop.inference import prior_mode_phi0                                # noqa: E402
from step9_phi_identification import (build_design, criteria, fit,        # noqa: E402
                                      _penalized, load_pairs)

# phi_0 follows the paper's rule by default. Pass --s2-fixed 1.0 to pin it.
S2 = None


def resolve_s2(X, y, override):
    return float(prior_mode_phi0(X, y)) if override is None else float(override)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-fixed", type=float, default=None,
                    help="pin phi_0 instead of using the paper's rule")
    args = ap.parse_args()

    for dim, grid in ((None, (1, 2, 3, 4, 5)), ("policy", (1, 2, 3)),
                      ("cost", (1, 2, 3))):
        pairs = load_pairs(dim_name=dim) if dim else load_pairs()
        X, y = build_design(pairs)
        nJ, A1 = X.shape
        A = A1 + 1
        s2 = resolve_s2(X, y, args.s2_fixed)
        label = dim or "overall"
        print("=" * 96)
        print(f"[{label}] n={nJ}, phi_0={s2:.4f} "
              f"({'pinned' if args.s2_fixed is not None else 'prior_mode_phi0'}), "
              f"both fits from the same kridge start")
        print(f"{'P':>3} {'fixed pll':>13} {'free pll':>13} {'free-fixed':>12} "
              f"{'ok?':>5} | {'fixed pBIC':>11} {'free pBIC':>11}")

        for k in grid:
            u0, _, _ = kridge(X, y, k)
            rf = fit(X, y, k, free_phi=False, s2=s2, u_init=u0)
            rg = fit(X, y, k, free_phi=True, s2=s2, u_init=u0)
            cf = criteria(rf, X, y, A, k, False)
            cg = criteria(rg, X, y, A, k, True)
            diff = rg["pll"] - rf["pll"]
            ok = "yes" if diff > -1e-6 else "NO"
            print(f"{k:>3} {rf['pll']:>13.4f} {rg['pll']:>13.4f} {diff:>12.4f} "
                  f"{ok:>5} | {cf['pBIC']:>11.2f} {cg['pBIC']:>11.2f}")

        print()
        print("free-fixed is the gain in the penalized objective from freeing phi,")
        print("from an identical start. It must be >= 0. A negative entry means the")
        print("free fit landed below the fixed one and is not comparable.")


if __name__ == "__main__":
    main()
