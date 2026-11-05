"""Free versus pinned dispersion, fitted by two independent implementations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.em import fit_gaussian_mixture                       # noqa: E402
from common.em_crosscheck import fit as fit_manual                  # noqa: E402
from common.params import (                                     # noqa: E402
    LAMBDA_SIGMA_OVER_N,
    lambda_u,
    lambda_pi,
)
from common.paths import result                                 # noqa: E402
from mop.criteria import (                                      # noqa: E402
    bic,
    degrees_of_freedom,
    lbic,
    log_prior_utilities,
    pbic,
)
from mop.general_fit import GaussianResponse                     # noqa: E402
from mop.laplace import ln_det_hessian                           # noqa: E402
from mop.inference import prior_mode_phi0                       # noqa: E402

from case_study.estimator import as_pcvector, design, load_pairs  # noqa: E402

GRID = (1, 2, 3, 4, 5, 6)
TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(result("specification.json")))
    args = ap.parse_args()

    pcv = as_pcvector(load_pairs())
    X, y = design(pcv)
    A = X.shape[1] + 1
    s2 = float(prior_mode_phi0(X, y))
    lam_sigma = LAMBDA_SIGMA_OVER_N * float(len(y))
    kw = {"lambda_u": lambda_u, "lambda_pi": lambda_pi, "lambda_sigma": lam_sigma, "s2": s2}

    out = {"design": {"n_judgments": int(len(y)), "A": int(A), "phi0": s2,
                      "grid": list(GRID),
                      "df_rule": "(A-1)K + (K-1) + K*[phi estimated]",
                      "inverse_gamma_counted_when_pinned": False},
           "specs": {}}

    for tag, free in (("free", True), ("pinned", False)):
        rows, worst = {}, 0.0
        for K in GRID:
            a = fit_manual(X, y, K, free, lam_u=lambda_u, lam_pi=lambda_pi,
                           lam_sigma=lam_sigma, s2=s2)
            b = fit_gaussian_mixture(X, y, K, phi_mode="free" if free else "one", **kw)
            worst = max(worst, abs(a["pll"] - b.pll))
            df = degrees_of_freedom(A, K, free)
            u_prior = log_prior_utilities(a["u"], a["phi"], lambda_u)
            ln_h = ln_det_hessian(
                X, y, a["u"], a["phi"], a["pi"],
                GaussianResponse() if free else GaussianResponse(phi_free=False),
                u_bar=None, lambda_u=lambda_u, lambda_pi=lambda_pi,
                lambda_sigma=(lam_sigma if free else None), s2=(s2 if free else None))
            rows[str(K)] = {"pll_a": a["pll"], "pll_b": float(b.pll),
                            "df": df, "lnH": ln_h,
                            "bic": bic(a["pll"], len(y), df),
                            "lbic": lbic(a["pll"], ln_h, df),
                            "pbic": pbic(a["pll"], len(y), df, u_prior),
                            "pbic_unprofiled": pbic(a["pll"], len(y), df),
                            "u_prior_term": -2.0 * u_prior,
                            "pi": a["pi"].tolist(), "phi": a["phi"].tolist()}
        best = min(rows, key=lambda k: rows[k]["bic"])
        out["specs"][tag] = {"curve": rows, "best_K": int(best),
                             "max_abs_pll_gap": worst}
        print(f"\n== dispersion {tag} (best K = {best}) ==")
        print("  K   pll_manual   pll_reference   df        BIC       LBIC")
        for K, r in rows.items():
            print(f"  {K}  {r['pll_a']:11.3f}   {r['pll_b']:11.3f}  {r['df']:3d}"
                  f"  {r['bic']:10.2f}  {r['lbic']:10.2f}")
        print(f"  max |pll gap| between implementations: {worst:.2e}")
        assert worst < TOL, f"{tag}: implementations disagree by {worst:.2e}"

    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
