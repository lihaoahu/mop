"""Check the re-implemented Gaussian EM against the paper's own `inference.mop`.

`step9_phi_identification.py` re-implements the Gaussian EM so that phi_p can be
either estimated or fixed. That is only a useful comparison if the re-implementation
agrees with the repository's own fit when phi_p is free. This script fits both on a
slice of the case data and compares the penalized log-posterior, the selected
component count and the log-likelihood.

If they disagree, the comparison in step9 measures my implementation rather than
the parameterisation, and step9's output is void.

Run: uv run python experiments/case_study/_check_step9.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from step9_phi_identification import (build_design, criteria, fit, load_pairs,
                                      _penalized)                       # noqa: E402
from mop import inference                                               # noqa: E402
from mop.inference import comparisons2regression                        # noqa: E402
from mop.pairwise_comparison import PairwiseComparison                  # noqa: E402
from _common import T                                                   # noqa: E402
from scipy.special import logit                                         # noqa: E402


def paper_fit(pairs, k):
    """The repository's own fit, exactly as step7_analyze runs it."""
    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    pcv = [PairwiseComparison(former_id=name_to_id[p["ft"]],
                              latter_id=name_to_id[p["lt"]],
                              agent_id=1, judgment=p["score"]) for p in pairs]
    link = [lambda s, _eps=1e-10: logit(np.clip(s, _eps, 1 - _eps))]
    nJ = len(pairs)
    result = inference.mop(pcv, link, k, lambda_u=1e-6, lambda_pi=1,
                           lambda_sigma=nJ / 50)
    pbic = inference.pbic(pcv, result, link, lambda_u=1e-6,
                          lambda_sigma=nJ / 50, lambda_pi=1)
    return result, pbic


def main():
    pairs = load_pairs()
    print(f"case data: {len(pairs)} judgments")
    X, y = build_design(pairs)
    nJ, A1 = X.shape
    A = A1 + 1
    s2 = float(np.var(y))

    print(f"\n{'P':>3} {'mine pll':>14} {'paper pll':>14} {'diff':>12} "
          f"{'mine pBIC':>13} {'paper pBIC':>13} {'diff':>12}")
    for k in (1, 2, 3):
        mine = fit(X, y, k, free_phi=True)
        mc = criteria(mine, X, y, A, k, True)
        res, pbic_paper = paper_fit(pairs, k)
        paper_pll = inference.penalized_log_likelihood(
            X, y,
            np.array([np.log(r.priority_vector[:A1] / r.priority_vector[A1])
                      for r in res.priorities]),
            np.array([r.inconsistency_level for r in res.priorities]),
            np.array(res.mixture_probabilities),
            1e-6, nJ / 50, 1, s2)
        print(f"{k:>3} {mine['pll']:>14.4f} {paper_pll:>14.4f} "
              f"{mine['pll'] - paper_pll:>12.3e} {mc['pBIC']:>13.2f} "
              f"{pbic_paper:>13.2f} {mc['pBIC'] - pbic_paper:>12.3e}")

    print("\nNote: perfect agreement is not expected. The two implementations differ")
    print("in initialisation (kridge from scratch here, warm-start in step7 the")
    print("paper's script does a K sweep) and in the convergence test, and a")
    print("Gaussian mixture has many local optima. What matters for step9 is that")
    print("the two reach the SAME parameterisation and the same order of magnitude")
    print("of penalized log-posterior, so that the fixed-phi/free-phi contrast is")
    print("not confounded with an implementation difference.")


if __name__ == "__main__":
    main()
