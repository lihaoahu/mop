"""Cross-check the two estimation routes against each other."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop import inference                                              # noqa: E402
from mop.general_fit import BinomialResponse, GaussianResponse, fit_map, pbic_map  # noqa: E402
from mop.pairwise_comparison import PairwiseComparison, comparisons2regression  # noqa: E402
from mop.winloss import fit_winloss, pbic as pbic_winloss             # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}{('  ' + detail) if detail else ''}")


def _pairs(rng, A=4, n_per_pair=70, scale=0.5):
    pc = []
    u_true = np.array([[1.2, 0.5, -0.6], [-1.2, -0.5, 0.6]])
    for i in range(A):
        for j in range(i + 1, A):
            row = np.zeros(A - 1)
            if i < A - 1:
                row[i] += 1
            if j < A - 1:
                row[j] -= 1
            for _ in range(n_per_pair):
                state = rng.choice(2)
                pc.append(PairwiseComparison(i + 1, j + 1, 1,
                                             row @ u_true[state] + rng.normal(0, scale)))
    return pc


def _u(model):
    return np.array([np.log(np.asarray(model[p].priority_vector[:-1])
                            / np.asarray(model[p].priority_vector[-1]))
                     for p in range(len(model))])


def gaussian_agreement():
    rng = np.random.default_rng(0)
    pc = _pairs(rng)
    X, y, _ = comparisons2regression(pc, [float])
    nJ = len(y)
    lam_s = nJ / 50.0
    s2 = float(inference.prior_mode_phi0(X, y))

    em = inference.mop(pc, [float], 2, lambda_u=1e-6, lambda_sigma=lam_s, lambda_pi=1.0, s2=s2)
    u_em = _u(em)
    sig_em = np.array([em[p].inconsistency_level for p in range(2)])
    pll_em = inference.penalized_log_likelihood(
        X, y, u_em, sig_em, np.asarray(em.mixture_probabilities), 1e-6, lam_s, 1.0, s2)

    gm = fit_map(pc, GaussianResponse(), 2, lambda_u=1e-6, lambda_sigma=lam_s, s2=s2)
    check("Gaussian: general fitter reaches the EM objective",
          abs(gm.penalized_log_posterior - pll_em) < 1e-2,
          f"general {gm.penalized_log_posterior:.4f} vs EM {pll_em:.4f}")
    check("Gaussian: utilities agree to 0.05",
          np.max(np.abs(np.sort(_u(gm), axis=0) - np.sort(u_em, axis=0))) < 0.05)
    check("Gaussian: dispersions agree to 0.05",
          np.max(np.abs(np.sort([gm[p].inconsistency_level for p in range(2)])
                        - np.sort(sig_em))) < 0.05)


def binomial_agreement():
    rng = np.random.default_rng(1)
    A, trials = 4, 30
    u_true = np.array([[1.5, 0.75, -0.75], [-1.5, -0.75, 0.75]])
    from scipy.special import expit
    pc, y = [], []
    for i in range(A):
        for j in range(i + 1, A):
            row = np.zeros(A - 1)
            if i < A - 1:
                row[i] += 1
            if j < A - 1:
                row[j] -= 1
            for _ in range(40):
                state = rng.choice(2)
                m = int(rng.binomial(trials, expit(row @ u_true[state])))
                pc.append(PairwiseComparison(i + 1, j + 1, 1, m / trials))
                y.append(m / trials)
    n = np.full(len(pc), float(trials))

    em = fit_winloss(pc, 2, omega=n)
    gm = fit_map(pc, BinomialResponse(), 2, omega=n)
    check("Binomial: general fitter reaches the EM objective",
          abs(gm.penalized_log_posterior - em.penalized_log_posterior) < 1e-2,
          f"general {gm.penalized_log_posterior:.4f} vs EM {em.penalized_log_posterior:.4f}")
    check("Binomial: criterion agrees",
          abs(pbic_map(pc, gm, BinomialResponse(), omega=n)
              - pbic_winloss(pc, em, omega=n)) < 1.0)


def main():
    gaussian_agreement()
    binomial_agreement()
    failed = [r for r in results if not r[1]]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED")
        return 1
    print(f"all {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
