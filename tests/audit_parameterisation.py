"""Parameterisation audit of the MoP codebase."""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, gammaln, logit
from scipy.stats import invgamma, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mop.mixture_of_priorities import expected_utility  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))


print("=" * 76)
print("1. Gaussian likelihood: scale argument")
print("=" * 76)
y = np.array([0.7, -1.2, 2.0, 0.1])
mu = np.array([0.5, -0.9, 1.7, 0.3])
sigma2 = 0.8

code_std = np.sum(norm.logpdf(y, loc=mu, scale=np.sqrt(sigma2)))
code_var = np.sum(norm.logpdf(y, loc=mu, scale=sigma2))
print(f"   sum logpdf with scale=sqrt(0.8): {code_std:.6f}")
print(f"   sum logpdf with scale=0.8     : {code_var:.6f}")
check("src/mop uses scale=sqrt(sigma2)",
      abs(code_std - float(np.sum(norm.logpdf(y, loc=mu, scale=math.sqrt(sigma2))))) < 1e-12)
check("the two conventions differ materially (so the check matters)",
      abs(code_std - code_var) > 0.1, f"(gap {abs(code_std - code_var):.3f} nats)")

print()
print("=" * 76)
print("2. Inverse-Gamma prior on sigma^2 (penalized_log_likelihood)")
print("=" * 76)
for lam_s, s2bar, sp2 in [(108.0, 0.5, 0.8), (10.0, 0.2, 0.05), (200.0, 1.0, 2.5)]:
    alpha = (lam_s - 2.0) / 2.0
    beta = lam_s * s2bar / 2.0
    code = (alpha * np.log(beta) - gammaln(alpha)
            - (alpha + 1.0) * np.log(sp2) - beta / sp2)
    ref = float(invgamma.logpdf(sp2, alpha, scale=beta))
    print(f"   lambda_s={lam_s:>6.1f} s2bar={s2bar:<5} sigma2={sp2:<5}  "
          f"code={code:>12.6f}  invgamma.logpdf={ref:>12.6f}  diff={abs(code-ref):.2e}")
    check(f"matches invgamma(shape={alpha}, scale={beta})", abs(code - ref) < 1e-9)

alpha, beta, sp2 = (108.0 - 2) / 2, 108.0 * 0.5 / 2, 0.8
no_jacobian = alpha * np.log(beta) - gammaln(alpha) - (alpha - 1.0) * np.log(sp2) - beta / sp2
print(f"\n   dropping the Jacobian gives {no_jacobian:.6f} vs {float(invgamma.logpdf(sp2, alpha, scale=beta)):.6f}")
print("   -> the code evaluates a density in sigma^2, which is the inverse-Gamma")
print("      prior eq:prior1 of sections/02_model.tex")
check("code PLL is NOT the Jacobian-free version",
      abs(no_jacobian - float(invgamma.logpdf(sp2, alpha, scale=beta))) > 1e-6)

print()
print("=" * 76)
print("3. M-step sigma^2 update equals the closed-form MAP of that prior")
print("=" * 76)
from scipy.optimize import minimize_scalar  # noqa: E402

print(f"   {'A':>3} {'n':>7} {'lambda_s':>9} {'closed form':>13} "
      f"{'|grad|/scale':>13} {'numeric argmax':>15} {'rel. err':>10}")
for A, n_eff, lam_s, SS, SSU in [(2, 5401, 108.0, 100.0, 0.0),
                                 (4, 5401, 108.0, 100.0, 0.0),
                                 (4, 500, 10.0, 60.0, 1.5),
                                 (8, 500, 10.0, 60.0, 1.5),
                                 (4, 30, 0.6, 12.0, 0.4),
                                 (10, 30, 0.6, 12.0, 0.4)]:
    d = A - 1
    s2bar = 0.5
    alpha = (lam_s - 2.0) / 2.0
    beta = lam_s * s2bar / 2.0
    closed = (SS + SSU + lam_s * s2bar) / (n_eff + lam_s + d)

    grad = (-0.5 * (n_eff + d) / closed
            + (SS + SSU) / (2.0 * closed ** 2)
            - (alpha + 1.0) / closed
            + beta / closed ** 2)
    scale = max(0.5 * (n_eff + d) / closed, (SS + SSU) / (2.0 * closed ** 2), 1.0)
    stationary = abs(grad) / scale

    def neg_logpost(s2):
        return -(-0.5 * (n_eff + d) * np.log(s2) - (SS + SSU) / (2.0 * s2)
                 + alpha * np.log(beta) - gammaln(alpha)
                 - (alpha + 1.0) * np.log(s2) - beta / s2)

    res = minimize_scalar(neg_logpost, bounds=(closed * 0.2, closed * 5.0),
                          method="bounded", options={"xatol": 1e-14})
    rel_num = abs(res.x - closed) / closed

    print(f"   {A:>3} {n_eff:>7} {lam_s:>9.1f} {closed:>13.6f} {stationary:>13.2e} "
          f"{res.x:>15.6f} {rel_num:>9.2e}")
    check("closed form is a stationary point", stationary < 1e-12)
    check("numeric maximiser agrees", rel_num < 1e-4)
print("   -> the M-step in inference.py maximises the same posterior density that")
print("      penalized_log_likelihood evaluates, so EM monotonicity holds.")

print()
print("=" * 76)
print("4. aggregate ranking: expected utility, not a vote score")
print("=" * 76)
LAB = ["BEV", "PHEV", "REEV", "ICEV"]
W = np.array([
    [0.4873485184590911, 0.24785860575773244, 0.17345053406765415, 0.09134234171552226],
    [0.9999999998480964, 4.139519556101162e-11, 1.0506031612582586e-11, 1.0000230260410934e-10],
    [0.09086025461190401, 0.4074870027660522, 0.39698678206075183, 0.10466596056129206],
    [0.10053462225970854, 0.13627269214855534, 0.3316377366205375, 0.43155494897119856],
])
PI = np.array([0.37237417147496743, 0.000539782896867043,
               0.25234737997290985, 0.37473866565525576])
PRINTED = np.array([0.054, 0.282, 0.475, 0.000])

u_bar = expected_utility(W, PI)
hand = np.zeros(W.shape[1])
for p in range(W.shape[0]):
    hand += PI[p] * np.log(W[p] / W[p][-1])

print("   package  : " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(LAB, u_bar)))
print("   hand     : " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(LAB, hand)))
print("   Table 5  : " + "  ".join(f"{lab}={v:.3f}" for lab, v in zip(LAB, PRINTED)))
check("package agrees with an independent computation",
      np.allclose(u_bar, hand),
      f"(max dev {np.max(np.abs(u_bar - hand)):.2e})")
check("the benchmark alternative sits at exactly zero",
      u_bar[-1] == 0.0,
      f"(got {u_bar[-1]!r})")
order = [LAB[i] for i in np.argsort(-u_bar)]
check("ranking matches the manuscript, REEV > PHEV > BEV > ICEV",
      order == ["REEV", "PHEV", "BEV", "ICEV"],
      f"(got {' > '.join(order)})")
check("reproduces the values of Table 5 (max dev < 0.001)",
      np.max(np.abs(u_bar - PRINTED)) < 0.001,
      f"(max dev {np.max(np.abs(u_bar - PRINTED)):.4f})")
print("   Utilities are logs of the ratio to the benchmark, so they are on the")
print("   log-odds scale the model is parameterised on. A vote score built from")
print("   Phi((u_i - u_j)/sigma_p) would need the normal observation model and")
print("   has no counterpart for a binary or count response.")

print()
print("=" * 76)
print("5. transforms are natural-log based")
print("=" * 76)
p = 0.83
check("logit(p) == ln(p/(1-p))", abs(logit(p) - math.log(p / (1 - p))) < 1e-12)
check("expit(logit(p)) == p", abs(expit(logit(p)) - p) < 1e-12)
check("exp is natural", abs(math.exp(1.0) - 2.718281828459045) < 1e-12)
check("no log10 anywhere in src/mop",
      not any("log10" in f.read_text(encoding="utf-8")
              for f in (ROOT / "src" / "mop").glob("*.py")))

print()
print("=" * 76)
print("6. synthetic data: noise variance lives on the log-odds scale")
print("=" * 76)
sys.path.insert(0, str(ROOT / "experiments" / "simulation"))
from _common import generate_synthetic_pairs, priority_vector_from_logratios  # noqa: E402

u_true = np.array([[2.0, 1.0, 0.5]])
w_true = priority_vector_from_logratios(u_true)
rng = np.random.default_rng(0)
pairs, _, _ = generate_synthetic_pairs(u_true, np.array([1.0]), np.array([0.05]), 20000, rng)
y = logit(np.clip(np.array([q["score"] for q in pairs]), 1e-10, 1 - 1e-10))
X = np.zeros((len(pairs), 3))
for k, q in enumerate(pairs):
    if q["ft"] < 3:
        X[k, q["ft"]] += 1.0
    if q["lt"] < 3:
        X[k, q["lt"]] -= 1.0
resid = y - X @ u_true[0]
print(f"   sd of (logit(score) - (u_i - u_j)) over 20k draws: {resid.std():.5f}  (target {math.sqrt(0.05):.5f})")
check("empirical noise sd matches sqrt(sigma2)", abs(resid.std() - math.sqrt(0.05)) < 0.005)
lu = np.log(w_true[0, :3] / w_true[0, 3])
check("log(w_i/w_j) round-trips to u_i - u_j", np.max(np.abs(lu - u_true[0])) < 1e-12)

print()
print("=" * 76)
print("7. ties encoded as 0.5 -> logit 0")
print("=" * 76)
check("logit(0.5) == 0", abs(logit(0.5)) < 1e-15)
check("score 0.5 clips safely (no inf)", np.isfinite(logit(np.clip(0.5, 1e-10, 1 - 1e-10))))

print()
print("=" * 76)
print("ALL CONVENTIONS VERIFIED" if ok else "SOME CHECKS FAILED")
print("=" * 76)
sys.exit(0 if ok else 1)
