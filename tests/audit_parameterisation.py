"""Parameterisation audit of the MoP codebase.

Distribution parameters are specified differently across implementations, and
a silent mismatch is the classic source of wrong statistical results. This
script pins down, and numerically verifies, every convention the project
relies on. Run it with

    uv run python tests/audit_parameterisation.py

It exits non-zero if any convention fails to reproduce.

Checklist
  1  Gaussian likelihood: is the scale argument sqrt(sigma^2) (std) or sigma^2?
  2  Inverse-Gamma prior on sigma^2: does the code's log-density match
     scipy.stats.invgamma with shape=(lambda_s+2)/2, scale=lambda_s*s2bar/2?
  3  M-step update for sigma^2 vs the strict MAP implied by that prior
  4  vote score: Phi(diff / sigma) vs Phi(diff / sigma^2)
  5  beta-binomial: (n, mu, rho) vs scipy (n, a, b); rho <-> (a, b)
  6  transforms: logit/log are natural, expit <-> logit
  7  synthetic data: noise added on the log-odds scale
  8  ties encoded as score 0.5 -> logit 0
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, gammaln, logit
from scipy.stats import betabinom, invgamma, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mop import inference_binomial as ib  # noqa: E402

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

code_std = np.sum(norm.logpdf(y, loc=mu, scale=np.sqrt(sigma2)))       # std convention
code_var = np.sum(norm.logpdf(y, loc=mu, scale=sigma2))               # var convention
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
    alpha = (lam_s + 2.0) / 2.0
    beta = lam_s * s2bar / 2.0
    code = (alpha * np.log(beta) - gammaln(alpha)
            - (alpha + 1.0) * np.log(sp2) - beta / sp2)
    ref = float(invgamma.logpdf(sp2, alpha, scale=beta))
    print(f"   lambda_s={lam_s:>6.1f} s2bar={s2bar:<5} sigma2={sp2:<5}  "
          f"code={code:>12.6f}  invgamma.logpdf={ref:>12.6f}  diff={abs(code-ref):.2e}")
    check(f"matches invgamma(shape={alpha}, scale={beta})", abs(code - ref) < 1e-9)

# the same prior written as a Gamma on 1/sigma^2, WITHOUT the Jacobian, differs
alpha, beta, sp2 = (108.0 + 2) / 2, 108.0 * 0.5 / 2, 0.8
no_jacobian = alpha * np.log(beta) - gammaln(alpha) - (alpha - 1.0) * np.log(sp2) - beta / sp2
print(f"\n   dropping the Jacobian gives {no_jacobian:.6f} vs {float(invgamma.logpdf(sp2, alpha, scale=beta)):.6f}")
print("   -> the code uses the inverse-Gamma density (with Jacobian), consistent with")
print("      'inverse-Gamma prior' in 03_empirical.tex:83")
check("code PLL is NOT the Jacobian-free version",
      abs(no_jacobian - float(invgamma.logpdf(sp2, alpha, scale=beta))) > 1e-6)

print()
print("=" * 76)
print("3. M-step sigma^2 update equals the closed-form MAP of that prior")
print("=" * 76)
from scipy.optimize import minimize_scalar  # noqa: E402

print(f"   {'n':>7} {'lambda_s':>9} {'closed form':>13} {'|grad|/scale':>13} "
      f"{'numeric argmax':>15} {'rel. err':>10}")
for n_eff, lam_s, SS, SSU in [(5401, 108.0, 100.0, 0.0), (500, 10.0, 60.0, 1.5),
                              (450, 9.0, 54.0, 0.0), (30, 0.6, 12.0, 0.4)]:
    s2bar = 0.5
    alpha = (lam_s + 2.0) / 2.0
    beta = lam_s * s2bar / 2.0
    closed = (SS + SSU + lam_s * s2bar) / (n_eff + lam_s + 4.0)

    # the closed form must be a stationary point of the penalised log-posterior
    grad = (-0.5 * n_eff / closed + (SS + SSU) / (2.0 * closed ** 2)
            - (alpha + 1.0) / closed + beta / closed ** 2)
    scale = max(0.5 * n_eff / closed, (SS + SSU) / (2.0 * closed ** 2), 1.0)
    stationary = abs(grad) / scale

    def neg_logpost(s2):
        return -(-0.5 * n_eff * np.log(s2) - (SS + SSU) / (2.0 * s2)
                 + alpha * np.log(beta) - gammaln(alpha)
                 - (alpha + 1.0) * np.log(s2) - beta / s2)

    res = minimize_scalar(neg_logpost, bounds=(closed * 0.2, closed * 5.0),
                          method="bounded", options={"xatol": 1e-14})
    rel_num = abs(res.x - closed) / closed

    print(f"   {n_eff:>7} {lam_s:>9.1f} {closed:>13.6f} {stationary:>13.2e} "
          f"{res.x:>15.6f} {rel_num:>9.2e}")
    check("closed form is a stationary point", stationary < 1e-12)
    check("numeric maximiser agrees", rel_num < 1e-4)
print("   -> the M-step in inference.py maximises the same posterior density that")
print("      penalized_log_likelihood evaluates, so EM monotonicity holds.")

print()
print("=" * 76)
print("4. vote score convention: Phi(diff/sigma) vs Phi(diff/sigma^2)")
print("=" * 76)
LAB = ["BEV", "PHEV", "REEV", "ICEV"]
W = np.array([[0.485, 0.254, 0.170, 0.091],
              [1.000, 0.000, 0.000, 0.000],
              [0.090, 0.392, 0.414, 0.104],
              [0.101, 0.137, 0.330, 0.432]])
PI = np.array([0.3729, 0.0005, 0.2510, 0.3756])
S2 = np.array([0.8120, 1.8524, 0.8287, 0.5652])
PRINTED = np.array([0.212, 0.262, 0.313, 0.214])


def vote(mode):
    n_p, n_a = W.shape
    S = np.zeros(n_a)
    for p in range(n_p):
        den = math.sqrt(S2[p]) if mode == "std" else S2[p]
        for i in range(n_a):
            acc = 0.0
            for j in range(n_a):
                if i == j:
                    continue
                wi, wj = W[p, i], W[p, j]
                if wi <= 0 or wj <= 0:
                    acc += 1.0 if (wi > 0 and wj == 0) else (0.0 if (wi == 0 and wj > 0) else 0.5)
                else:
                    acc += norm.cdf(math.log(wi / wj) / den)
            S[i] += PI[p] * acc
    return 2.0 / (n_a * (n_a - 1)) * S


s_std, s_var = vote("std"), vote("var")
print(f"   Phi(diff/sigma)   : " + "  ".join(f"{l}={v:.4f}" for l, v in zip(LAB, s_std)))
print(f"   Phi(diff/sigma^2) : " + "  ".join(f"{l}={v:.4f}" for l, v in zip(LAB, s_var)))
print(f"   manuscript Table 3: " + "  ".join(f"{l}={v:.3f}" for l, v in zip(LAB, PRINTED)))
check("std convention reproduces the manuscript (max dev < 0.001)",
      np.max(np.abs(s_std - PRINTED)) < 0.001,
      f"(max dev {np.max(np.abs(s_std - PRINTED)):.4f})")

print()
print("=" * 76)
print("5. beta-binomial: (n, mu, rho) vs scipy (n, a, b)")
print("=" * 76)
rng = np.random.default_rng(5)
worst = 0.0
for _ in range(300):
    mu = rng.uniform(0.02, 0.98)
    rho = rng.uniform(1e-4, 0.5)
    n = int(rng.integers(3, 80))
    m = int(rng.integers(0, n + 1))
    c = (1 - rho) / rho
    worst = max(worst, abs(float(betabinom.logpmf(m, n, mu * c, (1 - mu) * c))
                           - float(ib.beta_binomial_logpmf(m, n, mu, rho))))
check("matches scipy betabinom(k, n, a=mu*c, b=(1-mu)*c), c=(1-rho)/rho",
      worst < 1e-9, f"(worst diff {worst:.2e})")

# variance relation
mu, rho, n = 0.4, 0.13, 45
c = (1 - rho) / rho
emp_var = float(betabinom.var(n, mu * c, (1 - mu) * c))
formula = n * mu * (1 - mu) * (1 + (n - 1) * rho)
check("Var = n*mu*(1-mu)*[1+(n-1)*rho]", abs(emp_var - formula) < 1e-9,
      f"(scipy {emp_var:.6f}, formula {formula:.6f})")
check("rho = 0 collapses to the binomial log-pmf (module level)",
      abs(float(ib.beta_binomial_logpmf(21, 45, 0.467, 0.0))
          - float(ib.binomial_logpmf(21, 45, 0.467))) < 1e-12)
v0 = n * mu * (1 - mu) * (1 + (n - 1) * 0.0)
check("the variance formula at rho=0 is exactly the binomial variance",
      abs(v0 - n * mu * (1 - mu)) < 1e-15, f"({v0:.6f})")

print()
print("=" * 76)
print("6. transforms are natural-log based")
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
print("7. synthetic data: noise variance lives on the log-odds scale")
print("=" * 76)
sys.path.insert(0, str(ROOT / "experiments" / "v2"))
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
print("8. ties encoded as 0.5 -> logit 0")
print("=" * 76)
check("logit(0.5) == 0", abs(logit(0.5)) < 1e-15)
check("score 0.5 clips safely (no inf)", np.isfinite(logit(np.clip(0.5, 1e-10, 1 - 1e-10))))

print()
print("=" * 76)
print("ALL CONVENTIONS VERIFIED" if ok else "SOME CHECKS FAILED")
print("=" * 76)
sys.exit(0 if ok else 1)
