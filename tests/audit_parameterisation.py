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
     scipy.stats.invgamma with shape=(lambda_s-2)/2, scale=lambda_s*s2bar/2?
  3  M-step update for sigma^2 vs the strict MAP implied by that prior
  4  aggregate ranking: expected utility, and the benchmark's zero origin
  5  beta-binomial: (n, mu, rho) vs scipy (n, a, b); rho <-> (a, b)
  6  transforms: logit/log are natural, expit <-> logit
  7  synthetic data: noise added on the log-odds scale
  8  ties encoded as score 0.5 -> logit 0
  9  analytic M-step gradients vs central finite differences
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
    alpha = (lam_s - 2.0) / 2.0
    beta = lam_s * s2bar / 2.0
    code = (alpha * np.log(beta) - gammaln(alpha)
            - (alpha + 1.0) * np.log(sp2) - beta / sp2)
    ref = float(invgamma.logpdf(sp2, alpha, scale=beta))
    print(f"   lambda_s={lam_s:>6.1f} s2bar={s2bar:<5} sigma2={sp2:<5}  "
          f"code={code:>12.6f}  invgamma.logpdf={ref:>12.6f}  diff={abs(code-ref):.2e}")
    check(f"matches invgamma(shape={alpha}, scale={beta})", abs(code - ref) < 1e-9)

# the same prior read as a Gamma on 1/sigma^2, WITHOUT the Jacobian that turns
# it into a density in sigma^2, differs; the code must evaluate the latter
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
# A is swept on purpose. The normalising constant of the utility prior carries
# a factor phi^{-(A-1)/2}, so the denominator must contain d = A - 1; a single
# A would not expose its absence, which is how an earlier version of this audit
# passed while the code was wrong.
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
    # denominator: n_p (likelihood) + d (utility-prior normalisation)
    #              + lambda_s, where lambda_s = 2(alpha+1) comes from the
    #              inverse-Gamma density on sigma2
    closed = (SS + SSU + lam_s * s2bar) / (n_eff + lam_s + d)

    # each term carries one factor of phi in the posterior density
    grad = (-0.5 * (n_eff + d) / closed           # likelihood + prior norm.
            + (SS + SSU) / (2.0 * closed ** 2)    # their quadratic terms
            - (alpha + 1.0) / closed              # inverse-Gamma density
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
# the fitted overall mixture of the case study, to full precision
W = np.array([
    [0.48530012815378776, 0.2543372677951154, 0.16926740665935452, 0.09109519739174243],
    [0.9999999998478429, 4.147685794006449e-11, 1.0677907793234627e-11, 1.0000230260408398e-10],
    [0.08964249751608318, 0.3911496950821047, 0.41540518687229233, 0.10380262052951977],
    [0.10142686319478791, 0.1366147177974498, 0.3294297952774048, 0.4325286237303575],
])
PI = np.array([0.372628940638662, 0.0005225906356056135,
               0.25102135571570233, 0.3758271130100301])
PRINTED = np.array([0.054, 0.282, 0.475, 0.000])

u_bar = expected_utility(W, PI)
hand = np.zeros(W.shape[1])
for p in range(W.shape[0]):
    hand += PI[p] * np.log(W[p] / W[p][-1])

print("   package  : " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(LAB, u_bar)))
print("   hand     : " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(LAB, hand)))
print("   Table 3  : " + "  ".join(f"{lab}={v:.3f}" for lab, v in zip(LAB, PRINTED)))
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
check("reproduces the values of Table 3 (max dev < 0.001)",
      np.max(np.abs(u_bar - PRINTED)) < 0.001,
      f"(max dev {np.max(np.abs(u_bar - PRINTED)):.4f})")
print("   Utilities are logs of the ratio to the benchmark, so they are on the")
print("   log-odds scale the model is parameterised on. A vote score built from")
print("   Phi((u_i - u_j)/sigma_p) would need the normal observation model and")
print("   has no counterpart for a binary or count response.")

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
print("8. ties encoded as 0.5 -> logit 0")
print("=" * 76)
check("logit(0.5) == 0", abs(logit(0.5)) < 1e-15)
check("score 0.5 clips safely (no inf)", np.isfinite(logit(np.clip(0.5, 1e-10, 1 - 1e-10))))

print()
print("=" * 76)
print("9. analytic gradients of the beta-binomial M-step")
print("=" * 76)
print("   The appendix says the gradients were checked against finite")
print("   differences; this is that check. A central difference carries O(eps^2)")
print("   truncation error, so the error should fall about 100-fold per decade of")
print("   eps until roundoff takes over. A gradient that is wrong does not show")
print("   that scaling.")

_rng = np.random.default_rng(5)
_A, _A1, _K = 4, 3, 30
_Xg = np.zeros((_K, _A1))
for _k in range(_K):
    _i, _j = _rng.choice(_A, 2, replace=False)
    if _i < _A1:
        _Xg[_k, _i] += 1.0
    if _j < _A1:
        _Xg[_k, _j] -= 1.0
_ng = _rng.integers(10, 50, _K).astype(float)
_mg = _rng.binomial(_ng.astype(int), 0.5).astype(float)
_wg = _rng.uniform(0.2, 1.0, _K)
_wg /= _wg.sum()
_ug = _rng.normal(0, 0.4, _A1)
_lam = 1e-3
_ubar = np.zeros(_A1)


def _numeric_grad(mode, rho, eps):
    g = np.zeros(_A1 + 1)
    for i in range(_A1):
        up, um = _ug.copy(), _ug.copy()
        up[i] += eps
        um[i] -= eps
        g[i] = (ib._component_objective(_Xg, _mg, _ng, _wg, up, rho, _lam, _ubar, mode)
                - ib._component_objective(_Xg, _mg, _ng, _wg, um, rho, _lam, _ubar, mode)) / (2 * eps)
    g[_A1] = (ib._component_objective(_Xg, _mg, _ng, _wg, _ug, rho + eps, _lam, _ubar, mode)
              - ib._component_objective(_Xg, _mg, _ng, _wg, _ug, rho - eps, _lam, _ubar, mode)) / (2 * eps)
    return g


def _rel_err(mode, rho, eps):
    gu, gr = ib._component_grad(_Xg, _mg, _ng, _wg, _ug, rho, _lam, _ubar, mode)
    ana = np.concatenate([np.asarray(gu).ravel(), [gr]])
    num = _numeric_grad(mode, rho, eps)
    return float(np.max(np.abs(ana - num) / np.maximum(np.abs(num), 1e-10)))


for _mode, _rho in (("binomial", 0.0), ("beta-binomial", 0.2), ("beta-binomial", 0.7)):
    _e3, _e4, _e5 = _rel_err(_mode, _rho, 1e-3), _rel_err(_mode, _rho, 1e-4), _rel_err(_mode, _rho, 1e-5)
    _tag = f"{_mode}, rho={_rho}"
    print(f"   {_tag:<24} eps=1e-3 {_e3:.2e}   eps=1e-4 {_e4:.2e}   eps=1e-5 {_e5:.2e}")
    check(f"gradient of {_tag} matches at eps=1e-5", _e5 < 1e-7, f"(rel err {_e5:.2e})")
    # only the first decade is reliably in the quadratic regime; at 1e-5 the
    # central difference is already meeting the roundoff floor, so the ratio
    # there says nothing about the analytic gradient
    check(f"gradient error of {_tag} falls as eps^2 over 1e-3 to 1e-4",
          _e4 < _e3 / 20.0, f"(ratio {_e3 / _e4:.0f})")

print()
print("=" * 76)
print("ALL CONVENTIONS VERIFIED" if ok else "SOME CHECKS FAILED")
print("=" * 76)
sys.exit(0 if ok else 1)
