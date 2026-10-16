"""Why does the free-phi P=1 fit have a WORSE pBIC than the fixed-phi P=1 fit?

Freeing phi_p cannot make the model worse: phi = 1 is inside the free model's
parameter space. So either

  (a) the fixed-phi fit converges to a different (and here better) local optimum, or
  (b) the two penalized objectives differ by more than the phi terms, so they are not
      nested as they should be, or
  (c) one of the two is simply not at its optimum.

This checks the policy slice, where the anomaly appears, and compares against the
repository's own `inference.mop` at P=1 so that a third implementation arbitrates.

Run: uv run python experiments/case_study/_diag_policy_p1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "src"))

# The point of this script is to arbitrate between two implementations, so it must NOT
# carry its own copy of the parameters: a duplicate would only confirm that a copy
# agrees with itself. Everything comes from experiments/mop_params.json.
from _mop_params import (lambda_u as LAMBDA_U, lambda_pi as LAMBDA_PI,  # noqa: E402
                         lambda_sigma_for)

from _common import T                                                   # noqa: E402
from mop import inference                                               # noqa: E402
from mop.inference import comparisons2regression, prior_mode_phi0        # noqa: E402
from mop.pairwise_comparison import PairwiseComparison                  # noqa: E402
from mop.initialization import kridge                                   # noqa: E402
from step9_phi_identification import (build_design, fit, load_pairs,     # noqa: E402
                                      _penalized, criteria)

pairs = load_pairs(dim_name="policy")
X, y = build_design(pairs)
nJ, A1 = X.shape
A = A1 + 1
# Derived once from the shared config, never written out by hand.
lam_sigma = lambda_sigma_for(nJ)
print(f"parameters from mop_params.json: lambda_u={LAMBDA_U:g} "
      f"lambda_pi={LAMBDA_PI:g} lambda_sigma={lam_sigma:.4f}")
s2_internal = prior_mode_phi0(X, y)
print(f"policy slice: n={nJ}, A={A}")
print(f"  var(y)                  = {float(np.var(y)):.6f}   (what the fixed-phi run used as s2)")
print(f"  prior_mode_phi0(X, y)   = {s2_internal:.6f}   (what the free-phi run used as s2)")
print(f"  mean of y               = {float(np.mean(y)):.6f}")
print()

# ── (1) both variants from the SAME starting point ───────────────────────────
u0, phi0, pi0 = kridge(X, y, 1)
print(f"kridge start: u={np.round(np.atleast_2d(u0)[0], 4).tolist()}  "
      f"phi0={np.round(np.atleast_1d(phi0), 4).tolist()}")

r_fixed = fit(X, y, 1, free_phi=False, s2=s2_internal, u_init=u0)
r_free = fit(X, y, 1, free_phi=True, s2=s2_internal, u_init=u0)
print()
print("from the SAME start, with the SAME s2:")
print(f"  fixed phi: u={np.round(r_fixed['u'][0], 4).tolist()} phi={r_fixed['phi'].tolist()}")
print(f"             loglik={r_fixed['ll']:.4f}  pll={r_fixed['pll']:.4f}")
print(f"  free  phi: u={np.round(r_free['u'][0], 4).tolist()} phi={np.round(r_free['phi'], 4).tolist()}")
print(f"             loglik={r_free['ll']:.4f}  pll={r_free['pll']:.4f}")
print(f"  free minus fixed: loglik {r_free['ll'] - r_fixed['ll']:+.4f}, "
      f"pll {r_free['pll'] - r_fixed['pll']:+.4f}")
print("  (a free model must have loglik >= the fixed one; pll may go either way")
print("   because the two objectives differ by the phi priors)")

# ── (2) the penalized objective evaluated at the SAME u, both ways ───────────
print()
print("penalized log-posterior at one fixed u, under each objective:")
u_test = r_fixed["u"]
for tag, free in (("fixed", False), ("free ", True)):
    phi = np.ones(1) if not free else np.array([1.0])
    pll, ll = _penalized(X, y, u_test, phi, np.array([1.0]), LAMBDA_U, lam_sigma, LAMBDA_PI,
                         s2_internal, np.zeros((1, A1)), free)
    print(f"  {tag}: pll={pll:.4f}  loglik={ll:.4f}")

# ── (3) the repository's own fit, as a third opinion ────────────────────────
print()
name_to_id = {t: i + 1 for i, t in enumerate(T)}
pcv = [PairwiseComparison(former_id=name_to_id[p["ft"]],
                          latter_id=name_to_id[p["lt"]],
                          agent_id=1, judgment=p["score"]) for p in pairs]
link = [lambda s, _eps=1e-10: logit(np.clip(s, _eps, 1 - _eps))]
res = inference.mop(pcv, link, 1, lambda_u=LAMBDA_U, lambda_pi=LAMBDA_PI, lambda_sigma=lam_sigma)
pb = inference.pbic(pcv, res, link, lambda_u=LAMBDA_U, lambda_sigma=lam_sigma, lambda_pi=LAMBDA_PI)
u_paper = np.array([np.log(r.priority_vector[:A1] / r.priority_vector[A1])
                    for r in res.priorities])
phi_paper = np.array([r.inconsistency_level for r in res.priorities])
print(f"paper's inference.mop at P=1: phi={np.round(phi_paper, 4).tolist()} "
      f"pBIC={pb:.2f}")
print(f"  its u           = {np.round(u_paper[0], 4).tolist()}")
print(f"  my fixed-phi u  = {np.round(r_fixed['u'][0], 4).tolist()}")
print(f"  my free-phi u   = {np.round(r_free['u'][0], 4).tolist()}")

# ── (4) the phi prior at the two s2 values ──────────────────────────────────
print()
print("the Inverse-Gamma prior on phi, lambda_sigma = %.3f:" % lam_sigma)
from scipy.stats import invgamma
a_shape = (lam_sigma - 2) / 2
scale = lam_sigma * s2_internal / 2
for phi_val in (0.48, 1.0, 1.087):
    d = invgamma.logpdf(phi_val, a=a_shape, scale=scale)
    print(f"  log prior density at phi={phi_val}: {d:+.4f}")
print(f"  NOTE the prior MODE is s2 = {s2_internal:.4f}, so a phi near 1 is")
print("  penalised to the extent that s2 differs from 1. That is the asymmetry")
print("  between the two objectives, and it is a modelling fact, not a bug --")
print("  but it means the two pBIC values are not directly comparable.")
