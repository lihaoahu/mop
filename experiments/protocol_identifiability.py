"""Does the structural inconsistency carry information under each protocol?

The paper's revised thesis is that intrinsic variability and structural inconsistency
are separable *depending on the protocol*: identifiable for the Gaussian variants
(ratio, subjective probability), degenerate for the binomial ones (binary and repeated
choice), where Section 2.3 fixes `phi_p = 1`. That is a testable claim about
identifiability and it had not been tested.

The test
--------
Identifiability shows up as whether a richer parameterisation of `phi` changes the fit.
Two nested models are compared at the SAME number of preference states:

    shared-phi     one dispersion for every component      df = P(A-1) + (P-1) + 1
    per-state phi  a separate dispersion per component     df = P(A-1) + (P-1) + P

so the richer model uses exactly `P - 1` more parameters. The difference in the
penalized objective from those `P - 1` parameters is the measurement:

  * if `phi` is identified, the extra parameters buy a lot -- the dispersion genuinely
    differs across states and the likelihood improves substantially;
  * if `phi` is not identified -- which is the claim for the binomial protocol, since
    its variance `mu(1-mu)/omega` is a function of the mean -- the extra parameters
    buy almost nothing, because there is no state-specific dispersion to find.

Both protocols are fitted to the SAME judgments, read the way each protocol reads
them: the Gaussian variant takes the logit of the score, and the binomial variant
takes the score as a choice with `M` repetitions and weight `M`. So this is one
dataset asked two questions, which is what makes the contrast a property of the
protocol rather than of two different datasets.

Run:
    uv run python experiments/protocol_identifiability.py
    uv run python experiments/protocol_identifiability.py --dim policy --M 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

EXPERIMENTS = Path(__file__).resolve().parent
CASE = EXPERIMENTS / "case_study"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(EXPERIMENTS.parent / "src"))

import _mop_params as MP                                            # noqa: E402
from mop.inference import prior_mode_phi0                           # noqa: E402
from mop.initialization import kridge                               # noqa: E402

EPS = 1e-12


# ── the Gaussian (ratio / subjective probability) protocol ───────────────────

def fit_gaussian(X, y, nP, per_state_phi, lam_u, lam_sigma, s2, max_iter=2000,
                 tol=1e-8):
    """EM with either one dispersion for all components or one per component.

    ``y`` is the transformed judgment, so this is the protocol where phi is a variance
    on the transformed scale, free of the utility difference.
    """
    nJ, d = X.shape
    U = np.atleast_2d(kridge(X, y, nP)[0])
    pi = np.full(nP, 1.0 / nP)
    phi = np.full(nP, s2)
    prev, n_iter, conv = None, 0, False

    for n_iter in range(1, max_iter + 1):
        means = X @ U.T
        comp = (np.log(np.maximum(pi, EPS))[None, :]
                - 0.5 * np.log(2 * np.pi * phi)[None, :]
                - 0.5 * ((y[:, None] - means) ** 2) / phi[None, :])
        mx = comp.max(axis=1, keepdims=True)
        g = np.exp(comp - mx)
        g /= np.maximum(g.sum(axis=1, keepdims=True), EPS)

        pi = np.maximum((g.sum(axis=0) + MP.lambda_pi - 1.0)
                        / (nJ + MP.lambda_pi * nP - nP), 1e-10)
        pi /= pi.sum()
        for p in range(nP):
            gp = g[:, p]
            A_ = X.T @ (X * gp[:, None]) + lam_u * np.eye(d)
            b_ = X.T @ (y * gp)
            U[p] = np.linalg.solve(A_, b_)
        if per_state_phi:
            for p in range(nP):
                ss = float(np.sum(g[:, p] * (y - X @ U[p]) ** 2))
                prior_ss = lam_u * float(np.sum(U[p] ** 2))
                phi[p] = max((lam_sigma * s2 + ss + prior_ss)
                             / (lam_sigma + float(np.sum(g[:, p])) + d), 1e-10)
        else:
            ss = float(np.sum(g * (y[:, None] - X @ U.T) ** 2))
            prior_ss = lam_u * float(np.sum(U ** 2))
            phi[:] = max((lam_sigma * s2 + ss + prior_ss)
                         / (lam_sigma + nJ + nP * d), 1e-10)

        pll = _gauss_pll(X, y, U, pi, phi, lam_u, lam_sigma, s2)
        if prev is not None and abs(pll - prev) / (abs(pll) + 0.1) < tol:
            conv = True
            break
        prev = pll
    return {"U": U, "pi": pi, "phi": phi, "pll": pll, "n_iter": n_iter,
            "converged": conv}


def _gauss_pll(X, y, U, pi, phi, lam_u, lam_sigma, s2):
    means = X @ U.T
    comp = (np.log(np.maximum(pi, EPS))[None, :]
            - 0.5 * np.log(2 * np.pi * phi)[None, :]
            - 0.5 * ((y[:, None] - means) ** 2) / phi[None, :])
    mx = comp.max(axis=1, keepdims=True)
    total = float(np.sum(mx.ravel() + np.log(np.exp(comp - mx).sum(axis=1))))
    a = (lam_sigma - 2.0) / 2.0
    b = lam_sigma * s2 / 2.0
    for p in range(U.shape[0]):
        total -= 0.5 * lam_u / phi[p] * float(U[p] @ U[p])
        total -= 0.5 * X.shape[1] * np.log(2 * np.pi * phi[p] / lam_u)
        total += a * np.log(b) - gammaln(a) - (a + 1) * np.log(phi[p]) - b / phi[p]
    total += (MP.lambda_pi - 1.0) * float(np.sum(np.log(np.maximum(pi, EPS))))
    total += gammaln(U.shape[0] * MP.lambda_pi) - U.shape[0] * gammaln(MP.lambda_pi)
    return total


# ── the binomial (binary / repeated choice) protocol ─────────────────────────

def fit_binomial(X, m, n, omega, nP, per_state_phi, lam_u, max_iter=2000, tol=1e-8):
    """EM for the binomial protocol, one dispersion or one per component.

    The variance here is `mu(1-mu)/omega`, a function of the mean, so a free `phi_p`
    can only rescale it -- it has no state-specific information to find. Whether the
    fit improves when it is allowed to vary is the test.
    """
    nJ, d = X.shape
    U = np.atleast_2d(kridge(X, m / n, nP)[0])
    pi = np.full(nP, 1.0 / nP)
    phi = np.ones(nP)
    prev, n_iter, conv = None, 0, False

    for n_iter in range(1, max_iter + 1):
        eta = np.clip(X @ U.T, -30, 30)
        # phi enters as a scale on the log-likelihood contribution.
        comp = (np.log(np.maximum(pi, EPS))[None, :]
                + phi[None, :] * (m[:, None] * eta - n[:, None] * np.logaddexp(0.0, eta)))
        mx = comp.max(axis=1, keepdims=True)
        g = np.exp(comp - mx)
        g /= np.maximum(g.sum(axis=1, keepdims=True), EPS)

        pi = np.maximum((g.sum(axis=0) + MP.lambda_pi - 1.0)
                        / (nJ + MP.lambda_pi * nP - nP), 1e-10)
        pi /= pi.sum()
        for p in range(nP):
            w = omega * g[:, p] * phi[p]

            def obj(u, w=w):
                e = np.clip(X @ u, -30, 30)
                return -(float(np.sum(w * (m * e - n * np.logaddexp(0.0, e))))
                         - 0.5 * lam_u * float(u @ u))

            def grad(u, w=w):
                e = np.clip(X @ u, -30, 30)
                return -(X.T @ (w * (m - n * expit(e)))) + lam_u * u

            U[p] = minimize(obj, U[p], jac=grad, method="L-BFGS-B",
                            options={"maxiter": 500, "ftol": 1e-13}).x

        if per_state_phi:
            for p in range(nP):
                gp = g[:, p]
                r = m / n - expit(X @ U[p])
                ss = float(np.sum(omega * gp * r ** 2))
                phi[p] = max(1.0 + ss / max(float(np.sum(omega * gp)), EPS), 1e-6)
        else:
            r = m[:, None] / n[:, None] - expit(X @ U.T)
            ss = float(np.sum(omega[:, None] * g * r ** 2))
            phi[:] = max(1.0 + ss / max(nJ, EPS), 1e-6)

        pll = _binom_pll(X, m, n, omega, U, pi, phi, lam_u)
        if prev is not None and abs(pll - prev) / (abs(pll) + 0.1) < tol:
            conv = True
            break
        prev = pll
    return {"U": U, "pi": pi, "phi": phi, "pll": pll, "n_iter": n_iter,
            "converged": conv}


def _binom_pll(X, m, n, omega, U, pi, phi, lam_u):
    eta = np.clip(X @ U.T, -30, 30)
    comp = (np.log(np.maximum(pi, EPS))[None, :]
            + phi[None, :] * (m[:, None] * eta - n[:, None] * np.logaddexp(0.0, eta)))
    mx = comp.max(axis=1, keepdims=True)
    total = float(np.sum(mx.ravel() + np.log(np.exp(comp - mx).sum(axis=1))))
    for p in range(U.shape[0]):
        total -= 0.5 * lam_u * float(U[p] @ U[p])
    return total


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default=None)
    ap.add_argument("--components", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--M", type=int, default=20,
                    help="repetitions per judgment, i.e. the observation weight")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from step9_phi_identification import build_design, load_pairs

    pairs = load_pairs(dim_name=args.dim) if args.dim else load_pairs()
    X, y = build_design(pairs)
    nJ, A1 = X.shape
    A = A1 + 1
    label = args.dim or "overall"
    lam_sigma = MP.LAMBDA_SIGMA_OVER_N * nJ
    s2 = float(prior_mode_phi0(X, y))

    # The binomial reading of the same judgments: the score becomes m successes out
    # of M, with observation weight M.
    probs = expit(y)
    m = np.round(probs * args.M)
    n = np.full(nJ, float(args.M))
    omega = n.copy()

    print(f"[{label}] n={nJ}, A={A} judgments read under both protocols")
    print(f"  M = {args.M} repetitions per judgment for the binomial reading")
    print(f"  {MP.summary()}")
    print(f"  lambda_sigma = {lam_sigma:.4f}, phi_0 = {s2:.4f}\n")

    results = {"label": label, "n": int(nJ), "A": int(A), "M": args.M,
               "lambda_sigma": float(lam_sigma), "phi0": float(s2),
               "gaussian": {}, "binomial": {}}

    print(f"{'P':>3} | {'Gaussian shared':>16} {'per-state':>13} {'gain':>9} "
          f"{'phi':>24} | {'Binomial shared':>16} {'per-state':>13} {'gain':>9}")
    print("-" * 118)

    for k in args.components:
        gs = fit_gaussian(X, y, k, False, MP.lambda_u, lam_sigma, s2)
        gp = fit_gaussian(X, y, k, True, MP.lambda_u, lam_sigma, s2)
        g_gain = gp["pll"] - gs["pll"]

        bs = fit_binomial(X, m, n, omega, k, False, MP.lambda_u)
        bp = fit_binomial(X, m, n, omega, k, True, MP.lambda_u)
        b_gain = bp["pll"] - bs["pll"]

        results["gaussian"][str(k)] = {
            "shared_pll": gs["pll"], "per_state_pll": gp["pll"], "gain": g_gain,
            "phi_shared": [float(v) for v in gs["phi"]],
            "phi_per_state": [float(v) for v in gp["phi"]],
            "extra_parameters": int(k - 1),
        }
        results["binomial"][str(k)] = {
            "shared_pll": bs["pll"], "per_state_pll": bp["pll"], "gain": b_gain,
            "phi_shared": [float(v) for v in bs["phi"]],
            "phi_per_state": [float(v) for v in bp["phi"]],
            "extra_parameters": int(k - 1),
        }
        pltxt = f"shared={gs['phi'][0]:.2f} per-state={np.round(gp['phi'], 2).tolist()}"
        print(f"{k:>3} | {gs['pll']:>16.2f} {gp['pll']:>13.2f} {g_gain:>9.2f} "
              f"{pltxt:>24} | {bs['pll']:>16.2f} {bp['pll']:>13.2f} {b_gain:>9.2f}")

    print()
    print("Reading: 'gain' is what the P-1 extra dispersion parameters buy, in units")
    print("of the penalized objective. A protocol where the dispersion is identified")
    print("shows a substantial gain and per-state estimates that differ; a protocol")
    print("where it is not shows a gain near zero, because the extra parameters have")
    print("nothing state-specific to find.")

    for tag in ("gaussian", "binomial"):
        gains = [v["gain"] for v in results[tag].values()]
        print(f"  {tag:<9} gains: {[round(g, 2) for g in gains]}")

    out = Path(args.out) if args.out else (
        EXPERIMENTS /
        f"protocol_identifiability{'_' + args.dim if args.dim else ''}.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
