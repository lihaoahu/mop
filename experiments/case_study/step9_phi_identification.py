"""Does freeing the structural inconsistency change how many states are selected?

The question this answers
-------------------------
The paper's claim is that intrinsic variability (carried by the mixture) and
structural inconsistency (carried by phi_p) are separable, and that the second is
identifiable for the Gaussian protocols. If that separation has no consequence,
freeing phi_p changes nothing that matters and the paper has no methodological
claim. If it does have a consequence, the number of preference states the analyst
reports is different under the two parameterisations -- which is exactly what the
case study's "four-component structure" rests on.

So: fit the case-study ratio judgments twice, once with phi_p fixed at one and once
with phi_p free, over the same grid of component counts, and compare the selected
count.

Method
------
Both fits use the paper's own EM (Section 2, `eq:pi_general` and the Gaussian
M-step) and differ ONLY in whether phi_p is estimated. The same kridge/warm-start
initialisation, the same priors on u and pi, the same numerical constants. The
fixed-phi variant drops the phi terms from the penalized log-posterior, which is
what setting phi_p = 1 means, rather than fixing phi at one through the prior --
a fixed value must not carry a prior penalty it can no longer optimise against.

The criterion is reported two ways, because the paper's penalized BIC adds the
prior terms while an ordinary BIC does not, and a conclusion that depends on which
one is used would not be worth having:

  pBIC   -2 * penalized log-posterior + log(n) * df   (the paper's eq:pbic)
  BIC    -2 * observed-data log-likelihood + log(n) * df

df counts (A-1) utilities per component, P-1 mixture weights, and P dispersions
when phi is free (0 when it is fixed), plus the protocol-specific parameters.

Run:
    uv run python experiments/case_study/step9_phi_identification.py
    ... --dim cost          # one dimension
    ... --grid 1 2 3        # a shorter grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from _common import CFG, ENERGY_PAIRS, TYPE_MAP, T, DIMS, DIM_LABELS  # noqa: E402
from mop.inference import prior_mode_phi0                             # noqa: E402
from mop.initialization import kridge                                 # noqa: E402
from mop.pairwise_comparison import comparisons2regression            # noqa: E402

EPS = 1e-10
CLIP = 50.0


# ── data ─────────────────────────────────────────────────────────────────────

def load_pairs(dim_name=None, year=None):
    """The case study's ratio judgments, exactly as step7_analyze loads them."""
    from step7_analyze import load_pairs as _load
    return _load(dim_name=dim_name, year=year)


def build_design(pairs):
    """Design matrix and logit-transformed responses for the ratio protocol."""
    from mop.pairwise_comparison import PairwiseComparison
    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    pcv = [PairwiseComparison(former_id=name_to_id[p["ft"]],
                              latter_id=name_to_id[p["lt"]],
                              agent_id=1, judgment=p["score"]) for p in pairs]
    link = [lambda s, _eps=EPS: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcv, link)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


# ── the EM, with phi_p either estimated or fixed ─────────────────────────────

def _e_step(X, y, u, phi, pi):
    """Allocation posterior, `eq:e_gamma`."""
    means = X @ u.T
    sd = np.sqrt(phi)
    logp = np.log(np.maximum(pi, EPS))[None, :] \
        - 0.5 * np.log(2 * np.pi * phi)[None, :] \
        - 0.5 * ((y[:, None] - means) ** 2) / phi[None, :]
    mx = logp.max(axis=1, keepdims=True)
    g = np.exp(logp - mx)
    return g / np.maximum(g.sum(axis=1, keepdims=True), EPS)


def _penalized(X, y, u, phi, pi, lam_u, lam_sigma, lam_pi, s2, u_bar,
               free_phi):
    """Penalized log-posterior, the quantity pBIC penalizes (`eq:pbic`).

    With ``free_phi`` false the Inverse-Gamma prior on phi is dropped and the
    normalising constant of the normal prior loses its phi dependence, because
    phi is not a parameter of that model.
    """
    from scipy.special import gammaln
    nJ, A1 = X.shape
    nP = u.shape[0]
    # observed-data log-likelihood
    means = X @ u.T                                    # (nJ, nP)
    comp = (np.log(np.maximum(pi, EPS))[None, :]
            - 0.5 * np.log(2 * np.pi * phi)[None, :]
            - 0.5 * ((y[:, None] - means) ** 2) / phi[None, :])
    mx = comp.max(axis=1, keepdims=True)
    ll = float(np.sum(mx.ravel() + np.log(np.exp(comp - mx).sum(axis=1))))
    total = ll
    for p in range(nP):
        if free_phi:
            sp2 = max(phi[p], 1e-300)
            diff = u[p] - u_bar[p]
            total -= (lam_u / (2.0 * sp2)) * float(np.sum(diff ** 2))
            total -= (A1 / 2.0) * np.log(2.0 * np.pi * sp2 / lam_u)
            a = (lam_sigma - 2.0) / 2.0
            b = lam_sigma * s2 / 2.0
            total += a * np.log(b) - gammaln(a) - (a + 1.0) * np.log(sp2) - b / sp2
        else:
            diff = u[p] - u_bar[p]
            total -= (lam_u / 2.0) * float(np.sum(diff ** 2))
            total -= (A1 / 2.0) * np.log(2.0 * np.pi / lam_u)
    total += (lam_pi - 1.0) * float(np.sum(np.log(np.maximum(pi, EPS))))
    total += gammaln(nP * lam_pi) - nP * gammaln(lam_pi)
    return total, ll


def fit(X, y, nP, free_phi, lam_u=1e-6, lam_pi=1.0, lam_sigma=None,
        u_init=None, phi_init=None, pi_init=None, s2=None, u_bar=None,
        max_iter=5000, tol=1e-6):
    """EM for the Gaussian (ratio) variant, with phi_p estimated or fixed at one."""
    nJ, A1 = X.shape
    if s2 is None:
        s2 = prior_mode_phi0(X, y)
    if lam_sigma is None:
        lam_sigma = nJ / 50.0
    if u_bar is None:
        u_bar = np.zeros((nP, A1))
    else:
        u_bar = np.asarray(u_bar, dtype=float)
        if u_bar.ndim == 1:
            u_bar = np.tile(u_bar, (nP, 1))

    if u_init is None:
        u0, phi0, pi0 = kridge(X, y, nP)
        u = np.atleast_2d(u0)
        phi = np.full(nP, float(s2)) if phi_init is None else np.asarray(phi_init, float)
        pi = np.full(nP, 1.0 / nP) if pi_init is None else np.asarray(pi_init, float)
    else:
        u = np.atleast_2d(np.asarray(u_init, dtype=float)).copy()
        nP = u.shape[0]
        phi = (np.full(nP, float(s2)) if phi_init is None
               else np.asarray(phi_init, dtype=float).copy())
        pi = (np.full(nP, 1.0 / nP) if pi_init is None
              else np.asarray(pi_init, dtype=float).copy())
    if not free_phi:
        phi = np.ones(nP)

    prev, n_iter, converged = None, 0, False
    for n_iter in range(1, max_iter + 1):
        g = _e_step(X, y, u, phi, pi)
        sum_g = g.sum(axis=0)

        # mixture weights, `eq:pi_general` (all omega_k = 1 for this protocol)
        pi = (sum_g + lam_pi - 1.0) / (nJ + lam_pi * nP - nP)
        pi = np.maximum(pi, 1e-10)
        pi = pi / pi.sum()

        # utilities: the weighted ridge solve of the Gaussian M-step
        for p in range(nP):
            gp = g[:, p]
            XtWX = X.T @ (X * gp[:, None]) + lam_u * np.eye(A1)
            Xty = X.T @ (y * gp) + lam_u * u_bar[p]
            try:
                u[p] = np.linalg.solve(XtWX, Xty)
            except np.linalg.LinAlgError:
                u[p] = np.linalg.lstsq(XtWX, Xty, rcond=None)[0]

            if free_phi:
                ss = float(np.sum(gp * (y - X @ u[p]) ** 2))
                ss_prior = lam_u * float(np.sum((u[p] - u_bar[p]) ** 2))
                phi[p] = max((lam_sigma * s2 + ss + ss_prior)
                             / (lam_sigma + float(np.sum(gp)) + A1), 1e-10)

        pll, ll = _penalized(X, y, u, phi, pi, lam_u, lam_sigma, lam_pi, s2,
                             u_bar, free_phi)
        if not np.isfinite(pll):
            break
        if prev is not None and abs(pll - prev) / (abs(pll) + 0.1) < tol:
            converged = True
            break
        prev = pll

    pll, ll = _penalized(X, y, u, phi, pi, lam_u, lam_sigma, lam_pi, s2, u_bar,
                         free_phi)
    return {"u": u, "phi": phi, "pi": pi, "pll": pll, "ll": ll,
            "n_iter": n_iter, "converged": converged, "s2": s2}


def degrees_of_freedom(A, nP, free_phi):
    """Free parameters: utilities, weights, and dispersions when they are free."""
    return (A - 1) * nP + (nP - 1) + (nP if free_phi else 0)


def criteria(fitres, X, y, A, nP, free_phi):
    nJ = X.shape[0]
    df = degrees_of_freedom(A, nP, free_phi)
    return {"pBIC": -2.0 * fitres["pll"] + np.log(nJ) * df,
            "BIC": -2.0 * fitres["ll"] + np.log(nJ) * df,
            "df": df,
            "loglik": fitres["ll"], "pll": fitres["pll"]}


# ── comparison ───────────────────────────────────────────────────────────────

def compare(pairs, label, grid, verbose=True):
    X, y = build_design(pairs)
    nJ, A1 = X.shape
    A = A1 + 1
    out = {"label": label, "n_judgments": int(nJ), "n_alternatives": int(A),
           "grid": list(grid), "fixed_phi": {}, "free_phi": {}}

    # Fixed phi needs a warm start too, otherwise the comparison confounds the
    # parameterisation with the initialisation. Each family is warm-started
    # within itself, from the previous component count.
    for tag, free in (("fixed_phi", False), ("free_phi", True)):
        prev = None
        for k in grid:
            kw = {}
            if prev is not None:
                # Seed the new component where the current mixture explains worst,
                # the K-Ridge step, then keep the converged components.
                from mop.initialization import seed_component
                u_prev = prev["u"]
                ui = np.vstack([u_prev, seed_component(X, y, u_prev)])
                pi_i = np.concatenate([prev["pi"] * (k - 1) / k, [1.0 / k]])
                ph_i = np.concatenate([prev["phi"], [np.mean(prev["phi"])]])
                kw = {"u_init": ui, "phi_init": ph_i, "pi_init": pi_i}
            r = fit(X, y, k, free, **kw)
            crit = criteria(r, X, y, A, k, free)
            out[tag][str(k)] = {
                **crit,
                "mixture_weights": [float(x) for x in r["pi"]],
                "phi": [float(x) for x in r["phi"]],
                "n_iter": int(r["n_iter"]),
                "converged": bool(r["converged"]),
            }
            prev = r
            if verbose:
                print(f"    {tag:9s} P={k}  pBIC {crit['pBIC']:12.2f}  "
                      f"BIC {crit['BIC']:12.2f}  df {crit['df']:3d}  "
                      f"iters {r['n_iter']:4d}  conv {r['converged']}")

    for tag in ("fixed_phi", "free_phi"):
        for crit in ("pBIC", "BIC"):
            vals = {int(k): v[crit] for k, v in out[tag].items()}
            out[f"{tag}_best_by_{crit}"] = int(min(vals, key=vals.get))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default=None, help="one dimension, e.g. cost")
    ap.add_argument("--year", default=None)
    ap.add_argument("--grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dim:
        pairs = load_pairs(dim_name=args.dim)
        label = f"{DIM_LABELS.get(args.dim, args.dim)} ({args.dim})"
    elif args.year:
        pairs = load_pairs(year=args.year)
        label = args.year
    else:
        pairs = load_pairs()
        label = "overall"
    print(f"[{label}] n={len(pairs)} judgments, grid P={args.grid}")
    res = compare(pairs, label, args.grid)

    print(f"\n  selected component count")
    for tag in ("fixed_phi", "free_phi"):
        print(f"    {tag:9s}  by pBIC: P={res[f'{tag}_best_by_pBIC']}   "
              f"by BIC: P={res[f'{tag}_best_by_BIC']}")
    same_p = res["fixed_phi_best_by_pBIC"] == res["free_phi_best_by_pBIC"]
    same_b = res["fixed_phi_best_by_BIC"] == res["free_phi_best_by_BIC"]
    print(f"\n  phi changes the pBIC selection: {not same_p}")
    print(f"  phi changes the BIC  selection: {not same_b}")

    outpath = Path(args.out) if args.out else (
        HERE / "phi_identification_results.json")
    data = {}
    if outpath.exists():
        try:
            data = json.loads(outpath.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[label if args.dim or args.year else "overall"] = res
    outpath.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nwritten: {outpath}")


if __name__ == "__main__":
    main()
