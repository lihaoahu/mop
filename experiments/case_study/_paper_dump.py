"""Full MoP detail dump, for inspection.

Not a source for anything in the paper: no table or number in the manuscript
comes from this script. By default it now runs with the PAPER's parameters, read
from ``experiments/mop_params.json``, so that running it cannot produce numbers
that look like the paper's while resting on different hyperparameters. Pass
``--exploratory`` to use the older ``lambda_u = lambda_sigma = n/100`` setting
instead, which diverges from the paper's ``lambda_u = 1e-6`` and
``lambda_sigma = n/50``; that mode prints a banner naming every value it changes.
The credibility figures it prints sample an approximate posterior. Use
``step7_analyze.py`` for the paper's fits and ``_gen_tables.py`` for its tables.
"""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.special import logit
from mop import inference, pairwise_comparison
from mop.inference import comparisons2regression
from mop.initialization import warm_init

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mop_params import (lambda_u as PAPER_LAMBDA_U, lambda_pi as PAPER_LAMBDA_PI,
                         lambda_sigma_for, min_weight_share,
                         max_components, summary as params_summary)

from _common import ENERGY_PAIRS, TYPE_MAP, T

# Only used under --exploratory; the divisor that produced the old settings.
D = 100
EXPLORATORY = False
EPS = 1e-10

def load_pairs(dim_name=None, year=None):
    pairs = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if year and r.get("publish_time", "")[:4] != year:
                continue
            for p in r.get("comparison_pairs", []):
                if dim_name and p.get("dimension") != dim_name:
                    continue
                ft = TYPE_MAP.get(p.get("former_type", ""), "")
                lt = TYPE_MAP.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": ft, "lt": lt, "score": p.get("score", 0.5)})
    return pairs

def run_mop_full(pairs, label=""):
    nJ = len(pairs)
    name_to_id = {t: i+1 for i,t in enumerate(T)}
    pcvector = [pairwise_comparison.PairwiseComparison(
        former_id=name_to_id[p["ft"]], latter_id=name_to_id[p["lt"]],
        agent_id=1, judgment=p["score"]) for p in pairs]
    link_functions = [lambda s, _eps=EPS: logit(np.clip(s, _eps, 1-_eps))]
    X, y, _ = comparisons2regression(pcvector, link_functions)
    nA = X.shape[1]
    if EXPLORATORY:
        # Deliberately NOT the paper's settings. Named individually so a reader of the
        # output cannot mistake it for the case study's fit.
        lu = lpi = ls = nJ / D
        print(f"    [EXPLORATORY] lambda_u = lambda_pi = lambda_sigma = nJ/{D} "
              f"= {ls:.4f}")
        print(f"    [EXPLORATORY] the paper uses lambda_u = {PAPER_LAMBDA_U:g}, "
              f"lambda_pi = {PAPER_LAMBDA_PI:g}, "
              f"lambda_sigma = {lambda_sigma_for(nJ):.4f}")
    else:
        lu, lpi, ls = PAPER_LAMBDA_U, PAPER_LAMBDA_PI, lambda_sigma_for(nJ)
    # φ₀ comes from inference.prior_mode_phi0, which the fit and pBIC share
    phi0 = inference.prior_mode_phi0(X, y)

    prev = None
    best_k, best_bic, best_result = None, np.inf, None
    for k in range(1, max_components + 1):
        kw = {}
        if prev is not None:
            pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
            ps = np.array([p.inconsistency_level for p in prev.priorities])
            pm = np.array(prev.mixture_probabilities)
            u_init, sigma2_init, mp_init = warm_init(
                X, y, pu, ps, pm, min_weight_share=min_weight_share,
                n_components=k)
            kw = {"u_init": u_init, "sigma2_init": sigma2_init, "mp_init": mp_init}
        result = inference.mop(pcvector, link_functions, k,
                               lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls, **kw)
        val = inference.pbic(pcvector, result, link_functions,
                             lambda_u=lu, lambda_sigma=ls, lambda_pi=lpi)
        if val < best_bic:
            best_bic = val; best_k = k; best_result = result
        prev = result

        # If minprior is turned on and a candidate loses a component to it,
        # raising the candidate further can only reproduce that mixture, so the
        # search stops there. With the default of no removal the test never
        # fires and the whole grid is searched.
        if len(result) < k:
            break

    # Compute credibility via posterior sampling
    nP = best_k
    u_mat = np.zeros((nP, nA))
    for p in range(nP):
        w = best_result[p].priority_vector
        u_mat[p, :] = np.log(w[:nA] / w[-1])
    sigma2_arr = np.array([best_result[p].inconsistency_level for p in range(nP)])
    mp_arr = best_result.mixture_probabilities

    # Hard assignment for posterior params
    gkc = inference.e_gamma(X, y, u_mat, sigma2_arr, mp_arr)
    hard = np.argmax(gkc, axis=1)
    credibility = []
    n_samples = 10000
    for p in range(nP):
        idx_p = np.where(hard == p)[0]
        if len(idx_p) < nA:
            credibility.append(None)
            continue
        Xp = X[idx_p]; yp = y[idx_p]
        Sp_inv = Xp.T @ Xp + lu * np.eye(nA)
        try:
            Sp = np.linalg.inv(Sp_inv)
            mp_vec = Sp @ (Xp.T @ yp + lu * np.zeros(nA))
        except:
            credibility.append(None)
            continue
        # Sample from posterior
        modal_rank = tuple(np.argsort(-mp_vec))
        count = 0
        for _ in range(n_samples):
            # v_p = (lambda_sigma + n_p + A - 3)/2, with A = nA + 1 alternatives
            s2_sample = 1 / np.random.gamma((ls + len(idx_p) + nA - 2)/2,
                                             1/(ls * phi0 / 2 + 1e-10) if ls * phi0 > 0 else 1)
            # Actually proper sampling requires q_p which is complex. Approximate.
            u_sample = np.random.multivariate_normal(mp_vec, sigma2_arr[p] * Sp)
            rank = tuple(np.argsort(-u_sample))
            if rank == modal_rank:
                count += 1
        credibility.append(count / n_samples)

    # Weighted ranking
    weighted = np.zeros(len(T))
    for pi, pri in zip(mp_arr, [best_result[p].priority_vector for p in range(nP)]):
        weighted += np.array(pri) * pi

    print(f"\n[{label}] n={nJ} K={best_k} pBIC={best_bic:.1f}")
    print(f"  lambda = {lu:.1f}")
    print(f"  Weighted: ", end="")
    ranked = sorted(zip(T, weighted), key=lambda x: -x[1])
    print(" > ".join(f"{t}({v:.3f})" for t,v in ranked))
    for i in range(nP):
        pi = mp_arr[i]
        pri = best_result[i].priority_vector
        inc = sigma2_arr[i]
        cred = credibility[i]
        r = sorted(zip(T, pri), key=lambda x: -x[1])
        cs = f"{cred*100:.1f}%" if cred is not None else "N/A"
        print(f"    C{i+1}: pi={pi:.4f} sigma2={inc:.4f} cred={cs}")
        print(f"           {' > '.join(f'{t}({v:.3f})' for t,v in r)}")
    return best_k, best_result, weighted, nJ, credibility

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exploratory", action="store_true",
                    help=f"use lambda_u = lambda_pi = lambda_sigma = nJ/{D} instead "
                         f"of the paper's parameters; NOT a source for the paper")
    _args = ap.parse_args()
    if _args.exploratory:
        EXPLORATORY = True
        print("=" * 78)
        print("EXPLORATORY MODE: these are NOT the paper's parameters. No number or")
        print("table in the manuscript may be taken from this run. Use")
        print("step7_analyze.py for the paper's fits.")
        print("=" * 78)
    else:
        print(f"using the paper's parameters: {params_summary()}")

    for label, dim, year in [("整体", None, None),
                              ("Cost", "cost", None),
                              ("Driving", "driving_experience", None),
                              ("Range", "range", None),
                              ("Refueling", "refueling_convenience", None),
                              ("Reliability", "reliability", None),
                              ("Policy", "policy", None),
                              ("Other", "other", None),
                              ("2025", None, "2025"),
                              ("2026", None, "2026")]:
        pairs = load_pairs(dim_name=dim, year=year)
        if len(pairs) >= 30:
            run_mop_full(pairs, label)
