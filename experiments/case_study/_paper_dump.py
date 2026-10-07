"""Full MoP detail dump, for inspection.

Not a source for anything in the paper: no table or number in the manuscript
comes from this script. It sets its own hyperparameters (``D = 100``, so
``lambda_u = lambda_sigma = n/100``, against the paper's ``lambda_u = 1e-6``
and ``lambda_sigma = n/50``) and the credibility figures it prints sample an
approximate posterior. Use ``step7_analyze.py`` for the paper's fits and
``_gen_tables.py`` for its tables.
"""
import sys, json, math
import numpy as np
from scipy.special import logit
from mop import inference, pairwise_comparison
from mop.inference import comparisons2regression
from _common import ENERGY_PAIRS, TYPE_MAP, T

D = 100
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
    nA = X.shape[1]; lu = lpi = ls = nJ / D
    u_pool = np.linalg.solve(X.T @ X + 1e-10*np.eye(nA), X.T @ y)
    s2_bar = max(np.sum((X @ u_pool - y)**2) / max(nJ - nA + 1, 1), 1e-10)

    prev = None
    best_k, best_bic, best_result = None, np.inf, None
    for k in range(1, 10):
        kw = {}
        if prev is not None:
            pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
            ps = np.array([p.inconsistency_level for p in prev.priorities])
            ok = len(pu); nk = ok + 1
            u_new = np.zeros((nk, nA)); u_new[:ok] = pu
            min_rse = np.full(X.shape[0], np.inf)
            for p in range(ok):
                min_rse = np.minimum(min_rse, (X @ pu[p] - y)**2)
            n_edge = min(nA+1, X.shape[0])
            idx = min_rse >= np.sort(min_rse)[-n_edge]
            if np.sum(idx) > 0:
                Xe, ye = X[idx], y[idx]
                u_new[ok] = np.linalg.solve(Xe.T @ Xe + 1e-10*np.eye(nA), Xe.T @ ye)
            else:
                u_new[ok] = np.random.randn(nA)*0.1
            s_new = np.ones(nk); s_new[:ok]=ps; s_new[ok]=np.mean(ps)
            pm = np.array(prev.mixture_probabilities)
            m_new = np.zeros(nk); m_new[:ok]=pm*(ok/nk); m_new[ok]=1.0/nk
            floor = 1.0/(nk*20); m_new = np.maximum(m_new,floor); m_new/=np.sum(m_new)
            kw = {"u_init":u_new, "sigma2_init":s_new, "mp_init":m_new}
        result = inference.mop(pcvector, link_functions, k,
                               lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls, **kw)
        val = inference.pbic(pcvector, result, link_functions,
                             lambda_u=lu, lambda_sigma=ls, lambda_pi=lpi, s2=s2_bar)
        if val < best_bic:
            best_bic = val; best_k = k; best_result = result
        prev = result

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
                                             1/(ls * s2_bar / 2 + 1e-10) if ls * s2_bar > 0 else 1)
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
