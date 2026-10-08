"""Compute revelatory metrics: what MoP discovers that P=1 cannot.

Usage:
    uv run python experiments/simulation/step4_revelatory.py
"""

import json

import numpy as np
from scipy.special import logit

from mop import inference
from mop.pairwise_comparison import PairwiseComparison, comparisons2regression

from _common import SIM_DIR, ENERGY_PAIRS, CANONICAL_TYPES, TYPE_MAP
from mop import expected_utility
from mop.initialization import warm_init

EPS = 1e-10
T = CANONICAL_TYPES
TYPE_MAP_FULL = {**TYPE_MAP, "燃油": "燃油", "纯电": "纯电", "插混": "插混", "增程": "增程"}

# ── Load data ─────────────────────────────────────────────────────

def load_pairs(dim=None):
    pairs = []
    with open(ENERGY_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for p in r.get("comparison_pairs", []):
                if dim and p.get("dimension") != dim:
                    continue
                ft = TYPE_MAP_FULL.get(p.get("former_type", ""), "")
                lt = TYPE_MAP_FULL.get(p.get("latter_type", ""), "")
                if ft in T and lt in T and ft != lt:
                    pairs.append({"ft": T.index(ft), "lt": T.index(lt),
                                  "score": p.get("score", 0.5)})
    return pairs


# ── Fit MoP (same as step1) ───────────────────────────────────────

def fit_mop(pairs, nA, max_K=8):
    name_to_id = {i: i+1 for i in range(nA)}
    pcv = [PairwiseComparison(former_id=name_to_id[p["ft"]],
                               latter_id=name_to_id[p["lt"]],
                               agent_id=1, judgment=p["score"]) for p in pairs]
    link_fns = [lambda s: logit(np.clip(s, EPS, 1-EPS))]
    X, y, _ = comparisons2regression(pcv, link_fns)
    nJ, nA_m1 = len(pcv), X.shape[1]
    lu = 1e-6; lpi = 1; ls = nJ / 50
    u_pool = np.linalg.solve(X.T@X + 1e-10*np.eye(nA_m1), X.T@y)
    s2_bar = max(np.sum((X@u_pool - y)**2)/max(nJ-nA_m1+1, 1), 1e-10)

    prev = None
    best_k, best_bic, best_result, best_ll = None, np.inf, None, None
    models = {}

    for k in range(1, max_K+1):
        kw = {}
        if prev is not None:
            pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
            ps = np.array([p.inconsistency_level for p in prev.priorities])
            pm = np.array(prev.mixture_probabilities)
            u_init, sigma2_init, mp_init = warm_init(
                X, y, pu, ps, pm, min_weight_share=0.05,
                n_components=k)
            kw = {"u_init": u_init, "sigma2_init": sigma2_init, "mp_init": mp_init}

        result = inference.mop(pcv, link_fns, k, lambda_u=lu, lambda_pi=lpi, lambda_sigma=ls, **kw)
        bic_val = inference.pbic(pcv, result, link_fns, lambda_u=lu, lambda_sigma=ls, lambda_pi=lpi, s2=s2_bar)
        ll = inference.log_likelihood_from_pc(pcv, result, link_fns)

        models[k] = {"pis": result.mixture_probabilities.copy(),
                      "ws": np.array([pr.priority_vector for pr in result.priorities]),
                      "sigma2": np.array([pr.inconsistency_level for pr in result.priorities]),
                      "bic": bic_val, "ll": ll}
        if bic_val < best_bic:
            best_bic = bic_val; best_k = k; best_result = result; best_ll = ll
        prev = result

        # If minprior is turned on and a candidate loses a component to it,
        # raising the candidate further can only reproduce that mixture, so the
        # search stops there. With the default of no removal the test never
        # fires and the whole grid is searched.
        if len(result) < k:
            break

    return best_k, models


# ── BT log-likelihood ─────────────────────────────────────────────

def bt_own_log_likelihood(pairs, w):
    """BT's own log-likelihood (Bernoulli on probability scale)."""
    ll = 0.0
    eps = 1e-10
    for p in pairs:
        i, j, s = int(p["ft"]), int(p["lt"]), p["score"]
        prob = np.clip(w[i] / (w[i] + w[j] + eps), eps, 1 - eps)
        ll += s * np.log(prob) + (1.0 - s) * np.log(1.0 - prob)
    return float(ll)


def fit_bt(pairs, nA):
    """Fit BT, return (w, log-likelihood)."""
    wins = np.zeros(nA)
    N = np.zeros((nA, nA))
    for p in pairs:
        i, j, s = int(p["ft"]), int(p["lt"]), p["score"]
        wins[i] += s
        wins[j] += (1.0 - s)
        N[i, j] += 1.0
        N[j, i] += 1.0
    w = np.ones(nA) / nA
    for _ in range(500):
        w_old = w.copy()
        for i in range(nA):
            denom = 0.0
            for j in range(nA):
                if i != j and N[i, j] > 0:
                    denom += N[i, j] / (w[i] + w[j])
            w[i] = wins[i] / denom if denom > 0 else 1.0 / nA
        w /= w.sum()
        if np.max(np.abs(w - w_old)) < 1e-8:
            break
    return w, bt_own_log_likelihood(pairs, w)


# ── Revelatory metrics ────────────────────────────────────────────

def component_divergence(ws):
    """Mean pairwise cosine distance between component priority vectors."""
    K = len(ws)
    if K < 2:
        return 0.0
    dists = []
    for i in range(K):
        for j in range(i+1, K):
            sim = np.dot(ws[i], ws[j]) / (np.linalg.norm(ws[i])*np.linalg.norm(ws[j]) + 1e-300)
            dists.append(1.0 - sim)
    return float(np.mean(dists))


def information_gain(ll_mop, ll_p1, n_params_mop, n_params_p1, n_pairs):
    """AIC-like information gain per comparison."""
    # LL gain
    ll_gain = ll_mop - ll_p1
    # Parameter cost
    delta_params = n_params_mop - n_params_p1  # each extra component: (nA-1) w + 1 pi + 1 sigma2
    # Net gain per comparison
    net_per_pair = (ll_gain - delta_params) / n_pairs
    return {"ll_gain": float(ll_gain), "delta_params": delta_params,
            "net_per_pair": float(net_per_pair), "bic_gain": float(2*ll_gain - np.log(n_pairs)*delta_params)}


def analyze(label, pairs, nA=4):
    n = len(pairs)
    if n < 30:
        return None

    best_k, models = fit_mop(pairs, nA)

    # P=1 is K=1
    p1 = models[1]
    mop = models[best_k]

    ws = mop["ws"]
    pis = mop["pis"]
    sigmas = mop["sigma2"]

    # Component divergence
    div = component_divergence(ws)

    # LL gain: MoP(P) vs MoP(1) — nested comparison, same likelihood
    ll_gain = models[best_k]["ll"] - models[1]["ll"]

    # Rankings
    def rank_str(w):
        idx = np.argsort(-w)
        return " > ".join([T[i] for i in idx])

    print(f"\n{'='*65}")
    print(f"  {label}  (n={n},  candidate P={best_k}, fitted {len(pis)})")
    print(f"{'='*65}")
    print(f"  P=1 LL: {models[1]['ll']:.1f}    MoP LL: {models[best_k]['ll']:.1f}    Δℓ: {ll_gain:.1f}")
    print(f"  Component divergence: {div:.4f}")
    print(f"  Components:")
    # if minprior is on the fitted mixture can be smaller than the candidate P
    for i in range(len(pis)):
        print(f"    C{i+1}: π={pis[i]:.4f}  σ²={sigmas[i]:.4f}  {rank_str(ws[i])}")
    # Expected utility, the aggregate the manuscript reports
    u_bar = expected_utility(ws, pis)
    print(f"    Expected utility: {rank_str(u_bar)}")
    print(f"  P=1: π=1.0000  σ²={models[1]['sigma2'][0]:.4f}  {rank_str(models[1]['ws'][0])}")

    return {
        "n": n, "best_k": best_k,
        "p1_ll": float(models[1]["ll"]),
        "mop_ll": float(models[best_k]["ll"]),
        "ll_gain": float(ll_gain),
        "component_divergence": div,
        "p1_sigma2": float(models[1]["sigma2"][0]),
        "components": [{"pi": float(pis[i]), "sigma2": float(sigmas[i]),
                         "priority": ws[i].tolist(),
                         "ranking": rank_str(ws[i])}
                        for i in range(len(pis))],
        "p1_ranking": rank_str(models[1]["ws"][0]),
        "expected_utility_ranking": rank_str(u_bar),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    results = {}

    # Overall
    results["overall"] = analyze("Overall", load_pairs())

    # Per dimension
    for dim in ["cost", "driving_experience", "range", "refueling_convenience",
                "reliability", "policy", "other"]:
        pairs = load_pairs(dim=dim)
        r = analyze(dim.replace("_", " ").title(), pairs)
        if r:
            results[dim] = r

    # Save
    out_path = SIM_DIR / "revelatory_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
