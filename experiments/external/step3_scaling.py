"""Does more data fix the confounding, or does the likelihood have to change?

The previous experiment (step2_separability.py) fitted one model per
participant x set unit, so every unit had only 10 observations. That conflates
two different limitations: too little data, and a variance function that is
locked at n * mu * (1 - mu).

This script separates them. A population of D decision makers shares one set of
latent preference components and one mixture weight vector; each decision maker
contributes one count per pair (45 repetitions). Increasing D raises the number
of observations without touching the number of alternatives -- which matches
both the published dataset (30 participants x 10 pairs) and the case study
(thousands of authors over 4 energy types).

Prediction under test:

  locked rho      over-selection should NOT shrink as D grows, because extra
                  dispersion has nowhere else to go; more data makes the
                  spurious component *more* certain, not less.
  free rho        over-selection should shrink as D grows, and rho_hat should
                  approach rho_gen.

If that is what happens, the repair is the likelihood, not the sample size.

Usage:
    uv run python experiments/external/step3_scaling.py
    uv run python experiments/external/step3_scaling.py --n-rep 4 --p-max 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import betabinom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mop import inference_binomial as ib  # noqa: E402

HERE = Path(__file__).resolve().parent
N_ALT = 5                       # 10 pairs, as in the published dataset
PAIRS = [(i, j) for i in range(N_ALT) for j in range(i + 1, N_ALT)]
N_TRIALS = 45
MODES = ("binomial", "beta-binomial")


def simulate_population(u_true, pi_true, rho_gen, n_agents, rng):
    """D decision makers share (u_true, pi_true); each judges every pair 45 times."""
    n_comp = u_true.shape[0]
    u_full = np.hstack([u_true, np.zeros((n_comp, 1))])   # last alternative at 0
    counts = []
    for _ in range(n_agents):
        for i, j in PAIRS:
            c = int(rng.choice(n_comp, p=pi_true))
            mu = float(expit(u_full[c, i] - u_full[c, j]))
            if rho_gen <= 0.0:
                m = int(rng.binomial(N_TRIALS, mu))
            else:
                conc = (1.0 - rho_gen) / rho_gen
                m = int(betabinom.rvs(N_TRIALS, mu * conc, (1.0 - mu) * conc,
                                      random_state=rng))
            counts.append({"former": i, "latter": j,
                           "successes": float(m), "trials": N_TRIALS})
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--p-max", type=int, default=4)
    ap.add_argument("--n-starts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260202)
    ap.add_argument("--out", type=Path, default=HERE / "scaling_results.json")
    args = ap.parse_args()

    agents = [1, 10, 30]
    p_true_vals = [1, 2]
    rho_vals = [0.0, 0.25]
    cells = []

    for p_true in p_true_vals:
        for rho_gen in rho_vals:
            for n_agents in agents:
                for rep in range(args.n_rep):
                    rng = np.random.default_rng(
                        args.seed + 7919 * rep + 131 * p_true
                        + 17 * n_agents + int(rho_gen * 1000))
                    u_true = rng.normal(0.0, 1.0, (p_true, N_ALT - 1))
                    pi_true = (np.full(p_true, 1.0 / p_true) if p_true == 1
                               else rng.dirichlet(np.full(p_true, 2.0)))
                    counts = simulate_population(u_true, pi_true, rho_gen, n_agents, rng)

                    record = {"P_true": p_true, "rho_gen": rho_gen,
                              "n_agents": n_agents, "n_obs": len(counts), "replicate": rep}
                    for mode in MODES:
                        best_p, results = ib.select_components(
                            counts, N_ALT, p_max=args.p_max, phi_mode=mode,
                            n_starts=args.n_starts)
                        model = results[best_p]["model"]
                        record[mode] = {
                            "selected_P": int(best_p),
                            "mean_rho": float(np.mean(model.rho)),
                        }
                    cells.append(record)
                print(f"  P_true={p_true}  rho_gen={rho_gen:<5}  D={n_agents:<3} "
                      f"({len(counts):>4} obs)  done", flush=True)

    print()
    print("=" * 92)
    print("selected number of components vs number of decision makers")
    print("=" * 92)
    summary = []
    for p_true in p_true_vals:
        for rho_gen in rho_vals:
            print(f"\n  P_true = {p_true},  rho_gen = {rho_gen}")
            print(f"  {'D':>4} {'obs':>5} | {'locked rho: mean P':>19} {'= P_true':>9} | "
                  f"{'free rho: mean P':>18} {'= P_true':>9} {'mean rho_hat':>13}")
            print("  " + "-" * 88)
            for n_agents in agents:
                sub = [r for r in cells if r["P_true"] == p_true
                       and r["rho_gen"] == rho_gen and r["n_agents"] == n_agents]
                ps_b = np.array([r["binomial"]["selected_P"] for r in sub])
                ps_f = np.array([r["beta-binomial"]["selected_P"] for r in sub])
                rh = np.array([r["beta-binomial"]["mean_rho"] for r in sub])
                row = {"P_true": p_true, "rho_gen": rho_gen, "n_agents": n_agents,
                       "n_obs": sub[0]["n_obs"],
                       "binomial": {"mean_P": float(ps_b.mean()),
                                    "frac_correct": float((ps_b == p_true).mean())},
                       "beta-binomial": {"mean_P": float(ps_f.mean()),
                                         "frac_correct": float((ps_f == p_true).mean()),
                                         "mean_rho_hat": float(rh.mean())}}
                summary.append(row)
                print(f"  {n_agents:>4} {row['n_obs']:>5} | {ps_b.mean():>19.2f} "
                      f"{(ps_b == p_true).mean():>9.2f} | {ps_f.mean():>18.2f} "
                      f"{(ps_f == p_true).mean():>9.2f} {rh.mean():>13.4f}")

    payload = {"n_rep": args.n_rep, "p_max": args.p_max, "n_starts": args.n_starts,
               "seed": args.seed, "summary": summary, "replicates": cells}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
