"""Did externalising the parameters change the case-study fits?

`step7_analyze.py` used to hard-code `lu = 1e-6; lpi = 1; ls = nJ / 50` and
`min_weight_share = 0.05` and search `range(1, 10)`. Those are now read from
`experiments/mop_params.json`. Re-running the script after the change reported K = 3
for the corpus where `analysis_output.txt` records K = 4, so either the externalised
values differ from the hard-coded ones or the edit introduced something else.

This prints the full pBIC curve under each setting, so the two can be compared where
they diverge instead of only at the argmin.

Run: python _diag_step7_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import logit

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / ".." / "src"))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE))

import _mop_params                                            # noqa: E402
from mop import inference                                     # noqa: E402
from mop.initialization import warm_init as _warm             # noqa: E402
from mop.inference import comparisons2regression              # noqa: E402
from mop.pairwise_comparison import PairwiseComparison         # noqa: E402

import step7_analyze as S                                     # noqa: E402
from _common import T                                         # noqa: E402


def curve(pairs, ks, lu, lpi, ls, mws):
    """pBIC at each K, using exactly the code path step7_analyze uses."""
    nJ = len(pairs)
    name_to_id = {t: i + 1 for i, t in enumerate(T)}
    pcv = [PairwiseComparison(former_id=name_to_id[p["ft"]],
                              latter_id=name_to_id[p["lt"]],
                              agent_id=1, judgment=p["score"]) for p in pairs]
    eps = 1e-10
    link = [lambda s, _eps=eps: logit(np.clip(s, _eps, 1 - _eps))]
    X, y, _ = comparisons2regression(pcv, link)
    nA = X.shape[1]

    prev, out = None, {}
    for k in ks:
        kw = {}
        if prev is not None:
            pu = np.array([p.priority_vector[:-1] for p in prev.priorities])
            ps = np.array([p.inconsistency_level for p in prev.priorities])
            pm = np.array(prev.mixture_probabilities)
            ui, si, mi = _warm(X, y, pu, ps, pm, min_weight_share=mws,
                               n_components=k)
            kw = {"u_init": ui, "sigma2_init": si, "mp_init": mi}
        r = inference.mop(pcv, link, k, lambda_u=lu, lambda_pi=lpi,
                          lambda_sigma=ls, **kw)
        out[k] = inference.pbic(pcv, r, link, lambda_u=lu, lambda_sigma=ls,
                                lambda_pi=lpi)
        prev = r
    return out


def main():
    pairs = S.load_pairs()
    nJ = len(pairs)
    print(f"corpus: n={nJ} judgments")
    print(f"analysis_output.txt records: K=4, pBIC=18346.1\n")

    print("config in force (experiments/mop_params.json):")
    print(f"  lambda_u={_mop_params.lambda_u:g} lambda_pi={_mop_params.lambda_pi:g} "
          f"lambda_sigma={_mop_params.LAMBDA_SIGMA_OVER_N:g}*nJ="
          f"{_mop_params.lambda_sigma_for(nJ):.4f} "
          f"min_weight_share={_mop_params.min_weight_share:g} "
          f"max_K={_mop_params.max_components}")

    print("\npBIC curve, config values:")
    a = curve(pairs, range(1, _mop_params.max_components + 1),
              _mop_params.lambda_u, _mop_params.lambda_pi,
              _mop_params.lambda_sigma_for(nJ), _mop_params.min_weight_share)
    for k, v in a.items():
        print(f"  K={k}  pBIC={v:12.2f}")
    print(f"  argmin: K={min(a, key=a.get)}")

    print("\npBIC curve, the previously hard-coded values:")
    print(f"  lambda_u=1e-06 lambda_pi=1 lambda_sigma={nJ / 50:.4f} "
          f"min_weight_share=0.05")
    b = curve(pairs, range(1, 10), 1e-6, 1, nJ / 50, 0.05)
    for k, v in b.items():
        print(f"  K={k}  pBIC={v:12.2f}")
    print(f"  argmin: K={min(b, key=b.get)}")

    print("\ndifference (config minus hard-coded):")
    for k in sorted(set(a) & set(b)):
        print(f"  K={k}  {a[k] - b[k]:+.6f}")
    same = all(abs(a[k] - b[k]) < 1e-6 for k in sorted(set(a) & set(b)))
    print(f"\ncurves identical where both computed: {same}")


if __name__ == "__main__":
    main()
