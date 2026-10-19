"""Load the MoP fitting parameters from the single shared config.

Why this module exists
----------------------
The same five values were once copied into four experiment scripts --

    step7_analyze.py, simulation/step1_simulation.py,
    simulation/step4_revelatory.py, simulation/_baselines.py

each carrying its own ``lu = 1e-6; lpi = 1; ls = nJ / 50``, and the two
``config.json`` files described those values without any script reading them. A
change to one copy would therefore have disagreed silently with the others, and the
config would have gone on describing the old numbers. Both the duplicate literals and
the unread config are gone: everything comes from ``experiments/mop_params.json``
through this module.

Usage
-----
    from _mop_params import lambda_u, lambda_pi, lambda_sigma_for, min_weight_share

``lambda_sigma`` depends on the number of judgments being fitted, so it is not a
constant: call ``lambda_sigma_for(nJ)``.
"""
from __future__ import annotations

import json
from pathlib import Path

PARAMS_PATH = Path(__file__).resolve().parent / "mop_params.json"

with open(PARAMS_PATH, encoding="utf-8") as _f:
    PARAMS = json.load(_f)


def _get(key):
    if key not in PARAMS or key.startswith("_"):
        raise KeyError(f"{PARAMS_PATH.name} has no key {key!r}")
    return PARAMS[key]


#: ridge coefficient on the utilities
lambda_u: float = _get("lambda_u")

#: Dirichlet concentration on the mixture weights
lambda_pi: float = _get("lambda_pi")

#: multiplier turning nJ into lambda_sigma
LAMBDA_SIGMA_OVER_N: float = _get("lambda_sigma_over_n")

#: floor on a component's share when warm-starting from P-1 to P components
min_weight_share: float = _get("min_weight_share")

#: upper end of the component search
max_components: int = _get("max_components")

#: how phi_0 is set: "data" for the paper's rule, or a number
prior_mode_phi0: object = _get("prior_mode_phi0")

selection_criterion: str = _get("selection_criterion")

reference_type: str = _get("reference_type")


def lambda_sigma_for(n_judgments: int) -> float:
    """lambda_sigma = 0.02 * n_judgments, the paper's nJ / 50.

    The coefficient in the config is 0.02, not 0.05, because 1/50 = 0.02. It was
    first written as 0.05, which is nJ/20 -- a different prior by a factor of 2.5 --
    and the error was caught only because `step7_analyze.py` stopped reproducing the
    case study's recorded output: at 0.05 the corpus pBIC curve bottoms out at K = 3
    where the recorded run has K = 4. _mop_params_check.py now pins the identity
    against the recorded output so this cannot drift again.

    Section 2.4 needs lambda_sigma > 2 for the prior mode on phi to exist.
    """
    value = LAMBDA_SIGMA_OVER_N * float(n_judgments)
    return value


def resolve_phi0(X, y, override=None):
    """The prior mode phi_0, as a float.

    ``override`` is the same knob ``step9_phi_identification.py --s2`` exposes: None
    means use ``prior_mode_phi0`` as configured, a number pins it. Whatever the
    choice, the caller must pass the SAME value to every model it compares -- fixing
    phi at one is itself a claim about phi_0.
    """
    from mop.inference import prior_mode_phi0 as _pm
    mode = prior_mode_phi0 if override is None else override
    if mode == "data":
        return float(_pm(X, y))
    return float(mode)


def summary() -> str:
    """One-line description of the parameters in force, for run logs."""
    return (f"lambda_u={lambda_u:g} lambda_pi={lambda_pi:g} "
            f"lambda_sigma={LAMBDA_SIGMA_OVER_N:g}*nJ "
            f"min_weight_share={min_weight_share:g} max_K={max_components} "
            f"phi0={prior_mode_phi0} criterion={selection_criterion}")


if __name__ == "__main__":
    print(f"loaded from {PARAMS_PATH}")
    for k, v in PARAMS.items():
        if not k.startswith("_"):
            print(f"  {k} = {v}")
    print(f"\nsummary: {summary()}")
