"""Load the MoP fitting parameters from the single shared config."""
from __future__ import annotations

import json
import math
from pathlib import Path

PARAMS_PATH = Path(__file__).resolve().parents[1] / "config" / "mop_params.json"

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

#: upper end of the component search
max_components: int = _get("max_components")

#: how phi_0 is set: "data" for the paper's rule, or a number
prior_mode_phi0: object = _get("prior_mode_phi0")

selection_criterion: str = _get("selection_criterion")

reference_type: str = _get("reference_type")


def max_components_for(n_alternatives: int) -> int:
    """Upper end of the component search for a problem with ``A`` alternatives."""
    return min(int(max_components), math.factorial(int(n_alternatives)))


def lambda_sigma_for(n_judgments: int) -> float:
    """lambda_sigma = 0.02 * n_judgments, the paper's nJ / 50."""
    value = LAMBDA_SIGMA_OVER_N * float(n_judgments)
    return value


def resolve_phi0(X, y, override=None):
    """The prior mode phi_0, as a float."""
    from mop.inference import prior_mode_phi0 as _pm
    mode = prior_mode_phi0 if override is None else override
    if mode == "data":
        return float(_pm(X, y))
    return float(mode)


def summary() -> str:
    """One-line description of the parameters in force, for run logs."""
    return (f"lambda_u={lambda_u:g} lambda_pi={lambda_pi:g} "
            f"lambda_sigma={LAMBDA_SIGMA_OVER_N:g}*nJ "
            f"max_K={max_components} "
            f"phi0={prior_mode_phi0} criterion={selection_criterion}")


if __name__ == "__main__":
    print(f"loaded from {PARAMS_PATH}")
    for k, v in PARAMS.items():
        if not k.startswith("_"):
            print(f"  {k} = {v}")
    print(f"\nsummary: {summary()}")
