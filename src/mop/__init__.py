"""Mixture of Preferences: model, inference and selection criteria."""

from .criteria import bic, degrees_of_freedom, lbic, pbic
from .general_fit import BinomialResponse, GaussianResponse, bic_map, fit_map
from .inference import (
    e_gamma,
    e_gamma_from_pc,
    fit_mixture,
    log_likelihood,
    penalized_log_likelihood,
)
from .initialization import kridge, ridge_init, seed_component
from .mixture_of_priorities import MixtureOfPriorities, Priority, expected_utility
from .pairwise_comparison import PairwiseComparison, comparisons2regression
from .winloss import (
    binom_logpdf,
    e_gamma_winloss,
    fit_winloss,
    log_likelihood_from_pc as log_likelihood_winloss_from_pc,
    penalized_log_posterior,
    pbic as pbic_winloss,
    select_n_components,
)

__all__ = [
    "BinomialResponse",
    "GaussianResponse",
    "MixtureOfPriorities",
    "PairwiseComparison",
    "Priority",
    "bic",
    "bic_map",
    "binom_logpdf",
    "comparisons2regression",
    "degrees_of_freedom",
    "e_gamma",
    "e_gamma_from_pc",
    "e_gamma_winloss",
    "expected_utility",
    "fit_map",
    "fit_mixture",
    "fit_winloss",
    "kridge",
    "lbic",
    "log_likelihood",
    "log_likelihood_winloss_from_pc",
    "pbic",
    "pbic_winloss",
    "penalized_log_likelihood",
    "penalized_log_posterior",
    "ridge_init",
    "seed_component",
    "select_n_components",
]
