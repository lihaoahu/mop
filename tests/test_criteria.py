"""The selection criterion, and the identities it has to satisfy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "experiments")):
    sys.path.insert(0, p)

from mop.criteria import (bic, degrees_of_freedom, df_from_design,   # noqa: E402
                          lbic, log_prior_utilities, pbic)


def _reference_log_prior(u, phi, lambda_u):
    """Independent transcription of the normal prior's log density."""
    total = 0.0
    for p in range(u.shape[0]):
        total += -0.5 * lambda_u * float(np.sum(u[p] ** 2)) / phi[p]
        total += -0.5 * u.shape[1] * np.log(2.0 * np.pi * phi[p] / lambda_u)
    return total


def test_log_prior_utilities_matches_definition():
    rng = np.random.default_rng(7)
    u = rng.normal(scale=0.4, size=(3, 3))
    phi = np.array([0.6, 1.0, 1.7])
    for lam in (1e-6, 1e-9):
        got = log_prior_utilities(u, phi, lam)
        assert got == pytest.approx(_reference_log_prior(u, phi, lam), rel=0, abs=1e-10)


def test_log_prior_utilities_accepts_scalar_phi():
    u = np.array([[0.3, -0.2, 0.1]])
    one = log_prior_utilities(u, 1.0, 1e-6)
    arr = log_prior_utilities(u, np.array([1.0]), 1e-6)
    assert one == pytest.approx(arr)


def test_profile_utilities_makes_pbic_invariant_to_lambda_u():
    """A criterion that moves when a numerics setting moves is not a criterion."""
    rng = np.random.default_rng(11)
    u = rng.normal(scale=0.4, size=(4, 3))
    phi = np.array([0.72, 0.81, 0.65, 0.90])
    log_prior_phi, log_prior_pi = 210.4, -3.58
    ll = -8_890.0
    n_eff, df = 5_401.0, 19

    values = []
    for lam in (1e-6, 1e-8, 1e-10):
        # the fit maximises a posterior that contains the utility prior
        pll = ll + log_prior_phi + log_prior_pi + log_prior_utilities(u, phi, lam)
        values.append(pbic(pll, n_eff, df, log_prior_utilities(u, phi, lam)))
    assert max(values) - min(values) == pytest.approx(0.0, abs=1e-9)


def test_leaving_the_utility_prior_in_costs_about_47_per_component():
    """What the unpaired half of the Laplace approximation used to charge."""
    u = np.zeros((1, 3))
    lam = 1e-6
    for phi in (1.0, 0.917, 0.5):
        charge = -2.0 * log_prior_utilities(u, phi, lam)
        assert charge == pytest.approx(3.0 * np.log(2.0 * np.pi * phi / lam),
                                      rel=1e-12)
        if phi == 1.0:
            assert charge == pytest.approx(46.96, abs=0.02)


def test_pbic_minus_bic_is_prior_mass_only():
    n_eff, k, a = 5_401.0, 4, 4
    df = degrees_of_freedom(a, k, phi_free=True)
    ll = -8_889.606
    u = np.array([[0.5, -0.3, 0.2]] * k)
    phi = np.array([0.72, 0.81, 0.65, 0.90])
    # log densities: inverse-Gamma at these dispersions, uniform Dirichlet at K=4
    log_f_phi, log_f_pi = -110.245, np.log(6.0)

    pll = ll + log_f_phi + log_f_pi + log_prior_utilities(u, phi, 1e-6)
    gap = pbic(pll, n_eff, df, log_prior_utilities(u, phi, 1e-6)) \
        - (-2.0 * ll + np.log(n_eff) * df)
    assert gap == pytest.approx(-2.0 * (log_f_phi + log_f_pi), abs=1e-8)
    assert gap == pytest.approx(216.91, abs=0.05)


def test_degrees_of_freedom_counts_a_pinned_dispersion_as_zero():
    assert degrees_of_freedom(4, 3, phi_free=True) == 14
    assert degrees_of_freedom(4, 3, phi_free=False) == 11
    assert df_from_design(3, 3, phi_free=True) == 14


# the paper's own fit, from the record the supplement's table is built from
_PLL_4, _N_EFF, _DF_4, _LNH_4 = -9_088.27219239304, 5_401.0, 19, 104.50050107653557


def test_bic_is_the_dimension_term_and_the_penalised_likelihood():
    assert bic(_PLL_4, _N_EFF, _DF_4) == pytest.approx(
        -2.0 * _PLL_4 + _DF_4 * np.log(_N_EFF), rel=0, abs=1e-9)


def test_bic_and_lbic_differ_by_exactly_the_determinant_and_the_constant():
    """The two criteria are one approximation apart, not two criteria."""
    gap = (lbic(_PLL_4, _LNH_4, _DF_4) - bic(_PLL_4, _N_EFF, _DF_4))
    assert gap == pytest.approx(
        _LNH_4 - _DF_4 * np.log(_N_EFF) - _DF_4 * np.log(2.0 * np.pi),
        rel=0, abs=1e-9)
    # on this fit the stand-in overcharges: 163.29 - 104.50 - 34.92
    assert gap == pytest.approx(-58.8 - 34.9, abs=0.05)


def test_bic_never_returns_inf_but_lbic_does_without_a_mode():
    assert np.isfinite(bic(_PLL_4, _N_EFF, _DF_4))
    assert lbic(_PLL_4, float("nan"), _DF_4) == float("inf")


def test_bic_refuses_a_sample_size_that_is_not_positive():
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            bic(_PLL_4, bad, _DF_4)
