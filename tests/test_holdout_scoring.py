"""The scoring rules of the holdout evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import expit, logit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]


def _load_scorer():
    """Import the holdout scorer as a package module."""
    sys.path.insert(0, str(ROOT / "experiments"))
    import case_study.holdout as mod

    return mod


S = _load_scorer()


# One judgment: the first energy type against the second, utility gap 0.8.
X = np.array([[1.0, 0.0, 0.0]])
MU = np.array([[0.8, 0.0, 0.0]])
SCORE = 0.62
Y = np.array([logit(SCORE)])


def test_crps_reduces_to_absolute_error_for_a_point_forecast():
    """A forecast concentrated on one value has to score its absolute error."""
    _, _, med, Fm = S.predictive(MU, [1e-9], [1.0], X)
    got = float(S.crps(Fm, Y)[0])
    assert got == pytest.approx(abs(SCORE - expit(med[0])), abs=2e-4)


def test_crps_matches_a_fine_reference_integration():
    """The grid integral agrees with a dense integration on the probability scale."""
    mu = np.array([[0.8, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    _, _, _, Fm = S.predictive(mu, [0.2, 0.9], [0.3, 0.7], X)
    got = float(S.crps(Fm, Y)[0])

    grid = np.linspace(1e-9, 1 - 1e-9, 2_000_001)
    z = np.log(grid / (1.0 - grid))
    F = (0.3 * norm.cdf((z - 0.8) / np.sqrt(0.2))
         + 0.7 * norm.cdf((z + 0.5) / np.sqrt(0.9)))
    integrand = np.where(grid < SCORE, F ** 2, (1.0 - F) ** 2)
    assert got == pytest.approx(float(trapezoid(integrand, grid)), abs=2e-5)


def test_mixture_mean_is_the_weighted_average_of_component_means():
    mu = np.array([[0.8, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    pis = np.array([0.3, 0.7])
    mean, _, _, _ = S.predictive(mu, [0.2, 0.4], pis, X)
    parts = [S.predictive(mu[i:i + 1], [v], [1.0], X)[0][0]
             for i, v in enumerate((0.2, 0.4))]
    assert float(mean[0]) == pytest.approx(float(pis @ np.array(parts)), rel=1e-10)


def test_log_score_is_the_log_of_the_mixture_density():
    mu = np.array([[0.8, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    s2 = np.array([0.2, 0.4])
    pis = np.array([0.3, 0.7])
    got = float(S.log_score(mu, s2, pis, X, Y)[0])
    z = logit(SCORE)
    want = float(np.log(sum(p * norm.pdf(z, m[0], np.sqrt(v))
                            for p, m, v in zip(pis, mu, s2))))
    assert got == pytest.approx(want, rel=1e-10)


def test_median_sits_at_the_cdf_crossing():
    _, _, med, _ = S.predictive(np.array([[1.2, -0.4, 0.2]]), [0.6], [1.0], X)
    half = norm.cdf((med[0] - 1.2) / np.sqrt(0.6))
    assert half == pytest.approx(0.5, abs=1e-6)


def test_predictions_are_formed_without_the_held_out_scores():
    """The distribution does not move when the labels it is scored against do."""
    mu = np.array([[0.8, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    s2, pis = [0.2, 0.4], [0.3, 0.7]
    a = S.score_one(mu, s2, pis, X, Y)
    b = S.score_one(mu, s2, pis, X, np.array([logit(0.31)]))
    assert np.array_equal(a["pred_sd"], b["pred_sd"])
    assert np.array_equal(a["pred_med"], b["pred_med"])
    assert not np.allclose(a["crps"], b["crps"])
