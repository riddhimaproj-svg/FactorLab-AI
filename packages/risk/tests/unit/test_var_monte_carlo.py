"""Tests for Monte Carlo VaR / ES."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_risk.errors import RiskInputError
from factorlab_risk.var import monte_carlo as MC
from factorlab_risk.var import parametric as P


def test_mc_var_converges_to_parametric_normal() -> None:
    mc = MC.monte_carlo_var(mean=0.0, std=0.02, confidence=0.95, n_simulations=500_000, seed=1)
    param = P.parametric_var(mean=0.0, std=0.02, confidence=0.95)
    assert mc == pytest.approx(param, rel=0.02)


def test_mc_es_converges() -> None:
    mc = MC.monte_carlo_expected_shortfall(
        mean=0.0, std=0.02, confidence=0.95, n_simulations=500_000, seed=2
    )
    param = P.parametric_expected_shortfall(mean=0.0, std=0.02, confidence=0.95)
    assert mc == pytest.approx(param, rel=0.03)


def test_mc_reproducible_with_seed() -> None:
    a = MC.monte_carlo_var(mean=0.0, std=0.02, n_simulations=10_000, seed=7)
    b = MC.monte_carlo_var(mean=0.0, std=0.02, n_simulations=10_000, seed=7)
    assert a == b


def test_mc_from_returns(rng) -> None:
    r = rng.normal(0.0, 0.015, 2000)
    mc = MC.monte_carlo_var(r, confidence=0.95, n_simulations=200_000, seed=3)
    assert mc > 0


def test_mc_t_distribution_fatter() -> None:
    normal = MC.monte_carlo_var(
        mean=0.0, std=0.02, confidence=0.99, distribution="normal", n_simulations=300_000, seed=4
    )
    t = MC.monte_carlo_var(
        mean=0.0, std=0.02, confidence=0.99, distribution="t", dof=4,
        n_simulations=300_000, seed=4,
    )
    assert t > normal


def test_portfolio_mc_var(weights, covariance) -> None:
    mean_vec = np.array([0.001, 0.0008, 0.0012])
    v = MC.monte_carlo_portfolio_var(
        weights, mean_vec, covariance, confidence=0.95, n_simulations=200_000, seed=5
    )
    es = MC.monte_carlo_portfolio_var(
        weights, mean_vec, covariance, confidence=0.95, n_simulations=200_000, seed=5,
        expected_shortfall=True,
    )
    assert es > v > 0


def test_simulate_portfolio_returns_shape(weights, covariance) -> None:
    mean_vec = np.zeros(3)
    sims = MC.simulate_portfolio_returns(weights, mean_vec, covariance, n_simulations=1000, seed=0)
    assert sims.shape == (1000,)


def test_validation() -> None:
    with pytest.raises(RiskInputError):
        MC.monte_carlo_var(confidence=0.95)  # no returns and no mean/std
    with pytest.raises(RiskInputError):
        MC.monte_carlo_var(mean=0.0, std=0.02, distribution="t", dof=1.5)
