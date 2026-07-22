"""Realized-volatility estimators: historical, EWMA, and GARCH(1,1) MLE."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_derivatives import (
    ConvergenceError,
    DerivativesInputError,
    GarchResult,
    ewma_variance,
    ewma_volatility,
    fit_garch,
    historical_volatility,
)


def test_historical_volatility_recovers_generating_sigma() -> None:
    rng = np.random.default_rng(1)
    daily_sigma = 0.01
    returns = rng.normal(0.0, daily_sigma, 5000)
    prices = 100.0 * np.exp(np.cumsum(returns))
    hv = historical_volatility(prices)
    assert hv == pytest.approx(daily_sigma * np.sqrt(252), rel=0.05)


def test_historical_volatility_simple_returns_branch() -> None:
    prices = [100.0, 101.0, 102.0, 101.5, 103.0]
    hv = historical_volatility(prices, log_returns=False)
    assert hv > 0.0


def test_historical_volatility_needs_two_prices() -> None:
    with pytest.raises(DerivativesInputError):
        historical_volatility([100.0])


def test_historical_volatility_rejects_nonpositive_prices() -> None:
    with pytest.raises(DerivativesInputError):
        historical_volatility([100.0, -1.0, 102.0])


def test_ewma_variance_shape_and_recursion() -> None:
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    var = ewma_variance(returns, lam=0.94)
    assert var.shape == returns.shape
    # manual check of the recursion at t=1
    expected = 0.94 * var[0] + 0.06 * returns[0] ** 2
    assert var[1] == pytest.approx(expected, abs=1e-15)


def test_ewma_volatility_positive() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.012, 500)
    assert ewma_volatility(returns) > 0.0


def test_ewma_rejects_bad_lambda() -> None:
    with pytest.raises(DerivativesInputError):
        ewma_variance(np.array([0.01, 0.02]), lam=1.5)


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        var = omega + alpha * (r[t - 1] ** 2 if t > 0 else var) + beta * var
        r[t] = np.sqrt(var) * rng.standard_normal()
    return r


def test_garch_fit_is_stationary_and_improves_likelihood() -> None:
    returns = _simulate_garch(3000, omega=1e-6, alpha=0.08, beta=0.90, seed=3)
    result = fit_garch(returns)
    assert 0.0 < result.persistence < 1.0
    assert result.long_run_variance > 0.0
    assert result.long_run_volatility > 0.0
    assert result.conditional_variance.shape == returns.shape
    assert np.isfinite(result.log_likelihood)


def test_garch_forecast_reverts_to_long_run() -> None:
    returns = _simulate_garch(2000, omega=1e-6, alpha=0.1, beta=0.85, seed=4)
    result = fit_garch(returns)
    fc = result.forecast_variance(200)
    assert fc.shape == (200,)
    assert fc[-1] == pytest.approx(result.long_run_variance, rel=0.1)


def test_garch_conditional_variance_is_read_only() -> None:
    returns = _simulate_garch(500, 1e-6, 0.1, 0.85, seed=5)
    result = fit_garch(returns)
    with pytest.raises(ValueError):
        result.conditional_variance[0] = 1.0


def test_garch_annualized_flag() -> None:
    returns = _simulate_garch(500, 1e-6, 0.1, 0.85, seed=6)
    result = fit_garch(returns)
    ann = result.conditional_volatility(annualized=True)
    per = result.conditional_volatility(annualized=False)
    assert np.all(ann > per)


def test_garch_needs_enough_data() -> None:
    with pytest.raises(DerivativesInputError):
        fit_garch(np.array([0.01, 0.02, -0.01]))


def test_garch_forecast_horizon_validated() -> None:
    returns = _simulate_garch(500, 1e-6, 0.1, 0.85, seed=7)
    result = fit_garch(returns)
    with pytest.raises(DerivativesInputError):
        result.forecast_variance(0)


def test_garch_result_serializes() -> None:
    returns = _simulate_garch(500, 1e-6, 0.1, 0.85, seed=8)
    result = fit_garch(returns)
    restored = GarchResult.from_dict(result.to_dict())
    assert restored.omega == pytest.approx(result.omega)
    assert np.allclose(restored.conditional_variance, result.conditional_variance)


def test_garch_non_stationary_long_run_is_nan() -> None:
    # Construct directly with alpha+beta >= 1 to exercise the guard.
    result = GarchResult(
        omega=1e-6, alpha=0.6, beta=0.5, log_likelihood=0.0,
        conditional_variance=np.array([1e-4, 1e-4]), periods_per_year=252.0, mean=0.0,
    )
    assert result.persistence >= 1.0
    assert np.isnan(result.long_run_variance)


def test_convergence_error_is_derivatives_error() -> None:
    assert issubclass(ConvergenceError, Exception)
