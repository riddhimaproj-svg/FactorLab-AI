"""Tests for rolling VaR/ES and the portfolio VaR decomposition."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from factorlab_risk.errors import DimensionMismatchError, RiskInputError
from factorlab_risk.var import decomposition as D
from factorlab_risk.var import rolling as R
from factorlab_risk.var.historical import historical_var


# -- Rolling --------------------------------------------------------------- #
def test_rolling_var_shape_and_nan_prefix(returns) -> None:
    out = R.rolling_var(returns, window=100, confidence=0.95)
    assert out.shape == returns.shape
    assert np.all(np.isnan(out[:99]))
    assert np.all(np.isfinite(out[99:]))


def test_rolling_var_matches_window(returns) -> None:
    out = R.rolling_var(returns, window=100, confidence=0.95)
    assert out[99] == pytest.approx(historical_var(returns[:100], 0.95))


def test_rolling_parametric_method(returns) -> None:
    out = R.rolling_var(returns, window=60, confidence=0.95, method="parametric")
    assert np.all(np.isfinite(out[59:]))


def test_rolling_es(returns) -> None:
    out = R.rolling_expected_shortfall(returns, window=80, confidence=0.95)
    assert out.shape == returns.shape
    assert np.all(np.isfinite(out[79:]))


def test_rolling_validation(returns) -> None:
    with pytest.raises(RiskInputError):
        R.rolling_var(returns, window=1)
    with pytest.raises(RiskInputError):
        R.rolling_var(returns, window=100000)
    with pytest.raises(RiskInputError):
        R.rolling_var(returns, window=50, method="bogus")


# -- Decomposition --------------------------------------------------------- #
def test_portfolio_var_closed_form(weights, covariance) -> None:
    z = float(stats.norm.ppf(0.99))
    sigma_p = np.sqrt(weights @ covariance @ weights)
    assert D.portfolio_var(weights, covariance, 0.99) == pytest.approx(z * sigma_p)


def test_component_var_sums_to_total(weights, covariance) -> None:
    total = D.portfolio_var(weights, covariance, 0.975)
    comp = D.component_var(weights, covariance, 0.975)
    assert np.sum(comp) == pytest.approx(total)


def test_marginal_var_definition(weights, covariance) -> None:
    z = float(stats.norm.ppf(0.95))
    sigma_p = np.sqrt(weights @ covariance @ weights)
    expected = z * (covariance @ weights) / sigma_p
    np.testing.assert_allclose(D.marginal_var(weights, covariance, 0.95), expected)


def test_percent_contribution_sums_to_one(weights, covariance) -> None:
    pct = D.percent_contribution_var(weights, covariance, 0.95)
    assert np.sum(pct) == pytest.approx(1.0)


def test_incremental_var(weights, covariance) -> None:
    delta = np.array([0.05, -0.05, 0.0])
    inc = D.incremental_var(weights, covariance, delta, 0.95)
    manual = D.portfolio_var(weights + delta, covariance, 0.95) - D.portfolio_var(
        weights, covariance, 0.95
    )
    assert inc == pytest.approx(manual)


def test_zero_vol_marginal_is_zero() -> None:
    w = np.array([0.5, 0.5])
    cov = np.zeros((2, 2))
    np.testing.assert_allclose(D.marginal_var(w, cov, 0.95), [0.0, 0.0])


def test_decomposition_dim_mismatch(covariance) -> None:
    with pytest.raises(DimensionMismatchError):
        D.portfolio_var(np.array([0.5, 0.5]), covariance, 0.95)
    with pytest.raises(DimensionMismatchError):
        D.incremental_var(np.array([0.4, 0.35, 0.25]), covariance, np.array([0.1, 0.0]), 0.95)
