"""Tests for portfolio-risk statistics."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_risk import portfolio_risk as PR
from factorlab_risk.errors import DimensionMismatchError, RiskInputError


def test_portfolio_volatility(weights, covariance) -> None:
    assert PR.portfolio_volatility(weights, covariance) == pytest.approx(
        np.sqrt(weights @ covariance @ weights)
    )


def test_volatility_annualization() -> None:
    r = np.array([0.0, 0.02, -0.01, 0.015, -0.02])
    assert PR.volatility(r, 252) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))
    assert np.isnan(PR.volatility(np.array([0.01]), 252))


def test_rolling_volatility(returns) -> None:
    out = PR.rolling_volatility(returns, window=50, periods_per_year=252)
    assert out.shape == returns.shape
    assert np.all(np.isnan(out[:49])) and np.all(np.isfinite(out[49:]))


def test_tracking_error_and_active_risk(rng) -> None:
    r = rng.normal(0, 0.01, 300)
    b = rng.normal(0, 0.009, 300)
    te = PR.tracking_error(r, b, 252)
    assert te == pytest.approx(np.std(r - b, ddof=1) * np.sqrt(252))
    assert PR.active_risk(r, b, 252) == te


def test_information_ratio(rng) -> None:
    r = rng.normal(0.001, 0.01, 300)
    b = rng.normal(0.0, 0.009, 300)
    active = r - b
    expected = np.mean(active) / np.std(active, ddof=1) * np.sqrt(252)
    assert PR.information_ratio(r, b, 252) == pytest.approx(expected)


def test_beta_and_rolling_beta(rng) -> None:
    b = rng.normal(0, 0.01, 300)
    r = 1.4 * b + rng.normal(0, 0.001, 300)
    assert PR.beta(r, b) == pytest.approx(1.4, abs=0.1)
    rb = PR.rolling_beta(r, b, window=60)
    assert rb.shape == r.shape
    assert rb[-1] == pytest.approx(1.4, abs=0.2)


def test_covariance_and_correlation(returns_matrix) -> None:
    cov = PR.covariance_matrix(returns_matrix)
    corr = PR.correlation_matrix(returns_matrix)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-9)
    np.testing.assert_allclose(cov, np.cov(returns_matrix, rowvar=False, ddof=1))


def test_rolling_covariance_correlation(returns_matrix) -> None:
    rcov = PR.rolling_covariance(returns_matrix, window=100)
    rcorr = PR.rolling_correlation(returns_matrix, window=100)
    assert rcov.shape == (500, 3, 3)
    assert np.all(np.isnan(rcov[98]))
    assert np.all(np.isfinite(rcov[99]))
    np.testing.assert_allclose(np.diag(rcorr[-1]), 1.0, atol=1e-9)


def test_diversification_ratio(weights, covariance) -> None:
    asset_vols = np.sqrt(np.diag(covariance))
    vol = np.sqrt(weights @ covariance @ weights)
    assert PR.diversification_ratio(weights, covariance) == pytest.approx(
        (weights @ asset_vols) / vol
    )


def test_herfindahl_and_effective_n() -> None:
    assert PR.herfindahl_index([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.25)
    assert PR.effective_number_of_assets([0.25, 0.25, 0.25, 0.25]) == pytest.approx(4.0)
    assert PR.herfindahl_index([1.0]) == pytest.approx(1.0)


def test_concentration_ratio_and_metrics() -> None:
    w = np.array([0.5, 0.3, 0.15, 0.05])
    assert PR.concentration_ratio(w, top_n=2) == pytest.approx(0.8)
    m = PR.concentration_metrics(w, top_n=2)
    assert m["max_weight"] == pytest.approx(0.5)
    assert m["effective_number_of_assets"] == pytest.approx(1.0 / np.sum(w**2))


def test_validation(weights, covariance) -> None:
    with pytest.raises(DimensionMismatchError):
        PR.portfolio_volatility(np.array([0.5, 0.5]), covariance)
    with pytest.raises(RiskInputError):
        PR.covariance_matrix(np.array([[0.01, 0.02]]))  # 1 obs
    with pytest.raises(RiskInputError):
        PR.rolling_volatility(np.array([0.01, 0.02, 0.03]), window=1)
