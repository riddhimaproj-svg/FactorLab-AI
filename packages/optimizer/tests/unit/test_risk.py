"""Tests for the risk-decomposition module."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import risk


def test_variance_and_volatility() -> None:
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    # var = 0.25*0.04 + 0.25*0.09 = 0.0325
    assert risk.portfolio_variance(w, cov) == pytest.approx(0.0325)
    assert risk.portfolio_volatility(w, cov) == pytest.approx(np.sqrt(0.0325))


def test_risk_contributions_sum_to_volatility() -> None:
    rng = np.random.default_rng(0)
    L = np.tril(rng.uniform(0.05, 0.2, size=(4, 4)))
    cov = L @ L.T + np.eye(4) * 1e-3
    w = np.array([0.4, 0.1, 0.3, 0.2])
    rc = risk.risk_contributions(w, cov)
    assert np.sum(rc) == pytest.approx(risk.portfolio_volatility(w, cov))


def test_variance_decomposition_sums_to_variance() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    vd = risk.variance_decomposition(w, cov)
    assert np.sum(vd) == pytest.approx(risk.portfolio_variance(w, cov))


def test_percent_risk_contributions_sum_to_one() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.5, 0.5])
    prc = risk.percent_risk_contributions(w, cov)
    assert np.sum(prc) == pytest.approx(1.0)


def test_marginal_contribution_zero_vol() -> None:
    cov = np.zeros((2, 2))
    w = np.array([0.5, 0.5])
    np.testing.assert_allclose(risk.marginal_risk_contributions(w, cov), [0.0, 0.0])
    assert np.sum(risk.percent_risk_contributions(w, cov)) == 0.0


def test_diversification_ratio() -> None:
    # uncorrelated equal-vol assets: DR = 1/sqrt(sum w^2 * ... ) > 1
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    w = np.array([0.5, 0.5])
    # weighted vol = 0.5*0.2+0.5*0.2 = 0.2 ; port vol = sqrt(0.25*0.04+0.25*0.04)=sqrt(0.02)
    assert risk.diversification_ratio(w, cov) == pytest.approx(0.2 / np.sqrt(0.02))


def test_diversification_ratio_zero_vol_is_nan() -> None:
    assert np.isnan(risk.diversification_ratio(np.array([0.0, 0.0]), np.zeros((2, 2))))
