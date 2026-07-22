"""Tests for risk attribution."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_risk import attribution as A
from factorlab_risk.errors import DimensionMismatchError, RiskInputError


def test_component_contributions_sum_to_volatility(weights, covariance) -> None:
    ccr = A.component_contribution_to_risk(weights, covariance)
    assert np.sum(ccr) == pytest.approx(A.portfolio_volatility(weights, covariance))


def test_percentage_contributions_sum_to_one(weights, covariance) -> None:
    pct = A.percentage_contribution_to_risk(weights, covariance)
    assert np.sum(pct) == pytest.approx(1.0)


def test_marginal_contribution_formula(weights, covariance) -> None:
    sigma = A.portfolio_volatility(weights, covariance)
    expected = (covariance @ weights) / sigma
    np.testing.assert_allclose(A.marginal_contribution_to_risk(weights, covariance), expected)


def test_risk_budget_equals_pct_contribution(weights, covariance) -> None:
    np.testing.assert_allclose(
        A.risk_budget(weights, covariance),
        A.percentage_contribution_to_risk(weights, covariance),
    )


def test_risk_budget_deviation(weights, covariance) -> None:
    target = np.array([1 / 3, 1 / 3, 1 / 3])
    dev = A.risk_budget_deviation(weights, covariance, target)
    assert np.sum(dev) == pytest.approx(0.0, abs=1e-12)  # both sum to 1


def test_asset_contribution_alias(weights, covariance) -> None:
    np.testing.assert_allclose(
        A.asset_contribution(weights, covariance),
        A.component_contribution_to_risk(weights, covariance),
    )


def test_factor_risk_attribution_decomposes(weights) -> None:
    B = np.array([[1.0, 0.2], [0.9, -0.1], [1.1, 0.3]])
    F = np.array([[0.04, 0.0], [0.0, 0.02]])
    d = np.array([0.01, 0.015, 0.008])
    fa = A.factor_risk_attribution(weights, B, F, d)
    assert fa.total_variance == pytest.approx(fa.systematic_variance + fa.specific_variance)
    # systematic variance matches w'BFB'w
    b_p = B.T @ weights
    assert fa.systematic_variance == pytest.approx(b_p @ F @ b_p)
    # factor contributions sum to systematic variance
    assert np.sum(fa.factor_variance_contributions) == pytest.approx(fa.systematic_variance)
    assert fa.systematic_fraction + fa.specific_fraction == pytest.approx(1.0)
    assert "systematic_variance" in fa.to_dict()


def test_factor_attribution_validation(weights) -> None:
    with pytest.raises(RiskInputError):
        A.factor_risk_attribution(weights, np.ones(3), np.eye(1), np.zeros(3))  # B not 2D
    with pytest.raises(DimensionMismatchError):
        A.factor_risk_attribution(
            weights, np.ones((3, 2)), np.eye(3), np.zeros(3)  # F wrong shape
        )
    with pytest.raises(RiskInputError):
        A.factor_risk_attribution(
            weights, np.ones((3, 1)), np.eye(1), np.array([-1.0, 0.0, 0.0])  # negative specific
        )


def test_sector_attribution(weights, covariance) -> None:
    sectors = ["tech", "tech", "energy"]
    result = A.sector_risk_attribution(weights, covariance, sectors)
    assert set(result) == {"tech", "energy"}
    # sector totals sum to portfolio volatility
    assert sum(result.values()) == pytest.approx(A.portfolio_volatility(weights, covariance))


def test_sector_attribution_dim_mismatch(weights, covariance) -> None:
    with pytest.raises(DimensionMismatchError):
        A.sector_risk_attribution(weights, covariance, ["tech", "energy"])
