"""Integration tests: risk engine consuming portfolio / optimizer / backtest objects."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_risk import integration as I
from factorlab_risk.errors import RiskInputError

pytestmark = pytest.mark.integration


def test_extract_returns_from_return_series() -> None:
    from factorlab_portfolio import ReturnSeries

    rs = ReturnSeries(np.array([0.01, -0.02, 0.03, -0.01]), periods_per_year=252)
    np.testing.assert_allclose(I.extract_returns(rs), rs.values)


def test_var_report_from_return_series(returns) -> None:
    from factorlab_portfolio import ReturnSeries

    rs = ReturnSeries(returns, periods_per_year=252, name="fund")
    report = I.var_report_from_returns(rs, confidence=0.95, method="historical")
    assert report.var > 0


def test_extract_weights_from_optimizer_result(covariance) -> None:
    from factorlab_optimizer import Constraint, MinVarianceOptimizer, OptimizationProblem

    mu = np.array([0.08, 0.10, 0.09])
    problem = OptimizationProblem(("A", "B", "C"), mu, covariance,
                                  constraints=(Constraint.long_only(),))
    result = MinVarianceOptimizer().optimize(problem)
    assets, weights = I.extract_weights(result)
    assert assets == ("A", "B", "C")
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_risk_report_from_optimizer_result(covariance) -> None:
    from factorlab_optimizer import Constraint, MinVarianceOptimizer, OptimizationProblem

    mu = np.array([0.08, 0.10, 0.09])
    problem = OptimizationProblem(("A", "B", "C"), mu, covariance,
                                  constraints=(Constraint.long_only(),))
    result = MinVarianceOptimizer().optimize(problem)
    report = I.risk_report_from_weights(result, covariance, confidence=0.95)
    assert report.volatility > 0
    assert report.parametric_var is not None
    # decomposition adds up
    total = sum(c.component_contribution for c in report.decomposition.contributions)
    assert total == pytest.approx(report.volatility)


def test_var_report_from_backtest_result() -> None:
    from factorlab_backtesting import (
        Backtest,
        EqualWeightStrategy,
        MarketData,
        RebalanceSchedule,
    )

    rng = np.random.default_rng(0)
    n = 300
    dates = np.datetime64("2020-01-01") + np.arange(n)
    prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=(n, 3)), axis=0)
    md = MarketData(dates, ("A", "B", "C"), prices)
    result = Backtest(md, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    report = I.var_report_from_returns(result, confidence=0.95, method="historical")
    assert report.var > 0


def test_extract_weights_from_mapping() -> None:
    assets, w = I.extract_weights({"A": 0.6, "B": 0.4})
    assert assets == ("A", "B")
    np.testing.assert_allclose(w, [0.6, 0.4])


def test_snapshot_overlays_historical(returns, covariance) -> None:
    weights = {"A": 0.4, "B": 0.35, "C": 0.25}
    snap = I.snapshot_from_returns_and_weights(returns, weights, covariance, confidence=0.95)
    # historical VaR overlaid onto the snapshot
    from factorlab_risk import historical_var

    assert snap.var_95 == pytest.approx(historical_var(returns, 0.95))


def test_extract_failures() -> None:
    with pytest.raises(RiskInputError):
        I.extract_returns(np.zeros((3, 3)))
    with pytest.raises(RiskInputError):
        I.extract_weights(12345)


def test_full_workflow_optimizer_to_risk(covariance) -> None:
    """Optimizer -> weights -> risk report, the tail of the platform workflow."""
    from factorlab_optimizer import Constraint, MaxSharpeOptimizer, OptimizationProblem

    mu = np.array([0.08, 0.12, 0.10])
    problem = OptimizationProblem(("A", "B", "C"), mu, covariance,
                                  constraints=(Constraint.long_only(),))
    result = MaxSharpeOptimizer().optimize(problem)
    report = I.risk_report_from_weights(result, covariance)
    assert "Portfolio Risk Report" in report.summary()
    assert report.concentration["herfindahl_index"] > 0
