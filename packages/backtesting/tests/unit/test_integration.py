"""End-to-end integration: expected returns -> optimizer -> backtest -> report.

Exercises the full platform workflow through the optimizer and portfolio
packages (the factor-model stage is represented by a custom expected-return
estimator feeding the optimizer, which is exactly the integration seam).
"""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import (
    Backtest,
    Benchmark,
    BrokerModel,
    ExecutionEngine,
    OptimizerStrategy,
    PercentageCommission,
    RebalanceSchedule,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def market_data(rng):
    from factorlab_backtesting import MarketData

    n, k = 500, 4
    dates = np.datetime64("2019-01-01") + np.arange(n)
    rets = rng.normal(0.0005, 0.012, size=(n, k))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return MarketData(dates, ("A", "B", "C", "D"), prices)


def test_min_variance_workflow(market_data) -> None:
    from factorlab_optimizer import Constraint, MinVarianceOptimizer

    strat = OptimizerStrategy(
        MinVarianceOptimizer(), lookback=90, constraints=(Constraint.long_only(),)
    )
    engine = ExecutionEngine(BrokerModel(PercentageCommission(0.0005)))
    result = Backtest(
        market_data, strat, RebalanceSchedule.monthly(), engine, periods_per_year=252
    ).run()

    bench = Benchmark.from_prices("EW", market_data.prices.mean(axis=1))
    report = result.report(benchmark=bench, risk_free=0.0)
    assert report.n_rebalances > 0
    assert np.isfinite(report.performance.sharpe_ratio)
    assert np.isfinite(report.beta)
    assert result.total_costs > 0.0


def test_max_sharpe_with_expected_returns_estimator(market_data) -> None:
    """A custom mean estimator stands in for a factor-model expected-return signal."""
    from factorlab_optimizer import Constraint, MaxSharpeOptimizer

    # "Factor model" expected returns: shrink the sample mean toward zero.
    def shrunk_mean(window: np.ndarray) -> np.ndarray:
        return 0.5 * window.mean(axis=0)

    strat = OptimizerStrategy(
        MaxSharpeOptimizer(),
        lookback=120,
        constraints=(Constraint.long_only(), Constraint.weight_bounds(0.0, 0.5)),
        mean_estimator=shrunk_mean,
    )
    result = Backtest(
        market_data, strat, RebalanceSchedule.quarterly(), periods_per_year=252
    ).run()
    assert result.final_value > 0.0
    # weight bound respected implicitly via optimizer; report is well-formed
    report = result.report()
    assert np.isfinite(report.performance.cagr)


def test_workflow_produces_return_series_and_performance_report(market_data) -> None:
    from factorlab_optimizer import Constraint, MinVarianceOptimizer
    from factorlab_portfolio import PerformanceReport, ReturnSeries

    strat = OptimizerStrategy(
        MinVarianceOptimizer(), lookback=60, constraints=(Constraint.long_only(),)
    )
    result = Backtest(market_data, strat, RebalanceSchedule.monthly()).run()
    rs = result.to_return_series()
    assert isinstance(rs, ReturnSeries)
    report = result.report()
    assert isinstance(report.performance, PerformanceReport)
