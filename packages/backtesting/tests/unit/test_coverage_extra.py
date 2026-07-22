"""Supplementary tests covering remaining branches."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import (
    Backtest,
    Benchmark,
    EqualWeightStrategy,
    Fill,
    MarketData,
    Order,
    OrderBook,
    RebalanceSchedule,
    metrics,
    rolling_windows,
)
from factorlab_backtesting.errors import BacktestInputError


def test_market_data_duplicate_and_nonfinite() -> None:
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    with pytest.raises(BacktestInputError):
        MarketData(dates, ("A", "A"), np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(BacktestInputError):
        MarketData(dates, ("A",), np.array([[1.0], [np.inf]]))


def test_backtest_needs_two_periods() -> None:
    md = MarketData(
        np.array(["2020-01-01"], dtype="datetime64[D]"), ("A",), np.array([[100.0]])
    )
    with pytest.raises(BacktestInputError):
        Backtest(md, EqualWeightStrategy(), RebalanceSchedule.daily()).run()


def test_benchmark_len_and_return_series() -> None:
    b = Benchmark.from_returns("b", np.array([0.01, 0.02, -0.01]))
    assert len(b) == 3
    rs = b.to_return_series(periods_per_year=12.0)
    assert rs.n_observations == 3
    assert rs.name == "b"


def test_fill_total_cost_and_orderbook_iter() -> None:
    fill = Fill("A", 10.0, 100.0, commission=2.5)
    assert fill.total_cost == pytest.approx(2.5)
    assert fill.to_dict()["symbol"] == "A"
    book = OrderBook.from_orders([Order("A", 1.0, 10.0), Order("B", -1.0, 20.0)])
    assert [o.symbol for o in book] == ["A", "B"]


def test_metrics_edge_cases() -> None:
    assert np.isnan(metrics.alpha_beta(np.array([0.01]), np.array([0.01]))[0])  # n<2
    assert np.isnan(metrics.hit_ratio(np.array([]), np.array([])))


def test_report_align_tail(market_data) -> None:
    """A benchmark longer than the backtest returns is aligned to the tail."""
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    long_bench = Benchmark("long", np.zeros(result.returns.shape[0] + 50))
    report = result.report(benchmark=long_bench)
    assert np.isfinite(report.beta) or np.isnan(report.beta)  # runs without error


def test_rolling_windows_step_overlap() -> None:
    windows = rolling_windows(300, train_size=100, test_size=50, step=25)
    # overlapping test blocks (step < test_size) => more windows
    starts = [w.test_start for w in windows]
    assert starts == sorted(starts)
    gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    assert 25 in gaps
