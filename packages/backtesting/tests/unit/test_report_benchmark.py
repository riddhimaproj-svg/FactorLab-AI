"""Tests for Benchmark and BacktestReport."""

from __future__ import annotations

import json

import numpy as np
import pytest

from factorlab_backtesting import (
    Backtest,
    Benchmark,
    EqualWeightStrategy,
    RebalanceSchedule,
)
from factorlab_backtesting.errors import BacktestInputError


def test_benchmark_from_prices() -> None:
    b = Benchmark.from_prices("idx", [100.0, 110.0, 121.0])
    np.testing.assert_allclose(b.returns, [0.1, 0.1])
    assert len(b) == 2


def test_benchmark_validation() -> None:
    with pytest.raises(BacktestInputError):
        Benchmark("x", np.array([0.1, np.nan]))
    with pytest.raises(BacktestInputError):
        Benchmark.from_prices("x", [100.0])


@pytest.fixture
def report(market_data):
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    bench = Benchmark.from_prices("bench", market_data.prices.mean(axis=1))
    return result.report(benchmark=bench, risk_free=0.0)


def test_report_fields(report) -> None:
    assert report.strategy_name == "equal_weight"
    assert np.isfinite(report.beta)
    assert np.isfinite(report.alpha)
    assert 0.0 <= report.win_rate <= 1.0
    assert 0.0 <= report.hit_ratio <= 1.0
    assert report.average_turnover >= 0.0
    assert report.n_rebalances > 0


def test_report_without_benchmark(market_data) -> None:
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    rep = result.report()
    assert np.isnan(rep.beta)
    assert np.isnan(rep.hit_ratio)
    assert np.isfinite(rep.win_rate)


def test_report_serialization(report) -> None:
    d = report.to_dict()
    payload = json.dumps(d)  # must be JSON-serializable
    assert json.loads(payload)["strategy_name"] == "equal_weight"


def test_report_summary(report) -> None:
    text = report.summary()
    assert "Backtest Report" in text
    assert "Alpha" in text and "Turnover" in text.replace("turnover", "Turnover")
    assert "Win rate" in text


def test_report_benchmark_length_mismatch(market_data) -> None:
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    short = Benchmark("s", np.zeros(10))  # far too short
    with pytest.raises(BacktestInputError):
        result.report(benchmark=short)
