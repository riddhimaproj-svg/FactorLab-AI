"""Tests for the Backtest engine."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import (
    Backtest,
    BrokerModel,
    EqualWeightStrategy,
    ExecutionEngine,
    PercentageCommission,
    RebalanceSchedule,
    StaticWeightStrategy,
)
from factorlab_backtesting.errors import BacktestInputError, InsufficientHistoryError


def test_basic_run(market_data) -> None:
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    assert result.values.shape == (400,)
    assert result.returns.shape == (399,)
    assert result.n_rebalances > 0
    assert result.final_value > 0


def test_buy_and_hold_tracks_asset(flat_market_data) -> None:
    """A static 100% position in X earns X's return (frictionless)."""
    strat = StaticWeightStrategy({"X": 1.0, "Y": 0.0})
    result = Backtest(
        flat_market_data, strat, RebalanceSchedule.daily(), periods_per_year=252
    ).run()
    # X compounds at 0.1%/day; final/initial ~ price ratio of X
    x_total = flat_market_data.prices[-1, 0] / flat_market_data.prices[0, 0]
    assert result.final_value / result.initial_value == pytest.approx(x_total, rel=1e-6)


def test_costs_reduce_final_value(market_data) -> None:
    sched = RebalanceSchedule.weekly()
    cheap = Backtest(market_data, EqualWeightStrategy(), sched).run()
    pricey = Backtest(
        market_data, EqualWeightStrategy(), sched,
        ExecutionEngine(BrokerModel(PercentageCommission(0.01))),
    ).run()
    assert pricey.final_value < cheap.final_value
    assert pricey.total_costs > 0.0


def test_cash_drag(flat_market_data) -> None:
    """Holding 50% cash (earning 0) underperforms full investment in a rising market."""
    invested = Backtest(
        flat_market_data, StaticWeightStrategy({"X": 1.0, "Y": 0.0}),
        RebalanceSchedule.daily(), periods_per_year=252,
    ).run()
    half_cash = Backtest(
        flat_market_data, StaticWeightStrategy({"X": 0.5, "Y": 0.0}),
        RebalanceSchedule.daily(), periods_per_year=252,
    ).run()
    assert half_cash.final_value < invested.final_value  # cash drag


def test_cash_rate_grows_uninvested_cash(flat_market_data) -> None:
    no_rate = Backtest(
        flat_market_data, StaticWeightStrategy({"X": 0.5, "Y": 0.0}),
        RebalanceSchedule.daily(), cash_rate=0.0, periods_per_year=252,
    ).run()
    with_rate = Backtest(
        flat_market_data, StaticWeightStrategy({"X": 0.5, "Y": 0.0}),
        RebalanceSchedule.daily(), cash_rate=0.0005, periods_per_year=252,
    ).run()
    assert with_rate.final_value > no_rate.final_value


def test_turnover_recorded(market_data) -> None:
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    assert result.turnovers.shape[0] == result.n_rebalances
    assert result.average_turnover >= 0.0


def test_insufficient_history_raises(market_data) -> None:
    # warmup beyond the data -> no rebalance dates remain
    with pytest.raises(InsufficientHistoryError):
        Backtest(
            market_data, EqualWeightStrategy(), RebalanceSchedule.quarterly(), warmup=10_000
        ).run()


def test_invalid_capital(market_data) -> None:
    with pytest.raises(BacktestInputError):
        Backtest(
            market_data, EqualWeightStrategy(), RebalanceSchedule.monthly(), initial_capital=0.0
        )


def test_result_serialization(market_data) -> None:
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    d = result.to_dict()
    assert d["strategy_name"] == "equal_weight"
    assert len(d["values"]) == 400


def test_return_series_integration(market_data) -> None:
    result = Backtest(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    rs = result.to_return_series()
    assert rs.n_observations == 399
    assert np.isfinite(rs.sharpe())
