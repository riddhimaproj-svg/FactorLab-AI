"""Property-based invariants for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_backtesting import (
    Backtest,
    BrokerModel,
    EqualWeightStrategy,
    ExecutionEngine,
    MarketData,
    PercentageCommission,
    RebalanceSchedule,
)

pytestmark = pytest.mark.property

_SETTINGS = settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@st.composite
def market_datas(draw):
    n = draw(st.integers(min_value=60, max_value=200))
    k = draw(st.integers(min_value=2, max_value=4))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2020-01-01") + np.arange(n)
    rets = rng.normal(0.0, 0.01, size=(n, k))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return MarketData(dates, tuple(f"A{i}" for i in range(k)), prices)


@_SETTINGS
@given(md=market_datas())
def test_portfolio_value_stays_positive(md) -> None:
    result = Backtest(md, EqualWeightStrategy(), RebalanceSchedule.weekly()).run()
    assert np.all(result.values > 0.0)


@_SETTINGS
@given(md=market_datas())
def test_returns_length_and_finiteness(md) -> None:
    result = Backtest(md, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    assert result.returns.shape[0] == md.n_periods - 1
    assert np.all(np.isfinite(result.returns))


@_SETTINGS
@given(md=market_datas())
def test_costs_never_increase_final_value(md) -> None:
    sched = RebalanceSchedule.weekly()
    cheap = Backtest(md, EqualWeightStrategy(), sched).run()
    pricey = Backtest(
        md, EqualWeightStrategy(), sched,
        ExecutionEngine(BrokerModel(PercentageCommission(0.005))),
    ).run()
    assert pricey.final_value <= cheap.final_value + 1e-6


@_SETTINGS
@given(md=market_datas())
def test_turnover_nonnegative(md) -> None:
    result = Backtest(md, EqualWeightStrategy(), RebalanceSchedule.monthly()).run()
    assert np.all(result.turnovers >= -1e-12)
