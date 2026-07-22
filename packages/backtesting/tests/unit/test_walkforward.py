"""Tests for walk-forward analysis."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import (
    EqualWeightStrategy,
    RebalanceSchedule,
    WalkForward,
    expanding_windows,
    rolling_windows,
)
from factorlab_backtesting.errors import BacktestInputError


def test_rolling_windows_structure() -> None:
    windows = rolling_windows(n_periods=400, train_size=100, test_size=50)
    assert len(windows) >= 1
    for w in windows:
        assert w.test_start - w.train_start == 100  # fixed train length
        assert w.test_end > w.test_start
        assert w.test_end <= 399


def test_expanding_windows_structure() -> None:
    windows = expanding_windows(n_periods=400, initial_train=100, test_size=50)
    assert all(w.train_start == 0 for w in windows)  # always from the start
    # train length grows across folds
    lengths = [w.test_start - w.train_start for w in windows]
    assert lengths == sorted(lengths)


def test_windows_validation() -> None:
    with pytest.raises(BacktestInputError):
        rolling_windows(100, train_size=1, test_size=50)
    with pytest.raises(BacktestInputError):
        expanding_windows(100, initial_train=50, test_size=1)


def test_walkforward_runs_oos(market_data) -> None:
    windows = rolling_windows(market_data.n_periods, train_size=100, test_size=50)
    wf = WalkForward(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly())
    result = wf.run(windows)
    assert result.n_windows == len(windows)
    assert result.returns.shape[0] > 0
    assert result.returns.shape[0] == result.dates.shape[0]
    assert np.isfinite(result.to_return_series().sharpe())


def test_walkforward_oos_is_subset_of_history(market_data) -> None:
    windows = rolling_windows(market_data.n_periods, train_size=120, test_size=60)
    wf = WalkForward(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly())
    result = wf.run(windows)
    # OOS dates fall within the market data range and are unique
    assert set(result.dates.tolist()).issubset(set(market_data.dates.tolist()))
    assert len(set(result.dates.tolist())) == len(result.dates)


def test_expanding_walkforward(market_data) -> None:
    windows = expanding_windows(market_data.n_periods, initial_train=120, test_size=60)
    wf = WalkForward(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly())
    result = wf.run(windows)
    report = result.performance_report()
    assert report.n_observations == result.returns.shape[0]


def test_walkforward_requires_windows(market_data) -> None:
    wf = WalkForward(market_data, EqualWeightStrategy(), RebalanceSchedule.monthly())
    with pytest.raises(BacktestInputError):
        wf.run([])
