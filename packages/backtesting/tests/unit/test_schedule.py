"""Tests for RebalanceSchedule across frequencies."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import RebalanceSchedule
from factorlab_backtesting.errors import ScheduleError


@pytest.fixture
def year_of_days() -> np.ndarray:
    # 2 years of consecutive calendar days
    return np.datetime64("2021-01-01") + np.arange(730)


def test_daily(year_of_days) -> None:
    idx = RebalanceSchedule.daily().rebalance_indices(year_of_days)
    assert len(idx) == 730


def test_monthly_first_of_month(year_of_days) -> None:
    dates = RebalanceSchedule.monthly().rebalance_dates(year_of_days)
    assert len(dates) == 24  # 24 months
    # each is the 1st of a month
    assert all(str(d).endswith("-01") for d in dates)


def test_quarterly(year_of_days) -> None:
    dates = RebalanceSchedule.quarterly().rebalance_dates(year_of_days)
    assert len(dates) == 8  # 8 quarters in 2 years


def test_weekly(year_of_days) -> None:
    idx = RebalanceSchedule.weekly().rebalance_indices(year_of_days)
    # ~52 weeks/year -> ~104-105 over two years
    assert 100 <= len(idx) <= 106


def test_custom_dates(year_of_days) -> None:
    sched = RebalanceSchedule.custom(["2021-03-15", "2021-06-20"])
    idx = sched.rebalance_indices(year_of_days)
    assert len(idx) == 2


def test_custom_predicate(year_of_days) -> None:
    # rebalance on the 15th of every month
    sched = RebalanceSchedule.from_predicate(lambda d: d.astype(object).day == 15)
    idx = sched.rebalance_indices(year_of_days)
    assert len(idx) == 24


def test_invalid_frequency() -> None:
    with pytest.raises(ScheduleError):
        RebalanceSchedule("hourly")


def test_custom_requires_dates_or_predicate() -> None:
    with pytest.raises(ScheduleError):
        RebalanceSchedule("custom")


def test_empty_dates() -> None:
    assert RebalanceSchedule.monthly().rebalance_indices(
        np.array([], dtype="datetime64[D]")
    ) == ()
