"""RebalanceSchedule: when the strategy rebalances.

Given a backtest's date index, a schedule returns the indices at which the
portfolio is re-weighted.  Built-in frequencies pick the **first trading day of
each period** (week/month/quarter); ``daily`` rebalances every day; ``custom``
accepts an explicit set of dates or a date predicate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
from numpy.typing import NDArray

from factorlab_backtesting.errors import ScheduleError

__all__ = ["Frequency", "RebalanceSchedule"]

DateArray = NDArray[np.datetime64]

Frequency = str
_FREQUENCIES = ("daily", "weekly", "monthly", "quarterly", "custom")


class RebalanceSchedule:
    """A rule mapping a date index to the indices at which to rebalance."""

    def __init__(
        self,
        frequency: Frequency = "monthly",
        *,
        custom_dates: Sequence[str | np.datetime64] | None = None,
        predicate: Callable[[np.datetime64], bool] | None = None,
    ) -> None:
        if frequency not in _FREQUENCIES:
            raise ScheduleError(f"frequency must be one of {_FREQUENCIES}, got {frequency!r}")
        if frequency == "custom" and custom_dates is None and predicate is None:
            raise ScheduleError("custom frequency requires custom_dates or predicate")
        self.frequency = frequency
        self._custom_dates = (
            None
            if custom_dates is None
            else set(np.asarray(list(custom_dates), dtype="datetime64[D]"))
        )
        self._predicate = predicate

    # -- Factories --------------------------------------------------------
    @classmethod
    def daily(cls) -> RebalanceSchedule:
        return cls("daily")

    @classmethod
    def weekly(cls) -> RebalanceSchedule:
        return cls("weekly")

    @classmethod
    def monthly(cls) -> RebalanceSchedule:
        return cls("monthly")

    @classmethod
    def quarterly(cls) -> RebalanceSchedule:
        return cls("quarterly")

    @classmethod
    def custom(cls, dates: Sequence[str | np.datetime64]) -> RebalanceSchedule:
        return cls("custom", custom_dates=dates)

    @classmethod
    def from_predicate(cls, predicate: Callable[[np.datetime64], bool]) -> RebalanceSchedule:
        return cls("custom", predicate=predicate)

    # -- Resolution -------------------------------------------------------
    def rebalance_indices(self, dates: DateArray) -> tuple[int, ...]:
        """Indices into ``dates`` at which to rebalance (sorted, unique)."""
        dates = np.asarray(dates, dtype="datetime64[D]")
        n = dates.shape[0]
        if n == 0:
            return ()
        if self.frequency == "daily":
            return tuple(range(n))
        if self.frequency == "custom":
            return self._custom_indices(dates)
        return self._first_of_period(dates)

    def rebalance_dates(self, dates: DateArray) -> DateArray:
        idx = list(self.rebalance_indices(dates))
        return np.asarray(dates, dtype="datetime64[D]")[idx]

    # -- Internals --------------------------------------------------------
    def _custom_indices(self, dates: DateArray) -> tuple[int, ...]:
        result: list[int] = []
        for i, d in enumerate(dates):
            by_date = self._custom_dates is not None and d in self._custom_dates
            by_pred = self._predicate is not None and self._predicate(d)
            if by_date or by_pred:
                result.append(i)
        return tuple(result)

    def _first_of_period(self, dates: DateArray) -> tuple[int, ...]:
        keys = [self._period_key(d.astype(date)) for d in dates]
        result: list[int] = []
        seen: set[tuple[int, int]] = set()
        for i, key in enumerate(keys):
            if key not in seen:
                seen.add(key)
                result.append(i)
        return tuple(result)

    def _period_key(self, d: date) -> tuple[int, int]:
        if self.frequency == "weekly":
            iso = d.isocalendar()
            return (iso[0], iso[1])
        if self.frequency == "monthly":
            return (d.year, d.month)
        # quarterly
        return (d.year, (d.month - 1) // 3)
