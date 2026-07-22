"""Exception hierarchy for the backtesting engine."""

from __future__ import annotations

__all__ = [
    "BacktestError",
    "BacktestInputError",
    "InsufficientHistoryError",
    "ScheduleError",
]


class BacktestError(Exception):
    """Base class for every error raised by ``factorlab_backtesting``."""


class BacktestInputError(BacktestError):
    """Malformed inputs (market data, weights, configuration)."""


class InsufficientHistoryError(BacktestError):
    """Not enough price history to satisfy the strategy's lookback window."""


class ScheduleError(BacktestError):
    """A rebalance schedule could not be constructed."""
