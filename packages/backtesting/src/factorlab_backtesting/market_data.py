"""MarketData: an immutable price panel driving a backtest.

Holds adjusted price levels for a universe of assets on a shared, strictly
increasing date index.  Prices (not returns) are stored so the engine can mark
positions to market and compute execution values; returns are derived on demand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from factorlab_backtesting.errors import BacktestInputError

__all__ = ["MarketData"]

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class MarketData:
    """Adjusted prices for ``assets`` over ``dates`` (``n_dates x n_assets``)."""

    dates: DateArray
    assets: tuple[str, ...]
    prices: FloatArray

    def __post_init__(self) -> None:
        dates = np.asarray(self.dates, dtype="datetime64[D]")
        prices = np.asarray(self.prices, dtype=np.float64)
        if prices.ndim != 2:
            raise BacktestInputError("prices must be 2-D (n_dates x n_assets)")
        if prices.shape != (dates.shape[0], len(self.assets)):
            raise BacktestInputError("prices shape must match (len(dates), len(assets))")
        if len(set(self.assets)) != len(self.assets):
            raise BacktestInputError("duplicate asset names")
        if dates.shape[0] >= 2 and np.any(np.diff(dates) <= np.timedelta64(0, "D")):
            raise BacktestInputError("dates must be strictly increasing")
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
            raise BacktestInputError("prices must be finite and positive")
        dates.setflags(write=False)
        prices.setflags(write=False)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "prices", prices)

    @property
    def n_periods(self) -> int:
        return int(self.dates.shape[0])

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def prices_at(self, index: int) -> FloatArray:
        """Price row at time index ``index``."""
        return np.asarray(self.prices[index], dtype=np.float64)

    def simple_returns(self) -> FloatArray:
        """Per-period simple returns, shape ``(n_periods - 1, n_assets)``."""
        return self.prices[1:] / self.prices[:-1] - 1.0

    def returns_window(self, end_index: int, lookback: int) -> FloatArray:
        """Trailing simple returns ending at ``end_index`` (inclusive).

        Uses only prices up to ``end_index`` -- no future data -- so strategies
        built on it are free of look-ahead bias.  Returns at most ``lookback``
        rows.
        """
        start = max(1, end_index - lookback + 1)
        block = self.prices[start - 1 : end_index + 1]
        return block[1:] / block[:-1] - 1.0

    @classmethod
    def from_prices(
        cls, dates: Sequence[str] | DateArray, assets: Sequence[str], prices: FloatArray
    ) -> MarketData:
        return cls(
            np.asarray(dates, dtype="datetime64[D]"),
            tuple(assets),
            np.asarray(prices, dtype=np.float64),
        )
