"""Benchmark: a reference return stream to evaluate a strategy against."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from factorlab_backtesting.errors import BacktestInputError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_portfolio import ReturnSeries

__all__ = ["Benchmark"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """A named per-period return series used as a comparison baseline."""

    name: str
    returns: FloatArray

    def __post_init__(self) -> None:
        returns = np.asarray(self.returns, dtype=np.float64)
        if returns.ndim != 1:
            raise BacktestInputError("benchmark returns must be 1-D")
        if not np.all(np.isfinite(returns)):
            raise BacktestInputError("benchmark returns contain non-finite values")
        returns.setflags(write=False)
        object.__setattr__(self, "returns", returns)

    def __len__(self) -> int:
        return int(self.returns.shape[0])

    @classmethod
    def from_prices(cls, name: str, prices: Sequence[float] | FloatArray) -> Benchmark:
        px = np.asarray(prices, dtype=np.float64)
        if px.shape[0] < 2:
            raise BacktestInputError("need >= 2 prices to build a benchmark")
        return cls(name, px[1:] / px[:-1] - 1.0)

    @classmethod
    def from_returns(cls, name: str, returns: Sequence[float] | FloatArray) -> Benchmark:
        return cls(name, np.asarray(returns, dtype=np.float64))

    def to_return_series(self, periods_per_year: float = 252.0) -> ReturnSeries:
        """Wrap as a ``factorlab_portfolio`` ReturnSeries."""
        from factorlab_portfolio import ReturnSeries

        return ReturnSeries(self.returns, periods_per_year=periods_per_year, name=self.name)
