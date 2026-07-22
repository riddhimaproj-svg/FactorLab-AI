"""Walk-forward analysis: honest out-of-sample evaluation.

A single backtest over the whole history invites **overfitting**: parameters
(and researcher choices) get tuned, implicitly or explicitly, to that one path.
**Walk-forward** validation splits the timeline into consecutive folds -- each
with an in-sample *train* window used only as history and an out-of-sample
*test* window on which the strategy trades and is scored -- then stitches the
test-window returns into one continuous OOS track record.

Two schemes govern the train window:

* **Rolling window** -- a fixed-length train block immediately precedes each test
  block (adapts to regime changes; discards distant history).
* **Expanding window** -- the train block grows from the start (uses all history;
  slower to adapt).

Because each fold's strategy sees only data up to its test window, walk-forward
results are free of look-ahead across folds and are the standard defense against
an over-fit backtest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from factorlab_backtesting.backtest import Backtest, BacktestResult
from factorlab_backtesting.errors import BacktestInputError
from factorlab_backtesting.execution import ExecutionEngine
from factorlab_backtesting.market_data import MarketData
from factorlab_backtesting.schedule import RebalanceSchedule
from factorlab_backtesting.strategy import Strategy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_portfolio import PerformanceReport, ReturnSeries

__all__ = [
    "WalkForward",
    "WalkForwardResult",
    "WalkForwardWindow",
    "expanding_windows",
    "rolling_windows",
]

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One fold: train is ``[train_start, test_start)``; test ``[test_start, test_end]``."""

    train_start: int
    test_start: int
    test_end: int


def rolling_windows(
    n_periods: int, train_size: int, test_size: int, step: int | None = None
) -> list[WalkForwardWindow]:
    """Fixed-length train block immediately before each (tiled) test block."""
    if train_size < 2 or test_size < 2:
        raise BacktestInputError("train_size and test_size must be >= 2")
    step = step if step is not None else test_size
    windows: list[WalkForwardWindow] = []
    test_start = train_size
    while test_start < n_periods - 1:
        test_end = min(test_start + test_size - 1, n_periods - 1)
        if test_end <= test_start:
            break
        windows.append(WalkForwardWindow(test_start - train_size, test_start, test_end))
        test_start += step
    return windows


def expanding_windows(
    n_periods: int, initial_train: int, test_size: int, step: int | None = None
) -> list[WalkForwardWindow]:
    """Train block grows from index 0 up to each (tiled) test block."""
    if initial_train < 2 or test_size < 2:
        raise BacktestInputError("initial_train and test_size must be >= 2")
    step = step if step is not None else test_size
    windows: list[WalkForwardWindow] = []
    test_start = initial_train
    while test_start < n_periods - 1:
        test_end = min(test_start + test_size - 1, n_periods - 1)
        if test_end <= test_start:
            break
        windows.append(WalkForwardWindow(0, test_start, test_end))
        test_start += step
    return windows


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Stitched out-of-sample returns across all folds."""

    returns: FloatArray
    dates: DateArray
    n_windows: int
    periods_per_year: float
    strategy_name: str
    fold_results: tuple[BacktestResult, ...]

    def to_return_series(self) -> ReturnSeries:
        from factorlab_portfolio import ReturnSeries

        return ReturnSeries(
            self.returns, self.dates, self.periods_per_year, f"{self.strategy_name}-oos"
        )

    def performance_report(self, risk_free: float = 0.0) -> PerformanceReport:
        return self.to_return_series().performance_report(risk_free=risk_free)


class WalkForward:
    """Runs a strategy out-of-sample across a set of walk-forward windows."""

    def __init__(
        self,
        market_data: MarketData,
        strategy: Strategy,
        schedule: RebalanceSchedule,
        execution_engine: ExecutionEngine | None = None,
        *,
        initial_capital: float = 1_000_000.0,
        cash_rate: float = 0.0,
        periods_per_year: float = 252.0,
    ) -> None:
        self.market_data = market_data
        self.strategy = strategy
        self.schedule = schedule
        self.engine = execution_engine
        self.initial_capital = initial_capital
        self.cash_rate = cash_rate
        self.periods_per_year = periods_per_year

    def run(self, windows: Sequence[WalkForwardWindow]) -> WalkForwardResult:
        if not windows:
            raise BacktestInputError("no walk-forward windows provided")
        md = self.market_data
        oos_returns: list[float] = []
        oos_dates: list[np.datetime64] = []
        fold_results: list[BacktestResult] = []

        for w in windows:
            sub_md = MarketData(
                md.dates[w.train_start : w.test_end + 1],
                md.assets,
                md.prices[w.train_start : w.test_end + 1],
            )
            warmup = w.test_start - w.train_start
            sub = Backtest(
                sub_md,
                self.strategy,
                self.schedule,
                self.engine,
                initial_capital=self.initial_capital,
                cash_rate=self.cash_rate,
                periods_per_year=self.periods_per_year,
                warmup=warmup,
            ).run()
            fold_results.append(sub)

            # datetime64[D].tolist() yields datetime.date objects; compare on those.
            oos_global = set(md.dates[w.test_start + 1 : w.test_end + 1].tolist())
            for rd, r in zip(sub.return_dates, sub.returns, strict=True):
                if rd.tolist() in oos_global:
                    oos_returns.append(float(r))
                    oos_dates.append(rd)

        return WalkForwardResult(
            returns=np.asarray(oos_returns, dtype=np.float64),
            dates=np.asarray(oos_dates, dtype="datetime64[D]"),
            n_windows=len(windows),
            periods_per_year=self.periods_per_year,
            strategy_name=self.strategy.name,
            fold_results=tuple(fold_results),
        )
