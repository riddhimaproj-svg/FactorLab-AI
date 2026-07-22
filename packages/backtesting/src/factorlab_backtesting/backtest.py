"""Backtest: the event-driven simulation loop, and its raw result.

At each period the engine marks positions to market, and on rebalance dates it
asks the :class:`Strategy` for target weights (using only trailing data), routes
the required trades through the :class:`ExecutionEngine` (slippage + commission),
and updates holdings and cash.  Uninvested cash grows at ``cash_rate`` (zero by
default, which produces the usual *cash drag* on partially-invested strategies).

The output :class:`BacktestResult` exposes the portfolio value path, its
:class:`~factorlab_portfolio.ReturnSeries`, and a full :class:`BacktestReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from factorlab_backtesting.errors import BacktestInputError, InsufficientHistoryError
from factorlab_backtesting.execution import ExecutionEngine
from factorlab_backtesting.market_data import MarketData
from factorlab_backtesting.schedule import RebalanceSchedule
from factorlab_backtesting.strategy import Strategy, StrategyContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_portfolio import ReturnSeries

    from factorlab_backtesting.benchmark import Benchmark
    from factorlab_backtesting.report import BacktestReport

__all__ = ["Backtest", "BacktestResult"]

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The raw output of a backtest: the value path and rebalance diagnostics."""

    dates: DateArray
    values: FloatArray
    returns: FloatArray
    return_dates: DateArray
    rebalance_dates: DateArray
    turnovers: FloatArray
    total_costs: float
    initial_value: float
    final_value: float
    periods_per_year: float
    strategy_name: str

    @property
    def n_rebalances(self) -> int:
        return int(self.rebalance_dates.shape[0])

    @property
    def average_turnover(self) -> float:
        return float(np.mean(self.turnovers)) if self.turnovers.size else 0.0

    def to_return_series(self) -> ReturnSeries:
        """Portfolio returns as a ``factorlab_portfolio`` ReturnSeries."""
        from factorlab_portfolio import ReturnSeries

        return ReturnSeries(
            self.returns, self.return_dates, self.periods_per_year, self.strategy_name
        )

    def report(
        self, benchmark: Benchmark | None = None, risk_free: float = 0.0
    ) -> BacktestReport:
        from factorlab_backtesting.report import BacktestReport

        return BacktestReport.from_result(self, benchmark=benchmark, risk_free=risk_free)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dates": [np.datetime_as_string(d, unit="D") for d in self.dates],
            "values": self.values.tolist(),
            "returns": self.returns.tolist(),
            "total_costs": self.total_costs,
            "n_rebalances": self.n_rebalances,
            "average_turnover": self.average_turnover,
            "strategy_name": self.strategy_name,
            "periods_per_year": self.periods_per_year,
        }


class Backtest:
    """Runs a strategy over market data on a rebalance schedule."""

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
        warmup: int | None = None,
    ) -> None:
        if initial_capital <= 0.0:
            raise BacktestInputError("initial_capital must be positive")
        self.market_data = market_data
        self.strategy = strategy
        self.schedule = schedule
        self.engine = execution_engine if execution_engine is not None else ExecutionEngine()
        self.initial_capital = initial_capital
        self.cash_rate = cash_rate
        self.periods_per_year = periods_per_year
        self.warmup = warmup if warmup is not None else strategy.lookback

    def run(self) -> BacktestResult:
        md = self.market_data
        assets = md.assets
        n = md.n_periods
        if n < 2:
            raise BacktestInputError("need at least 2 periods to run a backtest")

        rebalance_idx = set(self.schedule.rebalance_indices(md.dates))
        # First rebalance may occur at ``warmup`` (index 0 for a zero-lookback
        # strategy); returns_window() safely yields an empty window there.
        first_allowed = self.warmup
        effective_rebalances = sorted(i for i in rebalance_idx if i >= first_allowed)
        if not effective_rebalances:
            raise InsufficientHistoryError(
                "no rebalance dates fall after the warmup window; "
                "increase history or reduce lookback/warmup"
            )

        positions: dict[str, float] = dict.fromkeys(assets, 0.0)
        cash = self.initial_capital
        values = np.empty(n, dtype=np.float64)
        turnovers: list[float] = []
        rebalanced_dates: list[np.datetime64] = []
        total_costs = 0.0

        for t in range(n):
            if t > 0 and self.cash_rate != 0.0:
                cash *= 1.0 + self.cash_rate
            prices_row = md.prices_at(t)
            prices = dict(zip(assets, prices_row, strict=True))

            if t in rebalance_idx and t >= first_allowed:
                portfolio_value = cash + float(
                    sum(positions[a] * prices[a] for a in assets)
                )
                current_weights = {
                    a: (positions[a] * prices[a] / portfolio_value if portfolio_value else 0.0)
                    for a in assets
                }
                window = md.returns_window(t, max(self.strategy.lookback, 2))
                context = StrategyContext(md.dates[t], assets, window, current_weights)
                target = self.strategy.compute_weights(context)
                outcome = self.engine.rebalance(positions, cash, target, prices)
                positions = outcome.positions
                cash = outcome.cash
                total_costs += outcome.total_commission
                turnovers.append(outcome.turnover)
                rebalanced_dates.append(md.dates[t])

            values[t] = cash + float(sum(positions[a] * prices[a] for a in assets))

        returns = values[1:] / values[:-1] - 1.0
        return BacktestResult(
            dates=md.dates,
            values=values,
            returns=returns,
            return_dates=md.dates[1:],
            rebalance_dates=np.asarray(rebalanced_dates, dtype="datetime64[D]"),
            turnovers=np.asarray(turnovers, dtype=np.float64),
            total_costs=total_costs,
            initial_value=self.initial_capital,
            final_value=float(values[-1]),
            periods_per_year=self.periods_per_year,
            strategy_name=self.strategy.name,
        )
