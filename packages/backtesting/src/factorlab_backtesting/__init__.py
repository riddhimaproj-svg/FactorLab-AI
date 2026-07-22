"""FactorLab backtesting — event-driven strategy simulation.

A typed, dependency-light backtesting engine that turns a strategy and a price
history into a realistic, cost-aware track record and a full performance report.
It completes the platform workflow:

    Factor Model -> Expected Returns -> Optimizer -> Portfolio -> Backtester -> Report

Core pieces
-----------
* :class:`MarketData` -- immutable price panel.
* :class:`Strategy` (+ :class:`StaticWeightStrategy`, :class:`EqualWeightStrategy`,
  :class:`OptimizerStrategy`) -- maps trailing data to target weights (no look-ahead).
* :class:`RebalanceSchedule` -- daily / weekly / monthly / quarterly / custom.
* :class:`Order`, :class:`Fill`, :class:`OrderBook`, :class:`ExecutionEngine` --
  order generation and execution.
* Cost models: :class:`FixedCommission`, :class:`PercentageCommission`,
  :class:`FixedBpsSlippage`, :class:`BidAskSpreadSlippage`, via :class:`BrokerModel`.
* :class:`Backtest` / :class:`BacktestResult` / :class:`BacktestReport`.
* :class:`WalkForward` with rolling / expanding out-of-sample windows.

It reuses the approved portfolio analytics (``ReturnSeries`` / ``PerformanceReport``)
and integrates with ``factorlab_optimizer`` through :class:`OptimizerStrategy`.
"""

from __future__ import annotations

from factorlab_backtesting.backtest import Backtest, BacktestResult
from factorlab_backtesting.benchmark import Benchmark
from factorlab_backtesting.costs import (
    BidAskSpreadSlippage,
    BrokerModel,
    CompositeCostModel,
    FixedBpsSlippage,
    FixedCommission,
    PercentageCommission,
    SlippageModel,
    TransactionCostModel,
    ZeroCostModel,
    ZeroSlippage,
)
from factorlab_backtesting.errors import (
    BacktestError,
    BacktestInputError,
    InsufficientHistoryError,
    ScheduleError,
)
from factorlab_backtesting.execution import ExecutionEngine, RebalanceOutcome
from factorlab_backtesting.market_data import MarketData
from factorlab_backtesting.orders import Fill, Order, OrderBook
from factorlab_backtesting.report import BacktestReport
from factorlab_backtesting.schedule import RebalanceSchedule
from factorlab_backtesting.strategy import (
    EqualWeightStrategy,
    OptimizerStrategy,
    StaticWeightStrategy,
    Strategy,
    StrategyContext,
)
from factorlab_backtesting.walkforward import (
    WalkForward,
    WalkForwardResult,
    WalkForwardWindow,
    expanding_windows,
    rolling_windows,
)

__version__ = "0.1.0"

__all__ = [
    "Backtest",
    "BacktestError",
    "BacktestInputError",
    "BacktestReport",
    "BacktestResult",
    "Benchmark",
    "BidAskSpreadSlippage",
    "BrokerModel",
    "CompositeCostModel",
    "EqualWeightStrategy",
    "ExecutionEngine",
    "Fill",
    "FixedBpsSlippage",
    "FixedCommission",
    "InsufficientHistoryError",
    "MarketData",
    "OptimizerStrategy",
    "Order",
    "OrderBook",
    "PercentageCommission",
    "RebalanceOutcome",
    "RebalanceSchedule",
    "ScheduleError",
    "SlippageModel",
    "StaticWeightStrategy",
    "Strategy",
    "StrategyContext",
    "TransactionCostModel",
    "WalkForward",
    "WalkForwardResult",
    "WalkForwardWindow",
    "ZeroCostModel",
    "ZeroSlippage",
    "__version__",
    "expanding_windows",
    "rolling_windows",
]
