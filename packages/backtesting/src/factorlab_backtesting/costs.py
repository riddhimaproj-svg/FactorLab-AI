"""Transaction-cost, slippage, and broker models.

Transaction costs are the main reason paper strategies fail in production: every
rebalance pays commissions and crosses the bid-ask spread, and large orders push
the price (slippage).  Ignoring them makes a backtest optimistic and, worse,
biases it toward high-turnover strategies that look great gross but bleed net.

* **TransactionCostModel** -- explicit fees (fixed and/or percentage commission).
* **SlippageModel** -- the gap between the mid price and the realized fill price
  (a fixed number of basis points, or half the bid-ask spread on each side).
* **BrokerModel** -- combines the two to turn an :class:`Order` into a
  :class:`Fill` at a realistic price and cost.
"""

from __future__ import annotations

import abc

from factorlab_backtesting.errors import BacktestInputError
from factorlab_backtesting.orders import Fill, Order

__all__ = [
    "BidAskSpreadSlippage",
    "BrokerModel",
    "CompositeCostModel",
    "FixedBpsSlippage",
    "FixedCommission",
    "PercentageCommission",
    "SlippageModel",
    "TransactionCostModel",
    "ZeroCostModel",
    "ZeroSlippage",
]


# --------------------------------------------------------------------------- #
# Transaction costs                                                           #
# --------------------------------------------------------------------------- #
class TransactionCostModel(abc.ABC):
    """Maps a trade (order at an execution price) to a commission in cash."""

    @abc.abstractmethod
    def commission(self, order: Order, execution_price: float) -> float: ...


class ZeroCostModel(TransactionCostModel):
    def commission(self, order: Order, execution_price: float) -> float:
        return 0.0


class FixedCommission(TransactionCostModel):
    """A flat fee per (non-empty) trade."""

    def __init__(self, per_trade: float) -> None:
        if per_trade < 0.0:
            raise BacktestInputError("per_trade commission must be >= 0")
        self.per_trade = per_trade

    def commission(self, order: Order, execution_price: float) -> float:
        return self.per_trade if order.quantity != 0.0 else 0.0


class PercentageCommission(TransactionCostModel):
    """A commission proportional to traded notional (e.g. 0.001 = 10 bps)."""

    def __init__(self, rate: float) -> None:
        if rate < 0.0:
            raise BacktestInputError("commission rate must be >= 0")
        self.rate = rate

    def commission(self, order: Order, execution_price: float) -> float:
        return self.rate * abs(order.quantity) * execution_price


class CompositeCostModel(TransactionCostModel):
    """Sum of several cost models (e.g. fixed + percentage)."""

    def __init__(self, models: list[TransactionCostModel]) -> None:
        self.models = list(models)

    def commission(self, order: Order, execution_price: float) -> float:
        return sum(m.commission(order, execution_price) for m in self.models)


# --------------------------------------------------------------------------- #
# Slippage                                                                    #
# --------------------------------------------------------------------------- #
class SlippageModel(abc.ABC):
    """Maps a mid price to the price actually paid/received for an order."""

    @abc.abstractmethod
    def execution_price(self, order: Order, mid_price: float) -> float: ...


class ZeroSlippage(SlippageModel):
    def execution_price(self, order: Order, mid_price: float) -> float:
        return mid_price


class FixedBpsSlippage(SlippageModel):
    """Buys pay ``mid*(1+bps)``, sells receive ``mid*(1-bps)``."""

    def __init__(self, bps: float) -> None:
        if bps < 0.0:
            raise BacktestInputError("slippage bps must be >= 0")
        self.fraction = bps / 10_000.0

    def execution_price(self, order: Order, mid_price: float) -> float:
        direction = 1.0 if order.quantity > 0.0 else -1.0
        return mid_price * (1.0 + direction * self.fraction)


class BidAskSpreadSlippage(SlippageModel):
    """Cross half the bid-ask spread on each side (spread quoted in bps)."""

    def __init__(self, spread_bps: float) -> None:
        if spread_bps < 0.0:
            raise BacktestInputError("spread_bps must be >= 0")
        self.half_spread = (spread_bps / 10_000.0) / 2.0

    def execution_price(self, order: Order, mid_price: float) -> float:
        direction = 1.0 if order.quantity > 0.0 else -1.0
        return mid_price * (1.0 + direction * self.half_spread)


# --------------------------------------------------------------------------- #
# Broker                                                                       #
# --------------------------------------------------------------------------- #
class BrokerModel:
    """Executes an order: applies slippage to the price, then charges commission."""

    def __init__(
        self,
        cost_model: TransactionCostModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.cost_model = cost_model if cost_model is not None else ZeroCostModel()
        self.slippage_model = slippage_model if slippage_model is not None else ZeroSlippage()

    def execute(self, order: Order, mid_price: float) -> Fill:
        exec_price = self.slippage_model.execution_price(order, mid_price)
        commission = self.cost_model.commission(order, exec_price)
        return Fill(order.symbol, order.quantity, exec_price, commission)
