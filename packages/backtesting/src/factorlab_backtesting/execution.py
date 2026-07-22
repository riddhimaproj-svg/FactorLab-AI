"""ExecutionEngine: rebalance current holdings toward target weights.

Given the current share holdings, cash, target weights, and prevailing prices, the
engine sizes the trades needed to hit the targets, routes each through the
:class:`BrokerModel` (slippage + commission), and returns the resulting holdings,
cash, costs, and turnover.  It is pure: it mutates nothing and returns a new
outcome, so a backtest loop can call it once per rebalance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from factorlab_backtesting.costs import BrokerModel
from factorlab_backtesting.orders import Fill, Order, OrderBook

__all__ = ["ExecutionEngine", "RebalanceOutcome"]

_MIN_TRADE_NOTIONAL = 1e-9


@dataclass(frozen=True, slots=True)
class RebalanceOutcome:
    """Result of a rebalance: new holdings, cash, costs, and turnover."""

    positions: dict[str, float]
    cash: float
    total_commission: float
    traded_notional: float
    turnover: float
    order_book: OrderBook
    fills: tuple[Fill, ...] = field(default_factory=tuple)


class ExecutionEngine:
    """Turns a target-weight vector into executed fills via a broker model."""

    def __init__(self, broker: BrokerModel | None = None) -> None:
        self.broker = broker if broker is not None else BrokerModel()

    def rebalance(
        self,
        positions: Mapping[str, float],
        cash: float,
        target_weights: Mapping[str, float],
        prices: Mapping[str, float],
    ) -> RebalanceOutcome:
        """Trade from ``positions`` toward ``target_weights`` at ``prices``."""
        portfolio_value = cash + sum(
            positions.get(sym, 0.0) * px for sym, px in prices.items()
        )

        new_positions: dict[str, float] = {}
        orders: list[Order] = []
        fills: list[Fill] = []
        new_cash = cash
        total_commission = 0.0
        traded_notional = 0.0

        for symbol, price in prices.items():
            current_shares = positions.get(symbol, 0.0)
            target_value = target_weights.get(symbol, 0.0) * portfolio_value
            target_shares = target_value / price
            delta = target_shares - current_shares

            if abs(delta) * price <= _MIN_TRADE_NOTIONAL:
                new_positions[symbol] = current_shares
                continue

            order = Order(symbol, delta, price)
            fill = self.broker.execute(order, price)
            orders.append(order)
            fills.append(fill)
            new_cash += fill.cash_impact
            total_commission += fill.commission
            traded_notional += abs(delta) * price
            new_positions[symbol] = current_shares + fill.quantity

        turnover = 0.0 if portfolio_value == 0.0 else traded_notional / portfolio_value
        return RebalanceOutcome(
            positions=new_positions,
            cash=new_cash,
            total_commission=total_commission,
            traded_notional=traded_notional,
            turnover=turnover,
            order_book=OrderBook(tuple(orders)),
            fills=tuple(fills),
        )
