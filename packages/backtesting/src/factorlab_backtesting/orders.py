"""Order, Fill, and OrderBook value objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from factorlab_backtesting.errors import BacktestInputError

__all__ = ["Fill", "Order", "OrderBook"]


@dataclass(frozen=True, slots=True)
class Order:
    """An instruction to trade ``quantity`` (signed) shares of ``symbol``.

    Positive quantity buys, negative sells.  ``reference_price`` is the prevailing
    mid price used to size the order; the executed price may differ once slippage
    is applied.
    """

    symbol: str
    quantity: float
    reference_price: float
    order_type: str = "market"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise BacktestInputError("Order.symbol must be non-empty")
        if self.reference_price <= 0.0:
            raise BacktestInputError("Order.reference_price must be positive")

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0.0 else "sell"

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.reference_price


@dataclass(frozen=True, slots=True)
class Fill:
    """The realized execution of an order: ``quantity`` at ``price`` plus cost."""

    symbol: str
    quantity: float
    price: float
    commission: float

    @property
    def gross_value(self) -> float:
        """Signed traded value at the execution price (excludes commission)."""
        return self.quantity * self.price

    @property
    def cash_impact(self) -> float:
        """Effect on cash: buys reduce cash, sells increase it, net of commission."""
        return -self.quantity * self.price - self.commission

    @property
    def total_cost(self) -> float:
        """Commission on this fill (slippage is embedded in ``price``)."""
        return self.commission

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
        }


@dataclass(frozen=True, slots=True)
class OrderBook:
    """An immutable collection of orders for a single rebalance."""

    orders: tuple[Order, ...] = ()

    @classmethod
    def from_orders(cls, orders: Iterable[Order]) -> OrderBook:
        return cls(tuple(orders))

    def __len__(self) -> int:
        return len(self.orders)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.orders)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(o.symbol for o in self.orders)

    @property
    def total_notional(self) -> float:
        return float(sum(o.notional for o in self.orders))

    def to_dict(self) -> dict[str, Any]:
        return {"orders": [{"symbol": o.symbol, "quantity": o.quantity,
                            "reference_price": o.reference_price} for o in self.orders]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrderBook:
        return cls(tuple(
            Order(o["symbol"], float(o["quantity"]), float(o["reference_price"]))
            for o in data["orders"]
        ))
