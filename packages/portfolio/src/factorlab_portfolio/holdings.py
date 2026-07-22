"""Immutable position, holding, and trade models.

These are plain, framework-free value objects.  A :class:`Position` records what
is held (quantity at a price); a :class:`Holding` is a portfolio-level view of a
position (its market value and weight); a :class:`Trade` records a transaction.
Keeping them immutable makes portfolio state transitions explicit and auditable
and lets optimization/backtesting layers build on them later without surprises.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from factorlab_portfolio.errors import PortfolioValidationError

__all__ = ["Holding", "Position", "Trade"]


@dataclass(frozen=True, slots=True)
class Position:
    """A holding of ``quantity`` shares of ``symbol`` at ``price`` per share.

    ``quantity`` may be negative (a short position).  ``cost_basis`` is the
    average acquisition price per share, if known, enabling unrealized P&L.
    """

    symbol: str
    quantity: float
    price: float
    cost_basis: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise PortfolioValidationError("Position.symbol must be non-empty")
        if not _finite(self.quantity):
            raise PortfolioValidationError(f"quantity must be finite, got {self.quantity}")
        if not _finite(self.price) or self.price < 0.0:
            raise PortfolioValidationError(f"price must be finite and >= 0, got {self.price}")
        if self.cost_basis is not None and not _finite(self.cost_basis):
            raise PortfolioValidationError("cost_basis must be finite when provided")

    @property
    def market_value(self) -> float:
        """Current market value, ``quantity * price`` (signed for shorts)."""
        return float(self.quantity * self.price)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0.0

    @property
    def unrealized_pnl(self) -> float:
        """Mark-to-market gain vs cost basis; ``nan`` if cost basis is unknown."""
        if self.cost_basis is None:
            return float("nan")
        return float((self.price - self.cost_basis) * self.quantity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "cost_basis": self.cost_basis,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Position:
        return cls(
            symbol=str(data["symbol"]),
            quantity=float(data["quantity"]),
            price=float(data["price"]),
            cost_basis=None if data.get("cost_basis") is None else float(data["cost_basis"]),
        )


@dataclass(frozen=True, slots=True)
class Holding:
    """A portfolio-level view of a position: its market value and weight."""

    symbol: str
    market_value: float
    weight: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise PortfolioValidationError("Holding.symbol must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "market_value": self.market_value, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Holding:
        return cls(
            symbol=str(data["symbol"]),
            market_value=float(data["market_value"]),
            weight=float(data["weight"]),
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """A transaction: ``quantity`` (signed) of ``symbol`` at ``price``.

    Positive ``quantity`` is a buy, negative a sell.  ``fees`` are transaction
    costs.  ``date`` is an optional ISO date string.
    """

    symbol: str
    quantity: float
    price: float
    date: str | None = None
    fees: float = 0.0

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise PortfolioValidationError("Trade.symbol must be non-empty")
        if not _finite(self.quantity) or self.quantity == 0.0:
            raise PortfolioValidationError("Trade.quantity must be finite and non-zero")
        if not _finite(self.price) or self.price < 0.0:
            raise PortfolioValidationError("Trade.price must be finite and >= 0")
        if not _finite(self.fees) or self.fees < 0.0:
            raise PortfolioValidationError("Trade.fees must be finite and >= 0")

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0.0 else "sell"

    @property
    def notional(self) -> float:
        """Absolute traded value, ``|quantity| * price`` (excludes fees)."""
        return float(abs(self.quantity) * self.price)

    @property
    def cash_flow(self) -> float:
        """Effect on cash: negative for buys (cash out), positive for sells,
        always net of fees."""
        return float(-self.quantity * self.price - self.fees)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "date": self.date,
            "fees": self.fees,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trade:
        return cls(
            symbol=str(data["symbol"]),
            quantity=float(data["quantity"]),
            price=float(data["price"]),
            date=data.get("date"),
            fees=float(data.get("fees", 0.0)),
        )


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
