"""Immutable Portfolio and PortfolioSnapshot models.

A :class:`Portfolio` is a set of positions plus cash at a point in time.  It is
immutable: state transitions (e.g. applying a trade) return a *new* portfolio,
which keeps history reconstructable and makes the model safe to share across a
future backtesting or optimization layer.

A :class:`PortfolioSnapshot` is a lightweight, timestamped valuation record
(total value + weighted holdings) suitable for building a time series of a
portfolio's value from which returns are derived.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from factorlab_portfolio.errors import PortfolioValidationError
from factorlab_portfolio.holdings import Holding, Position, Trade

__all__ = ["Portfolio", "PortfolioSnapshot"]


@dataclass(frozen=True, slots=True)
class Portfolio:
    """A collection of positions plus cash, valued at a point in time."""

    positions: tuple[Position, ...]
    cash: float = 0.0
    base_currency: str = "USD"
    as_of: str | None = None
    name: str = "portfolio"

    def __init__(
        self,
        positions: Iterable[Position],
        cash: float = 0.0,
        base_currency: str = "USD",
        as_of: str | None = None,
        name: str = "portfolio",
    ) -> None:
        materialized = tuple(positions)
        symbols = [p.symbol for p in materialized]
        if len(set(symbols)) != len(symbols):
            raise PortfolioValidationError("Portfolio has duplicate position symbols")
        object.__setattr__(self, "positions", materialized)
        object.__setattr__(self, "cash", float(cash))
        object.__setattr__(self, "base_currency", base_currency)
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "name", name)

    # ------------------------------------------------------------------ #
    # Valuation                                                           #
    # ------------------------------------------------------------------ #
    @property
    def total_market_value(self) -> float:
        """Sum of position market values (excludes cash)."""
        return float(sum(p.market_value for p in self.positions))

    @property
    def total_value(self) -> float:
        """Net asset value: market value of positions plus cash."""
        return self.total_market_value + self.cash

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(p.symbol for p in self.positions)

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute position values (long + short)."""
        return float(sum(abs(p.market_value) for p in self.positions))

    def position(self, symbol: str) -> Position:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        raise KeyError(f"No position for symbol {symbol!r}")

    def weights(self) -> dict[str, float]:
        """Position weights as a fraction of ``total_value``.

        Weights sum to ``1 - cash/total_value`` (cash is not a holding).  Returns
        zeros when ``total_value`` is zero.
        """
        total = self.total_value
        if total == 0.0:
            return {p.symbol: 0.0 for p in self.positions}
        return {p.symbol: p.market_value / total for p in self.positions}

    def holdings(self) -> tuple[Holding, ...]:
        """Portfolio-level holdings (symbol, market value, weight)."""
        weights = self.weights()
        return tuple(
            Holding(p.symbol, p.market_value, weights[p.symbol]) for p in self.positions
        )

    # ------------------------------------------------------------------ #
    # State transition (pure)                                             #
    # ------------------------------------------------------------------ #
    def apply_trade(self, trade: Trade) -> Portfolio:
        """Return a new portfolio reflecting ``trade`` (immutable transition).

        Updates the affected position's quantity, marks it at the trade price,
        maintains a weighted-average cost basis, and adjusts cash by the trade's
        cash flow.  A position that nets to zero is removed.  This is a single,
        pure state transition -- not a backtesting loop.
        """
        others = [p for p in self.positions if p.symbol != trade.symbol]
        existing = next((p for p in self.positions if p.symbol == trade.symbol), None)

        if existing is None:
            new_positions = [
                *others,
                Position(trade.symbol, trade.quantity, trade.price, cost_basis=trade.price),
            ]
        else:
            new_qty = existing.quantity + trade.quantity
            if abs(new_qty) < 1e-12:
                new_positions = others  # position closed
            else:
                new_positions = [
                    *others,
                    Position(
                        trade.symbol,
                        new_qty,
                        trade.price,
                        cost_basis=_updated_cost_basis(existing, trade, new_qty),
                    ),
                ]
        return Portfolio(
            new_positions,
            cash=self.cash + trade.cash_flow,
            base_currency=self.base_currency,
            as_of=trade.date or self.as_of,
            name=self.name,
        )

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": [p.to_dict() for p in self.positions],
            "cash": self.cash,
            "base_currency": self.base_currency,
            "as_of": self.as_of,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Portfolio:
        return cls(
            positions=[Position.from_dict(p) for p in data["positions"]],
            cash=float(data.get("cash", 0.0)),
            base_currency=str(data.get("base_currency", "USD")),
            as_of=data.get("as_of"),
            name=str(data.get("name", "portfolio")),
        )


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """A timestamped valuation of a portfolio (total value + weighted holdings)."""

    as_of: str
    total_value: float
    holdings: tuple[Holding, ...]
    cash: float = 0.0

    @classmethod
    def from_portfolio(cls, portfolio: Portfolio, as_of: str | None = None) -> PortfolioSnapshot:
        stamp = as_of if as_of is not None else portfolio.as_of
        if stamp is None:
            raise PortfolioValidationError(
                "PortfolioSnapshot requires an as_of date (pass one or set it on the portfolio)"
            )
        return cls(
            as_of=stamp,
            total_value=portfolio.total_value,
            holdings=portfolio.holdings(),
            cash=portfolio.cash,
        )

    def weights(self) -> dict[str, float]:
        return {h.symbol: h.weight for h in self.holdings}

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "total_value": self.total_value,
            "holdings": [h.to_dict() for h in self.holdings],
            "cash": self.cash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PortfolioSnapshot:
        return cls(
            as_of=str(data["as_of"]),
            total_value=float(data["total_value"]),
            holdings=tuple(Holding.from_dict(h) for h in data["holdings"]),
            cash=float(data.get("cash", 0.0)),
        )


def _updated_cost_basis(existing: Position, trade: Trade, new_qty: float) -> float:
    old_cb = existing.cost_basis if existing.cost_basis is not None else existing.price
    same_direction = (existing.quantity > 0) == (trade.quantity > 0)
    crossed_zero = (existing.quantity > 0) != (new_qty > 0)
    if crossed_zero:
        return trade.price  # flipped long<->short; basis resets to trade price
    if same_direction:
        # Adding to the position: size-weighted average cost.
        return float((existing.quantity * old_cb + trade.quantity * trade.price) / new_qty)
    return old_cb  # partial reduction: basis unchanged
