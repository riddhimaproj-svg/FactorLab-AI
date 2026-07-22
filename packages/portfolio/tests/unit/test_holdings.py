"""Tests for Position, Holding, and Trade models."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_portfolio.errors import PortfolioValidationError
from factorlab_portfolio.holdings import Holding, Position, Trade


# -- Position -------------------------------------------------------------- #
def test_position_market_value_and_pnl() -> None:
    p = Position("AAPL", 10, 150.0, cost_basis=100.0)
    assert p.market_value == pytest.approx(1500.0)
    assert p.unrealized_pnl == pytest.approx(500.0)
    assert p.is_long and not p.is_short


def test_position_short() -> None:
    p = Position("TSLA", -5, 200.0)
    assert p.market_value == pytest.approx(-1000.0)
    assert p.is_short
    assert np.isnan(p.unrealized_pnl)  # no cost basis


def test_position_validation() -> None:
    with pytest.raises(PortfolioValidationError):
        Position("", 1, 10.0)
    with pytest.raises(PortfolioValidationError):
        Position("A", 1, -10.0)
    with pytest.raises(PortfolioValidationError):
        Position("A", float("nan"), 10.0)


def test_position_roundtrip() -> None:
    p = Position("AAPL", 10, 150.0, cost_basis=100.0)
    assert Position.from_dict(p.to_dict()) == p
    p2 = Position("MSFT", 3, 300.0)
    assert Position.from_dict(p2.to_dict()) == p2


# -- Holding --------------------------------------------------------------- #
def test_holding_roundtrip() -> None:
    h = Holding("AAPL", 1500.0, 0.375)
    assert Holding.from_dict(h.to_dict()) == h


def test_holding_validation() -> None:
    with pytest.raises(PortfolioValidationError):
        Holding("", 1.0, 0.5)


# -- Trade ----------------------------------------------------------------- #
def test_trade_buy_semantics() -> None:
    t = Trade("AAPL", 10, 150.0, date="2024-01-02", fees=1.0)
    assert t.side == "buy"
    assert t.notional == pytest.approx(1500.0)
    assert t.cash_flow == pytest.approx(-1501.0)  # cash out + fees


def test_trade_sell_semantics() -> None:
    t = Trade("AAPL", -10, 150.0, fees=1.0)
    assert t.side == "sell"
    assert t.cash_flow == pytest.approx(1499.0)  # cash in - fees


def test_trade_validation() -> None:
    with pytest.raises(PortfolioValidationError):
        Trade("AAPL", 0, 150.0)  # zero quantity
    with pytest.raises(PortfolioValidationError):
        Trade("AAPL", 1, -1.0)  # negative price
    with pytest.raises(PortfolioValidationError):
        Trade("AAPL", 1, 10.0, fees=-1.0)  # negative fees


def test_trade_roundtrip() -> None:
    t = Trade("AAPL", 10, 150.0, date="2024-01-02", fees=1.0)
    assert Trade.from_dict(t.to_dict()) == t
