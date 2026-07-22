"""Tests for Portfolio and PortfolioSnapshot."""

from __future__ import annotations

import pytest

from factorlab_portfolio.errors import PortfolioValidationError
from factorlab_portfolio.holdings import Position, Trade
from factorlab_portfolio.portfolio import Portfolio, PortfolioSnapshot


def _portfolio() -> Portfolio:
    return Portfolio(
        [Position("AAPL", 10, 150.0, cost_basis=100.0), Position("MSFT", 5, 300.0)],
        cash=1000.0,
        as_of="2024-01-01",
    )


def test_valuation() -> None:
    p = _portfolio()
    assert p.total_market_value == pytest.approx(1500.0 + 1500.0)
    assert p.total_value == pytest.approx(4000.0)
    assert p.gross_exposure == pytest.approx(3000.0)
    assert set(p.symbols) == {"AAPL", "MSFT"}


def test_weights_sum_to_invested_fraction() -> None:
    p = _portfolio()
    w = p.weights()
    assert w["AAPL"] == pytest.approx(0.375)
    assert w["MSFT"] == pytest.approx(0.375)
    assert sum(w.values()) == pytest.approx(0.75)  # 0.25 is cash


def test_holdings() -> None:
    holdings = _portfolio().holdings()
    assert {h.symbol for h in holdings} == {"AAPL", "MSFT"}
    assert holdings[0].weight == pytest.approx(0.375)


def test_zero_total_value_weights() -> None:
    p = Portfolio([Position("A", 0.0, 10.0)], cash=0.0)
    assert p.weights() == {"A": 0.0}


def test_position_lookup_and_missing() -> None:
    p = _portfolio()
    assert p.position("AAPL").quantity == 10
    with pytest.raises(KeyError):
        p.position("NVDA")


def test_duplicate_symbol_rejected() -> None:
    with pytest.raises(PortfolioValidationError):
        Portfolio([Position("A", 1, 10.0), Position("A", 2, 11.0)])


def test_apply_trade_new_position() -> None:
    p = Portfolio([], cash=1000.0)
    p2 = p.apply_trade(Trade("AAPL", 5, 100.0))
    assert p2.position("AAPL").quantity == 5
    assert p2.position("AAPL").cost_basis == pytest.approx(100.0)
    assert p2.cash == pytest.approx(500.0)
    # original unchanged (immutability)
    assert p.symbols == ()


def test_apply_trade_add_averages_cost_basis() -> None:
    p = Portfolio([Position("AAPL", 10, 100.0, cost_basis=100.0)], cash=0.0)
    p2 = p.apply_trade(Trade("AAPL", 10, 200.0))
    pos = p2.position("AAPL")
    assert pos.quantity == 20
    # weighted avg cost = (10*100 + 10*200)/20 = 150
    assert pos.cost_basis == pytest.approx(150.0)


def test_apply_trade_reduce_keeps_cost_basis() -> None:
    p = Portfolio([Position("AAPL", 10, 100.0, cost_basis=90.0)], cash=0.0)
    p2 = p.apply_trade(Trade("AAPL", -4, 120.0))
    pos = p2.position("AAPL")
    assert pos.quantity == 6
    assert pos.cost_basis == pytest.approx(90.0)  # unchanged on partial sell


def test_apply_trade_close_removes_position() -> None:
    p = Portfolio([Position("AAPL", 10, 100.0)], cash=0.0)
    p2 = p.apply_trade(Trade("AAPL", -10, 120.0))
    assert "AAPL" not in p2.symbols
    assert p2.cash == pytest.approx(1200.0)


def test_apply_trade_flip_resets_cost_basis() -> None:
    p = Portfolio([Position("AAPL", 5, 100.0, cost_basis=100.0)], cash=0.0)
    p2 = p.apply_trade(Trade("AAPL", -8, 120.0))  # now short 3
    pos = p2.position("AAPL")
    assert pos.quantity == pytest.approx(-3)
    assert pos.cost_basis == pytest.approx(120.0)  # reset to trade price


def test_portfolio_roundtrip() -> None:
    p = _portfolio()
    restored = Portfolio.from_dict(p.to_dict())
    assert restored.total_value == pytest.approx(p.total_value)
    assert restored.symbols == p.symbols
    assert restored.cash == pytest.approx(p.cash)


# -- PortfolioSnapshot ----------------------------------------------------- #
def test_snapshot_from_portfolio() -> None:
    snap = PortfolioSnapshot.from_portfolio(_portfolio())
    assert snap.as_of == "2024-01-01"
    assert snap.total_value == pytest.approx(4000.0)
    assert snap.weights()["AAPL"] == pytest.approx(0.375)


def test_snapshot_requires_as_of() -> None:
    p = Portfolio([Position("A", 1, 10.0)])  # no as_of
    with pytest.raises(PortfolioValidationError):
        PortfolioSnapshot.from_portfolio(p)


def test_snapshot_explicit_as_of_and_roundtrip() -> None:
    p = Portfolio([Position("A", 1, 10.0)])
    snap = PortfolioSnapshot.from_portfolio(p, as_of="2024-06-30")
    restored = PortfolioSnapshot.from_dict(snap.to_dict())
    assert restored.as_of == "2024-06-30"
    assert restored.total_value == pytest.approx(snap.total_value)
