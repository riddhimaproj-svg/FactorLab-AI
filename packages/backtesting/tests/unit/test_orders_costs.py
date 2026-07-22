"""Tests for orders, transaction-cost, slippage, and broker models."""

from __future__ import annotations

import pytest

from factorlab_backtesting import (
    BidAskSpreadSlippage,
    BrokerModel,
    CompositeCostModel,
    Fill,
    FixedBpsSlippage,
    FixedCommission,
    Order,
    OrderBook,
    PercentageCommission,
    ZeroCostModel,
    ZeroSlippage,
)
from factorlab_backtesting.errors import BacktestInputError


# -- Order / Fill / OrderBook --------------------------------------------- #
def test_order_semantics() -> None:
    o = Order("A", 10.0, 100.0)
    assert o.side == "buy"
    assert o.notional == pytest.approx(1000.0)
    assert Order("A", -5.0, 100.0).side == "sell"


def test_order_validation() -> None:
    with pytest.raises(BacktestInputError):
        Order("", 1.0, 100.0)
    with pytest.raises(BacktestInputError):
        Order("A", 1.0, -1.0)


def test_fill_cash_impact() -> None:
    buy = Fill("A", 10.0, 100.0, commission=5.0)
    assert buy.cash_impact == pytest.approx(-1005.0)  # pay 1000 + 5 fee
    sell = Fill("A", -10.0, 100.0, commission=5.0)
    assert sell.cash_impact == pytest.approx(995.0)  # receive 1000 - 5 fee
    assert buy.gross_value == pytest.approx(1000.0)


def test_orderbook() -> None:
    book = OrderBook.from_orders([Order("A", 1.0, 10.0), Order("B", -2.0, 20.0)])
    assert len(book) == 2
    assert book.total_notional == pytest.approx(10.0 + 40.0)
    assert set(book.symbols) == {"A", "B"}
    restored = OrderBook.from_dict(book.to_dict())
    assert len(restored) == 2


# -- Transaction cost models ---------------------------------------------- #
def test_zero_cost() -> None:
    assert ZeroCostModel().commission(Order("A", 10.0, 100.0), 100.0) == 0.0


def test_fixed_commission() -> None:
    model = FixedCommission(1.5)
    assert model.commission(Order("A", 10.0, 100.0), 100.0) == pytest.approx(1.5)


def test_percentage_commission() -> None:
    model = PercentageCommission(0.001)  # 10 bps
    # |10| * 100 * 0.001 = 1.0
    assert model.commission(Order("A", 10.0, 100.0), 100.0) == pytest.approx(1.0)


def test_composite_cost() -> None:
    model = CompositeCostModel([FixedCommission(1.0), PercentageCommission(0.001)])
    assert model.commission(Order("A", 10.0, 100.0), 100.0) == pytest.approx(2.0)


def test_cost_validation() -> None:
    with pytest.raises(BacktestInputError):
        FixedCommission(-1.0)
    with pytest.raises(BacktestInputError):
        PercentageCommission(-0.1)


# -- Slippage models ------------------------------------------------------- #
def test_zero_slippage() -> None:
    assert ZeroSlippage().execution_price(Order("A", 1.0, 100.0), 100.0) == 100.0


def test_fixed_bps_slippage_direction() -> None:
    slip = FixedBpsSlippage(50)  # 50 bps
    buy_price = slip.execution_price(Order("A", 1.0, 100.0), 100.0)
    sell_price = slip.execution_price(Order("A", -1.0, 100.0), 100.0)
    assert buy_price == pytest.approx(100.5)  # buys pay more
    assert sell_price == pytest.approx(99.5)  # sells receive less


def test_bid_ask_half_spread() -> None:
    slip = BidAskSpreadSlippage(40)  # 40 bps spread -> 20 bps half
    assert slip.execution_price(Order("A", 1.0, 100.0), 100.0) == pytest.approx(100.2)


def test_slippage_validation() -> None:
    with pytest.raises(BacktestInputError):
        FixedBpsSlippage(-1.0)
    with pytest.raises(BacktestInputError):
        BidAskSpreadSlippage(-1.0)


# -- Broker ---------------------------------------------------------------- #
def test_broker_combines_slippage_and_cost() -> None:
    broker = BrokerModel(PercentageCommission(0.001), FixedBpsSlippage(50))
    fill = broker.execute(Order("A", 10.0, 100.0), 100.0)
    assert fill.price == pytest.approx(100.5)  # slippage applied
    # commission on executed price: 10 * 100.5 * 0.001
    assert fill.commission == pytest.approx(10 * 100.5 * 0.001)


def test_broker_defaults_frictionless() -> None:
    fill = BrokerModel().execute(Order("A", 10.0, 100.0), 100.0)
    assert fill.price == 100.0
    assert fill.commission == 0.0
