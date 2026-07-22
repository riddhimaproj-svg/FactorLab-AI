"""Tests for the ExecutionEngine rebalancing logic."""

from __future__ import annotations

import pytest

from factorlab_backtesting import BrokerModel, ExecutionEngine, PercentageCommission
from factorlab_backtesting.costs import FixedBpsSlippage


def test_rebalance_from_cash_to_target() -> None:
    engine = ExecutionEngine()
    positions = {"A": 0.0, "B": 0.0}
    cash = 1000.0
    target = {"A": 0.6, "B": 0.4}
    prices = {"A": 10.0, "B": 20.0}
    out = engine.rebalance(positions, cash, target, prices)
    # 60% of 1000 = 600 in A at 10 -> 60 shares; 40% -> 400 in B at 20 -> 20 shares
    assert out.positions["A"] == pytest.approx(60.0)
    assert out.positions["B"] == pytest.approx(20.0)
    assert out.cash == pytest.approx(0.0, abs=1e-9)
    assert out.turnover == pytest.approx(1.0)  # fully traded from cash


def test_rebalance_conserves_value_without_costs() -> None:
    engine = ExecutionEngine()
    positions = {"A": 10.0, "B": 0.0}
    cash = 0.0
    prices = {"A": 10.0, "B": 20.0}
    pre_value = cash + 10 * 10.0
    out = engine.rebalance(positions, cash, {"A": 0.5, "B": 0.5}, prices)
    post_value = out.cash + sum(out.positions[s] * prices[s] for s in prices)
    assert post_value == pytest.approx(pre_value)  # frictionless is value-preserving


def test_costs_reduce_value() -> None:
    engine = ExecutionEngine(BrokerModel(PercentageCommission(0.01), FixedBpsSlippage(50)))
    positions = {"A": 0.0, "B": 0.0}
    prices = {"A": 10.0, "B": 20.0}
    out = engine.rebalance(positions, 1000.0, {"A": 0.5, "B": 0.5}, prices)
    post_value = out.cash + sum(out.positions[s] * prices[s] for s in prices)
    assert post_value < 1000.0  # costs bleed value
    assert out.total_commission > 0.0


def test_no_trade_when_already_on_target() -> None:
    engine = ExecutionEngine()
    prices = {"A": 10.0, "B": 20.0}
    # already 50/50 with 60 in each
    positions = {"A": 6.0, "B": 3.0}  # 60 in A, 60 in B
    out = engine.rebalance(positions, 0.0, {"A": 0.5, "B": 0.5}, prices)
    assert len(out.order_book) == 0
    assert out.turnover == pytest.approx(0.0)


def test_selling_generates_cash() -> None:
    engine = ExecutionEngine()
    prices = {"A": 10.0}
    positions = {"A": 100.0}  # 1000 in A
    out = engine.rebalance(positions, 0.0, {"A": 0.0}, prices)  # liquidate
    assert out.positions["A"] == pytest.approx(0.0)
    assert out.cash == pytest.approx(1000.0)
