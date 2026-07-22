"""Instruments, market data, and result models: validation and serialization."""

from __future__ import annotations

import pytest

from factorlab_derivatives import (
    BarrierOption,
    BarrierType,
    DerivativesInputError,
    DigitalKind,
    DigitalOption,
    ExerciseStyle,
    Greeks,
    MarketData,
    MonteCarloResult,
    Option,
    OptionType,
)


def test_option_type_sign() -> None:
    assert OptionType.CALL.sign == 1.0
    assert OptionType.PUT.sign == -1.0


def test_option_intrinsic_and_is_call() -> None:
    opt = Option(OptionType.CALL, strike=100.0, maturity=1.0)
    assert opt.is_call
    assert opt.intrinsic_value(120.0) == 20.0
    assert opt.intrinsic_value(80.0) == 0.0
    put = Option(OptionType.PUT, strike=100.0, maturity=1.0)
    assert put.intrinsic_value(80.0) == 20.0


def test_option_rejects_bad_strike() -> None:
    with pytest.raises(DerivativesInputError):
        Option(OptionType.CALL, strike=-1.0, maturity=1.0)


def test_option_rejects_negative_maturity() -> None:
    with pytest.raises(DerivativesInputError):
        Option(OptionType.CALL, strike=100.0, maturity=-1.0)


def test_option_serialization_round_trip() -> None:
    opt = Option(OptionType.PUT, 100.0, 0.5, exercise=ExerciseStyle.AMERICAN)
    assert Option.from_dict(opt.to_dict()) == opt


def test_digital_option_serialization() -> None:
    d = DigitalOption(OptionType.CALL, 100.0, 1.0, payout=5.0, kind=DigitalKind.ASSET_OR_NOTHING)
    assert DigitalOption.from_dict(d.to_dict()) == d


def test_digital_rejects_negative_payout() -> None:
    with pytest.raises(DerivativesInputError):
        DigitalOption(OptionType.CALL, 100.0, 1.0, payout=-1.0)


def test_barrier_option_serialization() -> None:
    b = BarrierOption(OptionType.CALL, 100.0, 1.0, barrier=90.0,
                      barrier_type=BarrierType.DOWN_AND_IN)
    assert BarrierOption.from_dict(b.to_dict()) == b


def test_barrier_type_flags() -> None:
    assert BarrierType.DOWN_AND_IN.is_knock_in
    assert BarrierType.DOWN_AND_IN.is_down
    assert not BarrierType.UP_AND_OUT.is_knock_in
    assert not BarrierType.UP_AND_OUT.is_down


def test_market_data_validation() -> None:
    with pytest.raises(DerivativesInputError):
        MarketData(spot=-1.0, rate=0.05, volatility=0.2)
    with pytest.raises(DerivativesInputError):
        MarketData(spot=100.0, rate=float("nan"), volatility=0.2)
    with pytest.raises(DerivativesInputError):
        MarketData(spot=100.0, rate=float("inf"), volatility=0.2)


def test_market_data_serialization() -> None:
    m = MarketData(spot=100.0, rate=0.05, volatility=0.2, dividend_yield=0.01)
    assert MarketData.from_dict(m.to_dict()) == m


def test_greeks_serialization() -> None:
    g = Greeks(delta=0.5, gamma=0.02, vega=30.0, theta=-6.0, rho=50.0)
    assert Greeks.from_dict(g.to_dict()) == g


def test_monte_carlo_confidence_interval() -> None:
    mc = MonteCarloResult(price=10.0, standard_error=0.5, n_paths=1000, method="x")
    lo, hi = mc.confidence_interval
    assert lo == pytest.approx(10.0 - 1.959963984540054 * 0.5)
    assert hi == pytest.approx(10.0 + 1.959963984540054 * 0.5)
    assert "1,000 paths" in mc.summary()


def test_enums_are_strings() -> None:
    # str-Enum members serialize as their value.
    assert OptionType.CALL.value == "call"
    assert ExerciseStyle.AMERICAN.value == "american"
    assert DigitalKind.CASH_OR_NOTHING.value == "cash_or_nothing"
