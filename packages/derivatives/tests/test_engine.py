"""Engine façade: method dispatch, Greeks wiring, and result metadata."""

from __future__ import annotations

import pytest

from factorlab_derivatives import (
    ExerciseStyle,
    MarketData,
    Option,
    OptionType,
    PricingMethod,
    PricingResult,
    binomial_price,
    black_scholes_price,
    price_option,
)


def test_european_dispatches_to_black_scholes(atm_call: Option, atm_market: MarketData) -> None:
    result = price_option(atm_call, atm_market)
    assert result.method == "black_scholes"
    assert result.greeks is not None
    expected = black_scholes_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    assert result.price == pytest.approx(expected, abs=1e-12)


def test_american_dispatches_to_binomial(atm_market: MarketData) -> None:
    opt = Option(OptionType.PUT, 100.0, 1.0, exercise=ExerciseStyle.AMERICAN)
    result = price_option(opt, atm_market, steps=400)
    assert result.method == "binomial"
    assert result.metadata["steps"] == 400
    expected = binomial_price(OptionType.PUT, 100, 100, 1.0, 0.05, 0.2, steps=400, american=True)
    assert result.price == pytest.approx(expected, abs=1e-12)


def test_american_greeks_via_finite_difference(atm_market: MarketData) -> None:
    opt = Option(OptionType.PUT, 100.0, 1.0, exercise=ExerciseStyle.AMERICAN)
    result = price_option(opt, atm_market, steps=200)
    assert result.greeks is not None
    assert result.greeks.delta < 0.0  # put delta
    assert result.greeks.gamma > 0.0


def test_with_greeks_false_skips_greeks(atm_call: Option, atm_market: MarketData) -> None:
    result = price_option(atm_call, atm_market, with_greeks=False)
    assert result.greeks is None


def test_explicit_method_override(atm_call: Option, atm_market: MarketData) -> None:
    result = price_option(atm_call, atm_market, method=PricingMethod.BINOMIAL, steps=500)
    assert result.method == "binomial"
    bs = black_scholes_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    assert result.price == pytest.approx(bs, abs=1e-2)


def test_binomial_without_greeks(atm_call: Option, atm_market: MarketData) -> None:
    result = price_option(
        atm_call, atm_market, method=PricingMethod.BINOMIAL, with_greeks=False
    )
    assert result.greeks is None


def test_result_summary_and_serialization(atm_call: Option, atm_market: MarketData) -> None:
    result = price_option(atm_call, atm_market)
    text = result.summary()
    assert "Call option" in text and "Delta" in text
    restored = PricingResult.from_dict(result.to_dict())
    assert restored.price == result.price
    assert restored.greeks == result.greeks


def test_summary_without_greeks(atm_call: Option, atm_market: MarketData) -> None:
    result = price_option(atm_call, atm_market, with_greeks=False)
    assert "Delta" not in result.summary()
