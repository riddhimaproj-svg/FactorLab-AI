"""Implied volatility: round-trip recovery, method fallback, and no-arbitrage bounds."""

from __future__ import annotations

import pytest

from factorlab_derivatives import (
    NoArbitrageError,
    OptionType,
    black_scholes_price,
    implied_volatility,
)


@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("true_vol", [0.05, 0.15, 0.4, 0.9])
@pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
def test_recovers_known_volatility(opt: OptionType, true_vol: float, strike: float) -> None:
    s, t, r, q = 100.0, 1.0, 0.05, 0.01
    price = black_scholes_price(opt, s, strike, t, r, true_vol, q)
    result = implied_volatility(price, opt, s, strike, t, r, q)
    assert result.converged
    # Solver converges to a price tolerance; for deep-OTM/low-vol points vega is
    # tiny, so the implied vol is recovered to ~1e-4 rather than machine precision.
    assert result.implied_volatility == pytest.approx(true_vol, abs=1e-4)
    # The recovered vol must reproduce the input price to high precision.
    reprice = black_scholes_price(opt, s, strike, t, r, result.implied_volatility, q)
    assert reprice == pytest.approx(price, abs=1e-7)


def test_newton_path_is_used_for_well_behaved_atm() -> None:
    price = black_scholes_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.25)
    result = implied_volatility(price, OptionType.CALL, 100, 100, 1.0, 0.05)
    assert result.method == "newton"
    assert result.iterations >= 1


def test_brent_fallback_recovers_deep_otm_vol() -> None:
    # Deep OTM long-dated: Newton can overshoot; Brent must still recover it.
    true_vol = 0.6
    price = black_scholes_price(OptionType.CALL, 100, 300, 2.0, 0.05, true_vol)
    result = implied_volatility(price, OptionType.CALL, 100, 300, 2.0, 0.05, initial=0.01)
    assert result.implied_volatility == pytest.approx(true_vol, abs=1e-5)


def test_price_below_intrinsic_raises() -> None:
    with pytest.raises(NoArbitrageError):
        implied_volatility(0.001, OptionType.CALL, 200, 100, 1.0, 0.05)


def test_price_above_upper_bound_raises() -> None:
    with pytest.raises(NoArbitrageError):
        implied_volatility(500.0, OptionType.CALL, 100, 100, 1.0, 0.05)


def test_result_serializes() -> None:
    from factorlab_derivatives import ImpliedVolatilityResult

    price = black_scholes_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    result = implied_volatility(price, OptionType.CALL, 100, 100, 1.0, 0.05)
    restored = ImpliedVolatilityResult.from_dict(result.to_dict())
    assert restored == result
