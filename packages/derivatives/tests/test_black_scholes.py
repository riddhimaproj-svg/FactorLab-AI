"""Black-Scholes pricing and Greeks: analytical references, parity, degenerate limits."""

from __future__ import annotations

import math

import pytest

from factorlab_derivatives import (
    DerivativesInputError,
    OptionType,
    black_scholes_greeks,
    black_scholes_price,
    d1_d2,
    finite_difference_greeks,
)


def test_atm_call_matches_textbook_value() -> None:
    # Hull's canonical S=K=100, r=5%, sigma=20%, T=1 -> 10.4506
    price = black_scholes_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    assert price == pytest.approx(10.450583572185565, abs=1e-9)


def test_atm_put_matches_textbook_value() -> None:
    price = black_scholes_price(OptionType.PUT, 100, 100, 1.0, 0.05, 0.2)
    assert price == pytest.approx(5.573526022256971, abs=1e-9)


def test_put_call_parity() -> None:
    s, k, t, r, q, sigma = 100.0, 95.0, 0.75, 0.03, 0.01, 0.25
    c = black_scholes_price(OptionType.CALL, s, k, t, r, sigma, q)
    p = black_scholes_price(OptionType.PUT, s, k, t, r, sigma, q)
    lhs = c - p
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_deep_itm_call_approaches_discounted_forward() -> None:
    price = black_scholes_price(OptionType.CALL, 1000, 100, 1.0, 0.05, 0.2)
    intrinsic = 1000 - 100 * math.exp(-0.05)
    assert price == pytest.approx(intrinsic, rel=1e-6)


def test_expired_option_is_intrinsic() -> None:
    assert black_scholes_price(OptionType.CALL, 120, 100, 0.0, 0.05, 0.2) == 20.0
    assert black_scholes_price(OptionType.PUT, 80, 100, 0.0, 0.05, 0.2) == 20.0
    assert black_scholes_price(OptionType.CALL, 80, 100, 0.0, 0.05, 0.2) == 0.0


def test_zero_vol_is_discounted_forward_intrinsic() -> None:
    price = black_scholes_price(OptionType.CALL, 100, 90, 1.0, 0.05, 0.0)
    fwd = 100 * math.exp(0.05)
    assert price == pytest.approx(math.exp(-0.05) * max(fwd - 90, 0.0), abs=1e-12)


@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_rejects_nonpositive_spot(bad: float) -> None:
    with pytest.raises(DerivativesInputError):
        black_scholes_price(OptionType.CALL, bad, 100, 1.0, 0.05, 0.2)


def test_rejects_negative_maturity() -> None:
    with pytest.raises(DerivativesInputError):
        black_scholes_price(OptionType.CALL, 100, 100, -1.0, 0.05, 0.2)


def test_d1_d2_relationship() -> None:
    d1, d2 = d1_d2(100, 100, 1.0, 0.05, 0.2, 0.0)
    assert d1 - d2 == pytest.approx(0.2 * math.sqrt(1.0), abs=1e-12)


@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_greeks_match_finite_difference(opt: OptionType) -> None:
    s, k, t, r, q, sigma = 100.0, 105.0, 1.5, 0.04, 0.02, 0.3
    analytic = black_scholes_greeks(opt, s, k, t, r, sigma, q)
    numeric = finite_difference_greeks(
        lambda spot, mat, rate, vol: black_scholes_price(opt, spot, k, mat, rate, vol, q),
        s, t, r, sigma,
    )
    assert analytic.delta == pytest.approx(numeric.delta, abs=1e-5)
    assert analytic.gamma == pytest.approx(numeric.gamma, abs=1e-4)
    assert analytic.vega == pytest.approx(numeric.vega, abs=1e-3)
    assert analytic.theta == pytest.approx(numeric.theta, abs=1e-3)
    assert analytic.rho == pytest.approx(numeric.rho, abs=1e-3)


def test_call_delta_bounds() -> None:
    g = black_scholes_greeks(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0


def test_put_delta_is_negative() -> None:
    g = black_scholes_greeks(OptionType.PUT, 100, 100, 1.0, 0.05, 0.2)
    assert -1.0 < g.delta < 0.0


def test_degenerate_greeks_are_intrinsic_indicator() -> None:
    g = black_scholes_greeks(OptionType.CALL, 120, 100, 0.0, 0.05, 0.2)
    assert g.delta == 1.0
    assert g.gamma == g.vega == g.theta == g.rho == 0.0
    g_otm = black_scholes_greeks(OptionType.CALL, 80, 100, 0.0, 0.05, 0.2)
    assert g_otm.delta == 0.0
