"""Black-76: equivalence to Black-Scholes on the forward, parity, and rho identity."""

from __future__ import annotations

import math

import pytest

from factorlab_derivatives import (
    DerivativesInputError,
    OptionType,
    black76_greeks,
    black76_price,
    black_scholes_price,
    finite_difference_greeks,
)


def test_equals_black_scholes_with_spot_forward_and_q_equal_r() -> None:
    f, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    b76 = black76_price(OptionType.CALL, f, k, t, r, sigma)
    # BS with S=F and dividend yield = rate reproduces Black-76.
    bs = black_scholes_price(OptionType.CALL, f, k, t, r, sigma, dividend=r)
    assert b76 == pytest.approx(bs, abs=1e-12)


def test_put_call_parity_on_forward() -> None:
    f, k, t, r, sigma = 105.0, 100.0, 0.5, 0.04, 0.3
    c = black76_price(OptionType.CALL, f, k, t, r, sigma)
    p = black76_price(OptionType.PUT, f, k, t, r, sigma)
    assert c - p == pytest.approx(math.exp(-r * t) * (f - k), abs=1e-10)


def test_rho_is_minus_t_times_price() -> None:
    f, k, t, r, sigma = 100.0, 100.0, 2.0, 0.05, 0.25
    price = black76_price(OptionType.CALL, f, k, t, r, sigma)
    g = black76_greeks(OptionType.CALL, f, k, t, r, sigma)
    assert g.rho == pytest.approx(-t * price, abs=1e-12)


@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_greeks_match_finite_difference(opt: OptionType) -> None:
    f, k, t, r, sigma = 100.0, 95.0, 1.0, 0.03, 0.22
    analytic = black76_greeks(opt, f, k, t, r, sigma)
    numeric = finite_difference_greeks(
        lambda fwd, mat, rate, vol: black76_price(opt, fwd, k, mat, rate, vol),
        f, t, r, sigma,
    )
    assert analytic.delta == pytest.approx(numeric.delta, abs=1e-5)
    assert analytic.gamma == pytest.approx(numeric.gamma, abs=1e-4)
    assert analytic.vega == pytest.approx(numeric.vega, abs=1e-3)
    assert analytic.theta == pytest.approx(numeric.theta, abs=1e-3)
    assert analytic.rho == pytest.approx(numeric.rho, abs=1e-3)


def test_expiry_and_zero_vol_limits() -> None:
    assert black76_price(OptionType.CALL, 110, 100, 0.0, 0.05, 0.2) == 10.0
    zero_vol = black76_price(OptionType.CALL, 110, 100, 1.0, 0.05, 0.0)
    assert zero_vol == pytest.approx(math.exp(-0.05) * 10.0, abs=1e-12)


def test_degenerate_greeks() -> None:
    g = black76_greeks(OptionType.CALL, 110, 100, 0.0, 0.05, 0.2)
    assert g.delta == 1.0 and g.gamma == 0.0


def test_rejects_bad_inputs() -> None:
    with pytest.raises(DerivativesInputError):
        black76_price(OptionType.CALL, -1, 100, 1.0, 0.05, 0.2)
