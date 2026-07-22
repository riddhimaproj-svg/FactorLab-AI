"""Digital options: BS cross-checks and cash/asset decomposition of a vanilla."""

from __future__ import annotations

import math

import pytest

from factorlab_derivatives import (
    DerivativesInputError,
    DigitalKind,
    OptionType,
    black_scholes_price,
    digital_price,
)


def test_cash_call_is_discounted_n_d2() -> None:
    # cash-or-nothing call with unit payout = e^{-rT} N(d2)
    price = digital_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    assert price == pytest.approx(0.5323248154537634, abs=1e-9)


def test_cash_put_plus_cash_call_equals_discount_factor() -> None:
    c = digital_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    p = digital_price(OptionType.PUT, 100, 100, 1.0, 0.05, 0.2)
    assert c + p == pytest.approx(math.exp(-0.05), abs=1e-10)


def test_vanilla_equals_asset_minus_cash_struck_at_k() -> None:
    s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    asset = digital_price(OptionType.CALL, s, k, t, r, sigma, kind=DigitalKind.ASSET_OR_NOTHING)
    cash = digital_price(
        OptionType.CALL, s, k, t, r, sigma, payout=k, kind=DigitalKind.CASH_OR_NOTHING
    )
    vanilla = black_scholes_price(OptionType.CALL, s, k, t, r, sigma)
    assert asset - cash == pytest.approx(vanilla, abs=1e-10)


def test_payout_scales_linearly() -> None:
    base = digital_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, payout=1.0)
    scaled = digital_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, payout=10.0)
    assert scaled == pytest.approx(10.0 * base, abs=1e-10)


def test_expiry_and_zero_vol_limits() -> None:
    itm = digital_price(OptionType.CALL, 120, 100, 0.0, 0.05, 0.2, payout=5.0)
    assert itm == 5.0
    otm = digital_price(OptionType.CALL, 80, 100, 0.0, 0.05, 0.2, payout=5.0)
    assert otm == 0.0
    asset = digital_price(
        OptionType.CALL, 120, 100, 0.0, 0.05, 0.2, kind=DigitalKind.ASSET_OR_NOTHING
    )
    assert asset == 120.0


def test_rejects_bad_inputs() -> None:
    with pytest.raises(DerivativesInputError):
        digital_price(OptionType.CALL, 100, -1, 1.0, 0.05, 0.2)
