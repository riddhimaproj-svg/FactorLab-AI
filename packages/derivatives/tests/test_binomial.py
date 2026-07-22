"""Binomial (CRR): convergence to Black-Scholes and American early-exercise premium."""

from __future__ import annotations

import pytest

from factorlab_derivatives import (
    DerivativesInputError,
    OptionType,
    binomial_price,
    black_scholes_price,
)


@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_european_converges_to_black_scholes(opt: OptionType) -> None:
    s, k, t, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    bs = black_scholes_price(opt, s, k, t, r, sigma, q)
    coarse = abs(binomial_price(opt, s, k, t, r, sigma, q, steps=25) - bs)
    fine = abs(binomial_price(opt, s, k, t, r, sigma, q, steps=2000) - bs)
    assert fine < coarse
    assert fine < 1e-2


def test_american_call_no_dividend_equals_european() -> None:
    # Without dividends an American call is never exercised early.
    euro = binomial_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, steps=500, american=False)
    amer = binomial_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, steps=500, american=True)
    assert amer == pytest.approx(euro, abs=1e-10)


def test_american_put_has_early_exercise_premium() -> None:
    euro = binomial_price(OptionType.PUT, 100, 110, 1.0, 0.08, 0.3, steps=500, american=False)
    amer = binomial_price(OptionType.PUT, 100, 110, 1.0, 0.08, 0.3, steps=500, american=True)
    assert amer > euro


def test_expiry_and_zero_vol_limits() -> None:
    assert binomial_price(OptionType.CALL, 120, 100, 0.0, 0.05, 0.2) == 20.0
    zv = binomial_price(OptionType.CALL, 100, 90, 1.0, 0.05, 0.0)
    assert zv > 0.0


def test_rejects_zero_steps() -> None:
    with pytest.raises(DerivativesInputError):
        binomial_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, steps=0)


def test_rejects_bad_spot() -> None:
    with pytest.raises(DerivativesInputError):
        binomial_price(OptionType.CALL, 0.0, 100, 1.0, 0.05, 0.2)
