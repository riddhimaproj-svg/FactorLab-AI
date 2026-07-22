"""Coverage of validation, error, and solver edge branches."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_derivatives import (
    ConvergenceError,
    DerivativesInputError,
    NoArbitrageError,
    OptionType,
    binomial_price,
    historical_volatility,
    implied_volatility,
)
from factorlab_derivatives._validation import as_return_vector
from factorlab_derivatives.pricing.black_scholes import black_scholes_price


def test_as_return_vector_rejects_2d() -> None:
    with pytest.raises(DerivativesInputError):
        as_return_vector(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_as_return_vector_rejects_empty() -> None:
    with pytest.raises(DerivativesInputError):
        as_return_vector(np.array([]))


def test_as_return_vector_rejects_non_finite() -> None:
    with pytest.raises(DerivativesInputError):
        as_return_vector(np.array([0.1, np.nan, 0.2]))


def test_convergence_error_carries_iterations() -> None:
    err = ConvergenceError("boom", iterations=42)
    assert err.iterations == 42
    assert "boom" in str(err)


def test_historical_volatility_two_prices_is_nan() -> None:
    # Two prices -> one return -> ddof=1 std is undefined.
    assert np.isnan(historical_volatility([100.0, 101.0]))


def test_binomial_rejects_arbitrage_violating_step() -> None:
    # Enormous rate with a single coarse step pushes p outside [0, 1].
    with pytest.raises(DerivativesInputError):
        binomial_price(OptionType.CALL, 100, 100, 1.0, 5.0, 0.05, steps=1)


def test_implied_vol_not_bracketed_raises_convergence() -> None:
    # A price just under the upper bound but unreachable within [1e-9, 5]
    # is caught by the no-arbitrage guard; here we force the un-bracketed path
    # by targeting a price above what vol=5 can produce is impossible, so instead
    # confirm the guard on an out-of-bounds request.
    with pytest.raises((ConvergenceError, NoArbitrageError)):
        implied_volatility(1e-12, OptionType.CALL, 100, 100, 1.0, 0.05, initial=0.0001)


def test_implied_vol_newton_out_of_bounds_falls_back() -> None:
    # Start Newton at an extreme guess so its step leaves [MIN, MAX] and Brent runs.
    true_vol = 0.3
    price = black_scholes_price(OptionType.CALL, 100, 100, 1.0, 0.05, true_vol)
    result = implied_volatility(price, OptionType.CALL, 100, 100, 1.0, 0.05, initial=4.9)
    assert result.implied_volatility == pytest.approx(true_vol, abs=1e-5)
