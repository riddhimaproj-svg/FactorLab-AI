"""Property-based invariants (Hypothesis): bounds, monotonicity, and parity."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from factorlab_derivatives import OptionType, black_scholes_price, monte_carlo_european

spots = st.floats(min_value=1.0, max_value=1000.0)
strikes = st.floats(min_value=1.0, max_value=1000.0)
maturities = st.floats(min_value=0.01, max_value=5.0)
rates = st.floats(min_value=-0.05, max_value=0.15)
vols = st.floats(min_value=0.01, max_value=1.5)


@settings(max_examples=200, deadline=None)
@given(spots, strikes, maturities, rates, vols)
def test_price_is_non_negative(s: float, k: float, t: float, r: float, sigma: float) -> None:
    assert black_scholes_price(OptionType.CALL, s, k, t, r, sigma) >= -1e-9
    assert black_scholes_price(OptionType.PUT, s, k, t, r, sigma) >= -1e-9


@settings(max_examples=200, deadline=None)
@given(spots, strikes, maturities, rates, vols)
def test_call_price_within_bounds(s: float, k: float, t: float, r: float, sigma: float) -> None:
    # max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}  (q = 0 here)
    c = black_scholes_price(OptionType.CALL, s, k, t, r, sigma)
    lower = max(s - k * math.exp(-r * t), 0.0)
    assert lower - 1e-7 <= c <= s + 1e-7


@settings(max_examples=200, deadline=None)
@given(spots, strikes, maturities, rates, vols)
def test_call_monotonic_increasing_in_spot(
    s: float, k: float, t: float, r: float, sigma: float
) -> None:
    lo = black_scholes_price(OptionType.CALL, s, k, t, r, sigma)
    hi = black_scholes_price(OptionType.CALL, s * 1.01, k, t, r, sigma)
    assert hi >= lo - 1e-7


@settings(max_examples=200, deadline=None)
@given(spots, strikes, maturities, rates, vols)
def test_put_call_parity_property(
    s: float, k: float, t: float, r: float, sigma: float
) -> None:
    c = black_scholes_price(OptionType.CALL, s, k, t, r, sigma)
    p = black_scholes_price(OptionType.PUT, s, k, t, r, sigma)
    assert c - p == float("inf") or abs((c - p) - (s - k * math.exp(-r * t))) < 1e-6


@settings(max_examples=200, deadline=None)
@given(spots, strikes, maturities, rates, vols)
def test_call_increasing_in_volatility(
    s: float, k: float, t: float, r: float, sigma: float
) -> None:
    lo = black_scholes_price(OptionType.CALL, s, k, t, r, sigma)
    hi = black_scholes_price(OptionType.CALL, s, k, t, r, min(sigma + 0.05, 2.0))
    assert hi >= lo - 1e-7


@settings(max_examples=30, deadline=None)
@given(
    st.floats(min_value=50.0, max_value=150.0),
    st.floats(min_value=50.0, max_value=150.0),
    st.floats(min_value=0.1, max_value=2.0),
)
def test_monte_carlo_matches_black_scholes(s: float, k: float, t: float) -> None:
    r, sigma = 0.05, 0.2
    bs = black_scholes_price(OptionType.CALL, s, k, t, r, sigma)
    mc = monte_carlo_european(OptionType.CALL, s, k, t, r, sigma, n_paths=80_000, seed=0)
    assert abs(mc.price - bs) < 6.0 * mc.standard_error + 0.05
