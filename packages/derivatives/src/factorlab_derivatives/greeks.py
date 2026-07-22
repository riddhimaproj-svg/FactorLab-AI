"""Generic finite-difference Greeks.

Given any price function ``f(spot, maturity, rate, volatility) -> price``, this
computes the Greeks by central differences.  It is the reference used to validate
the analytical Black-Scholes/Black-76 Greeks, and the practical way to obtain
Greeks for models without closed forms (e.g. American options via the binomial
tree).

``theta`` is ``dV/dt = -dV/dT`` (per year); ``vega`` and ``rho`` are per unit of
volatility and rate respectively — matching the analytical convention.
"""

from __future__ import annotations

from collections.abc import Callable

from factorlab_derivatives.reports import Greeks

__all__ = ["PriceFn", "finite_difference_greeks"]

# f(spot, maturity, rate, volatility) -> price
PriceFn = Callable[[float, float, float, float], float]


def finite_difference_greeks(
    price_fn: PriceFn,
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    *,
    relative_ds: float = 1e-4,
    d_vol: float = 1e-4,
    d_t: float = 1e-5,
    d_r: float = 1e-4,
) -> Greeks:
    """Central-difference Greeks of ``price_fn`` at the given point."""
    ds = spot * relative_ds
    base = price_fn(spot, maturity, rate, volatility)

    up_s = price_fn(spot + ds, maturity, rate, volatility)
    down_s = price_fn(spot - ds, maturity, rate, volatility)
    delta = (up_s - down_s) / (2.0 * ds)
    gamma = (up_s - 2.0 * base + down_s) / (ds * ds)

    vega = (
        price_fn(spot, maturity, rate, volatility + d_vol)
        - price_fn(spot, maturity, rate, volatility - d_vol)
    ) / (2.0 * d_vol)

    # theta = dV/dt = -dV/dT
    theta = -(
        price_fn(spot, maturity + d_t, rate, volatility)
        - price_fn(spot, maturity - d_t, rate, volatility)
    ) / (2.0 * d_t)

    rho = (
        price_fn(spot, maturity, rate + d_r, volatility)
        - price_fn(spot, maturity, rate - d_r, volatility)
    ) / (2.0 * d_r)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
