r"""Implied-volatility solver.

Given a market option price, find the Black-Scholes volatility that reproduces it.
Uses **Newton-Raphson** with vega (fast quadratic convergence when well-behaved),
falling back to **Brent's method** (bracketed, guaranteed) if Newton stalls.

The target price must respect static no-arbitrage bounds
``max(discounted intrinsic, 0) <= price <= upper bound`` (``S e^{-qT}`` for a
call, ``K e^{-rT}`` for a put); otherwise no real implied vol exists and a
:class:`NoArbitrageError` is raised.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from factorlab_derivatives._validation import check_non_negative, check_positive
from factorlab_derivatives.errors import ConvergenceError, NoArbitrageError
from factorlab_derivatives.instruments import OptionType
from factorlab_derivatives.pricing.black_scholes import black_scholes_greeks, black_scholes_price
from factorlab_derivatives.reports import ImpliedVolatilityResult

__all__ = ["implied_volatility"]

_MIN_VOL = 1e-9
_MAX_VOL = 5.0


def _bounds(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
) -> tuple[float, float]:
    disc_spot = spot * np.exp(-dividend * maturity)
    disc_strike = strike * np.exp(-rate * maturity)
    if option_type is OptionType.CALL:
        return max(disc_spot - disc_strike, 0.0), disc_spot
    return max(disc_strike - disc_spot, 0.0), disc_strike


def implied_volatility(
    target_price: float,
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    *,
    initial: float = 0.2,
    tol: float = 1e-8,
    max_iterations: int = 100,
) -> ImpliedVolatilityResult:
    """Solve for the Black-Scholes implied volatility of ``target_price``."""
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_positive(maturity, "maturity")
    check_non_negative(target_price, "target_price")

    lower, upper = _bounds(option_type, spot, strike, maturity, rate, dividend)
    if target_price < lower - 1e-10 or target_price > upper + 1e-10:
        raise NoArbitrageError(
            f"price {target_price} outside no-arbitrage bounds [{lower:.6f}, {upper:.6f}]"
        )

    def price_at(vol: float) -> float:
        return black_scholes_price(option_type, spot, strike, maturity, rate, vol, dividend)

    # --- Newton-Raphson ---
    vol = max(min(initial, _MAX_VOL), _MIN_VOL)
    for i in range(1, max_iterations + 1):
        diff = price_at(vol) - target_price
        if abs(diff) < tol:
            return ImpliedVolatilityResult(
                implied_volatility=vol, converged=True, iterations=i,
                method="newton", target_price=target_price,
            )
        vega = black_scholes_greeks(
            option_type, spot, strike, maturity, rate, vol, dividend
        ).vega
        if vega < 1e-10:
            break
        vol -= diff / vega
        if not _MIN_VOL <= vol <= _MAX_VOL:
            break

    # --- Brent fallback (bracketed) ---
    f_lo = price_at(_MIN_VOL) - target_price
    f_hi = price_at(_MAX_VOL) - target_price
    if f_lo * f_hi > 0.0:
        raise ConvergenceError(
            "implied volatility not bracketed in [1e-9, 5]; price near a bound",
            iterations=max_iterations,
        )
    root, results = optimize.brentq(
        lambda v: price_at(v) - target_price, _MIN_VOL, _MAX_VOL,
        xtol=tol, maxiter=max_iterations, full_output=True,
    )
    return ImpliedVolatilityResult(
        implied_volatility=float(root),
        converged=bool(results.converged),
        iterations=int(results.iterations),
        method="brent",
        target_price=target_price,
    )
