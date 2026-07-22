r"""High-level pricing engine — the ergonomic front door to the pure core.

:func:`price_option` accepts an :class:`Option` and a :class:`MarketData` and
dispatches to the appropriate model:

* **European** options -> analytical Black-Scholes (exact, with closed-form Greeks).
* **American** options -> CRR binomial tree, with Greeks by finite differences
  (there is no closed form for the American early-exercise premium).

Callers who want full control can always use the ``pricing`` functions directly;
this façade just wires the common cases together and returns a serializable
:class:`PricingResult`.
"""

from __future__ import annotations

from enum import StrEnum

from factorlab_derivatives.greeks import finite_difference_greeks
from factorlab_derivatives.instruments import ExerciseStyle, MarketData, Option
from factorlab_derivatives.pricing.binomial import binomial_price
from factorlab_derivatives.pricing.black_scholes import black_scholes_greeks, black_scholes_price
from factorlab_derivatives.reports import PricingResult

__all__ = ["PricingMethod", "price_option"]


class PricingMethod(StrEnum):
    """How to price the option.  ``AUTO`` picks by exercise style."""

    AUTO = "auto"
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"


def price_option(
    option: Option,
    market: MarketData,
    *,
    method: PricingMethod = PricingMethod.AUTO,
    steps: int = 500,
    with_greeks: bool = True,
) -> PricingResult:
    """Price ``option`` under ``market`` state, returning a :class:`PricingResult`."""
    resolved = _resolve_method(option, method)
    s, k, t = market.spot, option.strike, option.maturity
    r, sigma, q = market.rate, market.volatility, market.dividend_yield

    if resolved is PricingMethod.BLACK_SCHOLES:
        price = black_scholes_price(option.option_type, s, k, t, r, sigma, q)
        greeks = (
            black_scholes_greeks(option.option_type, s, k, t, r, sigma, q)
            if with_greeks
            else None
        )
        metadata: dict[str, object] = {"exercise": option.exercise.value}
    else:
        american = option.exercise is ExerciseStyle.AMERICAN
        price = binomial_price(
            option.option_type, s, k, t, r, sigma, q, steps=steps, american=american
        )
        greeks = None
        if with_greeks:

            def price_fn(
                spot: float, maturity: float, rate: float, volatility: float
            ) -> float:
                return binomial_price(
                    option.option_type, spot, k, maturity, rate, volatility, q,
                    steps=steps, american=american,
                )

            greeks = finite_difference_greeks(price_fn, s, t, r, sigma)
        metadata = {"exercise": option.exercise.value, "steps": steps}

    return PricingResult(
        price=price,
        method=resolved.value,
        option_type=option.option_type.value,
        greeks=greeks,
        metadata=metadata,
    )


def _resolve_method(option: Option, method: PricingMethod) -> PricingMethod:
    if method is not PricingMethod.AUTO:
        return method
    if option.exercise is ExerciseStyle.AMERICAN:
        return PricingMethod.BINOMIAL
    return PricingMethod.BLACK_SCHOLES
