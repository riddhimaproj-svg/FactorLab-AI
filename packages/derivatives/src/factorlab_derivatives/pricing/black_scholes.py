r"""Black-Scholes-Merton pricing and analytical Greeks (European options).

With continuous dividend yield :math:`q`, spot :math:`S`, strike :math:`K`,
maturity :math:`T`, rate :math:`r`, volatility :math:`\sigma`:

.. math::

    d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \quad
    d_2 = d_1 - \sigma\sqrt{T},

    C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
    P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).

Setting :math:`q = 0` recovers the classic Black-Scholes model.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from factorlab_derivatives._validation import check_non_negative, check_positive
from factorlab_derivatives.instruments import OptionType
from factorlab_derivatives.reports import Greeks

__all__ = ["black_scholes_greeks", "black_scholes_price", "d1_d2"]

_N = stats.norm.cdf
_n = stats.norm.pdf


def d1_d2(
    spot: float, strike: float, maturity: float, rate: float, volatility: float, dividend: float
) -> tuple[float, float]:
    """Return the Black-Scholes ``(d1, d2)`` terms."""
    vol_sqrt_t = volatility * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(d1), float(d2)


def black_scholes_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European option."""
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_non_negative(maturity, "maturity")
    check_non_negative(volatility, "volatility")
    phi = option_type.sign

    # Degenerate limits: at/after expiry, or zero volatility -> discounted intrinsic.
    if maturity <= 0.0:
        return max(phi * (spot - strike), 0.0)
    if volatility <= 0.0:
        forward = spot * np.exp((rate - dividend) * maturity)
        return float(np.exp(-rate * maturity) * max(phi * (forward - strike), 0.0))

    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend)
    disc_spot = spot * np.exp(-dividend * maturity)
    disc_strike = strike * np.exp(-rate * maturity)
    return float(phi * disc_spot * _N(phi * d1) - phi * disc_strike * _N(phi * d2))


def black_scholes_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
) -> Greeks:
    """Analytical Black-Scholes Greeks (raw partial derivatives)."""
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_non_negative(maturity, "maturity")
    check_non_negative(volatility, "volatility")
    phi = option_type.sign

    if maturity <= 0.0 or volatility <= 0.0:
        # Degenerate: delta is the (discounted) exercise indicator; the rest vanish.
        intrinsic_sign = 1.0 if phi * (spot - strike) > 0.0 else 0.0
        return Greeks(delta=phi * intrinsic_sign, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    sqrt_t = np.sqrt(maturity)
    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend)
    disc_q = np.exp(-dividend * maturity)
    disc_r = np.exp(-rate * maturity)
    pdf_d1 = _n(d1)

    delta = phi * disc_q * _N(phi * d1)
    gamma = disc_q * pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    theta = (
        -(spot * disc_q * pdf_d1 * volatility) / (2.0 * sqrt_t)
        - phi * rate * strike * disc_r * _N(phi * d2)
        + phi * dividend * spot * disc_q * _N(phi * d1)
    )
    rho = phi * strike * maturity * disc_r * _N(phi * d2)
    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )
