r"""Black-76 pricing and Greeks for options on futures / forwards.

Black's (1976) model prices an option on a forward/futures price :math:`F`:

.. math::

    d_1 = \frac{\ln(F/K) + \tfrac12\sigma^2 T}{\sigma\sqrt T}, \quad d_2 = d_1 - \sigma\sqrt T,

    C = e^{-rT}\bigl(F N(d_1) - K N(d_2)\bigr), \qquad
    P = e^{-rT}\bigl(K N(-d_2) - F N(-d_1)\bigr).

It equals Black-Scholes with spot replaced by the forward and dividend yield set
to the rate — except **rho**, which here is ``-T * price`` because the forward is
observed directly and does not move with :math:`r`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from factorlab_derivatives._validation import check_non_negative, check_positive
from factorlab_derivatives.instruments import OptionType
from factorlab_derivatives.reports import Greeks

__all__ = ["black76_greeks", "black76_price"]

_N = stats.norm.cdf
_n = stats.norm.pdf


def _d1_d2(
    forward: float, strike: float, maturity: float, volatility: float
) -> tuple[float, float]:
    vol_sqrt_t = volatility * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * volatility**2 * maturity) / vol_sqrt_t
    return float(d1), float(d1 - vol_sqrt_t)


def black76_price(
    option_type: OptionType,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """Black-76 price of a European option on a forward/futures."""
    check_positive(forward, "forward")
    check_positive(strike, "strike")
    check_non_negative(maturity, "maturity")
    check_non_negative(volatility, "volatility")
    phi = option_type.sign
    disc = np.exp(-rate * maturity)

    if maturity <= 0.0:
        return max(phi * (forward - strike), 0.0)
    if volatility <= 0.0:
        return float(disc * max(phi * (forward - strike), 0.0))

    d1, d2 = _d1_d2(forward, strike, maturity, volatility)
    return float(disc * (phi * forward * _N(phi * d1) - phi * strike * _N(phi * d2)))


def black76_greeks(
    option_type: OptionType,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> Greeks:
    """Analytical Black-76 Greeks (delta/gamma w.r.t. the forward ``F``)."""
    check_positive(forward, "forward")
    check_positive(strike, "strike")
    check_non_negative(maturity, "maturity")
    check_non_negative(volatility, "volatility")
    phi = option_type.sign

    if maturity <= 0.0 or volatility <= 0.0:
        intrinsic_sign = 1.0 if phi * (forward - strike) > 0.0 else 0.0
        return Greeks(delta=phi * intrinsic_sign, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    sqrt_t = np.sqrt(maturity)
    disc = np.exp(-rate * maturity)
    d1, d2 = _d1_d2(forward, strike, maturity, volatility)
    pdf_d1 = _n(d1)
    price = disc * (phi * forward * _N(phi * d1) - phi * strike * _N(phi * d2))

    delta = phi * disc * _N(phi * d1)
    gamma = disc * pdf_d1 / (forward * volatility * sqrt_t)
    vega = forward * disc * pdf_d1 * sqrt_t
    theta = rate * price - disc * forward * pdf_d1 * volatility / (2.0 * sqrt_t)
    rho = -maturity * price
    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )
