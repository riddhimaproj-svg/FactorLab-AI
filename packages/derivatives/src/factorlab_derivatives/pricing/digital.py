r"""Digital (binary) option pricing.

A **cash-or-nothing** option pays a fixed amount ``Q`` if it finishes in the
money; an **asset-or-nothing** option pays the asset itself.  Under Black-Scholes
(dividend yield :math:`q`):

.. math::

    \text{cash call} = Q e^{-rT} N(d_2), \qquad
    \text{asset call} = S e^{-qT} N(d_1),

with puts using :math:`N(-d_2)` / :math:`N(-d_1)`.  (A vanilla call equals an
asset-or-nothing call minus a cash-or-nothing call struck at :math:`K` with
:math:`Q = K` — a useful cross-check.)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from factorlab_derivatives._validation import check_non_negative, check_positive
from factorlab_derivatives.instruments import DigitalKind, OptionType
from factorlab_derivatives.pricing.black_scholes import d1_d2

__all__ = ["digital_price"]

_N = stats.norm.cdf


def digital_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    *,
    payout: float = 1.0,
    kind: DigitalKind = DigitalKind.CASH_OR_NOTHING,
) -> float:
    """Black-Scholes price of a European digital option."""
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_non_negative(maturity, "maturity")
    check_non_negative(volatility, "volatility")
    check_non_negative(payout, "payout")
    phi = option_type.sign

    if maturity <= 0.0 or volatility <= 0.0:
        in_the_money = phi * (spot - strike) > 0.0
        if not in_the_money:
            return 0.0
        disc = float(np.exp(-rate * maturity))
        if kind is DigitalKind.CASH_OR_NOTHING:
            return payout * disc
        return float(spot * np.exp(-dividend * maturity))

    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend)
    if kind is DigitalKind.CASH_OR_NOTHING:
        return float(payout * np.exp(-rate * maturity) * _N(phi * d2))
    return float(spot * np.exp(-dividend * maturity) * _N(phi * d1))
