r"""Cox-Ross-Rubinstein (CRR) binomial-tree pricing.

The CRR tree discretizes the underlying into ``steps`` periods of length
:math:`\Delta t = T/n` with up/down moves and risk-neutral probability

.. math::

    u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad
    p = \frac{e^{(r-q)\Delta t} - d}{u - d}.

European values are the discounted risk-neutral expectation; **American** values
apply early-exercise (``max(continuation, intrinsic)``) at every node via backward
induction.  As ``steps`` grows, the European price converges to Black-Scholes.
"""

from __future__ import annotations

import numpy as np

from factorlab_derivatives._validation import check_non_negative, check_positive
from factorlab_derivatives.errors import DerivativesInputError
from factorlab_derivatives.instruments import OptionType

__all__ = ["binomial_price"]


def binomial_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    *,
    steps: int = 500,
    american: bool = False,
) -> float:
    """CRR binomial price of a European or American option."""
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_non_negative(maturity, "maturity")
    check_non_negative(volatility, "volatility")
    if steps < 1:
        raise DerivativesInputError("steps must be >= 1")
    phi = option_type.sign

    if maturity <= 0.0 or volatility <= 0.0:
        # No time value: intrinsic (discounted forward intrinsic if vol is zero).
        if maturity <= 0.0:
            return max(phi * (spot - strike), 0.0)
        forward = spot * np.exp((rate - dividend) * maturity)
        return float(np.exp(-rate * maturity) * max(phi * (forward - strike), 0.0))

    dt = maturity / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-rate * dt)
    p = (np.exp((rate - dividend) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise DerivativesInputError(
            "risk-neutral probability outside [0, 1]; increase steps or check inputs"
        )

    # Terminal asset prices and payoffs.
    j = np.arange(steps + 1)
    asset = spot * u**j * d ** (steps - j)
    values = np.maximum(phi * (asset - strike), 0.0)

    # Backward induction.
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1 : i + 2] + (1.0 - p) * values[0 : i + 1])
        if american:
            asset_i = spot * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            values = np.maximum(values, phi * (asset_i - strike))
    return float(values[0])
