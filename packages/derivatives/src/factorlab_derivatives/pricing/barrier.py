r"""Single-barrier option pricing (Reiner-Rubinstein, continuous monitoring).

Knock-**in** options are priced with the closed-form Reiner-Rubinstein (1991)
formulas; knock-**out** options use in/out parity (``knock_in + knock_out =
vanilla`` for zero rebate), which is exact by construction.  Barriers are assumed
continuously monitored with no rebate.  If the barrier is already breached at
inception, an in-option is worth its vanilla and an out-option is worthless.

This module demonstrates the engine's extension pattern: it composes the
Black-Scholes core and the standard barrier-term algebra without touching it.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from factorlab_derivatives._validation import check_positive
from factorlab_derivatives.errors import DerivativesInputError
from factorlab_derivatives.instruments import BarrierType, OptionType
from factorlab_derivatives.pricing.black_scholes import black_scholes_price

__all__ = ["barrier_price"]

_N = stats.norm.cdf


def barrier_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    barrier: float,
    barrier_type: BarrierType,
    dividend: float = 0.0,
) -> float:
    """Price a single-barrier knock-in/out option (continuous monitoring, no rebate)."""
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_positive(barrier, "barrier")
    if maturity <= 0.0 or volatility <= 0.0:
        raise DerivativesInputError(
            "barrier pricing requires strictly positive maturity and volatility"
        )

    vanilla = black_scholes_price(
        option_type, spot, strike, maturity, rate, volatility, dividend
    )
    is_in = barrier_type.is_knock_in
    is_down = barrier_type.is_down

    # Already breached at inception?
    breached = (is_down and spot <= barrier) or ((not is_down) and spot >= barrier)
    if breached:
        knock_in = vanilla
        return knock_in if is_in else 0.0

    knock_in = _knock_in_value(
        option_type, spot, strike, maturity, rate, volatility, barrier, is_down, dividend
    )
    return knock_in if is_in else vanilla - knock_in


def _knock_in_value(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    barrier: float,
    is_down: bool,
    dividend: float,
) -> float:
    S, K, H, T = spot, strike, barrier, maturity
    phi = option_type.sign
    eta = 1.0 if is_down else -1.0
    b = rate - dividend
    vsig = volatility * np.sqrt(T)
    mu = (b - 0.5 * volatility**2) / volatility**2
    disc_q = np.exp((b - rate) * T)  # = e^{-qT}
    disc_r = np.exp(-rate * T)

    x1 = np.log(S / K) / vsig + (1.0 + mu) * vsig
    x2 = np.log(S / H) / vsig + (1.0 + mu) * vsig
    y1 = np.log(H**2 / (S * K)) / vsig + (1.0 + mu) * vsig
    y2 = np.log(H / S) / vsig + (1.0 + mu) * vsig
    hs = H / S

    a = phi * S * disc_q * _N(phi * x1) - phi * K * disc_r * _N(phi * x1 - phi * vsig)
    bb = phi * S * disc_q * _N(phi * x2) - phi * K * disc_r * _N(phi * x2 - phi * vsig)
    c = phi * S * disc_q * hs ** (2.0 * (mu + 1.0)) * _N(eta * y1) - phi * K * disc_r * hs ** (
        2.0 * mu
    ) * _N(eta * y1 - eta * vsig)
    d = phi * S * disc_q * hs ** (2.0 * (mu + 1.0)) * _N(eta * y2) - phi * K * disc_r * hs ** (
        2.0 * mu
    ) * _N(eta * y2 - eta * vsig)

    is_call = option_type is OptionType.CALL
    strike_above_barrier = K > H

    if is_down and is_call:
        value = c if strike_above_barrier else a - bb + d
    elif (not is_down) and is_call:
        value = a if strike_above_barrier else bb - c + d
    elif is_down and (not is_call):
        value = bb - c + d if strike_above_barrier else a
    else:  # up-and-in put
        value = a - bb + d if strike_above_barrier else c
    return float(value)
