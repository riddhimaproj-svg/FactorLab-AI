r"""Drawdown analytics.

A *drawdown* at time :math:`t` is the fractional decline of cumulative wealth
from its running peak:

.. math:: D_t = \frac{W_t}{\max_{s \le t} W_s} - 1 \le 0,
   \qquad W_t = \prod_{s \le t}(1 + r_s).

Drawdown captures the peak-to-trough pain an investor actually experiences, and
is the risk measure behind the Calmar ratio.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_portfolio.analytics.performance import wealth_index

__all__ = [
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "time_to_recovery",
]

FloatArray = NDArray[np.float64]


def drawdown_series(returns: FloatArray) -> FloatArray:
    """Per-period drawdown series (each value in ``(-1, 0]``)."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    wealth = wealth_index(arr)
    running_peak = np.maximum.accumulate(wealth)
    # A running peak of 0 means wealth was wiped out; drawdown there is -100%.
    with np.errstate(invalid="ignore", divide="ignore"):
        drawdown = wealth / running_peak - 1.0
    return np.where(running_peak > 0.0, drawdown, -1.0)


def max_drawdown(returns: FloatArray) -> float:
    r"""Maximum drawdown, :math:`\min_t D_t` (a non-positive number).

    Interpretation: the worst peak-to-trough loss over the period.  ``-0.30``
    means the portfolio fell 30% from a high-water mark.  ``nan`` for an empty
    series.
    """
    dd = drawdown_series(returns)
    return float(dd.min()) if dd.size else float("nan")


def max_drawdown_duration(returns: FloatArray) -> int:
    """Longest underwater stretch, in periods.

    The maximum number of consecutive periods spent below a prior peak (from the
    start of a drawdown until wealth recovers to that peak, or the series ends).
    Interpretation: how long the investor waited to get back to even in the worst
    case.
    """
    dd = drawdown_series(returns)
    longest = 0
    current = 0
    for value in dd:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def time_to_recovery(returns: FloatArray) -> int | None:
    """Periods from the deepest trough back to its prior peak.

    Returns ``None`` if the series never recovers from its maximum drawdown by
    the end of the sample (still underwater).
    """
    dd = drawdown_series(returns)
    if dd.size == 0:
        return None
    trough = int(np.argmin(dd))
    if dd[trough] == 0.0:
        return 0
    # Walk forward from the trough to the first full recovery (drawdown == 0).
    for i in range(trough, dd.size):
        if dd[i] >= 0.0:
            return int(i - trough)
    return None
