r"""Rolling (trailing-window) analytics.

Each function returns an array the same length as the input.  The first
``window - 1`` entries are ``nan`` (an incomplete window), and entry ``i`` is
computed from the trailing window ``returns[i - window + 1 : i + 1]``.  Rolling
views reveal how risk and performance evolve over time rather than collapsing to
a single number.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_portfolio.analytics.performance import sharpe_ratio
from factorlab_portfolio.analytics.relative import beta
from factorlab_portfolio.errors import DimensionMismatchError

__all__ = [
    "rolling_beta",
    "rolling_return",
    "rolling_sharpe",
    "rolling_volatility",
]

FloatArray = NDArray[np.float64]


def _validate_window(n: int, window: int) -> None:
    if window < 1:
        raise ValueError("window must be a positive integer")
    if window > n:
        raise ValueError(f"window ({window}) exceeds series length ({n})")


def _windows(arr: FloatArray, window: int):  # type: ignore[no-untyped-def]
    """Yield ``(i, trailing_window)`` for each complete window."""
    for i in range(window - 1, arr.shape[0]):
        yield i, arr[i - window + 1 : i + 1]


def rolling_return(returns: FloatArray, window: int) -> FloatArray:
    """Rolling compounded return over a trailing ``window``."""
    arr = np.asarray(returns, dtype=np.float64)
    _validate_window(arr.shape[0], window)
    out = np.full(arr.shape[0], np.nan)
    for i, w in _windows(arr, window):
        out[i] = np.prod(1.0 + w) - 1.0
    return out


def rolling_volatility(
    returns: FloatArray, window: int, periods_per_year: float = 252.0
) -> FloatArray:
    """Rolling annualized volatility over a trailing ``window``."""
    arr = np.asarray(returns, dtype=np.float64)
    _validate_window(arr.shape[0], window)
    out = np.full(arr.shape[0], np.nan)
    scale = np.sqrt(periods_per_year)
    for i, w in _windows(arr, window):
        out[i] = np.std(w, ddof=1) * scale
    return out


def rolling_sharpe(
    returns: FloatArray,
    window: int,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> FloatArray:
    """Rolling annualized Sharpe ratio over a trailing ``window``."""
    arr = np.asarray(returns, dtype=np.float64)
    _validate_window(arr.shape[0], window)
    out = np.full(arr.shape[0], np.nan)
    for i, w in _windows(arr, window):
        out[i] = sharpe_ratio(w, risk_free=risk_free, periods_per_year=periods_per_year)
    return out


def rolling_beta(returns: FloatArray, benchmark: FloatArray, window: int) -> FloatArray:
    """Rolling beta of ``returns`` on ``benchmark`` over a trailing ``window``."""
    arr = np.asarray(returns, dtype=np.float64)
    bench = np.asarray(benchmark, dtype=np.float64)
    if arr.shape != bench.shape:
        raise DimensionMismatchError(arr.shape[0], bench.shape[0], name="benchmark")
    _validate_window(arr.shape[0], window)
    out = np.full(arr.shape[0], np.nan)
    for i in range(window - 1, arr.shape[0]):
        sl = slice(i - window + 1, i + 1)
        out[i] = beta(arr[sl], bench[sl])
    return out
