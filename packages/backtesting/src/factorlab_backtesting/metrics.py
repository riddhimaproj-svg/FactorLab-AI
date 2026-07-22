r"""Backtest-specific performance metrics.

These complement the portfolio package's :class:`PerformanceReport` (return,
volatility, Sharpe, Sortino, Calmar, information ratio, Treynor, tracking error,
beta, max drawdown) with metrics particular to a *simulated strategy*: Jensen's
alpha, turnover, hit ratio, and win rate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_backtesting.errors import BacktestInputError

__all__ = ["alpha_beta", "annualized_turnover", "hit_ratio", "win_rate"]

FloatArray = NDArray[np.float64]
_ZERO_TOL = 1e-13


def alpha_beta(
    returns: FloatArray,
    benchmark: FloatArray,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> tuple[float, float]:
    r"""Jensen's alpha (annualized) and beta from the CAPM time-series regression
    :math:`r^e = \alpha + \beta\, b^e + \varepsilon`.

    Beta is the systematic sensitivity to the benchmark; alpha is the average
    return unexplained by that exposure, annualized by ``periods_per_year``.
    """
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if r.shape != b.shape:
        raise BacktestInputError("returns and benchmark must have the same length")
    if r.shape[0] < 2:
        return float("nan"), float("nan")
    excess_r = r - risk_free
    excess_b = b - risk_free
    var_b = np.var(excess_b, ddof=1)
    if var_b <= _ZERO_TOL**2:
        return float("nan"), float("nan")
    beta = float(np.cov(excess_r, excess_b, ddof=1)[0, 1] / var_b)
    alpha_per_period = float(np.mean(excess_r) - beta * np.mean(excess_b))
    return alpha_per_period * periods_per_year, beta


def win_rate(returns: FloatArray) -> float:
    """Fraction of periods with a strictly positive return."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0.0))


def hit_ratio(returns: FloatArray, benchmark: FloatArray) -> float:
    """Fraction of periods the strategy strictly beats the benchmark."""
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if r.shape != b.shape:
        raise BacktestInputError("returns and benchmark must have the same length")
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > b))


def annualized_turnover(average_turnover: float, rebalances_per_year: float) -> float:
    """Scale average per-rebalance (two-way) turnover to an annual figure."""
    return float(average_turnover * rebalances_per_year)
