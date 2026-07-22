r"""Benchmark-relative metrics.

These compare a portfolio's return series to a benchmark of equal length.
"Active" quantities are computed from the difference series
:math:`a_t = r_t - b_t`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_portfolio.errors import DimensionMismatchError

__all__ = [
    "active_return",
    "beta",
    "information_ratio",
    "tracking_error",
    "treynor_ratio",
]

FloatArray = NDArray[np.float64]

# Standard deviation at or below this level is treated as zero (see performance).
_ZERO_TOL = 1e-13


def _aligned(returns: FloatArray, benchmark: FloatArray) -> tuple[FloatArray, FloatArray]:
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if r.shape != b.shape:
        raise DimensionMismatchError(r.shape[0], b.shape[0], name="benchmark")
    return r, b


def beta(returns: FloatArray, benchmark: FloatArray) -> float:
    r"""Market/benchmark beta: :math:`\operatorname{Cov}(r, b)/\operatorname{Var}(b)`.

    Interpretation: sensitivity of the portfolio to the benchmark.  ``1`` moves
    one-for-one; ``>1`` amplifies; ``<1`` dampens; ``<0`` moves opposite.  ``nan``
    if benchmark variance is zero or fewer than two observations.
    """
    r, b = _aligned(returns, benchmark)
    if r.size < 2:
        return float("nan")
    var_b = np.var(b, ddof=1)
    if np.sqrt(var_b) <= _ZERO_TOL:
        return float("nan")
    cov = np.cov(r, b, ddof=1)[0, 1]
    return float(cov / var_b)


def active_return(
    returns: FloatArray, benchmark: FloatArray, periods_per_year: float = 252.0
) -> float:
    r"""Annualized mean active return, :math:`\overline{(r - b)}\,P`.

    Interpretation: the average outperformance (positive) or underperformance
    (negative) versus the benchmark, per year.
    """
    r, b = _aligned(returns, benchmark)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r - b) * periods_per_year)


def tracking_error(
    returns: FloatArray, benchmark: FloatArray, periods_per_year: float = 252.0
) -> float:
    r"""Annualized standard deviation of active returns,
    :math:`\sigma(r - b)\sqrt{P}`.

    Interpretation: how tightly the portfolio follows the benchmark.  Low
    tracking error means index-like behavior; high means large active bets.
    ``nan`` if fewer than two observations.
    """
    r, b = _aligned(returns, benchmark)
    if r.size < 2:
        return float("nan")
    return float(np.std(r - b, ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: FloatArray, benchmark: FloatArray, periods_per_year: float = 252.0
) -> float:
    r"""Information ratio: active return per unit of tracking error.

    .. math:: \mathrm{IR} = \frac{\overline{(r-b)}}{\sigma(r-b)}\sqrt{P}.

    Interpretation: the consistency and magnitude of a manager's outperformance;
    the "Sharpe ratio of active returns".  ``nan`` if tracking error is zero.
    """
    r, b = _aligned(returns, benchmark)
    if r.size < 2:
        return float("nan")
    active = r - b
    sd = np.std(active, ddof=1)
    if sd <= _ZERO_TOL:
        return float("nan")
    return float(np.mean(active) / sd * np.sqrt(periods_per_year))


def treynor_ratio(
    returns: FloatArray,
    benchmark: FloatArray,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    r"""Treynor ratio: annualized excess return per unit of beta,
    :math:`\overline{r^e}\,P / \beta`.

    Like Sharpe, but divides by *systematic* risk (beta) rather than total
    volatility -- appropriate for a well-diversified portfolio whose idiosyncratic
    risk is negligible.  ``nan`` if beta is (near) zero.
    """
    r, b = _aligned(returns, benchmark)
    if r.size < 2:
        return float("nan")
    portfolio_beta = beta(r, b)
    if np.isnan(portfolio_beta) or abs(portfolio_beta) < 1e-12:
        return float("nan")
    excess_mean = float(np.mean(r - risk_free))
    return float(excess_mean * periods_per_year / portfolio_beta)
