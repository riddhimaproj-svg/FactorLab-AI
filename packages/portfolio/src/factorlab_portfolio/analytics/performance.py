r"""Absolute performance and risk metrics.

All functions are pure and operate on 1-D arrays of **per-period simple returns**
in decimal units (``0.01`` = 1%).  Annualization uses ``periods_per_year`` (252
daily, 52 weekly, 12 monthly, 4 quarterly).

Conventions
-----------
* **Geometric** compounding for return/growth metrics (CAGR), because returns
  chain multiplicatively.
* **Sample** standard deviation (``ddof=1``) for volatility, the unbiased
  estimator standard in performance reporting.
* Metrics that require dispersion return ``nan`` when fewer than two
  observations are available or when the denominator is zero, rather than
  raising -- a report should degrade gracefully, not crash.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "cagr",
    "calmar_ratio",
    "cumulative_return",
    "downside_deviation",
    "mean_return",
    "omega_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "wealth_index",
]

FloatArray = NDArray[np.float64]

# Dispersion at or below this level (on decimal-return scale) is treated as zero:
# a std below 1e-13 is numerically a constant series, so ratios are undefined
# rather than an artefact of floating-point noise (~1e-17).
_ZERO_TOL = 1e-13


def _as_returns(returns: FloatArray) -> FloatArray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("returns must be a 1-D array")
    return arr


def wealth_index(returns: FloatArray, initial: float = 1.0) -> FloatArray:
    r"""Cumulative wealth (growth of ``initial``): :math:`\prod_{s\le t}(1+r_s)`."""
    arr = _as_returns(returns)
    return initial * np.cumprod(1.0 + arr)


def cumulative_return(returns: FloatArray) -> float:
    r"""Total compounded return over the whole series,
    :math:`\prod_t (1 + r_t) - 1`.  ``nan`` for an empty series."""
    arr = _as_returns(returns)
    if arr.size == 0:
        return float("nan")
    return float(np.prod(1.0 + arr) - 1.0)


def mean_return(returns: FloatArray) -> float:
    """Arithmetic mean per-period return."""
    arr = _as_returns(returns)
    return float(np.mean(arr)) if arr.size else float("nan")


def cagr(returns: FloatArray, periods_per_year: float) -> float:
    r"""Compound annual growth rate (geometric annualized return).

    .. math:: \mathrm{CAGR} = \left(\prod_t (1+r_t)\right)^{P/n} - 1,

    with ``P = periods_per_year`` and ``n`` observations.  Returns ``nan`` if the
    series is empty or wealth reaches zero/negative (a total wipe-out, where a
    growth rate is undefined).
    """
    arr = _as_returns(returns)
    if arr.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + arr))
    if growth <= 0.0:
        return float("nan")
    return float(growth ** (periods_per_year / arr.size) - 1.0)


def annualized_return(returns: FloatArray, periods_per_year: float) -> float:
    """Alias of :func:`cagr` (geometric annualized return)."""
    return cagr(returns, periods_per_year)


def annualized_volatility(returns: FloatArray, periods_per_year: float) -> float:
    r"""Annualized standard deviation of returns,
    :math:`\sigma \sqrt{P}` with sample :math:`\sigma` (``ddof=1``).

    Interpretation: the dispersion of returns; higher means a wider, riskier
    distribution of outcomes.  ``nan`` if fewer than two observations.
    """
    arr = _as_returns(returns)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: FloatArray, target: float = 0.0, periods_per_year: float = 1.0
) -> float:
    r"""Downside (semi-)deviation relative to ``target`` (a minimum acceptable
    return), annualized by :math:`\sqrt{P}`.

    .. math:: \mathrm{DD} = \sqrt{\tfrac{1}{n}\sum_t \min(r_t - \tau, 0)^2}\,\sqrt{P}.

    Only shortfalls below ``target`` contribute, so it measures "bad" volatility
    only -- the risk an investor actually dislikes.
    """
    arr = _as_returns(returns)
    if arr.size == 0:
        return float("nan")
    shortfall = np.minimum(arr - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: FloatArray, risk_free: float = 0.0, periods_per_year: float = 252.0
) -> float:
    r"""Annualized Sharpe ratio: excess return per unit of total volatility.

    .. math:: \mathrm{Sharpe} = \frac{\bar{r^e}}{\sigma(r^e)} \sqrt{P},
       \qquad r^e_t = r_t - r_f.

    Interpretation: risk-adjusted reward. Higher is better; it says how much
    excess return you earn per unit of return variability.  ``nan`` if the
    excess-return volatility is zero or fewer than two observations.
    """
    arr = _as_returns(returns)
    if arr.size < 2:
        return float("nan")
    excess = arr - risk_free
    sd = np.std(excess, ddof=1)
    if sd <= _ZERO_TOL:
        return float("nan")
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: FloatArray,
    risk_free: float = 0.0,
    target: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    r"""Annualized Sortino ratio: excess return per unit of *downside* deviation.

    Like Sharpe, but penalizes only downside volatility (below ``target``), so
    it does not punish upside variability.  Preferred when the return
    distribution is asymmetric.  ``nan`` if downside deviation is zero.
    """
    arr = _as_returns(returns)
    if arr.size < 1:
        return float("nan")
    excess_mean = float(np.mean(arr - risk_free))
    dd = downside_deviation(arr, target=target, periods_per_year=1.0)
    if np.isnan(dd) or dd <= _ZERO_TOL:
        return float("nan")
    return float(excess_mean / dd * np.sqrt(periods_per_year))


def calmar_ratio(returns: FloatArray, periods_per_year: float) -> float:
    r"""Calmar ratio: CAGR divided by the magnitude of maximum drawdown.

    Rewards steady compounding and penalizes deep peak-to-trough losses.  ``nan``
    when there is no drawdown.  (Depends on :func:`~.drawdown.max_drawdown`.)
    """
    from factorlab_portfolio.analytics.drawdown import max_drawdown

    mdd = abs(max_drawdown(returns))
    if mdd == 0.0 or np.isnan(mdd):
        return float("nan")
    growth = cagr(returns, periods_per_year)
    if np.isnan(growth):
        return float("nan")
    return float(growth / mdd)


def omega_ratio(returns: FloatArray, threshold: float = 0.0) -> float:
    r"""Omega ratio at a per-period ``threshold``:

    .. math:: \Omega(\tau) = \frac{\sum_t \max(r_t - \tau, 0)}
                                  {\sum_t \max(\tau - r_t, 0)}.

    The ratio of probability-weighted gains to losses about ``threshold``; it
    uses the entire return distribution (all moments), not just mean and
    variance.  ``> 1`` means gains outweigh losses at that threshold.  Returns
    ``inf`` when there are gains but no losses, ``nan`` for an empty series.
    """
    arr = _as_returns(returns)
    if arr.size == 0:
        return float("nan")
    gains = float(np.sum(np.maximum(arr - threshold, 0.0)))
    losses = float(np.sum(np.maximum(threshold - arr, 0.0)))
    if losses == 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return float(gains / losses)
