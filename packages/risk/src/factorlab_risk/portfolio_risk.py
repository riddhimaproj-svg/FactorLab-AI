r"""Portfolio-level risk statistics.

Volatility, benchmark-relative risk (tracking error, active risk, information
ratio, beta), the covariance/correlation matrices and their rolling variants, the
diversification ratio, and concentration measures (Herfindahl index, effective
number of holdings).  All functions are pure and operate on per-period simple
returns (decimal) and weight vectors; annualization uses ``periods_per_year``.

These mirror the conventions of the approved portfolio package (sample ``ddof=1``
standard deviation, :math:`\sqrt{P}` annualization) so numbers reconcile across
the platform, while remaining self-contained on plain arrays.
"""

from __future__ import annotations

import numpy as np

from factorlab_risk._validation import (
    FloatArray,
    as_covariance,
    as_return_matrix,
    as_return_vector,
    as_weights,
    check_lengths_match,
)
from factorlab_risk.errors import DimensionMismatchError, RiskInputError

__all__ = [
    "portfolio_volatility",
    "volatility",
    "rolling_volatility",
    "tracking_error",
    "active_risk",
    "information_ratio",
    "beta",
    "rolling_beta",
    "covariance_matrix",
    "correlation_matrix",
    "rolling_covariance",
    "rolling_correlation",
    "diversification_ratio",
    "herfindahl_index",
    "effective_number_of_assets",
    "concentration_ratio",
    "concentration_metrics",
]

_ZERO_TOL = 1e-13


# --------------------------------------------------------------------------- #
# Volatility                                                                  #
# --------------------------------------------------------------------------- #
def portfolio_volatility(weights: object, covariance: object) -> float:
    r"""Portfolio volatility :math:`\sqrt{w'\Sigma w}` (per-period)."""
    w = as_weights(weights)
    cov = as_covariance(covariance)
    if cov.shape[0] != w.shape[0]:
        raise DimensionMismatchError(w.shape[0], cov.shape[0], name="covariance")
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def volatility(returns: object, periods_per_year: float = 252.0) -> float:
    """Annualized volatility of a return series (sample stdev)."""
    r = as_return_vector(returns)
    if r.shape[0] < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def rolling_volatility(
    returns: object, window: int, periods_per_year: float = 252.0
) -> FloatArray:
    """Rolling annualized volatility over a trailing window."""
    r = as_return_vector(returns)
    _validate_window(r.shape[0], window)
    out = np.full(r.shape[0], np.nan)
    scale = np.sqrt(periods_per_year)
    for i in range(window - 1, r.shape[0]):
        out[i] = np.std(r[i - window + 1 : i + 1], ddof=1) * scale
    return out


# --------------------------------------------------------------------------- #
# Benchmark-relative                                                          #
# --------------------------------------------------------------------------- #
def tracking_error(
    returns: object, benchmark: object, periods_per_year: float = 252.0
) -> float:
    """Annualized standard deviation of active (returns - benchmark) returns."""
    r = as_return_vector(returns)
    b = as_return_vector(benchmark, name="benchmark")
    check_lengths_match(r, b)
    if r.shape[0] < 2:
        return float("nan")
    return float(np.std(r - b, ddof=1) * np.sqrt(periods_per_year))


def active_risk(returns: object, benchmark: object, periods_per_year: float = 252.0) -> float:
    """Alias for :func:`tracking_error` (the volatility of active returns)."""
    return tracking_error(returns, benchmark, periods_per_year)


def information_ratio(
    returns: object, benchmark: object, periods_per_year: float = 252.0
) -> float:
    """Annualized active return per unit of tracking error."""
    r = as_return_vector(returns)
    b = as_return_vector(benchmark, name="benchmark")
    check_lengths_match(r, b)
    if r.shape[0] < 2:
        return float("nan")
    active = r - b
    sd = np.std(active, ddof=1)
    if sd <= _ZERO_TOL:
        return float("nan")
    return float(np.mean(active) / sd * np.sqrt(periods_per_year))


def beta(returns: object, benchmark: object) -> float:
    r"""Beta to the benchmark, :math:`\operatorname{Cov}(r,b)/\operatorname{Var}(b)`."""
    r = as_return_vector(returns)
    b = as_return_vector(benchmark, name="benchmark")
    check_lengths_match(r, b)
    if r.shape[0] < 2:
        return float("nan")
    var_b = np.var(b, ddof=1)
    if np.sqrt(var_b) <= _ZERO_TOL:
        return float("nan")
    return float(np.cov(r, b, ddof=1)[0, 1] / var_b)


def rolling_beta(returns: object, benchmark: object, window: int) -> FloatArray:
    """Rolling beta over a trailing window."""
    r = as_return_vector(returns)
    b = as_return_vector(benchmark, name="benchmark")
    check_lengths_match(r, b)
    _validate_window(r.shape[0], window)
    out = np.full(r.shape[0], np.nan)
    for i in range(window - 1, r.shape[0]):
        sl = slice(i - window + 1, i + 1)
        out[i] = beta(r[sl], b[sl])
    return out


# --------------------------------------------------------------------------- #
# Covariance / correlation                                                    #
# --------------------------------------------------------------------------- #
def covariance_matrix(returns_matrix: object, ddof: int = 1) -> FloatArray:
    """Sample covariance matrix of asset returns (``n_obs x n_assets`` input)."""
    r = as_return_matrix(returns_matrix)
    if r.shape[0] < 2:
        raise RiskInputError("need >= 2 observations for a covariance matrix")
    return np.cov(r, rowvar=False, ddof=ddof)


def correlation_matrix(returns_matrix: object) -> FloatArray:
    """Sample correlation matrix of asset returns."""
    r = as_return_matrix(returns_matrix)
    if r.shape[0] < 2:
        raise RiskInputError("need >= 2 observations for a correlation matrix")
    return np.corrcoef(r, rowvar=False)


def rolling_covariance(returns_matrix: object, window: int) -> FloatArray:
    """Rolling covariance, shape ``(n_obs, k, k)`` (NaN before the first window)."""
    r = as_return_matrix(returns_matrix)
    n, k = r.shape
    _validate_window(n, window)
    out = np.full((n, k, k), np.nan)
    for i in range(window - 1, n):
        out[i] = np.cov(r[i - window + 1 : i + 1], rowvar=False, ddof=1)
    return out


def rolling_correlation(returns_matrix: object, window: int) -> FloatArray:
    """Rolling correlation, shape ``(n_obs, k, k)`` (NaN before the first window)."""
    r = as_return_matrix(returns_matrix)
    n, k = r.shape
    _validate_window(n, window)
    out = np.full((n, k, k), np.nan)
    for i in range(window - 1, n):
        out[i] = np.corrcoef(r[i - window + 1 : i + 1], rowvar=False)
    return out


# --------------------------------------------------------------------------- #
# Diversification & concentration                                             #
# --------------------------------------------------------------------------- #
def diversification_ratio(weights: object, covariance: object) -> float:
    r"""Diversification ratio :math:`(w'\sigma)/\sqrt{w'\Sigma w}`."""
    w = as_weights(weights)
    cov = as_covariance(covariance)
    if cov.shape[0] != w.shape[0]:
        raise DimensionMismatchError(w.shape[0], cov.shape[0], name="covariance")
    asset_vols = np.sqrt(np.diag(cov))
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if vol <= _ZERO_TOL:
        return float("nan")
    return float((w @ asset_vols) / vol)


def herfindahl_index(weights: object) -> float:
    r"""Herfindahl-Hirschman index :math:`\sum_i \tilde w_i^2` of the (absolute,
    renormalized) weights.  ``1`` = fully concentrated, ``1/n`` = equal-weighted."""
    w = np.abs(as_weights(weights))
    total = float(np.sum(w))
    if total <= _ZERO_TOL:
        return float("nan")
    shares = w / total
    return float(np.sum(shares**2))


def effective_number_of_assets(weights: object) -> float:
    """``1 / HHI`` -- the equivalent number of equally-weighted holdings."""
    hhi = herfindahl_index(weights)
    if not np.isfinite(hhi) or hhi == 0.0:
        return float("nan")
    return float(1.0 / hhi)


def concentration_ratio(weights: object, top_n: int = 5) -> float:
    """Fraction of gross weight held by the ``top_n`` largest positions."""
    w = np.abs(as_weights(weights))
    total = float(np.sum(w))
    if total <= _ZERO_TOL:
        return float("nan")
    shares = np.sort(w / total)[::-1]
    return float(np.sum(shares[:top_n]))


def concentration_metrics(weights: object, top_n: int = 5) -> dict[str, float]:
    """Bundle of concentration measures."""
    w = as_weights(weights)
    return {
        "herfindahl_index": herfindahl_index(w),
        "effective_number_of_assets": effective_number_of_assets(w),
        "max_weight": float(np.max(np.abs(w))) if w.size else float("nan"),
        f"top_{top_n}_concentration": concentration_ratio(w, top_n),
    }


def _validate_window(n: int, window: int) -> None:
    if window < 2:
        raise RiskInputError("window must be >= 2")
    if window > n:
        raise RiskInputError(f"window ({window}) exceeds series length ({n})")
