r"""Portfolio risk decomposition.

For weights :math:`w` and covariance :math:`\Sigma`, portfolio variance is
:math:`\sigma_p^2 = w'\Sigma w` and volatility :math:`\sigma_p = \sqrt{w'\Sigma w}`.
The **marginal risk contribution** of asset :math:`i` is
:math:`\mathrm{MRC}_i = (\Sigma w)_i / \sigma_p`, and its **risk contribution** is
:math:`\mathrm{RC}_i = w_i \mathrm{MRC}_i`.  By Euler's theorem the risk
contributions sum to total volatility, giving an exact additive decomposition of
risk across assets -- the foundation of risk-parity investing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "diversification_ratio",
    "marginal_risk_contributions",
    "percent_risk_contributions",
    "portfolio_variance",
    "portfolio_volatility",
    "risk_contributions",
    "variance_decomposition",
]

FloatArray = NDArray[np.float64]


def portfolio_variance(weights: FloatArray, covariance: FloatArray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float(w @ covariance @ w)


def portfolio_volatility(weights: FloatArray, covariance: FloatArray) -> float:
    return float(np.sqrt(max(portfolio_variance(weights, covariance), 0.0)))


def marginal_risk_contributions(weights: FloatArray, covariance: FloatArray) -> FloatArray:
    r"""``MRC = Sigma w / sigma_p`` (the derivative of volatility w.r.t. weights)."""
    w = np.asarray(weights, dtype=np.float64)
    vol = portfolio_volatility(w, covariance)
    if vol == 0.0:
        return np.zeros_like(w)
    return (covariance @ w) / vol


def risk_contributions(weights: FloatArray, covariance: FloatArray) -> FloatArray:
    r"""``RC_i = w_i * MRC_i``; sums to portfolio volatility (Euler decomposition)."""
    w = np.asarray(weights, dtype=np.float64)
    return w * marginal_risk_contributions(w, covariance)


def percent_risk_contributions(weights: FloatArray, covariance: FloatArray) -> FloatArray:
    """Risk contributions normalized to sum to 1."""
    rc = risk_contributions(weights, covariance)
    total = np.sum(rc)
    if total == 0.0:
        return np.zeros_like(rc)
    return rc / total


def variance_decomposition(weights: FloatArray, covariance: FloatArray) -> FloatArray:
    r"""Each asset's contribution to *variance*, ``w_i (Sigma w)_i``; sums to
    ``w' Sigma w``."""
    w = np.asarray(weights, dtype=np.float64)
    return w * (covariance @ w)


def diversification_ratio(weights: FloatArray, covariance: FloatArray) -> float:
    r"""Diversification ratio ``(w . sigma) / sigma_p``.

    The ratio of the weighted-average asset volatility to the portfolio
    volatility.  It equals 1 when assets are perfectly correlated and grows as
    diversification reduces portfolio risk below the weighted average.
    """
    w = np.asarray(weights, dtype=np.float64)
    asset_vols = np.sqrt(np.diag(covariance))
    vol = portfolio_volatility(w, covariance)
    if vol == 0.0:
        return float("nan")
    return float((w @ asset_vols) / vol)
