r"""Portfolio VaR and its exact decomposition into asset contributions.

Under a Normal (variance-covariance) model with mean set to zero over the risk
horizon, portfolio VaR is proportional to portfolio volatility:

.. math::

    \mathrm{VaR} = z_c\, \sigma_p\sqrt{h}, \qquad
    \sigma_p = \sqrt{w'\Sigma w}, \qquad z_c = \Phi^{-1}(c).

Because VaR is homogeneous of degree one in ``w``, Euler's theorem gives an exact
additive decomposition:

* **Marginal VaR** -- :math:`\partial\mathrm{VaR}/\partial w_i
  = z_c\sqrt{h}\,(\Sigma w)_i/\sigma_p` (the risk of a marginal unit of asset ``i``).
* **Component VaR** -- :math:`w_i\,\mathrm{MVaR}_i`, which **sum exactly to total
  VaR** -- each asset's share of portfolio risk.
* **Incremental VaR** -- the change in portfolio VaR from a finite trade
  :math:`\Delta w`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from factorlab_risk._validation import (
    FloatArray,
    as_covariance,
    as_weights,
    check_confidence,
)
from factorlab_risk.errors import DimensionMismatchError

__all__ = [
    "component_var",
    "incremental_var",
    "marginal_var",
    "percent_contribution_var",
    "portfolio_var",
    "portfolio_volatility",
]


def _z(confidence: float) -> float:
    return float(stats.norm.ppf(confidence))


def _check(weights: FloatArray, cov: FloatArray) -> None:
    if cov.shape[0] != weights.shape[0]:
        raise DimensionMismatchError(weights.shape[0], cov.shape[0], name="covariance")


def portfolio_volatility(weights: object, covariance: object) -> float:
    w = as_weights(weights)
    cov = as_covariance(covariance)
    _check(w, cov)
    return float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def portfolio_var(
    weights: object, covariance: object, confidence: float = 0.95, horizon: int = 1
) -> float:
    """Parametric portfolio VaR (positive loss fraction)."""
    check_confidence(confidence)
    sigma_p = portfolio_volatility(weights, covariance)
    return float(_z(confidence) * sigma_p * np.sqrt(horizon))


def marginal_var(
    weights: object, covariance: object, confidence: float = 0.95, horizon: int = 1
) -> FloatArray:
    r"""Marginal VaR per asset, :math:`\partial\mathrm{VaR}/\partial w_i`."""
    check_confidence(confidence)
    w = as_weights(weights)
    cov = as_covariance(covariance)
    _check(w, cov)
    sigma_p = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
    if sigma_p == 0.0:
        return np.zeros_like(w)
    return np.asarray(_z(confidence) * np.sqrt(horizon) * (cov @ w) / sigma_p, dtype=np.float64)


def component_var(
    weights: object, covariance: object, confidence: float = 0.95, horizon: int = 1
) -> FloatArray:
    r"""Component VaR, :math:`w_i\,\mathrm{MVaR}_i`; sums to total VaR."""
    w = as_weights(weights)
    return np.asarray(w * marginal_var(w, covariance, confidence, horizon), dtype=np.float64)


def percent_contribution_var(
    weights: object, covariance: object, confidence: float = 0.95, horizon: int = 1
) -> FloatArray:
    """Component VaR normalized to sum to 1."""
    comp = component_var(weights, covariance, confidence, horizon)
    total = float(np.sum(comp))
    if total == 0.0:
        return np.zeros_like(comp)
    return comp / total


def incremental_var(
    weights: object,
    covariance: object,
    weight_change: object,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    r"""Exact incremental VaR: :math:`\mathrm{VaR}(w+\Delta) - \mathrm{VaR}(w)`."""
    w = as_weights(weights)
    delta = as_weights(weight_change, name="weight_change")
    if delta.shape != w.shape:
        raise DimensionMismatchError(w.shape[0], delta.shape[0], name="weight_change")
    base = portfolio_var(w, covariance, confidence, horizon)
    shocked = portfolio_var(w + delta, covariance, confidence, horizon)
    return float(shocked - base)
