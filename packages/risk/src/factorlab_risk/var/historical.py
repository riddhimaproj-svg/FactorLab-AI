r"""Historical (non-parametric) Value-at-Risk and Expected Shortfall.

Historical simulation makes **no distributional assumption**: it reads the risk
directly off the empirical return distribution.  For confidence ``c`` and tail
``alpha = 1 - c``:

.. math::

    \mathrm{VaR}_c = -Q_\alpha(r), \qquad
    \mathrm{ES}_c  = -\mathbb{E}[\,r \mid r \le Q_\alpha(r)\,],

where :math:`Q_\alpha` is the empirical ``alpha``-quantile of returns.  Both are
returned as **positive loss magnitudes**.  Horizon scaling uses the
square-root-of-time rule (:math:`\sqrt{h}`), the standard short-horizon
approximation under i.i.d. returns.

Advantages: captures the actual fat tails and skew of the sample.  Limitation:
it can only produce losses that have already occurred in the window.
"""

from __future__ import annotations

import numpy as np

from factorlab_risk._validation import FloatArray, as_return_vector, tail_alpha

__all__ = [
    "historical_expected_shortfall",
    "historical_var",
    "tail_loss",
    "worst_loss",
]


def historical_var(returns: object, confidence: float = 0.95, horizon: int = 1) -> float:
    r"""Historical VaR as a positive loss fraction."""
    r = as_return_vector(returns)
    alpha = tail_alpha(confidence)
    quantile = float(np.quantile(r, alpha, method="linear"))
    return float(-quantile * np.sqrt(horizon))


def historical_expected_shortfall(
    returns: object, confidence: float = 0.95, horizon: int = 1
) -> float:
    r"""Historical Expected Shortfall / CVaR: mean loss beyond the VaR threshold."""
    r = as_return_vector(returns)
    alpha = tail_alpha(confidence)
    quantile = float(np.quantile(r, alpha, method="linear"))
    tail = r[r <= quantile]
    if tail.size == 0:
        tail = np.array([quantile])
    return float(-float(np.mean(tail)) * np.sqrt(horizon))


def tail_loss(returns: object, confidence: float = 0.95) -> float:
    """Expected loss conditional on being in the tail (alias of ES, horizon 1)."""
    return historical_expected_shortfall(returns, confidence, horizon=1)


def worst_loss(returns: object) -> float:
    """The single worst realized loss (positive magnitude)."""
    r = as_return_vector(returns)
    return float(-np.min(r))


def _empirical_quantile(returns: FloatArray, alpha: float) -> float:
    return float(np.quantile(returns, alpha, method="linear"))
