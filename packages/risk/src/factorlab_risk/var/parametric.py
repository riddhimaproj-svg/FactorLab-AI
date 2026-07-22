r"""Parametric (variance-covariance) VaR and Expected Shortfall.

Assumes returns follow a known distribution and derives risk in closed form.

**Normal.**  With mean :math:`\mu` and volatility :math:`\sigma`, for tail
:math:`\alpha = 1 - c` and :math:`z_\alpha = \Phi^{-1}(\alpha)`:

.. math::

    \mathrm{VaR}_c = -(\mu + \sigma z_\alpha), \qquad
    \mathrm{ES}_c  = -\mu + \sigma\,\frac{\phi(z_\alpha)}{\alpha}.

**Student-t.**  A standardized t with :math:`\nu` degrees of freedom (rescaled to
unit variance by :math:`\sqrt{(\nu-2)/\nu}`) gives fatter tails, a better fit for
financial returns.

Fast and smooth, but a Normal model *understates* tail risk when returns are
fat-tailed -- compare against :mod:`~factorlab_risk.var.historical` in practice.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from factorlab_risk._validation import as_return_vector, check_confidence, tail_alpha
from factorlab_risk.errors import RiskInputError

__all__ = ["parametric_var", "parametric_expected_shortfall"]


def _moments(
    returns: object | None, mean: float | None, std: float | None
) -> tuple[float, float]:
    if returns is not None:
        r = as_return_vector(returns)
        if r.size < 2:
            raise RiskInputError("need >= 2 observations to estimate moments")
        return float(np.mean(r)), float(np.std(r, ddof=1))
    if mean is None or std is None:
        raise RiskInputError("provide either returns or both mean and std")
    if std < 0.0:
        raise RiskInputError("std must be non-negative")
    return float(mean), float(std)


def parametric_var(
    returns: object | None = None,
    confidence: float = 0.95,
    horizon: int = 1,
    *,
    mean: float | None = None,
    std: float | None = None,
    distribution: str = "normal",
    dof: float = 6.0,
) -> float:
    """Parametric VaR as a positive loss fraction (``"normal"`` or ``"t"``)."""
    check_confidence(confidence)
    mu, sigma = _moments(returns, mean, std)
    alpha = tail_alpha(confidence)
    h = float(horizon)

    if distribution == "normal":
        z = float(stats.norm.ppf(alpha))
    elif distribution == "t":
        if dof <= 2.0:
            raise RiskInputError("t-distribution requires dof > 2 for finite variance")
        scale = np.sqrt((dof - 2.0) / dof)
        z = float(stats.t.ppf(alpha, dof) * scale)
    else:
        raise RiskInputError(f"unknown distribution {distribution!r}")

    return float(-(mu * h + sigma * np.sqrt(h) * z))


def parametric_expected_shortfall(
    returns: object | None = None,
    confidence: float = 0.95,
    horizon: int = 1,
    *,
    mean: float | None = None,
    std: float | None = None,
    distribution: str = "normal",
    dof: float = 6.0,
) -> float:
    """Parametric Expected Shortfall as a positive loss fraction."""
    check_confidence(confidence)
    mu, sigma = _moments(returns, mean, std)
    alpha = tail_alpha(confidence)
    h = float(horizon)

    if distribution == "normal":
        z = float(stats.norm.ppf(alpha))
        es_std = float(stats.norm.pdf(z) / alpha)
    elif distribution == "t":
        if dof <= 2.0:
            raise RiskInputError("t-distribution requires dof > 2 for finite variance")
        scale = np.sqrt((dof - 2.0) / dof)
        t_q = float(stats.t.ppf(alpha, dof))
        pdf = float(stats.t.pdf(t_q, dof))
        # ES of a standard t, then rescale to unit variance.
        es_std_raw = (dof + t_q**2) / (dof - 1.0) * pdf / alpha
        es_std = es_std_raw * scale
    else:
        raise RiskInputError(f"unknown distribution {distribution!r}")

    return float(-mu * h + sigma * np.sqrt(h) * es_std)
