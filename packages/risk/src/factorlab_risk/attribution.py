r"""Risk attribution: decomposing portfolio risk across assets, factors, sectors.

For weights :math:`w` and covariance :math:`\Sigma`, portfolio volatility
:math:`\sigma_p = \sqrt{w'\Sigma w}` decomposes exactly (Euler) into per-asset
contributions:

* **Marginal Contribution to Risk** :math:`\mathrm{MCR}_i = (\Sigma w)_i/\sigma_p`.
* **Component Contribution to Risk** :math:`\mathrm{CCR}_i = w_i\,\mathrm{MCR}_i`,
  which sum to :math:`\sigma_p`.
* **Percentage contribution** :math:`\mathrm{CCR}_i/\sigma_p`, summing to 1 --
  the realized *risk budget*.

**Factor attribution** splits variance into a systematic part carried by a factor
model (exposures :math:`B`, factor covariance :math:`F`) and an idiosyncratic
(specific) part:  :math:`w'\Sigma w = w'BFB'w + w'Dw`, with per-factor
contributions from the portfolio factor exposure :math:`b_p = B'w`.

**Sector attribution** simply groups the asset component contributions by sector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_risk._validation import (
    FloatArray,
    as_covariance,
    as_weights,
)
from factorlab_risk.errors import DimensionMismatchError, RiskInputError

__all__ = [
    "FactorRiskAttribution",
    "asset_contribution",
    "component_contribution_to_risk",
    "factor_risk_attribution",
    "marginal_contribution_to_risk",
    "percentage_contribution_to_risk",
    "portfolio_volatility",
    "risk_budget",
    "risk_budget_deviation",
    "sector_risk_attribution",
]


def portfolio_volatility(weights: object, covariance: object) -> float:
    w = as_weights(weights)
    cov = as_covariance(covariance)
    if cov.shape[0] != w.shape[0]:
        raise DimensionMismatchError(w.shape[0], cov.shape[0], name="covariance")
    return float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def marginal_contribution_to_risk(weights: object, covariance: object) -> FloatArray:
    w = as_weights(weights)
    cov = as_covariance(covariance)
    sigma = portfolio_volatility(w, cov)
    if sigma == 0.0:
        return np.zeros_like(w)
    return (cov @ w) / sigma


def component_contribution_to_risk(weights: object, covariance: object) -> FloatArray:
    w = as_weights(weights)
    return w * marginal_contribution_to_risk(w, covariance)


def percentage_contribution_to_risk(weights: object, covariance: object) -> FloatArray:
    ccr = component_contribution_to_risk(weights, covariance)
    total = float(np.sum(ccr))
    if total == 0.0:
        return np.zeros_like(ccr)
    return ccr / total


def asset_contribution(weights: object, covariance: object) -> FloatArray:
    """Per-asset component contribution to risk (alias, sums to volatility)."""
    return component_contribution_to_risk(weights, covariance)


def risk_budget(weights: object, covariance: object) -> FloatArray:
    """The realized risk budget: percentage risk contributions (sum to 1)."""
    return percentage_contribution_to_risk(weights, covariance)


def risk_budget_deviation(
    weights: object, covariance: object, target_budget: object
) -> FloatArray:
    """Difference between the realized risk budget and a target budget."""
    realized = risk_budget(weights, covariance)
    target = np.asarray(target_budget, dtype=np.float64)
    if target.shape != realized.shape:
        raise DimensionMismatchError(realized.shape[0], target.shape[0], name="target_budget")
    return realized - target


class FactorRiskAttribution:
    """Result of a factor risk decomposition (immutable-by-convention)."""

    __slots__ = (
        "factor_exposures",
        "factor_variance_contributions",
        "specific_variance",
        "systematic_variance",
        "total_variance",
    )

    def __init__(
        self,
        total_variance: float,
        systematic_variance: float,
        specific_variance: float,
        factor_variance_contributions: FloatArray,
        factor_exposures: FloatArray,
    ) -> None:
        self.total_variance = total_variance
        self.systematic_variance = systematic_variance
        self.specific_variance = specific_variance
        self.factor_variance_contributions = factor_variance_contributions
        self.factor_exposures = factor_exposures

    @property
    def total_volatility(self) -> float:
        return float(np.sqrt(max(self.total_variance, 0.0)))

    @property
    def systematic_fraction(self) -> float:
        if not self.total_variance:
            return float("nan")
        return self.systematic_variance / self.total_variance

    @property
    def specific_fraction(self) -> float:
        if not self.total_variance:
            return float("nan")
        return self.specific_variance / self.total_variance

    def to_dict(self) -> dict[str, object]:
        return {
            "total_variance": self.total_variance,
            "systematic_variance": self.systematic_variance,
            "specific_variance": self.specific_variance,
            "factor_variance_contributions": self.factor_variance_contributions.tolist(),
            "factor_exposures": self.factor_exposures.tolist(),
        }


def factor_risk_attribution(
    weights: object,
    exposures: object,
    factor_covariance: object,
    specific_variance: object,
) -> FactorRiskAttribution:
    r"""Decompose portfolio variance into factor and specific components.

    Parameters
    ----------
    weights:
        Portfolio weights, length ``n``.
    exposures:
        Factor loadings ``B``, shape ``n x k`` (asset ``i`` on factor ``j``).
    factor_covariance:
        Factor covariance ``F``, shape ``k x k``.
    specific_variance:
        Idiosyncratic variances ``d``, length ``n`` (the diagonal of ``D``).
    """
    w = as_weights(weights)
    B = np.asarray(exposures, dtype=np.float64)
    F = as_covariance(factor_covariance, name="factor_covariance")
    d = np.asarray(specific_variance, dtype=np.float64)
    n = w.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise RiskInputError("exposures must have shape (n_assets, n_factors)")
    k = B.shape[1]
    if F.shape != (k, k):
        raise DimensionMismatchError(k, F.shape[0], name="factor_covariance")
    if d.shape != (n,):
        raise DimensionMismatchError(n, d.shape[0], name="specific_variance")
    if np.any(d < 0.0):
        raise RiskInputError("specific_variance must be non-negative")

    b_p = B.T @ w  # portfolio factor exposure (k,)
    systematic = float(b_p @ F @ b_p)
    specific = float(np.sum(w**2 * d))
    total = systematic + specific
    # Per-factor variance contribution (Euler on factor exposures): b_j (F b)_j.
    factor_contrib: NDArray[np.float64] = b_p * (F @ b_p)
    return FactorRiskAttribution(total, systematic, specific, factor_contrib, b_p)


def sector_risk_attribution(
    weights: object, covariance: object, sectors: list[str]
) -> dict[str, float]:
    """Aggregate asset component risk contributions by sector."""
    w = as_weights(weights)
    if len(sectors) != w.shape[0]:
        raise DimensionMismatchError(w.shape[0], len(sectors), name="sectors")
    ccr = component_contribution_to_risk(w, covariance)
    result: dict[str, float] = {}
    for sector, contribution in zip(sectors, ccr, strict=True):
        result[sector] = result.get(sector, 0.0) + float(contribution)
    return result
