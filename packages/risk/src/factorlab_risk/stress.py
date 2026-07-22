r"""Stress testing: apply severe, named shocks and measure portfolio impact.

Stress tests ask "what would this portfolio lose if <bad thing> happened again?"
They complement VaR -- which describes typical tail losses -- with specific,
often historical or regulatory, adverse scenarios.  This module provides:

* **Scenario builders** -- market crash, interest-rate shock, factor shock,
  sector shock, and a historical scenario reconstructed from a past return
  window.  Each returns a :class:`~factorlab_risk.scenario.Scenario` for the
  :class:`~factorlab_risk.scenario.ScenarioEngine` to revalue.
* **Volatility shock** -- scale the covariance matrix and re-measure VaR (a vol
  shock changes *risk*, not directly P&L).
* **run_stress_test** -- revalue a portfolio across a battery of scenarios and
  bundle the results into a :class:`StressTestReport`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from factorlab_risk._validation import (
    as_covariance,
    as_return_matrix,
    as_weights,
    check_confidence,
)
from factorlab_risk.errors import DimensionMismatchError, RiskInputError
from factorlab_risk.scenario import Scenario, ScenarioEngine
from factorlab_risk.var.decomposition import portfolio_var

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_risk.reports import StressTestReport

__all__ = [
    "VolatilityShockResult",
    "factor_shock_scenario",
    "historical_scenario",
    "interest_rate_shock_scenario",
    "market_crash_scenario",
    "run_stress_test",
    "sector_shock_scenario",
    "volatility_shock",
]


def market_crash_scenario(
    assets: Sequence[str], magnitude: float = -0.30, betas: Sequence[float] | None = None
) -> Scenario:
    """A broad market decline of ``magnitude``; per-asset shock = ``beta * magnitude``.

    Without ``betas`` every asset takes the full ``magnitude`` (beta = 1).
    """
    if betas is None:
        shocks = dict.fromkeys(assets, magnitude)
    else:
        if len(betas) != len(assets):
            raise DimensionMismatchError(len(assets), len(betas), name="betas")
        shocks = {a: float(b) * magnitude for a, b in zip(assets, betas, strict=True)}
    return Scenario(
        name=f"market_crash_{magnitude:+.0%}",
        asset_shocks=shocks,
        description="Broad equity market decline scaled by beta.",
    )


def interest_rate_shock_scenario(
    assets: Sequence[str], rate_shock: float, rate_betas: Sequence[float]
) -> Scenario:
    """A parallel rate move; per-asset shock = ``rate_beta * rate_shock``.

    ``rate_shock`` is the change in rates (e.g. ``+0.01`` = +100bp); ``rate_betas``
    are per-asset return sensitivities to that move (negative for bond-like assets).
    """
    if len(rate_betas) != len(assets):
        raise DimensionMismatchError(len(assets), len(rate_betas), name="rate_betas")
    shocks = {a: float(rb) * rate_shock for a, rb in zip(assets, rate_betas, strict=True)}
    return Scenario(
        name=f"rate_shock_{rate_shock:+.2%}",
        asset_shocks=shocks,
        description="Parallel interest-rate move via per-asset rate sensitivities.",
    )


def factor_shock_scenario(factor: str, magnitude: float) -> Scenario:
    """A shock to a single factor (applied through the engine's exposures)."""
    return Scenario(
        name=f"factor_shock_{factor}_{magnitude:+.2%}",
        factor_shocks={factor: magnitude},
        description=f"Shock of {magnitude:+.2%} to factor {factor!r}.",
    )


def sector_shock_scenario(
    assets: Sequence[str], sectors: Sequence[str], sector: str, magnitude: float
) -> Scenario:
    """Shock every asset belonging to ``sector`` by ``magnitude``."""
    if len(sectors) != len(assets):
        raise DimensionMismatchError(len(assets), len(sectors), name="sectors")
    shocks = {
        a: magnitude for a, s in zip(assets, sectors, strict=True) if s == sector
    }
    if not shocks:
        raise RiskInputError(f"no assets in sector {sector!r}")
    return Scenario(
        name=f"sector_shock_{sector}_{magnitude:+.2%}",
        asset_shocks=shocks,
        description=f"Shock of {magnitude:+.2%} applied to sector {sector!r}.",
    )


def historical_scenario(
    name: str, returns_window: object, assets: Sequence[str]
) -> Scenario:
    """Reconstruct a scenario from a historical return window.

    The per-asset shock is the compounded return over the window
    :math:`\\prod_t (1 + r_{i,t}) - 1`, i.e. "what if that episode repeated".
    """
    r = as_return_matrix(returns_window)
    if r.shape[1] != len(assets):
        raise DimensionMismatchError(len(assets), r.shape[1], name="assets")
    cumulative = np.prod(1.0 + r, axis=0) - 1.0
    shocks = {a: float(c) for a, c in zip(assets, cumulative, strict=True)}
    return Scenario(name=name, asset_shocks=shocks, description=f"Historical episode: {name}.")


class VolatilityShockResult:
    """Base vs shocked VaR when the covariance is scaled by a volatility multiplier."""

    __slots__ = ("base_var", "shocked_var", "vol_multiplier")

    def __init__(self, vol_multiplier: float, base_var: float, shocked_var: float) -> None:
        self.vol_multiplier = vol_multiplier
        self.base_var = base_var
        self.shocked_var = shocked_var

    @property
    def var_increase(self) -> float:
        return self.shocked_var - self.base_var

    def to_dict(self) -> dict[str, float]:
        return {
            "vol_multiplier": self.vol_multiplier,
            "base_var": self.base_var,
            "shocked_var": self.shocked_var,
            "var_increase": self.var_increase,
        }


def volatility_shock(
    weights: object,
    covariance: object,
    vol_multiplier: float,
    confidence: float = 0.95,
    horizon: int = 1,
) -> VolatilityShockResult:
    """Recompute portfolio VaR when volatilities are scaled by ``vol_multiplier``.

    Scaling volatilities by ``m`` scales the covariance by ``m**2`` (correlations
    unchanged), so VaR scales by ``m``.
    """
    check_confidence(confidence)
    if vol_multiplier <= 0.0:
        raise RiskInputError("vol_multiplier must be positive")
    w = as_weights(weights)
    cov = as_covariance(covariance)
    base = portfolio_var(w, cov, confidence, horizon)
    shocked = portfolio_var(w, cov * vol_multiplier**2, confidence, horizon)
    return VolatilityShockResult(vol_multiplier, base, shocked)


def run_stress_test(
    engine: ScenarioEngine,
    weights: object,
    scenarios: Sequence[Scenario],
    portfolio_value: float = 1.0,
) -> StressTestReport:
    """Revalue ``weights`` across ``scenarios`` and return a StressTestReport."""
    from factorlab_risk.reports import StressTestReport

    outcomes = [engine.revalue(weights, s, portfolio_value) for s in scenarios]
    return StressTestReport(tuple(outcomes), portfolio_value)
