r"""Scenario analysis: revalue a portfolio under hypothetical shocks.

A :class:`Scenario` describes a hypothetical move as per-asset return shocks
and/or per-factor shocks.  The :class:`ScenarioEngine` revalues a portfolio under
a scenario:

.. math::

    r_i^{\text{shock}} = a_i + \sum_j B_{ij} f_j, \qquad
    \Delta_{\text{portfolio}} = \sum_i w_i\, r_i^{\text{shock}},

where :math:`a_i` are direct asset shocks, :math:`f_j` factor shocks, and
:math:`B` the factor exposures.  The engine supports comparing scenarios and
one-dimensional **sensitivity** analysis (P&L as a single shock varies).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factorlab_risk._validation import FloatArray, as_weights
from factorlab_risk.errors import DimensionMismatchError, RiskInputError

__all__ = ["Scenario", "ScenarioEngine", "ScenarioOutcome", "SensitivityResult"]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named shock: per-asset return shocks and/or per-factor shocks."""

    name: str
    asset_shocks: dict[str, float] = field(default_factory=dict)
    factor_shocks: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "asset_shocks": dict(self.asset_shocks),
            "factor_shocks": dict(self.factor_shocks),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scenario:
        return cls(
            name=str(data["name"]),
            asset_shocks=dict(data.get("asset_shocks", {})),
            factor_shocks=dict(data.get("factor_shocks", {})),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    """Result of revaluing a portfolio under one scenario."""

    scenario_name: str
    portfolio_return: float
    pnl: float
    asset_returns: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "portfolio_return": self.portfolio_return,
            "pnl": self.pnl,
            "asset_returns": dict(self.asset_returns),
        }


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """P&L as a single shock is swept over a grid of values."""

    target: str
    kind: str  # "asset" or "factor"
    shock_values: FloatArray
    portfolio_returns: FloatArray
    pnls: FloatArray

    @property
    def delta(self) -> float:
        """First-order sensitivity (slope) of P&L to the shock."""
        if self.shock_values.shape[0] < 2:
            return float("nan")
        slope = np.polyfit(self.shock_values, self.pnls, 1)[0]
        return float(slope)


class ScenarioEngine:
    """Revalues portfolios under scenarios for a fixed asset universe."""

    def __init__(
        self,
        assets: Sequence[str],
        *,
        exposures: FloatArray | None = None,
        factor_names: Sequence[str] | None = None,
    ) -> None:
        self.assets = tuple(assets)
        self._index = {a: i for i, a in enumerate(self.assets)}
        self.exposures = None if exposures is None else np.asarray(exposures, dtype=np.float64)
        self.factor_names = tuple(factor_names) if factor_names is not None else ()
        if self.exposures is not None:
            n = len(self.assets)
            k = len(self.factor_names)
            if self.exposures.shape != (n, k):
                raise DimensionMismatchError(
                    n * max(k, 1), self.exposures.size, name="exposures"
                )

    def asset_shock_vector(self, scenario: Scenario) -> FloatArray:
        """The total per-asset return shock implied by a scenario."""
        shocks = np.zeros(len(self.assets), dtype=np.float64)
        for asset, value in scenario.asset_shocks.items():
            if asset not in self._index:
                raise RiskInputError(f"scenario references unknown asset {asset!r}")
            shocks[self._index[asset]] += value
        if scenario.factor_shocks:
            if self.exposures is None:
                raise RiskInputError("factor shocks require exposures/factor_names")
            factor_vec = np.zeros(len(self.factor_names), dtype=np.float64)
            for factor, value in scenario.factor_shocks.items():
                if factor not in self.factor_names:
                    raise RiskInputError(f"scenario references unknown factor {factor!r}")
                factor_vec[self.factor_names.index(factor)] += value
            shocks = shocks + self.exposures @ factor_vec
        return shocks

    def revalue(
        self, weights: object, scenario: Scenario, portfolio_value: float = 1.0
    ) -> ScenarioOutcome:
        """Revalue a portfolio under ``scenario``."""
        w = as_weights(weights)
        if w.shape[0] != len(self.assets):
            raise DimensionMismatchError(len(self.assets), w.shape[0], name="weights")
        shocks = self.asset_shock_vector(scenario)
        portfolio_return = float(w @ shocks)
        return ScenarioOutcome(
            scenario_name=scenario.name,
            portfolio_return=portfolio_return,
            pnl=portfolio_value * portfolio_return,
            asset_returns=dict(zip(self.assets, shocks.tolist(), strict=True)),
        )

    def compare(
        self, weights: object, scenarios: Sequence[Scenario], portfolio_value: float = 1.0
    ) -> list[ScenarioOutcome]:
        """Revalue under several scenarios, sorted worst P&L first."""
        outcomes = [self.revalue(weights, s, portfolio_value) for s in scenarios]
        return sorted(outcomes, key=lambda o: o.pnl)

    def sensitivity(
        self,
        weights: object,
        target: str,
        shock_values: object,
        *,
        kind: str = "asset",
        portfolio_value: float = 1.0,
    ) -> SensitivityResult:
        """Sweep a single asset/factor shock and record portfolio P&L."""
        values = np.asarray(shock_values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise RiskInputError("shock_values must be a non-empty 1-D array")
        returns = np.empty(values.shape[0], dtype=np.float64)
        for i, v in enumerate(values):
            if kind == "asset":
                scenario = Scenario("sens", asset_shocks={target: float(v)})
            elif kind == "factor":
                scenario = Scenario("sens", factor_shocks={target: float(v)})
            else:
                raise RiskInputError("kind must be 'asset' or 'factor'")
            returns[i] = self.revalue(weights, scenario, portfolio_value).portfolio_return
        pnls = returns * portfolio_value
        return SensitivityResult(target, kind, values, returns, pnls)
