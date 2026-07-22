"""Immutable, serializable risk data models and report aggregates.

These are the public "currency" of the risk engine: a computation produces one of
these frozen dataclasses, which serialize cleanly (``to_dict`` / ``from_dict``)
and render a human-readable ``summary``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from factorlab_risk import attribution, portfolio_risk
from factorlab_risk.var import (
    historical_expected_shortfall,
    historical_var,
    parametric_expected_shortfall,
    parametric_var,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_risk.scenario import ScenarioOutcome

__all__ = [
    "RiskContribution",
    "RiskDecomposition",
    "RiskReport",
    "RiskSnapshot",
    "ScenarioReport",
    "StressTestReport",
    "VaRReport",
]


@dataclass(frozen=True, slots=True)
class RiskContribution:
    """One asset's contribution to portfolio risk."""

    asset: str
    weight: float
    marginal_contribution: float
    component_contribution: float
    percentage_contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "weight": self.weight,
            "marginal_contribution": self.marginal_contribution,
            "component_contribution": self.component_contribution,
            "percentage_contribution": self.percentage_contribution,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RiskContribution:
        return cls(
            asset=str(data["asset"]),
            weight=float(data["weight"]),
            marginal_contribution=float(data["marginal_contribution"]),
            component_contribution=float(data["component_contribution"]),
            percentage_contribution=float(data["percentage_contribution"]),
        )


@dataclass(frozen=True, slots=True)
class RiskDecomposition:
    """Portfolio volatility split into per-asset contributions (sum to total)."""

    total_volatility: float
    contributions: tuple[RiskContribution, ...]

    @classmethod
    def from_weights_covariance(
        cls, assets: Sequence[str], weights: object, covariance: object
    ) -> RiskDecomposition:
        w = np.asarray(weights, dtype=np.float64)
        total = portfolio_risk.portfolio_volatility(w, covariance)
        mcr = attribution.marginal_contribution_to_risk(w, covariance)
        ccr = attribution.component_contribution_to_risk(w, covariance)
        pct = attribution.percentage_contribution_to_risk(w, covariance)
        contributions = tuple(
            RiskContribution(a, float(w[i]), float(mcr[i]), float(ccr[i]), float(pct[i]))
            for i, a in enumerate(assets)
        )
        return cls(total, contributions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_volatility": self.total_volatility,
            "contributions": [c.to_dict() for c in self.contributions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RiskDecomposition:
        return cls(
            total_volatility=float(data["total_volatility"]),
            contributions=tuple(
                RiskContribution.from_dict(c) for c in data["contributions"]
            ),
        )

    def summary(self) -> str:
        lines = [f"Risk decomposition (total volatility = {self.total_volatility:.4%})", "-" * 60]
        lines.append(f"{'asset':<14}{'weight':>10}{'component':>14}{'% of risk':>12}")
        for c in self.contributions:
            lines.append(
                f"{c.asset:<14}{c.weight:>10.4f}{c.component_contribution:>14.6f}"
                f"{c.percentage_contribution:>12.2%}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class VaRReport:
    """VaR and Expected Shortfall at a confidence level and horizon."""

    confidence: float
    horizon: int
    method: str
    var: float
    expected_shortfall: float

    @classmethod
    def from_returns(
        cls, returns: object, confidence: float = 0.95, horizon: int = 1, method: str = "historical"
    ) -> VaRReport:
        if method == "historical":
            v = historical_var(returns, confidence, horizon)
            es = historical_expected_shortfall(returns, confidence, horizon)
        elif method == "parametric":
            v = parametric_var(returns, confidence, horizon)
            es = parametric_expected_shortfall(returns, confidence, horizon)
        else:
            from factorlab_risk.errors import RiskInputError

            raise RiskInputError(f"unknown method {method!r}")
        return cls(confidence, horizon, method, v, es)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "horizon": self.horizon,
            "method": self.method,
            "var": self.var,
            "expected_shortfall": self.expected_shortfall,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VaRReport:
        return cls(
            confidence=float(data["confidence"]),
            horizon=int(data["horizon"]),
            method=str(data["method"]),
            var=float(data["var"]),
            expected_shortfall=float(data["expected_shortfall"]),
        )

    def summary(self) -> str:
        return (
            f"VaR ({self.method}, {self.confidence:.0%}, {self.horizon}p): "
            f"{self.var:.4%}   ES: {self.expected_shortfall:.4%}"
        )


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    """A point-in-time bundle of headline portfolio-risk metrics."""

    as_of: str | None
    volatility: float
    var_95: float
    expected_shortfall_95: float
    diversification_ratio: float
    herfindahl_index: float
    effective_number_of_assets: float
    max_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "volatility": self.volatility,
            "var_95": self.var_95,
            "expected_shortfall_95": self.expected_shortfall_95,
            "diversification_ratio": self.diversification_ratio,
            "herfindahl_index": self.herfindahl_index,
            "effective_number_of_assets": self.effective_number_of_assets,
            "max_weight": self.max_weight,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RiskSnapshot:
        return cls(
            as_of=data.get("as_of"),
            volatility=float(data["volatility"]),
            var_95=float(data["var_95"]),
            expected_shortfall_95=float(data["expected_shortfall_95"]),
            diversification_ratio=float(data["diversification_ratio"]),
            herfindahl_index=float(data["herfindahl_index"]),
            effective_number_of_assets=float(data["effective_number_of_assets"]),
            max_weight=float(data["max_weight"]),
        )


@dataclass(frozen=True, slots=True)
class StressTestReport:
    """Portfolio P&L across a battery of stress scenarios."""

    outcomes: tuple[ScenarioOutcome, ...]
    portfolio_value: float = 1.0

    @property
    def worst_case(self) -> ScenarioOutcome | None:
        return min(self.outcomes, key=lambda o: o.pnl) if self.outcomes else None

    @property
    def best_case(self) -> ScenarioOutcome | None:
        return max(self.outcomes, key=lambda o: o.pnl) if self.outcomes else None

    def as_table(self) -> list[tuple[str, float, float]]:
        """``(scenario_name, portfolio_return, pnl)`` sorted worst first."""
        rows = [(o.scenario_name, o.portfolio_return, o.pnl) for o in self.outcomes]
        return sorted(rows, key=lambda r: r[2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_value": self.portfolio_value,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def summary(self) -> str:
        lines = ["=" * 60, "Stress Test Report", "=" * 60,
                 f"{'scenario':<32}{'return':>12}{'pnl':>14}", "-" * 60]
        for name, ret, pnl in self.as_table():
            lines.append(f"{name:<32}{ret:>12.2%}{pnl:>14.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    """A comparison of scenario outcomes for a portfolio."""

    outcomes: tuple[ScenarioOutcome, ...]
    portfolio_value: float = 1.0

    def ranked(self) -> tuple[ScenarioOutcome, ...]:
        """Outcomes sorted from worst to best P&L."""
        return tuple(sorted(self.outcomes, key=lambda o: o.pnl))

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_value": self.portfolio_value,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def summary(self) -> str:
        lines = ["Scenario comparison", "-" * 50]
        for o in self.ranked():
            lines.append(f"  {o.scenario_name:<30}{o.portfolio_return:>10.2%}{o.pnl:>14.2f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RiskReport:
    """Top-level portfolio risk report: volatility, VaR/ES, and decomposition."""

    assets: tuple[str, ...]
    volatility: float
    diversification_ratio: float
    concentration: dict[str, float]
    parametric_var: VaRReport | None
    decomposition: RiskDecomposition
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_portfolio(
        cls,
        assets: Sequence[str],
        weights: object,
        covariance: object,
        *,
        confidence: float = 0.95,
        horizon: int = 1,
        periods_per_year: float = 252.0,
    ) -> RiskReport:
        w = np.asarray(weights, dtype=np.float64)
        vol = portfolio_risk.portfolio_volatility(w, covariance)
        div = portfolio_risk.diversification_ratio(w, covariance)
        conc = portfolio_risk.concentration_metrics(w)
        # Parametric portfolio VaR/ES from the (Normal) portfolio volatility.
        from factorlab_risk.var.decomposition import portfolio_var as pv

        var_value = pv(w, covariance, confidence, horizon)
        es_value = parametric_expected_shortfall(
            None, confidence, horizon, mean=0.0, std=vol
        )
        var_report = VaRReport(confidence, horizon, "parametric", var_value, es_value)
        decomp = RiskDecomposition.from_weights_covariance(assets, w, covariance)
        return cls(
            assets=tuple(assets),
            volatility=vol,
            diversification_ratio=div,
            concentration=conc,
            parametric_var=var_report,
            decomposition=decomp,
            metadata={"periods_per_year": periods_per_year},
        )

    def snapshot(self, as_of: str | None = None) -> RiskSnapshot:
        var = self.parametric_var
        return RiskSnapshot(
            as_of=as_of,
            volatility=self.volatility,
            var_95=var.var if var is not None else float("nan"),
            expected_shortfall_95=var.expected_shortfall if var is not None else float("nan"),
            diversification_ratio=self.diversification_ratio,
            herfindahl_index=self.concentration["herfindahl_index"],
            effective_number_of_assets=self.concentration["effective_number_of_assets"],
            max_weight=self.concentration["max_weight"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": list(self.assets),
            "volatility": self.volatility,
            "diversification_ratio": self.diversification_ratio,
            "concentration": dict(self.concentration),
            "parametric_var": self.parametric_var.to_dict() if self.parametric_var else None,
            "decomposition": self.decomposition.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RiskReport:
        var = data.get("parametric_var")
        return cls(
            assets=tuple(data["assets"]),
            volatility=float(data["volatility"]),
            diversification_ratio=float(data["diversification_ratio"]),
            concentration=dict(data["concentration"]),
            parametric_var=None if var is None else VaRReport.from_dict(var),
            decomposition=RiskDecomposition.from_dict(data["decomposition"]),
            metadata=dict(data.get("metadata", {})),
        )

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Portfolio Risk Report",
            "=" * 60,
            f"Volatility (per period):   {self.volatility:>12.4%}",
            f"Diversification ratio:     {self.diversification_ratio:>12.4f}",
            f"Herfindahl index:          {self.concentration['herfindahl_index']:>12.4f}",
            f"Effective # assets:        {self.concentration['effective_number_of_assets']:>12.2f}",
        ]
        if self.parametric_var is not None:
            lines.append("-" * 60)
            lines.append(self.parametric_var.summary())
        lines.append("-" * 60)
        lines.append(self.decomposition.summary())
        lines.append("=" * 60)
        return "\n".join(lines)
