"""PerformanceReport: a serializable bundle of computed metrics.

A report snapshots every headline performance and risk metric for a return
series (optionally versus a benchmark).  It is immutable and JSON-serializable,
so it can be stored, compared across periods, or handed to a reporting layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

__all__ = ["PerformanceReport"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_portfolio.returns import ReturnSeries


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Computed performance and risk metrics for a return series."""

    name: str
    periods_per_year: float
    n_observations: int
    risk_free: float
    # Absolute
    total_return: float
    cagr: float
    annualized_volatility: float
    downside_deviation: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    # Relative (NaN / -1 when no benchmark supplied)
    has_benchmark: bool
    beta: float
    active_return: float
    tracking_error: float
    information_ratio: float
    treynor_ratio: float

    @classmethod
    def from_series(
        cls,
        series: ReturnSeries,
        benchmark: ReturnSeries | None = None,
        risk_free: float = 0.0,
    ) -> PerformanceReport:
        """Compute every metric for ``series`` (and vs ``benchmark`` if given)."""
        has_benchmark = benchmark is not None
        nan = float("nan")
        return cls(
            name=series.name,
            periods_per_year=series.periods_per_year,
            n_observations=series.n_observations,
            risk_free=risk_free,
            total_return=series.total_return(),
            cagr=series.cagr(),
            annualized_volatility=series.volatility(),
            downside_deviation=series.downside_deviation(),
            sharpe_ratio=series.sharpe(risk_free),
            sortino_ratio=series.sortino(risk_free),
            calmar_ratio=series.calmar(),
            omega_ratio=series.omega(),
            max_drawdown=series.max_drawdown(),
            max_drawdown_duration=series.max_drawdown_duration(),
            has_benchmark=has_benchmark,
            beta=series.beta(benchmark) if benchmark is not None else nan,
            active_return=series.active_return(benchmark) if benchmark is not None else nan,
            tracking_error=series.tracking_error(benchmark) if benchmark is not None else nan,
            information_ratio=(
                series.information_ratio(benchmark) if benchmark is not None else nan
            ),
            treynor_ratio=(
                series.treynor(benchmark, risk_free) if benchmark is not None else nan
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "periods_per_year": self.periods_per_year,
            "n_observations": self.n_observations,
            "risk_free": self.risk_free,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_volatility": self.annualized_volatility,
            "downside_deviation": self.downside_deviation,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "has_benchmark": self.has_benchmark,
            "beta": self.beta,
            "active_return": self.active_return,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "treynor_ratio": self.treynor_ratio,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerformanceReport:
        return cls(
            name=str(data["name"]),
            periods_per_year=float(data["periods_per_year"]),
            n_observations=int(data["n_observations"]),
            risk_free=float(data["risk_free"]),
            total_return=float(data["total_return"]),
            cagr=float(data["cagr"]),
            annualized_volatility=float(data["annualized_volatility"]),
            downside_deviation=float(data["downside_deviation"]),
            sharpe_ratio=float(data["sharpe_ratio"]),
            sortino_ratio=float(data["sortino_ratio"]),
            calmar_ratio=float(data["calmar_ratio"]),
            omega_ratio=float(data["omega_ratio"]),
            max_drawdown=float(data["max_drawdown"]),
            max_drawdown_duration=int(data["max_drawdown_duration"]),
            has_benchmark=bool(data["has_benchmark"]),
            beta=float(data["beta"]),
            active_return=float(data["active_return"]),
            tracking_error=float(data["tracking_error"]),
            information_ratio=float(data["information_ratio"]),
            treynor_ratio=float(data["treynor_ratio"]),
        )

    def summary(self) -> str:
        """A formatted, human-readable performance report."""
        lines = [
            "=" * 60,
            f"Performance Report — {self.name}",
            "=" * 60,
            f"Observations: {self.n_observations:>10d}   "
            f"Periods/yr: {self.periods_per_year:g}",
            "-" * 60,
            "Return",
            f"  Total return:          {self.total_return:>12.4%}",
            f"  CAGR:                  {self.cagr:>12.4%}",
            "-" * 60,
            "Risk",
            f"  Annualized volatility: {self.annualized_volatility:>12.4%}",
            f"  Downside deviation:    {self.downside_deviation:>12.4%}",
            f"  Max drawdown:          {self.max_drawdown:>12.4%}",
            f"  Max DD duration:       {self.max_drawdown_duration:>10d} periods",
            "-" * 60,
            "Risk-adjusted",
            f"  Sharpe ratio:          {self.sharpe_ratio:>12.4f}",
            f"  Sortino ratio:         {self.sortino_ratio:>12.4f}",
            f"  Calmar ratio:          {self.calmar_ratio:>12.4f}",
            f"  Omega ratio:           {self.omega_ratio:>12.4f}",
        ]
        if self.has_benchmark:
            lines.extend(
                [
                    "-" * 60,
                    "Benchmark-relative",
                    f"  Beta:                  {self.beta:>12.4f}",
                    f"  Active return:         {self.active_return:>12.4%}",
                    f"  Tracking error:        {self.tracking_error:>12.4%}",
                    f"  Information ratio:     {self.information_ratio:>12.4f}",
                    f"  Treynor ratio:         {self.treynor_ratio:>12.4f}",
                ]
            )
        lines.append("=" * 60)
        return "\n".join(lines)
