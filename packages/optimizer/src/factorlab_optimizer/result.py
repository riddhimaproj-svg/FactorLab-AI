"""OptimizationResult: the immutable output of an optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from factorlab_optimizer import risk
from factorlab_optimizer.weights import PortfolioWeights

__all__ = ["OptimizationResult"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Solved weights plus the portfolio's return/risk profile."""

    weights: PortfolioWeights
    optimizer: str
    success: bool
    message: str
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    objective_value: float
    n_iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        weights: PortfolioWeights,
        optimizer: str,
        success: bool,
        message: str,
        expected_returns: FloatArray,
        covariance: FloatArray,
        risk_free_rate: float,
        objective_value: float,
        n_iterations: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        """Construct a result, deriving the return/risk/Sharpe from the weights."""
        w = weights.values
        exp_ret = float(w @ expected_returns)
        vol = risk.portfolio_volatility(w, covariance)
        sharpe = float("nan") if vol == 0.0 else (exp_ret - risk_free_rate) / vol
        return cls(
            weights=weights,
            optimizer=optimizer,
            success=success,
            message=message,
            expected_return=exp_ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            objective_value=objective_value,
            n_iterations=n_iterations,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "optimizer": self.optimizer,
            "success": self.success,
            "message": self.message,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "objective_value": self.objective_value,
            "n_iterations": self.n_iterations,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationResult:
        return cls(
            weights=PortfolioWeights.from_dict(data["weights"]),
            optimizer=str(data["optimizer"]),
            success=bool(data["success"]),
            message=str(data["message"]),
            expected_return=float(data["expected_return"]),
            expected_volatility=float(data["expected_volatility"]),
            sharpe_ratio=float(data["sharpe_ratio"]),
            objective_value=float(data["objective_value"]),
            n_iterations=int(data.get("n_iterations", 0)),
            metadata=dict(data.get("metadata", {})),
        )

    def summary(self) -> str:
        lines = [
            "=" * 58,
            f"Optimization Result — {self.optimizer}",
            "=" * 58,
            f"Success: {self.success}   ({self.message})",
            f"Expected return:     {self.expected_return:>12.6f}",
            f"Expected volatility: {self.expected_volatility:>12.6f}",
            f"Sharpe ratio:        {self.sharpe_ratio:>12.4f}",
            "-" * 58,
            "Weights",
        ]
        for asset, w in self.weights.as_dict().items():
            if abs(w) > 1e-6:
                lines.append(f"  {asset:<16}{w:>12.4%}")
        lines.append("=" * 58)
        return "\n".join(lines)
