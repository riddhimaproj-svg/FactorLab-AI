"""OptimizerConfig: solver settings and default bounds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from factorlab_optimizer.errors import OptimizationInputError

__all__ = ["OptimizerConfig"]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Configuration shared by every optimizer.

    Parameters
    ----------
    risk_free_rate:
        Per-period risk-free rate (same units as expected returns), used for
        Sharpe-based objectives and the capital allocation line.
    risk_aversion:
        The ``gamma`` in mean-variance utility ``mu'w - gamma/2 w'Sigma w``.
    min_weight, max_weight:
        Default per-asset box bounds.  ``min_weight=None`` resolves to ``0`` for
        long-only or ``-max_weight`` when ``allow_short`` is True.
    allow_short:
        Convenience toggle for the default lower bound (see ``min_weight``).
    budget:
        Default sum-of-weights target (``1.0`` = fully invested).  ``None`` adds
        no automatic budget constraint (use explicit cash/leverage constraints).
    covariance_regularization:
        Ridge term added to the covariance diagonal for numerical stability.
    max_iterations, tolerance:
        SLSQP solver controls.
    """

    risk_free_rate: float = 0.0
    risk_aversion: float = 1.0
    min_weight: float | None = None
    max_weight: float = 1.0
    allow_short: bool = False
    budget: float | None = 1.0
    covariance_regularization: float = 0.0
    max_iterations: int = 1000
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.risk_aversion <= 0:
            raise OptimizationInputError("risk_aversion must be positive")
        if self.covariance_regularization < 0:
            raise OptimizationInputError("covariance_regularization must be >= 0")

    def default_bounds(self) -> tuple[float, float]:
        """Resolve the default per-asset ``(lower, upper)`` box bounds."""
        if self.min_weight is not None:
            lower = self.min_weight
        else:
            lower = -abs(self.max_weight) if self.allow_short else 0.0
        return lower, self.max_weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_free_rate": self.risk_free_rate,
            "risk_aversion": self.risk_aversion,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "allow_short": self.allow_short,
            "budget": self.budget,
            "covariance_regularization": self.covariance_regularization,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerConfig:
        return cls(**dict(data))
