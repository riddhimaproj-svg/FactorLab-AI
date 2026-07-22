"""Exception hierarchy for the optimization engine."""

from __future__ import annotations

__all__ = [
    "InfeasibleProblemError",
    "OptimizationFailedError",
    "OptimizationInputError",
    "OptimizerError",
]


class OptimizerError(Exception):
    """Base class for every error raised by ``factorlab_optimizer``."""


class OptimizationInputError(OptimizerError):
    """Inputs (returns, covariance, constraints) were malformed or inconsistent."""


class InfeasibleProblemError(OptimizerError):
    """The constraint set admits no feasible portfolio."""


class OptimizationFailedError(OptimizerError):
    """The numerical solver did not converge to a solution."""

    def __init__(self, message: str, *, optimizer: str = "") -> None:
        self.optimizer = optimizer
        prefix = f"[{optimizer}] " if optimizer else ""
        super().__init__(f"{prefix}{message}")
