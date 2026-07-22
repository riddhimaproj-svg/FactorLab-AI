"""Exception hierarchy for the derivatives engine."""

from __future__ import annotations

__all__ = [
    "ConvergenceError",
    "DerivativesError",
    "DerivativesInputError",
    "NoArbitrageError",
]


class DerivativesError(Exception):
    """Base class for every error raised by ``factorlab_derivatives``."""


class DerivativesInputError(DerivativesError):
    """Malformed inputs (negative spot/strike/maturity, bad option type, …)."""


class ConvergenceError(DerivativesError):
    """An iterative solver (implied vol, GARCH MLE) failed to converge."""

    def __init__(self, message: str, *, iterations: int = 0) -> None:
        self.iterations = iterations
        super().__init__(message)


class NoArbitrageError(DerivativesError):
    """A quoted price violates static no-arbitrage bounds (e.g. below intrinsic)."""
