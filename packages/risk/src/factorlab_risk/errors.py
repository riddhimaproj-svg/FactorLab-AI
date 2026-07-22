"""Exception hierarchy for the risk engine."""

from __future__ import annotations

__all__ = [
    "RiskError",
    "RiskInputError",
    "InsufficientDataError",
    "DimensionMismatchError",
]


class RiskError(Exception):
    """Base class for every error raised by ``factorlab_risk``."""


class RiskInputError(RiskError):
    """Malformed inputs (returns, weights, covariance, confidence level)."""


class InsufficientDataError(RiskError):
    """Too few observations to compute a statistic reliably."""

    def __init__(self, n_obs: int, minimum: int, statistic: str = "statistic") -> None:
        self.n_obs = n_obs
        self.minimum = minimum
        super().__init__(
            f"{statistic} needs at least {minimum} observation(s); got {n_obs}."
        )


class DimensionMismatchError(RiskInputError):
    """Two aligned inputs disagree on length or shape."""

    def __init__(self, expected: int, received: int, name: str = "input") -> None:
        self.expected = expected
        self.received = received
        super().__init__(
            f"Dimension mismatch for {name!r}: expected {expected}, got {received}."
        )
