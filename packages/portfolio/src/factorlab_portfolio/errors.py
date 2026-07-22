"""Exception hierarchy for the portfolio analytics package."""

from __future__ import annotations

__all__ = [
    "DimensionMismatchError",
    "InsufficientDataError",
    "PortfolioError",
    "PortfolioValidationError",
]


class PortfolioError(Exception):
    """Base class for every error raised by ``factorlab_portfolio``."""


class PortfolioValidationError(PortfolioError):
    """A model (position, trade, portfolio, return series) was malformed."""


class DimensionMismatchError(PortfolioError):
    """Two series that must align disagree on length."""

    def __init__(self, expected: int, received: int, name: str = "series") -> None:
        self.expected = expected
        self.received = received
        super().__init__(
            f"Dimension mismatch for {name!r}: expected {expected}, got {received}."
        )


class InsufficientDataError(PortfolioError):
    """Too few observations to compute a statistic reliably."""

    def __init__(self, n_obs: int, minimum: int, statistic: str = "statistic") -> None:
        self.n_obs = n_obs
        self.minimum = minimum
        super().__init__(
            f"{statistic} needs at least {minimum} observation(s); got {n_obs}."
        )
