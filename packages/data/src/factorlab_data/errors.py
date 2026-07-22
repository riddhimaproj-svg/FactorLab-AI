"""Exception hierarchy for the FactorLab data layer.

Mirrors the quant engine's approach: every failure is a typed subclass of
:class:`FactorDataError`, so callers can catch the whole family or react to a
specific condition.
"""

from __future__ import annotations

__all__ = [
    "DatasetNotFoundError",
    "FactorAlignmentError",
    "FactorCacheError",
    "FactorDataError",
    "FactorFetchError",
    "FactorNotFoundError",
    "FactorParseError",
    "FactorValidationError",
]


class FactorDataError(Exception):
    """Base class for every error raised by ``factorlab_data``."""


class FactorParseError(FactorDataError):
    """Raw provider content could not be parsed into a factor panel."""


class FactorValidationError(FactorDataError):
    """A parsed panel or dataset failed a validation invariant."""


class FactorNotFoundError(FactorDataError, KeyError):
    """A requested factor (column) is absent from a panel or dataset."""

    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.available = available
        joined = ", ".join(available) or "<none>"
        super().__init__(f"No factor named {name!r}; available: {joined}.")


class DatasetNotFoundError(FactorDataError, KeyError):
    """A requested dataset id / frequency is not known to an adapter or registry."""

    def __init__(self, dataset_id: str, available: tuple[str, ...] = ()) -> None:
        self.dataset_id = dataset_id
        self.available = available
        joined = ", ".join(sorted(available)) or "<none>"
        super().__init__(f"Unknown dataset {dataset_id!r}. Known datasets: {joined}.")


class FactorAlignmentError(FactorDataError):
    """Series/panels could not be aligned onto a common date index."""


class FactorCacheError(FactorDataError):
    """A cache read/write operation failed."""


class FactorFetchError(FactorDataError):
    """Remote retrieval failed or was attempted without a configured fetcher."""
