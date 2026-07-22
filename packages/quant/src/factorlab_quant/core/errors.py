"""Exception hierarchy for the FactorLab quant engine.

The engine communicates every failure mode through a small, typed hierarchy so
that callers (the API service, a notebook, a CLI) can react programmatically
rather than pattern-matching on error strings.  Every exception derives from
:class:`QuantError`, so a caller may catch the whole family with a single
``except QuantError``.
"""

from __future__ import annotations

__all__ = [
    "CollinearityError",
    "ConstantFactorError",
    "DataValidationError",
    "DimensionMismatchError",
    "DuplicateFactorError",
    "DuplicateObservationError",
    "EstimationError",
    "FrequencyMismatchError",
    "InsufficientDataError",
    "ModelAlreadyRegisteredError",
    "ModelNotFoundError",
    "NonFiniteError",
    "QuantError",
    "SerializationError",
]


class QuantError(Exception):
    """Base class for every error raised by ``factorlab_quant``."""


class DataValidationError(QuantError):
    """Input data violated a precondition of the model or estimator.

    Raised before any numerical work begins.  A validation error means the
    *inputs* were malformed, not that the estimation itself was unstable.
    """


class InsufficientDataError(DataValidationError):
    """Too few observations to estimate the requested model reliably.

    A regression with ``k`` parameters needs at least ``k + 1`` observations to
    have positive residual degrees of freedom; most inferential statistics need
    materially more.  The engine enforces a caller-configurable minimum.
    """

    def __init__(self, n_obs: int, minimum: int, detail: str | None = None) -> None:
        self.n_obs = n_obs
        self.minimum = minimum
        message = (
            f"Received {n_obs} usable observation(s); at least {minimum} are "
            f"required for reliable estimation."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


class DimensionMismatchError(DataValidationError):
    """Two aligned inputs disagreed on length or shape."""

    def __init__(self, expected: int, received: int, name: str = "input") -> None:
        self.expected = expected
        self.received = received
        super().__init__(
            f"Dimension mismatch for {name!r}: expected length {expected}, "
            f"received {received}."
        )


class DuplicateFactorError(DataValidationError):
    """A :class:`~factorlab_quant.models.factors.FactorSet` contained repeated
    factor names, which would make the design matrix rank-deficient and the
    coefficients unidentifiable."""

    def __init__(self, duplicates: tuple[str, ...]) -> None:
        self.duplicates = duplicates
        joined = ", ".join(repr(d) for d in duplicates)
        super().__init__(f"Duplicate factor name(s): {joined}.")


class ConstantFactorError(DataValidationError):
    """A factor has (near-)zero variance.

    A constant factor is perfectly collinear with the intercept, so its loading
    cannot be separated from alpha.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"Factor {name!r} is constant (zero variance); it is collinear with "
            f"the intercept and its loading is not identifiable."
        )


class FrequencyMismatchError(DataValidationError):
    """Factors combined into one model declared inconsistent frequencies."""

    def __init__(self, frequencies: tuple[str, ...]) -> None:
        self.frequencies = frequencies
        joined = ", ".join(repr(f) for f in frequencies)
        super().__init__(
            f"Factors have mismatched frequencies ({joined}); align them to a "
            f"common sampling frequency before combining."
        )


class DuplicateObservationError(DataValidationError):
    """Identical observation rows were found where uniqueness was required."""

    def __init__(self, n_duplicates: int) -> None:
        self.n_duplicates = n_duplicates
        super().__init__(
            f"Found {n_duplicates} duplicate observation row(s); duplicated rows "
            f"understate standard errors and bias inference."
        )


class EstimationError(QuantError):
    """The numerical estimation could not be completed."""


class CollinearityError(EstimationError):
    """Design matrix is rank-deficient (perfect / near-perfect collinearity).

    Carries the observed condition number so callers can decide whether to drop
    a factor, add data, or proceed with a weaker guarantee.
    """

    def __init__(self, condition_number: float, threshold: float) -> None:
        self.condition_number = condition_number
        self.threshold = threshold
        super().__init__(
            f"Design matrix is ill-conditioned (condition number "
            f"{condition_number:.3e} exceeds threshold {threshold:.3e}); the "
            f"regressors are collinear and coefficients are not identifiable."
        )


class NonFiniteError(EstimationError):
    """A NaN or infinity appeared where a finite number was required."""


class ModelNotFoundError(QuantError):
    """A model name was requested from the registry but is not registered."""

    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.available = available
        joined = ", ".join(sorted(available)) or "<none>"
        super().__init__(
            f"No model registered under {name!r}. Registered models: {joined}."
        )


class ModelAlreadyRegisteredError(QuantError):
    """Attempted to register a second, different model under an existing key."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"A different model is already registered under {name!r}; choose a "
            f"unique key or unregister the existing one first."
        )


class SerializationError(QuantError):
    """A result could not be (de)serialized, e.g. an unknown schema version."""
