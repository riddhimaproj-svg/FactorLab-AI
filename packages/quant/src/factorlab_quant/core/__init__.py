"""Core, framework-free primitives: typed results, protocols, and errors."""

from __future__ import annotations

from factorlab_quant.core.errors import (
    CollinearityError,
    ConstantFactorError,
    DataValidationError,
    DimensionMismatchError,
    DuplicateFactorError,
    DuplicateObservationError,
    EstimationError,
    FrequencyMismatchError,
    InsufficientDataError,
    ModelAlreadyRegisteredError,
    ModelNotFoundError,
    NonFiniteError,
    QuantError,
    SerializationError,
)
from factorlab_quant.core.protocols import Estimator, FactorModel
from factorlab_quant.core.types import (
    CoefficientEstimate,
    CovarianceType,
    FloatArray,
    RegressionDiagnostics,
    RegressionResult,
)

__all__ = [
    "CoefficientEstimate",
    "CollinearityError",
    "ConstantFactorError",
    "CovarianceType",
    "DataValidationError",
    "DimensionMismatchError",
    "DuplicateFactorError",
    "DuplicateObservationError",
    "EstimationError",
    "Estimator",
    "FactorModel",
    "FloatArray",
    "FrequencyMismatchError",
    "InsufficientDataError",
    "ModelAlreadyRegisteredError",
    "ModelNotFoundError",
    "NonFiniteError",
    "QuantError",
    "RegressionDiagnostics",
    "RegressionResult",
    "SerializationError",
]
