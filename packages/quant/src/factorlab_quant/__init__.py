"""FactorLab quant engine — institutional multi-factor asset-pricing analytics.

A pure, typed, side-effect-free library.  It knows nothing about HTTP,
databases, or data providers: it takes return arrays in and returns immutable,
fully-documented result objects.  That isolation is what makes it independently
installable and testable, and reusable from a notebook, a CLI, a batch job, or
the FactorLab API service.

Architecture
------------
The package is a generic *linear factor modeling framework*.  A single engine
(:class:`LinearFactorModel`) performs all regression work; concrete models are
thin subclasses that specify only their factors.  Data flows in as immutable
:class:`Factor` / :class:`FactorSet` objects and out as a reusable, serializable
:class:`FactorModelResult`.

Quick start
-----------
>>> import numpy as np
>>> from factorlab_quant import CAPM
>>> rng = np.random.default_rng(42)
>>> market_excess = rng.normal(0.006, 0.043, size=360)
>>> asset_excess = 0.0012 + 0.95 * market_excess + rng.normal(0, 0.02, size=360)
>>> result = CAPM().fit(asset_excess, market_excess, returns_are_excess=True)
>>> print(result.summary())  # doctest: +SKIP
"""

from __future__ import annotations

from factorlab_quant.core import (
    CoefficientEstimate,
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
    RegressionDiagnostics,
    RegressionResult,
    SerializationError,
)
from factorlab_quant.estimation import OLS
from factorlab_quant.models import (
    CAPM,
    SCHEMA_VERSION,
    AbstractFactorModel,
    CAPMResult,
    CarhartModel,
    CarhartResult,
    Factor,
    FactorModelResult,
    FactorSet,
    FamaFrench3Model,
    FamaFrench3Result,
    FamaFrench5Model,
    FamaFrench5Result,
    LinearFactorModel,
    create_model,
    get_model,
    is_registered,
    list_models,
    register_model,
    unregister_model,
)

__version__ = "0.2.0"

__all__ = [
    "CAPM",
    "OLS",
    "SCHEMA_VERSION",
    "AbstractFactorModel",
    "CAPMResult",
    "CarhartModel",
    "CarhartResult",
    "CoefficientEstimate",
    "CollinearityError",
    "ConstantFactorError",
    "DataValidationError",
    "DimensionMismatchError",
    "DuplicateFactorError",
    "DuplicateObservationError",
    "EstimationError",
    "Factor",
    "FactorModelResult",
    "FactorSet",
    "FamaFrench3Model",
    "FamaFrench3Result",
    "FamaFrench5Model",
    "FamaFrench5Result",
    "FrequencyMismatchError",
    "InsufficientDataError",
    "LinearFactorModel",
    "ModelAlreadyRegisteredError",
    "ModelNotFoundError",
    "NonFiniteError",
    "QuantError",
    "RegressionDiagnostics",
    "RegressionResult",
    "SerializationError",
    "__version__",
    "create_model",
    "get_model",
    "is_registered",
    "list_models",
    "register_model",
    "unregister_model",
]
