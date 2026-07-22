"""Asset-pricing factor models and the generic framework they share.

The framework is layered:

* :class:`AbstractFactorModel` -- thin base: estimator injection + identity.
* :class:`LinearFactorModel` -- generic engine holding all regression logic.
* :class:`Factor` / :class:`FactorSet` -- the immutable factor data abstraction
  every model consumes.
* :class:`FactorModelResult` -- the reusable, serializable result with a full
  prediction API.
* :func:`register_model` / :func:`get_model` / :func:`list_models` -- the model
  registry.

Concrete models (CAPM today; FF3, FF5, Carhart, HXZ-q, APT next) are thin
subclasses of :class:`LinearFactorModel` that specify only their factors.
Importing this package registers every bundled model.
"""

from __future__ import annotations

from factorlab_quant.models.base import AbstractFactorModel
from factorlab_quant.models.capm import CAPM, CAPMResult
from factorlab_quant.models.carhart import CarhartModel, CarhartResult
from factorlab_quant.models.factors import Factor, FactorSet
from factorlab_quant.models.fama_french_3 import FamaFrench3Model, FamaFrench3Result
from factorlab_quant.models.fama_french_5 import FamaFrench5Model, FamaFrench5Result
from factorlab_quant.models.linear_factor_model import (
    SCHEMA_VERSION,
    FactorModelResult,
    LinearFactorModel,
)
from factorlab_quant.models.registry import (
    create_model,
    get_model,
    is_registered,
    list_models,
    register_model,
    unregister_model,
)

__all__ = [
    "CAPM",
    "SCHEMA_VERSION",
    "AbstractFactorModel",
    "CAPMResult",
    "CarhartModel",
    "CarhartResult",
    "Factor",
    "FactorModelResult",
    "FactorSet",
    "FamaFrench3Model",
    "FamaFrench3Result",
    "FamaFrench5Model",
    "FamaFrench5Result",
    "LinearFactorModel",
    "create_model",
    "get_model",
    "is_registered",
    "list_models",
    "register_model",
    "unregister_model",
]
