"""Abstract base for asset-pricing factor models.

Every model in this package is a time-series regression of an asset's (excess)
return on one or more factor returns.  The base captures the two things all
models share -- an injected estimator and a stable identity (name + factor
names) -- while leaving the concrete ``fit`` signature to subclasses, since it
differs by model (CAPM takes one market series; Fama--French takes several).

The heavy lifting (design construction, alignment, estimation, diagnostics,
prediction, serialization) lives in
:class:`~factorlab_quant.models.linear_factor_model.LinearFactorModel`, which
subclasses this base.  ``AbstractFactorModel`` itself stays deliberately thin so
that alternative model families could inherit from it without inheriting the
linear machinery.
"""

from __future__ import annotations

import abc

import numpy as np

from factorlab_quant.core.types import FloatArray
from factorlab_quant.estimation.ols import OLS
from factorlab_quant.utils.validation import as_float_vector, check_lengths_match

__all__ = ["AbstractFactorModel"]


class AbstractFactorModel(abc.ABC):
    """Common identity and estimator injection for every factor model.

    Parameters
    ----------
    estimator:
        The OLS engine used to fit the model.  Injected (Dependency Inversion)
        so a caller can supply a differently-configured estimator -- e.g. a
        stricter collinearity threshold -- without subclassing.  Defaults to a
        stock :class:`~factorlab_quant.estimation.ols.OLS`.
    """

    def __init__(self, estimator: OLS | None = None) -> None:
        self.estimator = estimator if estimator is not None else OLS()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable model name, e.g. ``"Capital Asset Pricing Model"``."""

    @property
    @abc.abstractmethod
    def factor_names(self) -> tuple[str, ...]:
        """Ordered factor labels excluding the intercept."""

    # ------------------------------------------------------------------ #
    # Shared helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_excess(
        returns: FloatArray, risk_free: FloatArray | float | None
    ) -> FloatArray:
        """Convert raw returns to excess returns over the risk-free rate.

        ``risk_free`` may be ``None`` (returns already in excess form), a scalar
        (a constant per-period rate), or a per-period vector aligned with
        ``returns``.
        """
        if risk_free is None:
            return returns
        if np.isscalar(risk_free):
            return np.asarray(returns - float(risk_free), dtype=np.float64)  # type: ignore[arg-type]
        rf = as_float_vector(risk_free, name="risk_free")  # type: ignore[arg-type]
        check_lengths_match(("returns", returns), ("risk_free", rf))
        return np.asarray(returns - rf, dtype=np.float64)
