"""Structural interfaces (Protocols) for the estimation and model layers.

These protocols express *what* an estimator or factor model must provide
without binding callers to any concrete class.  Outer layers (the API service,
tests, alternative estimators) depend on these abstractions -- satisfying the
Dependency Inversion Principle -- so a new estimator or model is a drop-in
replacement rather than a refactor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from factorlab_quant.core.types import CovarianceType, FloatArray, RegressionResult

__all__ = ["Estimator", "FactorModel"]


@runtime_checkable
class Estimator(Protocol):
    """Fits ``y = X @ beta + e`` and returns a full :class:`RegressionResult`.

    Implementations must not mutate their inputs and must be deterministic:
    the same ``(y, X)`` and configuration always yield the same result.
    """

    def fit(
        self,
        y: FloatArray,
        X: FloatArray,
        *,
        param_names: tuple[str, ...] | None = ...,
        covariance_type: CovarianceType = ...,
        conf_level: float = ...,
    ) -> RegressionResult: ...


@runtime_checkable
class FactorModel(Protocol):
    """A named asset-pricing model that maps return data to a fitted result.

    Every model in :mod:`factorlab_quant.models` conforms to this protocol.
    The concrete ``fit`` signatures differ (CAPM needs one market factor,
    Fama--French needs several), so the protocol fixes only the shared surface:
    a human-readable name, the ordered factor labels, and a ``summary`` string.
    """

    @property
    def name(self) -> str: ...

    @property
    def factor_names(self) -> tuple[str, ...]: ...
