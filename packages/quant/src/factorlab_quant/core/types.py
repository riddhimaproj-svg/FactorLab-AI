"""Immutable, typed result objects returned by the estimation and model layers.

Design principles
-----------------
* **Framework-free.**  These types depend only on the standard library and
  NumPy.  They contain no reference to HTTP, pandas, or any data provider, so
  they serialize cleanly and can be mapped to API DTOs by an outer layer.
* **Immutable.**  Every result is a frozen dataclass.  A fitted model's output
  is a fact about a dataset; it should never be mutated in place.
* **Self-describing.**  Each object carries enough metadata (degrees of
  freedom, covariance type, lag configuration) to reproduce and audit the
  inference that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CoefficientEstimate",
    "CovarianceType",
    "FloatArray",
    "RegressionDiagnostics",
    "RegressionResult",
]

FloatArray = NDArray[np.float64]

CovarianceType = Literal["nonrobust", "HC0", "HC1", "HAC"]
"""Supported covariance estimators.

* ``nonrobust`` -- classical OLS covariance, valid under homoskedasticity and
  no autocorrelation.
* ``HC0`` / ``HC1`` -- White heteroskedasticity-consistent estimators
  (``HC1`` applies the ``n / (n - k)`` small-sample correction).
* ``HAC`` -- Newey--West heteroskedasticity- and autocorrelation-consistent
  estimator with a Bartlett kernel.  The default for financial time series.
"""


@dataclass(frozen=True, slots=True)
class CoefficientEstimate:
    """Point estimate and inference for a single regression coefficient."""

    name: str
    estimate: float
    std_error: float
    t_statistic: float
    p_value: float
    conf_int_lower: float
    conf_int_upper: float
    conf_level: float

    @property
    def is_significant(self) -> bool:
        """True when the coefficient differs from zero at ``1 - conf_level``."""
        return self.p_value < (1.0 - self.conf_level)

    def significance_stars(self) -> str:
        """Conventional significance markers used in econometric tables."""
        if self.p_value < 0.01:
            return "***"
        if self.p_value < 0.05:
            return "**"
        if self.p_value < 0.10:
            return "*"
        return ""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible mapping."""
        return {
            "name": self.name,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int_lower,
            "conf_int_upper": self.conf_int_upper,
            "conf_level": self.conf_level,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoefficientEstimate:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            name=str(data["name"]),
            estimate=float(data["estimate"]),
            std_error=float(data["std_error"]),
            t_statistic=float(data["t_statistic"]),
            p_value=float(data["p_value"]),
            conf_int_lower=float(data["conf_int_lower"]),
            conf_int_upper=float(data["conf_int_upper"]),
            conf_level=float(data["conf_level"]),
        )


@dataclass(frozen=True, slots=True)
class RegressionDiagnostics:
    """Goodness-of-fit and specification diagnostics for a fitted regression.

    Attributes
    ----------
    r_squared, adj_r_squared:
        Coefficient of determination and its degrees-of-freedom adjustment.
    f_statistic, f_p_value:
        Joint test that all slope coefficients are zero.
    log_likelihood, aic, bic:
        Gaussian log-likelihood and information criteria for model comparison.
    durbin_watson:
        Test statistic for first-order residual autocorrelation.  Values near
        2 indicate no autocorrelation; toward 0 positive, toward 4 negative.
    jarque_bera, jarque_bera_p_value:
        Test of residual normality based on skewness and excess kurtosis.
    breusch_pagan, breusch_pagan_p_value:
        Lagrange-multiplier test of conditional heteroskedasticity.
    skewness, excess_kurtosis:
        Third and (excess) fourth standardized moments of the residuals.
    condition_number:
        2-norm condition number of the design matrix; large values flag
        multicollinearity.
    """

    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    log_likelihood: float
    aic: float
    bic: float
    durbin_watson: float
    jarque_bera: float
    jarque_bera_p_value: float
    breusch_pagan: float
    breusch_pagan_p_value: float
    skewness: float
    excess_kurtosis: float
    condition_number: float

    def to_dict(self) -> dict[str, float]:
        """Serialize to a JSON-compatible mapping of statistic name to value."""
        return {
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "f_statistic": self.f_statistic,
            "f_p_value": self.f_p_value,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "durbin_watson": self.durbin_watson,
            "jarque_bera": self.jarque_bera,
            "jarque_bera_p_value": self.jarque_bera_p_value,
            "breusch_pagan": self.breusch_pagan,
            "breusch_pagan_p_value": self.breusch_pagan_p_value,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "condition_number": self.condition_number,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> RegressionDiagnostics:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{key: float(value) for key, value in data.items()})


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """Complete, auditable output of a linear regression estimation.

    This is the canonical currency of the engine: every factor model produces
    one (or composes several) of these.  Arrays are stored read-only to
    preserve immutability guarantees.
    """

    coefficients: tuple[CoefficientEstimate, ...]
    n_observations: int
    n_parameters: int
    degrees_of_freedom: int
    residual_std_error: float
    fitted_values: FloatArray
    residuals: FloatArray
    covariance_matrix: FloatArray
    covariance_type: CovarianceType
    diagnostics: RegressionDiagnostics
    cov_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce read-only arrays so a caller cannot silently corrupt results.
        for arr in (self.fitted_values, self.residuals, self.covariance_matrix):
            arr.setflags(write=False)

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.coefficients)

    def coefficient(self, name: str) -> CoefficientEstimate:
        """Return the estimate for ``name`` or raise ``KeyError``."""
        for c in self.coefficients:
            if c.name == name:
                return c
        raise KeyError(
            f"No coefficient named {name!r}; available: {list(self.coefficient_names)}"
        )

    @property
    def params(self) -> FloatArray:
        """Coefficient point estimates as an array, in model order."""
        return np.array([c.estimate for c in self.coefficients], dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible mapping (arrays become nested lists)."""
        return {
            "coefficients": [c.to_dict() for c in self.coefficients],
            "n_observations": self.n_observations,
            "n_parameters": self.n_parameters,
            "degrees_of_freedom": self.degrees_of_freedom,
            "residual_std_error": self.residual_std_error,
            "fitted_values": self.fitted_values.tolist(),
            "residuals": self.residuals.tolist(),
            "covariance_matrix": self.covariance_matrix.tolist(),
            "covariance_type": self.covariance_type,
            "diagnostics": self.diagnostics.to_dict(),
            "cov_config": dict(self.cov_config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RegressionResult:
        """Reconstruct from :meth:`to_dict` output."""
        coefficients = tuple(
            CoefficientEstimate.from_dict(c) for c in data["coefficients"]
        )
        return cls(
            coefficients=coefficients,
            n_observations=int(data["n_observations"]),
            n_parameters=int(data["n_parameters"]),
            degrees_of_freedom=int(data["degrees_of_freedom"]),
            residual_std_error=float(data["residual_std_error"]),
            fitted_values=np.asarray(data["fitted_values"], dtype=np.float64),
            residuals=np.asarray(data["residuals"], dtype=np.float64),
            covariance_matrix=np.asarray(data["covariance_matrix"], dtype=np.float64),
            covariance_type=data["covariance_type"],
            diagnostics=RegressionDiagnostics.from_dict(data["diagnostics"]),
            cov_config=dict(data["cov_config"]),
        )
