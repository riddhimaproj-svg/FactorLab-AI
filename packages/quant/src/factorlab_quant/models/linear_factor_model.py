r"""Generic linear factor model and its reusable result object.

Theory
------
A *linear factor model* explains an asset's (excess) return as a linear
combination of systematic factor returns plus an intercept and an idiosyncratic
error:

.. math::

    R_{i,t} - R_{f,t}
        = \alpha_i + \sum_{k=1}^{K} \beta_{i,k}\, F_{k,t} + \varepsilon_{i,t}.

Every model in this package is a special case distinguished only by *which*
factors :math:`F_{k}` enter the sum:

============================  ===================================================
Model                          Factors :math:`F_k`
============================  ===================================================
CAPM                           Mkt-RF
Fama--French 3                 Mkt-RF, SMB, HML
Carhart 4                      Mkt-RF, SMB, HML, MOM
Fama--French 5                 Mkt-RF, SMB, HML, RMW, CMA
Hou--Xue--Zhang q              Mkt-RF, ME, I/A, ROE
APT / user-defined             arbitrary set of priced factors
============================  ===================================================

Because the estimation, inference, diagnostics, prediction, and serialization
are identical across all of these, they live *once* in this module.  A concrete
model supplies only its factor specification; it writes no regression code.

This module defines:

* :class:`FactorModelResult` -- the immutable, serializable output shared by
  every model, with a full prediction API.
* :class:`LinearFactorModel` -- the generic estimator that turns a
  :class:`~factorlab_quant.models.factors.FactorSet` and a response series into
  a :class:`FactorModelResult`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from factorlab_quant.core.types import (
    CoefficientEstimate,
    CovarianceType,
    FloatArray,
    RegressionDiagnostics,
    RegressionResult,
)
from factorlab_quant.estimation.ols import OLS
from factorlab_quant.models.base import AbstractFactorModel
from factorlab_quant.models.factors import FactorSet
from factorlab_quant.utils.validation import as_float_vector

__all__ = ["SCHEMA_VERSION", "FactorModelResult", "LinearFactorModel"]

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class FactorModelResult:
    """Immutable, serializable output of a linear factor-model regression.

    This is the single result type reused by every model.  It composes a
    :class:`RegressionResult` (the raw estimation output) with factor-model
    semantics: which columns are factors, which is the intercept, the sampling
    frequency for annualization, arbitrary metadata, and a prediction API.
    """

    model_name: str
    param_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    has_intercept: bool
    intercept_name: str
    regression: RegressionResult
    design_matrix: FloatArray
    response: FloatArray
    periods_per_year: int
    uses_t_distribution: bool
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        for arr in (self.design_matrix, self.response):
            arr.setflags(write=False)

    # ------------------------------------------------------------------ #
    # Coefficient access                                                 #
    # ------------------------------------------------------------------ #
    @property
    def coefficients(self) -> tuple[CoefficientEstimate, ...]:
        return self.regression.coefficients

    @property
    def params(self) -> FloatArray:
        """Point estimates in design-column order."""
        return self.regression.params

    @property
    def standard_errors(self) -> FloatArray:
        return np.array([c.std_error for c in self.coefficients], dtype=np.float64)

    @property
    def t_statistics(self) -> FloatArray:
        return np.array([c.t_statistic for c in self.coefficients], dtype=np.float64)

    @property
    def p_values(self) -> FloatArray:
        return np.array([c.p_value for c in self.coefficients], dtype=np.float64)

    @property
    def confidence_intervals(self) -> FloatArray:
        """``k x 2`` array of ``[lower, upper]`` per coefficient."""
        return np.array(
            [[c.conf_int_lower, c.conf_int_upper] for c in self.coefficients],
            dtype=np.float64,
        )

    @property
    def covariance_matrix(self) -> FloatArray:
        return self.regression.covariance_matrix

    def coefficient(self, name: str) -> CoefficientEstimate:
        """Return the estimate named ``name`` (intercept or a factor)."""
        return self.regression.coefficient(name)

    def factor_loading(self, name: str) -> CoefficientEstimate:
        """Return the loading (beta) on factor ``name``."""
        if name not in self.factor_names:
            raise KeyError(
                f"{name!r} is not a factor; factors are {list(self.factor_names)}"
            )
        return self.regression.coefficient(name)

    @property
    def intercept(self) -> CoefficientEstimate:
        """The intercept coefficient (alpha).  Requires ``has_intercept``."""
        if not self.has_intercept:
            raise AttributeError("model was fitted without an intercept")
        return self.regression.coefficient(self.intercept_name)

    # ------------------------------------------------------------------ #
    # Fit statistics & diagnostics                                       #
    # ------------------------------------------------------------------ #
    @property
    def diagnostics(self) -> RegressionDiagnostics:
        return self.regression.diagnostics

    @property
    def r_squared(self) -> float:
        return self.regression.diagnostics.r_squared

    @property
    def adj_r_squared(self) -> float:
        return self.regression.diagnostics.adj_r_squared

    @property
    def aic(self) -> float:
        return self.regression.diagnostics.aic

    @property
    def bic(self) -> float:
        return self.regression.diagnostics.bic

    @property
    def log_likelihood(self) -> float:
        return self.regression.diagnostics.log_likelihood

    @property
    def residuals(self) -> FloatArray:
        return self.regression.residuals

    @property
    def fitted_values(self) -> FloatArray:
        return self.regression.fitted_values

    @property
    def n_observations(self) -> int:
        return self.regression.n_observations

    # ------------------------------------------------------------------ #
    # Prediction API                                                     #
    # ------------------------------------------------------------------ #
    def predict(
        self, factor_values: Mapping[str, object] | FactorSet | FloatArray | Sequence[float]
    ) -> float | FloatArray:
        r"""Point prediction of the response for new factor realizations.

        In asset-pricing usage the response is an excess return, so this is the
        model-implied excess return :math:`\hat\alpha + \sum_k \hat\beta_k F_k`.

        ``factor_values`` may be:

        * a mapping ``{factor_name: value_or_series}``,
        * a :class:`FactorSet`,
        * a 1-D array of length ``K`` (one scenario) or ``2-D`` ``m x K``.

        Returns a float for a single scenario, else an array of length ``m``.
        """
        design = self._prediction_design(factor_values)
        predictions = design @ self.params
        return self._scalar_or_array(predictions)

    def predict_excess_return(
        self, factor_values: Mapping[str, object] | FactorSet | FloatArray | Sequence[float]
    ) -> float | FloatArray:
        """Alias of :meth:`predict`, named for asset-pricing clarity."""
        return self.predict(factor_values)

    def expected_return(
        self,
        factor_values: Mapping[str, object] | FactorSet | FloatArray | Sequence[float],
        risk_free: float = 0.0,
    ) -> float | FloatArray:
        """Model-implied *total* return: predicted excess return plus ``risk_free``."""
        excess = self.predict(factor_values)
        if isinstance(excess, np.ndarray):
            return excess + float(risk_free)
        return float(excess) + float(risk_free)

    def confidence_interval(
        self,
        factor_values: Mapping[str, object] | FactorSet | FloatArray | Sequence[float],
        level: float = 0.95,
    ) -> tuple[float, float] | tuple[FloatArray, FloatArray]:
        r"""Confidence interval for the *mean* response at ``factor_values``.

        Uses :math:`\operatorname{Var}(\hat y_0) = x_0' \widehat{\Sigma}_\beta x_0`,
        the sampling uncertainty of the fitted mean (excludes idiosyncratic
        noise).
        """
        design = self._prediction_design(factor_values)
        point = design @ self.params
        var_mean = np.einsum("ij,jk,ik->i", design, self.covariance_matrix, design)
        se = np.sqrt(np.clip(var_mean, 0.0, None))
        crit = self._critical_value(level)
        lower = point - crit * se
        upper = point + crit * se
        return self._scalar_or_array(lower), self._scalar_or_array(upper)  # type: ignore[return-value]

    def prediction_interval(
        self,
        factor_values: Mapping[str, object] | FactorSet | FloatArray | Sequence[float],
        level: float = 0.95,
    ) -> tuple[float, float] | tuple[FloatArray, FloatArray]:
        r"""Prediction interval for a *new* observation at ``factor_values``.

        Adds the idiosyncratic residual variance to the mean's sampling
        variance: :math:`\operatorname{Var}(y_0) = x_0'\widehat{\Sigma}_\beta x_0
        + \hat\sigma_\varepsilon^2`.  Always wider than the confidence interval.
        """
        design = self._prediction_design(factor_values)
        point = design @ self.params
        var_mean = np.einsum("ij,jk,ik->i", design, self.covariance_matrix, design)
        sigma2 = self.regression.residual_std_error**2
        se = np.sqrt(np.clip(var_mean + sigma2, 0.0, None))
        crit = self._critical_value(level)
        lower = point - crit * se
        upper = point + crit * se
        return self._scalar_or_array(lower), self._scalar_or_array(upper)  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Prediction helpers                                                 #
    # ------------------------------------------------------------------ #
    def _prediction_design(
        self, factor_values: Mapping[str, object] | FactorSet | FloatArray | Sequence[float]
    ) -> FloatArray:
        """Assemble an ``m x k`` design matrix consistent with the fitted model."""
        factor_matrix = self._factor_matrix_from_input(factor_values)
        if self.has_intercept:
            ones = np.ones((factor_matrix.shape[0], 1), dtype=np.float64)
            return np.column_stack([ones, factor_matrix])
        return factor_matrix

    def _factor_matrix_from_input(
        self, factor_values: Mapping[str, object] | FactorSet | FloatArray | Sequence[float]
    ) -> FloatArray:
        n_factors = len(self.factor_names)
        if isinstance(factor_values, FactorSet):
            return factor_values.select(self.factor_names).matrix()
        if isinstance(factor_values, Mapping):
            missing = [n for n in self.factor_names if n not in factor_values]
            if missing:
                raise KeyError(f"Missing factor value(s): {missing}")
            columns = [
                np.atleast_1d(np.asarray(factor_values[name], dtype=np.float64))
                for name in self.factor_names
            ]
            lengths = {c.shape[0] for c in columns}
            if len(lengths) != 1:
                raise ValueError("all factor value arrays must share a length")
            return np.column_stack(columns)
        arr = np.asarray(factor_values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != n_factors:
            raise ValueError(
                f"expected factor array with {n_factors} column(s), got shape {arr.shape}"
            )
        return arr

    def _critical_value(self, level: float) -> float:
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly in (0, 1)")
        alpha = 1.0 - level
        if self.uses_t_distribution:
            return float(stats.t.ppf(1.0 - alpha / 2.0, df=self.regression.degrees_of_freedom))
        return float(stats.norm.ppf(1.0 - alpha / 2.0))

    @staticmethod
    def _scalar_or_array(values: FloatArray) -> float | FloatArray:
        return float(values[0]) if values.shape[0] == 1 else values

    # ------------------------------------------------------------------ #
    # Presentation                                                       #
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """A generic, formatted regression report usable by any model."""
        return _format_generic_summary(self)

    # ------------------------------------------------------------------ #
    # Serialization                                                      #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible mapping (see :meth:`from_dict`)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "result_type": type(self).__name__,
            "model_name": self.model_name,
            "param_names": list(self.param_names),
            "factor_names": list(self.factor_names),
            "has_intercept": self.has_intercept,
            "intercept_name": self.intercept_name,
            "periods_per_year": self.periods_per_year,
            "uses_t_distribution": self.uses_t_distribution,
            "metadata": dict(self.metadata),
            "design_matrix": self.design_matrix.tolist(),
            "response": self.response.tolist(),
            "regression": self.regression.to_dict(),
        }

    def to_json(self, **json_kwargs: object) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, payload: str) -> FactorModelResult:
        """Reconstruct a result from a JSON string produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def _base_kwargs_from_dict(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "model_name": str(data["model_name"]),
            "param_names": tuple(data["param_names"]),
            "factor_names": tuple(data["factor_names"]),
            "has_intercept": bool(data["has_intercept"]),
            "intercept_name": str(data["intercept_name"]),
            "regression": RegressionResult.from_dict(data["regression"]),
            "design_matrix": np.asarray(data["design_matrix"], dtype=np.float64),
            "response": np.asarray(data["response"], dtype=np.float64),
            "periods_per_year": int(data["periods_per_year"]),
            "uses_t_distribution": bool(data["uses_t_distribution"]),
            "metadata": dict(data["metadata"]),
        }

    @classmethod
    def _find_result_subclass(cls, name: str) -> type[FactorModelResult] | None:
        """Depth-first search of the subclass tree for a class named ``name``."""
        if cls.__name__ == name:
            return cls
        stack = list(cls.__subclasses__())
        while stack:
            candidate = stack.pop()
            if candidate.__name__ == name:
                return candidate
            stack.extend(candidate.__subclasses__())
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorModelResult:
        """Reconstruct a result, dispatching to the correct subclass.

        Calling ``FactorModelResult.from_dict`` on a payload produced by a
        subclass (e.g. ``CAPMResult``) returns an instance of that subclass.
        The subclass must be imported so it is discoverable.
        """
        if cls is FactorModelResult:
            result_type = str(data.get("result_type", "FactorModelResult"))
            target = cls._find_result_subclass(result_type)
            if target is not None and target is not FactorModelResult:
                return target.from_dict(data)
        return cls(**cls._base_kwargs_from_dict(data))


class LinearFactorModel(AbstractFactorModel):
    """Generic linear factor model: fits any :class:`FactorSet` to a response.

    All shared regression work -- alignment, design construction, estimation,
    inference, diagnostics, and result assembly -- lives here.  Concrete models
    (CAPM, FF3, ...) subclass this and supply only their factor specification,
    typically by overriding ``fit`` to build the appropriate ``FactorSet`` and
    delegating to :meth:`fit_factor_set`.

    Parameters
    ----------
    name:
        Human-readable model name.
    factor_names:
        The ordered factor specification.  When non-empty, an incoming
        ``FactorSet`` is selected/reordered to exactly these factors, enforcing
        the model's contract.  When empty, the set's own factors are used as-is
        (the arbitrary user-defined case).
    estimator:
        Injected OLS engine (Dependency Inversion); defaults to ``OLS()``.
    intercept:
        Whether to include an alpha intercept term.
    intercept_name:
        Column name for the intercept.
    response_name:
        Label for the response series (used in validation messages/metadata).
    metadata:
        Static metadata merged into every result's ``metadata``.
    """

    def __init__(
        self,
        name: str,
        factor_names: Sequence[str] = (),
        *,
        estimator: OLS | None = None,
        intercept: bool = True,
        intercept_name: str = "alpha",
        response_name: str = "response",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(estimator)
        self._name = name
        self._factor_names = tuple(factor_names)
        self.intercept = intercept
        self.intercept_name = intercept_name
        self.response_name = response_name
        self._metadata = dict(metadata or {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self._factor_names

    def fit(
        self,
        response: FloatArray | Sequence[float],
        factors: FactorSet | Mapping[str, object] | FloatArray,
        *,
        covariance_type: CovarianceType = "HAC",
        conf_level: float = 0.95,
        hac_lags: int | None = None,
        small_sample_correction: bool = True,
        use_t: bool | None = None,
        periods_per_year: int = 12,
        reject_duplicate_observations: bool = False,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> FactorModelResult:
        """Fit the model and return a :class:`FactorModelResult`.

        See :meth:`fit_factor_set` for parameter semantics; this method coerces
        ``factors`` to a :class:`FactorSet` first.
        """
        factor_set = self._coerce_factor_set(factors)
        return self.fit_factor_set(
            response,
            factor_set,
            covariance_type=covariance_type,
            conf_level=conf_level,
            hac_lags=hac_lags,
            small_sample_correction=small_sample_correction,
            use_t=use_t,
            periods_per_year=periods_per_year,
            reject_duplicate_observations=reject_duplicate_observations,
            extra_metadata=extra_metadata,
        )

    def fit_factor_set(
        self,
        response: FloatArray | Sequence[float],
        factor_set: FactorSet,
        *,
        covariance_type: CovarianceType = "HAC",
        conf_level: float = 0.95,
        hac_lags: int | None = None,
        small_sample_correction: bool = True,
        use_t: bool | None = None,
        periods_per_year: int = 12,
        reject_duplicate_observations: bool = False,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> FactorModelResult:
        """Core estimation routine shared by every model.

        Steps: enforce the factor specification, validate & align observations,
        run regularity checks, build the design matrix, call the estimator, and
        assemble the result.
        """
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer")

        if self._factor_names:
            factor_set = factor_set.select(self._factor_names)

        response_arr = as_float_vector(response, name=self.response_name)
        response_aligned, aligned = factor_set.align(response_arr)

        aligned.assert_regular()
        if reject_duplicate_observations:
            aligned.assert_unique_observations(response_aligned)

        design, param_names = aligned.to_design_matrix(
            intercept=self.intercept, intercept_name=self.intercept_name
        )
        regression = self.estimator.fit(
            response_aligned,
            design,
            param_names=param_names,
            covariance_type=covariance_type,
            conf_level=conf_level,
            hac_lags=hac_lags,
            small_sample_correction=small_sample_correction,
            use_t=use_t,
        )

        resolved_use_t = (covariance_type == "nonrobust") if use_t is None else use_t
        metadata = self._build_metadata(aligned, covariance_type, extra_metadata)

        return self._make_result(
            param_names=param_names,
            factor_names=aligned.names,
            regression=regression,
            design=design,
            response=response_aligned,
            periods_per_year=periods_per_year,
            uses_t_distribution=resolved_use_t,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Extension hooks                                                    #
    # ------------------------------------------------------------------ #
    def _make_result(
        self,
        *,
        param_names: tuple[str, ...],
        factor_names: tuple[str, ...],
        regression: RegressionResult,
        design: FloatArray,
        response: FloatArray,
        periods_per_year: int,
        uses_t_distribution: bool,
        metadata: dict[str, object],
    ) -> FactorModelResult:
        """Build the result object.

        Subclasses that return a richer result type (e.g. ``CAPMResult``)
        override this or post-process the returned base result.
        """
        return FactorModelResult(
            model_name=self.name,
            param_names=param_names,
            factor_names=factor_names,
            has_intercept=self.intercept,
            intercept_name=self.intercept_name,
            regression=regression,
            design_matrix=design,
            response=response,
            periods_per_year=periods_per_year,
            uses_t_distribution=uses_t_distribution,
            metadata=metadata,
        )

    def _build_metadata(
        self,
        factor_set: FactorSet,
        covariance_type: CovarianceType,
        extra_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "model": self.name,
            "covariance_type": covariance_type,
            "frequency": factor_set.frequency,
            "factors": factor_set.metadata(),
            **self._metadata,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _coerce_factor_set(
        self, factors: FactorSet | Mapping[str, object] | FloatArray
    ) -> FactorSet:
        if isinstance(factors, FactorSet):
            return factors
        if isinstance(factors, Mapping):
            return FactorSet.from_mapping(factors)  # type: ignore[arg-type]
        arr = np.asarray(factors, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = self._factor_names or tuple(f"factor_{i}" for i in range(arr.shape[1]))
        return FactorSet.from_matrix(arr, names)


# ---------------------------------------------------------------------------- #
# Generic summary                                                              #
# ---------------------------------------------------------------------------- #
def _format_generic_summary(result: FactorModelResult) -> str:
    reg = result.regression
    diag = reg.diagnostics
    cov_desc: str = reg.covariance_type
    if reg.covariance_type == "HAC":
        cov_desc = f"HAC (Newey-West, {reg.cov_config.get('lags')} lags, Bartlett)"

    header = f"Linear Factor Model — {result.model_name}"
    lines = [
        "=" * 74,
        header,
        f"Factors: {', '.join(result.factor_names)}",
        "=" * 74,
        f"Observations: {reg.n_observations:>7d}    Resid. DoF: {reg.degrees_of_freedom:>6d}"
        f"    Cov: {cov_desc}",
        f"R-squared:   {diag.r_squared:>8.4f}    Adj. R-sq: {diag.adj_r_squared:>7.4f}"
        f"    AIC: {diag.aic:>9.2f}   BIC: {diag.bic:>9.2f}",
        "-" * 74,
        f"{'coef':<10}{'estimate':>12}{'std err':>12}{'t':>10}{'P>|t|':>10}"
        f"{'[0.025':>10}{'0.975]':>10}",
        "-" * 74,
    ]
    lines.extend(_coef_row(c) for c in result.coefficients)
    lines.extend(
        [
            "-" * 74,
            "Diagnostics",
            f"  Durbin-Watson: {diag.durbin_watson:>7.4f}    "
            f"Jarque-Bera: {diag.jarque_bera:>8.4f} (p={diag.jarque_bera_p_value:.4f})",
            f"  Breusch-Pagan: {diag.breusch_pagan:>7.4f} (p={diag.breusch_pagan_p_value:.4f})"
            f"    Cond. no.: {diag.condition_number:>8.2f}",
            "=" * 74,
            "Significance:  *** p<0.01   ** p<0.05   * p<0.10",
        ]
    )
    return "\n".join(lines)


def _coef_row(c: CoefficientEstimate) -> str:
    return (
        f"{c.name:<10}{c.estimate:>12.6f}{c.std_error:>12.6f}"
        f"{c.t_statistic:>10.4f}{c.p_value:>10.4f}"
        f"{c.conf_int_lower:>10.5f}{c.conf_int_upper:>10.5f} {c.significance_stars()}"
    )
