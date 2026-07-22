r"""Fama--French Five-Factor Model (2015).

Why FF5 extends FF3
-------------------
The three-factor model (Fama & French, 1993) leaves systematic patterns in
average returns.  Two are especially robust:

* firms with **higher profitability** earn higher average returns than the
  three factors predict;
* firms that **invest more aggressively** earn *lower* average returns.

Fama & French (2015) add two factors that span these patterns, producing a model
that "leaves fewer anomalies on the table" than FF3.  FF5 nests FF3: set the RMW
and CMA loadings to zero and you recover the three-factor model.

Economic intuition of the new factors
--------------------------------------
* **RMW** ("Robust Minus Weak") -- the return of portfolios of highly profitable
  firms minus weakly profitable firms.  A positive RMW loading means the asset
  behaves like high-quality, profitable companies.
* **CMA** ("Conservative Minus Aggressive") -- the return of firms that invest
  conservatively minus those that invest aggressively.  A positive CMA loading
  means the asset behaves like low-investment ("conservative") firms.

Both are motivated by the dividend-discount identity: holding valuation fixed,
higher expected profitability implies higher expected returns, while higher
expected investment implies lower expected returns.  RMW and CMA are the
mimicking portfolios for those two channels.

Mathematical specification
--------------------------
.. math::

    R_{i,t} - R_{f,t}
        = \alpha_i
        + \beta_{i}^{\mathrm{MKT}} (R_{m,t} - R_{f,t})
        + \beta_{i}^{\mathrm{SMB}} \mathrm{SMB}_t
        + \beta_{i}^{\mathrm{HML}} \mathrm{HML}_t
        + \beta_{i}^{\mathrm{RMW}} \mathrm{RMW}_t
        + \beta_{i}^{\mathrm{CMA}} \mathrm{CMA}_t
        + \varepsilon_{i,t}.

Estimation is OLS with Newey--West (HAC) inference by default -- inherited
unchanged from :class:`LinearFactorModel`.  FF5 adds no estimation code.

Data flow (Kenneth French -> Factor Layer -> FF5)
-------------------------------------------------
Unlike FF3 (which takes raw factor arrays), FF5 consumes factor data **through
the factor-data layer**.  The intended flow is:

1. ``KennethFrenchAdapter`` parses the FF5 file into a ``FactorDataset``.
2. ``FactorLoader`` validates, caches, and returns it.
3. ``FactorPanel.to_factor_set()`` emits a ``FactorSet`` (Mkt-RF, SMB, HML, RMW,
   CMA; RF excluded).
4. ``FamaFrench5Model().fit(asset_excess, factor_set)`` regresses.

FF5 depends only on the *structural* ``to_factor_set`` capability, not on the
data package, so the quant engine stays free of any data-provider dependency.

Interpretation, assumptions, limitations
-----------------------------------------
* :math:`\hat\alpha` is the five-factor alpha; ``H0: alpha = 0`` tests whether
  market, size, value, profitability, and investment exposures fully explain the
  asset's average excess return.
* Assumptions and limitations mirror FF3 (linearity, constant loadings, factors
  spanning priced risks).  FF5 additionally makes HML somewhat redundant for
  many assets once RMW and CMA are included (Fama & French, 2015, note HML is a
  "redundant" factor for describing average returns in their sample).  FF5 still
  omits momentum (Carhart, 1997) and does not subsume the q-factor model
  (Hou--Xue--Zhang, 2015).

When FF5 outperforms FF3
------------------------
Whenever the asset has a profitability or investment tilt -- quality strategies,
low-investment ("conservative") portfolios, many factor funds -- FF3 leaves a
profitability/investment pattern in its alpha that RMW and CMA absorb.

References
----------
Fama, E. F., & French, K. R. (2015). "A Five-Factor Asset Pricing Model."
    *Journal of Financial Economics* 116(1), 1--22.
Fama, E. F., & French, K. R. (1993). "Common Risk Factors in the Returns on
    Stocks and Bonds." *Journal of Financial Economics* 33(1), 3--56.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from factorlab_quant.core.types import (
    CoefficientEstimate,
    CovarianceType,
    FloatArray,
    RegressionResult,
)
from factorlab_quant.estimation.ols import OLS
from factorlab_quant.models.factors import FactorSet
from factorlab_quant.models.linear_factor_model import FactorModelResult, LinearFactorModel
from factorlab_quant.models.registry import register_model
from factorlab_quant.utils.validation import as_float_vector

__all__ = ["FamaFrench5Model", "FamaFrench5Result"]

_MARKET = "Mkt-RF"
_SMB = "SMB"
_HML = "HML"
_RMW = "RMW"
_CMA = "CMA"
_ALPHA = "alpha"
_FACTORS = (_MARKET, _SMB, _HML, _RMW, _CMA)


@dataclass(frozen=True, slots=True)
class FamaFrench5Result(FactorModelResult):
    """Fitted Fama--French five-factor model.

    Adds named, finance-specific accessors on top of the generic
    :class:`FactorModelResult`.  It introduces no new stored fields, so it
    inherits the base serialization unchanged.
    """

    @property
    def alpha(self) -> CoefficientEstimate:
        """Five-factor alpha (per period), with HAC-based inference."""
        return self.intercept

    @property
    def market_beta(self) -> CoefficientEstimate:
        return self.factor_loading(_MARKET)

    @property
    def smb_loading(self) -> CoefficientEstimate:
        return self.factor_loading(_SMB)

    @property
    def hml_loading(self) -> CoefficientEstimate:
        return self.factor_loading(_HML)

    @property
    def rmw_loading(self) -> CoefficientEstimate:
        """RMW loading -- profitability tilt (positive => robust/high-quality)."""
        return self.factor_loading(_RMW)

    @property
    def cma_loading(self) -> CoefficientEstimate:
        """CMA loading -- investment tilt (positive => conservative)."""
        return self.factor_loading(_CMA)

    @property
    def annualized_alpha(self) -> float:
        r"""Geometrically annualized alpha, :math:`(1 + \hat\alpha)^{P} - 1`."""
        return float((1.0 + self.alpha.estimate) ** self.periods_per_year - 1.0)

    @property
    def size_tilt(self) -> str:
        return "small-cap" if self.smb_loading.estimate > 0.0 else "large-cap"

    @property
    def value_tilt(self) -> str:
        return "value" if self.hml_loading.estimate > 0.0 else "growth"

    @property
    def profitability_tilt(self) -> str:
        return "robust" if self.rmw_loading.estimate > 0.0 else "weak"

    @property
    def investment_tilt(self) -> str:
        return "conservative" if self.cma_loading.estimate > 0.0 else "aggressive"

    def summary(self) -> str:
        return _format_ff5_summary(self)


@register_model("FF5")
@register_model("FamaFrench5")
class FamaFrench5Model(LinearFactorModel):
    """Fama--French five-factor model estimator.

    Factors: ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``.  A thin subclass of
    :class:`LinearFactorModel` that consumes factor data through the factor-data
    layer (any object exposing ``to_factor_set()``), a ``FactorSet``, or a
    ``{name: values}`` mapping.

    Examples
    --------
    >>> import numpy as np
    >>> from factorlab_quant.models import FactorSet, Factor
    >>> from factorlab_quant.models.fama_french_5 import FamaFrench5Model
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> cols = {k: rng.normal(0, 0.03, n) for k in ("Mkt-RF","SMB","HML","RMW","CMA")}
    >>> asset = 0.001 + 1.1*cols["Mkt-RF"] + 0.4*cols["RMW"] + rng.normal(0, 0.01, n)
    >>> res = FamaFrench5Model().fit(asset, cols, returns_are_excess=True)
    >>> round(res.rmw_loading.estimate, 1)
    0.4
    """

    def __init__(self, estimator: OLS | None = None) -> None:
        super().__init__(
            name="Fama-French 5-Factor Model",
            factor_names=_FACTORS,
            estimator=estimator,
            intercept=True,
            intercept_name=_ALPHA,
            response_name="asset_excess_return",
            metadata={
                "family": "linear_factor_model",
                "n_factors": 5,
                "reference": "Fama & French (2015)",
            },
        )

    def fit(  # type: ignore[override]
        self,
        asset_returns: FloatArray,
        factors: object,
        risk_free: FloatArray | float | None = None,
        *,
        returns_are_excess: bool = False,
        covariance_type: CovarianceType = "HAC",
        conf_level: float = 0.95,
        hac_lags: int | None = None,
        small_sample_correction: bool = True,
        use_t: bool | None = None,
        periods_per_year: int = 12,
        reject_duplicate_observations: bool = False,
    ) -> FamaFrench5Result:
        r"""Estimate the Fama--French five-factor regression.

        Parameters
        ----------
        asset_returns:
            The asset (or portfolio) return series -- the dependent variable.
        factors:
            The five factors, supplied through the factor-data layer.  Accepts
            any object exposing ``to_factor_set()`` (e.g. a
            :class:`factorlab_data.FactorPanel` / ``FactorDataset``), a
            :class:`FactorSet`, or a ``{name: values}`` mapping.  Must contain
            ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA`` (extra factors, such
            as ``RF``, are ignored).
        risk_free:
            Per-period risk-free rate used only to convert ``asset_returns`` to
            excess form.  ``None`` when ``returns_are_excess=True``.
        returns_are_excess:
            When ``True``, ``asset_returns`` is already an excess return and
            ``risk_free`` must be ``None``.  The factors are always used as-is
            (``Mkt-RF`` is already an excess return; the rest are spreads).
        covariance_type, conf_level, hac_lags, small_sample_correction, use_t,
        periods_per_year, reject_duplicate_observations:
            Inference / behavior options; see
            :meth:`LinearFactorModel.fit_factor_set`.

        Returns
        -------
        FamaFrench5Result

        Raises
        ------
        ValueError
            If ``returns_are_excess`` is True but ``risk_free`` is provided.
        TypeError
            If ``factors`` is not convertible to a factor set.
        KeyError
            If any of the five required factors is missing.
        """
        if returns_are_excess and risk_free is not None:
            raise ValueError(
                "risk_free must be None when returns_are_excess=True; the asset "
                "return is assumed to already be an excess return."
            )

        factor_set = self._resolve_factor_set(factors)
        asset = as_float_vector(asset_returns, name="asset_returns")
        asset_excess = asset if returns_are_excess else self._to_excess(asset, risk_free)

        result = self.fit_factor_set(
            asset_excess,
            factor_set,
            covariance_type=covariance_type,
            conf_level=conf_level,
            hac_lags=hac_lags,
            small_sample_correction=small_sample_correction,
            use_t=use_t,
            periods_per_year=periods_per_year,
            reject_duplicate_observations=reject_duplicate_observations,
            extra_metadata={"specification": "Fama-French (2015) five-factor"},
        )
        return cast(FamaFrench5Result, result)

    @staticmethod
    def _resolve_factor_set(factors: object) -> FactorSet:
        """Coerce a factors source into a :class:`FactorSet`.

        Accepts a ``FactorSet``, any object with a ``to_factor_set()`` method
        (the factor-data layer's panels/datasets), or a ``{name: values}``
        mapping.  The duck-typed ``to_factor_set`` path is what lets FF5 consume
        the data layer without the quant engine importing it.
        """
        if isinstance(factors, FactorSet):
            return factors
        to_factor_set = getattr(factors, "to_factor_set", None)
        if callable(to_factor_set):
            produced = to_factor_set()
            if not isinstance(produced, FactorSet):
                raise TypeError(
                    "to_factor_set() must return a FactorSet, got "
                    f"{type(produced).__name__}"
                )
            return produced
        if isinstance(factors, Mapping):
            return FactorSet.from_mapping(factors)
        raise TypeError(
            "factors must be a FactorSet, a mapping, or an object exposing "
            f"to_factor_set(); got {type(factors).__name__}"
        )

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
    ) -> FamaFrench5Result:
        return FamaFrench5Result(
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


# ---------------------------------------------------------------------------- #
# Presentation                                                                 #
# ---------------------------------------------------------------------------- #
def _format_ff5_summary(result: FamaFrench5Result) -> str:
    reg = result.regression
    diag = reg.diagnostics
    cov_desc: str = reg.covariance_type
    if reg.covariance_type == "HAC":
        cov_desc = f"HAC (Newey-West, {reg.cov_config.get('lags')} lags, Bartlett)"

    rows = [
        ("alpha", result.alpha),
        ("beta_MKT (Mkt-RF)", result.market_beta),
        ("beta_SMB (SMB)", result.smb_loading),
        ("beta_HML (HML)", result.hml_loading),
        ("beta_RMW (RMW)", result.rmw_loading),
        ("beta_CMA (CMA)", result.cma_loading),
    ]

    lines = [
        "=" * 78,
        "Fama-French 5-Factor Model",
        "  R_i - R_f = alpha + b_MKT(Mkt-RF) + b_SMB(SMB) + b_HML(HML)",
        "              + b_RMW(RMW) + b_CMA(CMA) + e",
        "=" * 78,
        f"Observations: {reg.n_observations:>8d}    "
        f"Resid. DoF: {reg.degrees_of_freedom:>6d}    Cov: {cov_desc}",
        f"R-squared:    {diag.r_squared:>8.4f}    Adj. R-sq: {diag.adj_r_squared:>7.4f}"
        f"    F p-value: {diag.f_p_value:>.4f}",
        f"AIC: {diag.aic:>10.2f}    BIC: {diag.bic:>10.2f}    "
        f"LogLik: {diag.log_likelihood:>10.2f}",
        "-" * 78,
        f"{'coef':<20}{'estimate':>12}{'std err':>12}{'t':>10}{'P>|t|':>10}"
        f"{'[0.025':>7}{'0.975]':>9}",
        "-" * 78,
    ]
    lines.extend(_ff5_coef_row(label, c) for label, c in rows)
    lines.extend(
        [
            "-" * 78,
            "Style interpretation",
            f"  Annualized alpha:  {result.annualized_alpha:>10.4%}    "
            f"H0 alpha=0: t={result.alpha.t_statistic:>7.3f} "
            f"p={result.alpha.p_value:.4f} {result.alpha.significance_stars()}",
            f"  Size: {result.size_tilt:<10} Value: {result.value_tilt:<8} "
            f"Profitability: {result.profitability_tilt:<8} "
            f"Investment: {result.investment_tilt}",
            "-" * 78,
            "Diagnostics",
            f"  Durbin-Watson: {diag.durbin_watson:>7.4f}    "
            f"Jarque-Bera: {diag.jarque_bera:>8.4f} (p={diag.jarque_bera_p_value:.4f})",
            f"  Breusch-Pagan: {diag.breusch_pagan:>7.4f} (p={diag.breusch_pagan_p_value:.4f})"
            f"    Cond. no.: {diag.condition_number:>8.2f}",
            "=" * 78,
            "Significance:  *** p<0.01   ** p<0.05   * p<0.10",
        ]
    )
    return "\n".join(lines)


def _ff5_coef_row(label: str, c: CoefficientEstimate) -> str:
    return (
        f"{label:<20}{c.estimate:>12.6f}{c.std_error:>12.6f}"
        f"{c.t_statistic:>10.4f}{c.p_value:>10.4f}"
        f"{c.conf_int_lower:>9.4f}{c.conf_int_upper:>9.4f} {c.significance_stars()}"
    )
