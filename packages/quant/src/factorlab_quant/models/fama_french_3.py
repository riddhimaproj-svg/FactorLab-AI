r"""Fama--French Three-Factor Model (1993).

Economic intuition
-------------------
The CAPM says a single factor -- the market -- prices all assets.  Empirically it
does not: portfolios of *small* stocks and of *value* (high book-to-market)
stocks earn average returns the CAPM cannot explain.  Fama and French (1993)
augment the market factor with two return-spread ("zero-cost, long-short")
factors that proxy for these pervasive sources of common risk:

* **SMB** ("Small Minus Big") -- the return of small-cap portfolios minus
  large-cap portfolios.  A positive SMB loading means the asset behaves like
  small-cap stocks (a *size* tilt).
* **HML** ("High Minus Low") -- the return of high book-to-market (value)
  portfolios minus low book-to-market (growth) portfolios.  A positive HML
  loading means the asset behaves like value stocks (a *value* tilt).

Both are constructed to be (approximately) uncorrelated mimicking portfolios, so
their loadings are interpretable as risk exposures rather than mechanical
artifacts.

Mathematical specification
--------------------------
The model is a time-series regression of an asset's excess return on the three
factors (Fama & French, 1993, eq. for stock returns):

.. math::

    R_{i,t} - R_{f,t}
        = \alpha_i
        + \beta_{i}^{\mathrm{MKT}} (R_{m,t} - R_{f,t})
        + \beta_{i}^{\mathrm{SMB}} \, \mathrm{SMB}_t
        + \beta_{i}^{\mathrm{HML}} \, \mathrm{HML}_t
        + \varepsilon_{i,t}.

* ``Mkt-RF`` is *already* an excess return (the market return over the
  risk-free rate), exactly as distributed in the Kenneth French Data Library, so
  it is used as-is -- the risk-free rate is **not** subtracted from it again.
* ``SMB`` and ``HML`` are self-financing return spreads, so they too enter the
  regression unadjusted.
* Only the dependent variable, the asset return, is converted to an excess
  return :math:`R_i - R_f`.

Estimation is OLS with Newey--West (HAC) standard errors by default, identical to
every other model in the framework -- FF3 adds **no** estimation code.

Interpretation
--------------
* :math:`\hat\alpha` -- the three-factor alpha.  ``H0: alpha = 0`` tests whether
  the asset's average excess return is fully explained by its market, size, and
  value exposures.  A reliably positive alpha is evidence of skill or of a risk
  the three factors miss.
* :math:`\hat\beta^{\mathrm{MKT}}` -- market (systematic) risk exposure.
* :math:`\hat\beta^{\mathrm{SMB}}`, :math:`\hat\beta^{\mathrm{HML}}` -- size and
  value style exposures.

Assumptions
-----------
Standard linear-model assumptions for valid OLS point estimates (linearity,
factors uncorrelated with the error in expectation).  HAC covariance relaxes the
homoskedasticity/no-autocorrelation assumptions for *inference*.  Interpreting
alpha as "abnormal return" further assumes the three factors span the relevant
priced risks.

Limitations
-----------
* FF3 omits profitability and investment premia; Fama & French (2015) add RMW
  and CMA (the five-factor model) precisely because FF3 leaves patterns in
  average returns (notably among profitable and conservatively-investing firms).
* It does not capture momentum; Carhart (1997) adds a fourth (MOM) factor.
* Loadings are assumed constant over the estimation window; regime shifts break
  this (address with rolling estimation).
* SMB/HML are empirically motivated; their status as compensation for risk
  versus mispricing remains debated.

When FF3 outperforms CAPM
-------------------------
Whenever the test asset has a non-trivial size or value tilt -- small-cap funds,
value/growth strategies, most equity portfolios -- CAPM leaves a size/value
pattern in its residual alpha that FF3 absorbs, typically raising adjusted
:math:`R^2` and shrinking |alpha| toward zero.  For a broad-market,
large-cap-like asset with no style tilt, FF3 and CAPM largely agree.

References
----------
Fama, E. F., & French, K. R. (1993). "Common Risk Factors in the Returns on
    Stocks and Bonds." *Journal of Financial Economics* 33(1), 3--56.
Fama, E. F., & French, K. R. (2015). "A Five-Factor Asset Pricing Model."
    *Journal of Financial Economics* 116(1), 1--22.
Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance."
    *Journal of Finance* 52(1), 57--82.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from factorlab_quant.core.types import (
    CoefficientEstimate,
    CovarianceType,
    FloatArray,
    RegressionResult,
)
from factorlab_quant.estimation.ols import OLS
from factorlab_quant.models.factors import Factor, FactorSet
from factorlab_quant.models.linear_factor_model import FactorModelResult, LinearFactorModel
from factorlab_quant.models.registry import register_model
from factorlab_quant.utils.validation import as_float_vector, check_lengths_match

__all__ = ["FamaFrench3Model", "FamaFrench3Result"]

_MARKET = "Mkt-RF"
_SMB = "SMB"
_HML = "HML"
_ALPHA = "alpha"
_FACTORS = (_MARKET, _SMB, _HML)
_SOURCE = "Kenneth French Data Library"


@dataclass(frozen=True, slots=True)
class FamaFrench3Result(FactorModelResult):
    """Fitted Fama--French three-factor model.

    Adds named, finance-specific accessors on top of the generic
    :class:`FactorModelResult` (which already provides coefficients, standard
    errors, confidence intervals, t/p-values, adjusted :math:`R^2`, residuals,
    predictions, diagnostics, metadata, and serialization).  It introduces no
    new stored fields -- every quantity below derives from the regression -- so
    it inherits the base serialization unchanged.
    """

    # -- Named coefficient accessors --------------------------------------
    @property
    def alpha(self) -> CoefficientEstimate:
        """Three-factor alpha (per period), with HAC-based inference."""
        return self.intercept

    @property
    def market_beta(self) -> CoefficientEstimate:
        """Market (MKT) loading -- systematic risk exposure."""
        return self.factor_loading(_MARKET)

    @property
    def smb_loading(self) -> CoefficientEstimate:
        """SMB loading -- size tilt (positive => small-cap-like)."""
        return self.factor_loading(_SMB)

    @property
    def hml_loading(self) -> CoefficientEstimate:
        """HML loading -- value tilt (positive => value-like)."""
        return self.factor_loading(_HML)

    # -- Interpretation helpers -------------------------------------------
    @property
    def annualized_alpha(self) -> float:
        r"""Geometrically annualized alpha, :math:`(1 + \hat\alpha)^{P} - 1`."""
        return float((1.0 + self.alpha.estimate) ** self.periods_per_year - 1.0)

    @property
    def size_tilt(self) -> str:
        """``"small-cap"`` if the SMB loading is positive, else ``"large-cap"``."""
        return "small-cap" if self.smb_loading.estimate > 0.0 else "large-cap"

    @property
    def value_tilt(self) -> str:
        """``"value"`` if the HML loading is positive, else ``"growth"``."""
        return "value" if self.hml_loading.estimate > 0.0 else "growth"

    def summary(self) -> str:
        """A FF3-specific formatted report (overrides the generic summary)."""
        return _format_ff3_summary(self)


@register_model("FF3")
@register_model("FamaFrench3")
class FamaFrench3Model(LinearFactorModel):
    """Fama--French three-factor model estimator (``Mkt-RF``, ``SMB``, ``HML``).

    A thin subclass of :class:`LinearFactorModel`: it defines only the factor
    identity, metadata, and the excess-return construction specific to FF3.  All
    estimation, inference, diagnostics, prediction, and serialization are
    inherited from the framework.

    Examples
    --------
    >>> import numpy as np
    >>> from factorlab_quant.models.fama_french_3 import FamaFrench3Model
    >>> rng = np.random.default_rng(0)
    >>> n = 240
    >>> mkt = rng.normal(0.005, 0.04, n)
    >>> smb = rng.normal(0.001, 0.02, n)
    >>> hml = rng.normal(0.002, 0.03, n)
    >>> asset = 0.001 + 1.1*mkt - 0.3*smb + 0.6*hml + rng.normal(0, 0.01, n)
    >>> res = FamaFrench3Model().fit(asset, mkt, smb, hml, returns_are_excess=True)
    >>> round(res.hml_loading.estimate, 1)
    0.6
    """

    def __init__(self, estimator: OLS | None = None) -> None:
        super().__init__(
            name="Fama-French 3-Factor Model",
            factor_names=_FACTORS,
            estimator=estimator,
            intercept=True,
            intercept_name=_ALPHA,
            response_name="asset_excess_return",
            metadata={
                "family": "linear_factor_model",
                "n_factors": 3,
                "reference": "Fama & French (1993)",
            },
        )

    def fit(  # type: ignore[override]
        self,
        asset_returns: FloatArray,
        mkt_rf: FloatArray,
        smb: FloatArray,
        hml: FloatArray,
        risk_free: FloatArray | float | None = None,
        *,
        returns_are_excess: bool = False,
        frequency: str | None = None,
        covariance_type: CovarianceType = "HAC",
        conf_level: float = 0.95,
        hac_lags: int | None = None,
        small_sample_correction: bool = True,
        use_t: bool | None = None,
        periods_per_year: int = 12,
        reject_duplicate_observations: bool = False,
    ) -> FamaFrench3Result:
        r"""Estimate the Fama--French three-factor regression.

        Parameters
        ----------
        asset_returns:
            The asset (or portfolio) return series -- the dependent variable.
        mkt_rf:
            The market **excess** return factor (``Mkt-RF``), used as-is.
        smb, hml:
            The SMB and HML return-spread factors, used as-is.
        risk_free:
            Per-period risk-free rate used only to convert ``asset_returns`` to
            excess form.  ``None`` if the asset is already an excess return
            (``returns_are_excess=True``), else a scalar or an aligned vector.
        returns_are_excess:
            When ``True``, ``asset_returns`` is treated as already-excess and
            ``risk_free`` must be ``None``.  The three factors are *always*
            used unadjusted regardless of this flag.
        frequency:
            Optional sampling-frequency tag (``"monthly"``, ``"daily"``, ...)
            attached to every factor for provenance and downstream checks.
        covariance_type, conf_level, hac_lags, small_sample_correction, use_t:
            Inference options; see
            :meth:`LinearFactorModel.fit_factor_set`.  Defaults to Newey--West
            ``"HAC"``.
        periods_per_year:
            Observations per year for annualization (12 monthly, 252 daily).
        reject_duplicate_observations:
            When ``True``, raise on exactly-duplicated observation rows.

        Returns
        -------
        FamaFrench3Result

        Raises
        ------
        ValueError
            If ``returns_are_excess`` is True but ``risk_free`` is provided.
        DimensionMismatchError, NonFiniteError, InsufficientDataError,
        ConstantFactorError, DuplicateFactorError, CollinearityError
            Propagated from framework validation and estimation.
        """
        if returns_are_excess and risk_free is not None:
            raise ValueError(
                "risk_free must be None when returns_are_excess=True; the asset "
                "return is assumed to already be an excess return."
            )

        asset = as_float_vector(asset_returns, name="asset_returns")
        market = as_float_vector(mkt_rf, name="mkt_rf")
        smb_arr = as_float_vector(smb, name="smb")
        hml_arr = as_float_vector(hml, name="hml")
        check_lengths_match(
            ("asset_returns", asset),
            ("mkt_rf", market),
            ("smb", smb_arr),
            ("hml", hml_arr),
        )

        asset_excess = asset if returns_are_excess else self._to_excess(asset, risk_free)

        factor_set = FactorSet(
            [
                Factor(
                    name=_MARKET,
                    values=market,
                    display_name="Market excess return (Mkt-RF)",
                    frequency=frequency,
                    source=_SOURCE,
                    description="Value-weighted market return minus the risk-free rate.",
                ),
                Factor(
                    name=_SMB,
                    values=smb_arr,
                    display_name="Small Minus Big (SMB)",
                    frequency=frequency,
                    source=_SOURCE,
                    description="Average small-cap return minus average large-cap return.",
                ),
                Factor(
                    name=_HML,
                    values=hml_arr,
                    display_name="High Minus Low (HML)",
                    frequency=frequency,
                    source=_SOURCE,
                    description="Average value (high B/M) return minus growth (low B/M) return.",
                ),
            ]
        )

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
            extra_metadata={"specification": "Fama-French (1993) three-factor"},
        )
        return cast(FamaFrench3Result, result)

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
    ) -> FamaFrench3Result:
        """Build the FF3-specific result (adds no fields to the base)."""
        return FamaFrench3Result(
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
def _format_ff3_summary(result: FamaFrench3Result) -> str:
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
    ]

    lines = [
        "=" * 78,
        "Fama-French 3-Factor Model",
        "  R_i - R_f = alpha + b_MKT(Mkt-RF) + b_SMB(SMB) + b_HML(HML) + e",
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
    lines.extend(_ff3_coef_row(label, c) for label, c in rows)
    lines.extend(
        [
            "-" * 78,
            "Style interpretation",
            f"  Annualized alpha:  {result.annualized_alpha:>10.4%}    "
            f"H0 alpha=0: t={result.alpha.t_statistic:>7.3f} "
            f"p={result.alpha.p_value:.4f} {result.alpha.significance_stars()}",
            f"  Size tilt:  {result.size_tilt:<10}  "
            f"Value tilt: {result.value_tilt}",
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


def _ff3_coef_row(label: str, c: CoefficientEstimate) -> str:
    return (
        f"{label:<20}{c.estimate:>12.6f}{c.std_error:>12.6f}"
        f"{c.t_statistic:>10.4f}{c.p_value:>10.4f}"
        f"{c.conf_int_lower:>9.4f}{c.conf_int_upper:>9.4f} {c.significance_stars()}"
    )
