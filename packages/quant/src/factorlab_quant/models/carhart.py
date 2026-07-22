r"""Carhart Four-Factor Model (1997).

Why Carhart extends FF3
-----------------------
The Fama--French three-factor model does not explain the *momentum* anomaly:
stocks that performed well over the past ~3--12 months (winners) continue to
outperform stocks that performed poorly (losers) over the following months
(Jegadeesh & Titman, 1993).  FF3 loadings leave this pattern in alpha.  Carhart
(1997) adds a momentum factor to FF3, giving a four-factor model widely used to
evaluate fund performance -- a fund's "alpha" after controlling for market,
size, value, **and** momentum exposure is a much stronger claim of skill.

Carhart nests FF3: set the momentum loading to zero and the three-factor model
is recovered.

Economic intuition of momentum (MOM / UMD / WML)
------------------------------------------------
* **MOM** ("Momentum", also called UMD = Up Minus Down, or WML = Winners Minus
  Losers) is the return of a portfolio long recent winners and short recent
  losers.  A positive momentum loading means the asset behaves like a
  trend-following / winner portfolio; a negative loading means it behaves like a
  contrarian / loser portfolio.
* Candidate explanations include under-reaction to news, delayed overreaction,
  and behavioral herding.  Whether momentum is compensation for risk or a
  behavioral mispricing remains debated.

Momentum crashes
----------------
Momentum earns high average returns but exhibits **crash risk**: in sharp market
rebounds following large declines (e.g. 1932, 2009), the short-loser leg rallies
violently and the momentum factor suffers severe, sudden losses (Daniel &
Moskowitz, 2016).  Momentum returns are negatively skewed and fat-tailed, so a
positive momentum loading carries left-tail exposure that volatility alone
understates -- something to read alongside the residual skew/kurtosis diagnostics
this framework already reports.

Mathematical specification
--------------------------
.. math::

    R_{i,t} - R_{f,t}
        = \alpha_i
        + \beta_{i}^{\mathrm{MKT}} (R_{m,t} - R_{f,t})
        + \beta_{i}^{\mathrm{SMB}} \mathrm{SMB}_t
        + \beta_{i}^{\mathrm{HML}} \mathrm{HML}_t
        + \beta_{i}^{\mathrm{MOM}} \mathrm{MOM}_t
        + \varepsilon_{i,t}.

Estimation is OLS with Newey--West (HAC) inference by default -- inherited
unchanged from :class:`LinearFactorModel`.  Carhart adds no estimation code.

Data flow
---------
The momentum factor comes from the factor-data layer: the Kenneth French adapter
already exposes ``F-F_Momentum_Factor`` (monthly) and ``F-F_Momentum_Factor_daily``.
A caller aligns the FF3 factor panel with the momentum panel (via the layer's
``FactorAlignment``) and hands the combined factor set to this model.  The
library's momentum column is named ``Mom``; this model accepts that alias (and
``UMD`` / ``WML``) and normalizes it to ``MOM`` -- alias handling lives here, not
in the generic adapter.

References
----------
Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance."
    *Journal of Finance* 52(1), 57--82.
Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling
    Losers." *Journal of Finance* 48(1), 65--91.
Daniel, K., & Moskowitz, T. J. (2016). "Momentum Crashes."
    *Journal of Financial Economics* 122(2), 221--247.
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
from factorlab_quant.models.factors import Factor, FactorSet
from factorlab_quant.models.linear_factor_model import FactorModelResult, LinearFactorModel
from factorlab_quant.models.registry import register_model
from factorlab_quant.utils.validation import as_float_vector

__all__ = ["CarhartModel", "CarhartResult"]

_MARKET = "Mkt-RF"
_SMB = "SMB"
_HML = "HML"
_MOM = "MOM"
_ALPHA = "alpha"
_FACTORS = (_MARKET, _SMB, _HML, _MOM)

# Common names for the momentum factor across data sources, normalized to MOM.
_MOMENTUM_ALIASES = {"MOM", "Mom", "MOMENTUM", "Momentum", "UMD", "WML"}


@dataclass(frozen=True, slots=True)
class CarhartResult(FactorModelResult):
    """Fitted Carhart four-factor model.

    Adds named, finance-specific accessors on top of the generic
    :class:`FactorModelResult`.  Introduces no new stored fields, so it inherits
    the base serialization unchanged.
    """

    @property
    def alpha(self) -> CoefficientEstimate:
        """Four-factor alpha (per period), with HAC-based inference."""
        return self.intercept

    @property
    def market_beta(self) -> CoefficientEstimate:
        return self.factor_loading(_MARKET)

    @property
    def size_loading(self) -> CoefficientEstimate:
        """SMB loading -- size tilt (positive => small-cap-like)."""
        return self.factor_loading(_SMB)

    @property
    def value_loading(self) -> CoefficientEstimate:
        """HML loading -- value tilt (positive => value-like)."""
        return self.factor_loading(_HML)

    @property
    def momentum_loading(self) -> CoefficientEstimate:
        """MOM loading -- momentum tilt (positive => winner/trend-like)."""
        return self.factor_loading(_MOM)

    @property
    def annualized_alpha(self) -> float:
        r"""Geometrically annualized alpha, :math:`(1 + \hat\alpha)^{P} - 1`."""
        return float((1.0 + self.alpha.estimate) ** self.periods_per_year - 1.0)

    @property
    def size_tilt(self) -> str:
        return "small-cap" if self.size_loading.estimate > 0.0 else "large-cap"

    @property
    def value_tilt(self) -> str:
        return "value" if self.value_loading.estimate > 0.0 else "growth"

    @property
    def momentum_tilt(self) -> str:
        return "winner" if self.momentum_loading.estimate > 0.0 else "loser"

    def summary(self) -> str:
        return _format_carhart_summary(self)


@register_model("Carhart4")
@register_model("Carhart")
@register_model("MomentumFactorModel")
class CarhartModel(LinearFactorModel):
    """Carhart four-factor model estimator (``Mkt-RF``, ``SMB``, ``HML``, ``MOM``).

    A thin subclass of :class:`LinearFactorModel` that consumes factor data
    through the factor-data layer (any object exposing ``to_factor_set()``), a
    ``FactorSet``, or a ``{name: values}`` mapping.  The momentum factor may be
    supplied under any of ``MOM`` / ``Mom`` / ``UMD`` / ``WML``.

    Examples
    --------
    >>> import numpy as np
    >>> from factorlab_quant.models.carhart import CarhartModel
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> cols = {k: rng.normal(0, 0.03, n) for k in ("Mkt-RF", "SMB", "HML", "Mom")}
    >>> asset = 0.001 + 1.0*cols["Mkt-RF"] + 0.5*cols["Mom"] + rng.normal(0, 0.01, n)
    >>> res = CarhartModel().fit(asset, cols, returns_are_excess=True)
    >>> round(res.momentum_loading.estimate, 1)
    0.5
    """

    def __init__(self, estimator: OLS | None = None) -> None:
        super().__init__(
            name="Carhart 4-Factor Model",
            factor_names=_FACTORS,
            estimator=estimator,
            intercept=True,
            intercept_name=_ALPHA,
            response_name="asset_excess_return",
            metadata={
                "family": "linear_factor_model",
                "n_factors": 4,
                "reference": "Carhart (1997)",
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
    ) -> CarhartResult:
        r"""Estimate the Carhart four-factor regression.

        Parameters
        ----------
        asset_returns:
            The asset (or portfolio) return series -- the dependent variable.
        factors:
            The four factors, supplied through the factor-data layer.  Accepts
            any object exposing ``to_factor_set()`` (e.g. a
            :class:`factorlab_data.FactorPanel`), a :class:`FactorSet`, or a
            ``{name: values}`` mapping.  Must contain ``Mkt-RF``, ``SMB``,
            ``HML``, and a momentum factor (``MOM``/``Mom``/``UMD``/``WML``).
        risk_free:
            Per-period risk-free rate used only to convert ``asset_returns`` to
            excess form.  ``None`` when ``returns_are_excess=True``.
        returns_are_excess, covariance_type, conf_level, hac_lags,
        small_sample_correction, use_t, periods_per_year,
        reject_duplicate_observations:
            As documented on :meth:`LinearFactorModel.fit_factor_set`; inference
            defaults to Newey--West ``"HAC"``.

        Returns
        -------
        CarhartResult
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
            extra_metadata={"specification": "Carhart (1997) four-factor"},
        )
        return cast(CarhartResult, result)

    @staticmethod
    def _resolve_factor_set(factors: object) -> FactorSet:
        """Coerce a factors source into a :class:`FactorSet`, normalizing the
        momentum factor's name to ``MOM``.

        The coercion (FactorSet / ``to_factor_set()`` / mapping) mirrors the
        other multi-factor models; the momentum-alias normalization is the only
        Carhart-specific step, and it lives here rather than in the generic
        adapter.
        """
        if isinstance(factors, FactorSet):
            factor_set = factors
        else:
            to_factor_set = getattr(factors, "to_factor_set", None)
            if callable(to_factor_set):
                produced = to_factor_set()
                if not isinstance(produced, FactorSet):
                    raise TypeError(
                        "to_factor_set() must return a FactorSet, got "
                        f"{type(produced).__name__}"
                    )
                factor_set = produced
            elif isinstance(factors, Mapping):
                factor_set = FactorSet.from_mapping(factors)
            else:
                raise TypeError(
                    "factors must be a FactorSet, a mapping, or an object exposing "
                    f"to_factor_set(); got {type(factors).__name__}"
                )
        return _normalize_momentum(factor_set)

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
    ) -> CarhartResult:
        return CarhartResult(
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


def _normalize_momentum(factor_set: FactorSet) -> FactorSet:
    """Rename an aliased momentum factor to ``MOM`` (idempotent)."""
    if _MOM in factor_set.names:
        return factor_set
    rebuilt: list[Factor] = []
    renamed = False
    for factor in factor_set:
        if not renamed and factor.name in _MOMENTUM_ALIASES:
            rebuilt.append(
                Factor(
                    name=_MOM,
                    values=factor.values,
                    display_name=factor.display_name or factor.name,
                    frequency=factor.frequency,
                    source=factor.source,
                    description=factor.description,
                )
            )
            renamed = True
        else:
            rebuilt.append(factor)
    return FactorSet(rebuilt)


# ---------------------------------------------------------------------------- #
# Presentation                                                                 #
# ---------------------------------------------------------------------------- #
def _format_carhart_summary(result: CarhartResult) -> str:
    reg = result.regression
    diag = reg.diagnostics
    cov_desc: str = reg.covariance_type
    if reg.covariance_type == "HAC":
        cov_desc = f"HAC (Newey-West, {reg.cov_config.get('lags')} lags, Bartlett)"

    rows = [
        ("alpha", result.alpha),
        ("beta_MKT (Mkt-RF)", result.market_beta),
        ("beta_SMB (SMB)", result.size_loading),
        ("beta_HML (HML)", result.value_loading),
        ("beta_MOM (MOM)", result.momentum_loading),
    ]

    lines = [
        "=" * 78,
        "Carhart 4-Factor Model",
        "  R_i - R_f = alpha + b_MKT(Mkt-RF) + b_SMB(SMB) + b_HML(HML) + b_MOM(MOM) + e",
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
    lines.extend(_carhart_coef_row(label, c) for label, c in rows)
    lines.extend(
        [
            "-" * 78,
            "Style interpretation",
            f"  Annualized alpha:  {result.annualized_alpha:>10.4%}    "
            f"H0 alpha=0: t={result.alpha.t_statistic:>7.3f} "
            f"p={result.alpha.p_value:.4f} {result.alpha.significance_stars()}",
            f"  Size: {result.size_tilt:<10} Value: {result.value_tilt:<8} "
            f"Momentum: {result.momentum_tilt}",
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


def _carhart_coef_row(label: str, c: CoefficientEstimate) -> str:
    return (
        f"{label:<20}{c.estimate:>12.6f}{c.std_error:>12.6f}"
        f"{c.t_statistic:>10.4f}{c.p_value:>10.4f}"
        f"{c.conf_int_lower:>9.4f}{c.conf_int_upper:>9.4f} {c.significance_stars()}"
    )
