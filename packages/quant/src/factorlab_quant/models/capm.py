r"""Capital Asset Pricing Model (CAPM) — the one-factor special case.

Theory
------
The CAPM (Sharpe, 1964; Lintner, 1965; Mossin, 1966) states that, in
equilibrium, the expected excess return on any asset is proportional to its
covariance with the market portfolio:

.. math::

    \mathbb{E}[R_i] - R_f = \beta_i \,\bigl(\mathbb{E}[R_m] - R_f\bigr),
    \qquad
    \beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)}.

Empirical specification (Jensen, 1968) — a time-series regression of the asset's
excess return on the market's excess return:

.. math::

    R_{i,t} - R_{f,t}
        = \alpha_i + \beta_i \,(R_{m,t} - R_{f,t}) + \varepsilon_{i,t}.

Architectural role
------------------
CAPM is the **one-factor** instance of
:class:`~factorlab_quant.models.linear_factor_model.LinearFactorModel`: its only
factor is ``Mkt-RF``.  All estimation, inference, diagnostics, prediction and
serialization are inherited; this module adds only (a) the excess-return
construction specific to CAPM and (b) the finance-specific convenience layer on
the result (Jensen's alpha, the ``H0: beta = 1`` test, Treynor ratio, and the
systematic/idiosyncratic variance decomposition).

References
----------
Sharpe, W. F. (1964). *Journal of Finance* 19(3), 425--442.
Lintner, J. (1965). *Review of Economics and Statistics* 47(1), 13--37.
Jensen, M. C. (1968). *Journal of Finance* 23(2), 389--416.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import stats

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

__all__ = ["CAPM", "CAPMResult"]

_MARKET = "Mkt-RF"
_ALPHA = "alpha"


@dataclass(frozen=True, slots=True)
class CAPMResult(FactorModelResult):
    """CAPM fit: a :class:`FactorModelResult` plus CAPM-specific statistics.

    Inherits the entire generic result API (coefficients, diagnostics,
    prediction, serialization) and adds Jensen's-alpha framing, the market-beta
    hypothesis test, and the systematic/idiosyncratic risk decomposition.
    """

    mean_asset_excess: float
    mean_market_excess: float
    beta_t_vs_one: float
    beta_p_vs_one: float

    # -- Core coefficients -------------------------------------------------
    @property
    def alpha(self) -> CoefficientEstimate:
        """Jensen's alpha (per period), with HAC-based inference."""
        return self.intercept

    @property
    def beta(self) -> CoefficientEstimate:
        """Market beta (systematic risk exposure)."""
        return self.factor_loading(_MARKET)

    # -- Annualization -----------------------------------------------------
    @property
    def annualized_alpha(self) -> float:
        r"""Geometrically annualized Jensen's alpha,
        :math:`(1 + \hat\alpha)^{P} - 1`, where ``P = periods_per_year``.
        """
        return float((1.0 + self.alpha.estimate) ** self.periods_per_year - 1.0)

    # -- Risk decomposition ------------------------------------------------
    @property
    def systematic_variance_ratio(self) -> float:
        r""":math:`R^2`: share of the asset's excess-return variance explained by
        the market factor (systematic risk)."""
        return self.r_squared

    @property
    def idiosyncratic_variance_ratio(self) -> float:
        """Complement of :attr:`systematic_variance_ratio` (diversifiable risk)."""
        return 1.0 - self.r_squared

    @property
    def idiosyncratic_volatility(self) -> float:
        r"""Per-period idiosyncratic volatility :math:`\hat\sigma_\varepsilon`."""
        return self.regression.residual_std_error

    @property
    def annualized_idiosyncratic_volatility(self) -> float:
        r"""Idiosyncratic volatility scaled by :math:`\sqrt{P}`."""
        return float(self.idiosyncratic_volatility * np.sqrt(self.periods_per_year))

    # -- Performance ratios ------------------------------------------------
    @property
    def treynor_ratio(self) -> float:
        r"""Per-period Treynor ratio :math:`\bar R^e_i / \hat\beta_i`;
        ``nan`` when beta is (near) zero."""
        b = self.beta.estimate
        if abs(b) < 1e-12:
            return float("nan")
        return float(self.mean_asset_excess / b)

    def summary(self) -> str:
        """A CAPM-specific formatted report (overrides the generic summary)."""
        return _format_capm_summary(self)

    # -- Serialization overrides ------------------------------------------
    def to_dict(self) -> dict[str, object]:
        # Explicit parent call: zero-arg super() is unreliable inside a
        # slots=True dataclass (the class is recreated, orphaning __class__).
        data = FactorModelResult.to_dict(self)
        data.update(
            mean_asset_excess=self.mean_asset_excess,
            mean_market_excess=self.mean_market_excess,
            beta_t_vs_one=self.beta_t_vs_one,
            beta_p_vs_one=self.beta_p_vs_one,
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CAPMResult:
        base = cls._base_kwargs_from_dict(data)
        return cls(
            **base,
            mean_asset_excess=float(data["mean_asset_excess"]),
            mean_market_excess=float(data["mean_market_excess"]),
            beta_t_vs_one=float(data["beta_t_vs_one"]),
            beta_p_vs_one=float(data["beta_p_vs_one"]),
        )


@register_model("CAPM")
class CAPM(LinearFactorModel):
    """Capital Asset Pricing Model estimator (one-factor: ``Mkt-RF``).

    Examples
    --------
    >>> import numpy as np
    >>> from factorlab_quant import CAPM
    >>> rng = np.random.default_rng(0)
    >>> mkt = rng.normal(0.01, 0.04, size=240)
    >>> asset = 0.001 + 1.2 * mkt + rng.normal(0, 0.02, size=240)
    >>> result = CAPM().fit(asset, mkt, returns_are_excess=True)
    >>> round(result.beta.estimate, 1)
    1.2
    """

    def __init__(self, estimator: OLS | None = None) -> None:
        super().__init__(
            name="Capital Asset Pricing Model",
            factor_names=(_MARKET,),
            estimator=estimator,
            intercept=True,
            intercept_name=_ALPHA,
            response_name="asset_excess_return",
            metadata={"family": "linear_factor_model", "n_factors": 1},
        )

    def fit(  # type: ignore[override]
        self,
        asset_returns: FloatArray,
        market_returns: FloatArray,
        risk_free: FloatArray | float | None = None,
        *,
        returns_are_excess: bool = False,
        covariance_type: CovarianceType = "HAC",
        conf_level: float = 0.95,
        hac_lags: int | None = None,
        small_sample_correction: bool = True,
        use_t: bool | None = None,
        periods_per_year: int = 12,
    ) -> CAPMResult:
        r"""Estimate the CAPM time-series regression.

        Parameters mirror the previous CAPM API for backward compatibility.
        ``asset_returns``/``market_returns`` may be raw returns (supply
        ``risk_free``) or already-excess returns (``returns_are_excess=True``).
        See :meth:`LinearFactorModel.fit_factor_set` for the covariance and
        inference options.

        Returns
        -------
        CAPMResult
        """
        if returns_are_excess and risk_free is not None:
            raise ValueError(
                "risk_free must be None when returns_are_excess=True; the inputs "
                "are assumed to already be excess returns."
            )

        asset = as_float_vector(asset_returns, name="asset_returns")
        market = as_float_vector(market_returns, name="market_returns")
        check_lengths_match(("asset_returns", asset), ("market_returns", market))

        asset_excess = asset if returns_are_excess else self._to_excess(asset, risk_free)
        market_excess = market if returns_are_excess else self._to_excess(market, risk_free)

        factor = Factor(
            name=_MARKET,
            values=market_excess,
            display_name="Market excess return (Mkt-RF)",
            description="Excess return of the market portfolio over the risk-free rate.",
        )
        factor_set = FactorSet([factor])

        result = self.fit_factor_set(
            asset_excess,
            factor_set,
            covariance_type=covariance_type,
            conf_level=conf_level,
            hac_lags=hac_lags,
            small_sample_correction=small_sample_correction,
            use_t=use_t,
            periods_per_year=periods_per_year,
            extra_metadata={"specification": "Jensen (1968) time-series CAPM"},
        )
        return cast(CAPMResult, result)

    # ------------------------------------------------------------------ #
    # Result construction hook                                           #
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
    ) -> CAPMResult:
        market_idx = param_names.index(_MARKET)
        beta = regression.coefficient(_MARKET)
        beta_t, beta_p = self._test_beta_equals_one(
            beta, regression.degrees_of_freedom, uses_t_distribution
        )
        return CAPMResult(
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
            mean_asset_excess=float(np.mean(response)),
            mean_market_excess=float(np.mean(design[:, market_idx])),
            beta_t_vs_one=beta_t,
            beta_p_vs_one=beta_p,
        )

    @staticmethod
    def _test_beta_equals_one(
        beta: CoefficientEstimate, dof: int, uses_t: bool
    ) -> tuple[float, float]:
        r"""Two-sided test of :math:`H_0: \beta = 1`, using the same reference
        distribution as the estimator's own inference."""
        se = beta.std_error
        if se == 0.0 or not np.isfinite(se):
            return float("nan"), float("nan")
        t_stat = (beta.estimate - 1.0) / se
        if uses_t:
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df=dof))
        else:
            p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
        return float(t_stat), p_value


# ---------------------------------------------------------------------------- #
# Presentation                                                                 #
# ---------------------------------------------------------------------------- #
def _format_capm_summary(result: CAPMResult) -> str:
    reg = result.regression
    diag = reg.diagnostics
    a = result.alpha
    b = result.beta
    cov_desc: str = reg.covariance_type
    if reg.covariance_type == "HAC":
        cov_desc = f"HAC (Newey-West, {reg.cov_config.get('lags')} lags, Bartlett)"

    lines = [
        "=" * 72,
        "Capital Asset Pricing Model  —  R_i - R_f = alpha + beta (R_m - R_f) + e",
        "=" * 72,
        f"Observations: {reg.n_observations:>8d}    "
        f"Resid. DoF: {reg.degrees_of_freedom:>6d}    "
        f"Cov: {cov_desc}",
        f"R-squared:    {diag.r_squared:>8.4f}    "
        f"Adj. R-sq:  {diag.adj_r_squared:>6.4f}    "
        f"F p-value: {diag.f_p_value:>.4f}",
        "-" * 72,
        f"{'coef':<14}{'estimate':>12}{'std err':>12}{'t':>10}{'P>|t|':>10}",
        "-" * 72,
        _capm_coef_row("alpha", a),
        _capm_coef_row(f"beta ({_MARKET})", b),
        "-" * 72,
        "Hypothesis tests",
        f"  H0: alpha = 0  ->  t = {a.t_statistic:>8.4f}   p = {a.p_value:>.4f}  "
        f"{a.significance_stars()}",
        f"  H0: beta  = 1  ->  t = {result.beta_t_vs_one:>8.4f}   "
        f"p = {result.beta_p_vs_one:>.4f}",
        "-" * 72,
        "Risk & performance",
        f"  Annualized alpha:            {result.annualized_alpha:>12.4%}",
        f"  Systematic variance (R^2):   {result.systematic_variance_ratio:>12.4f}",
        f"  Idiosyncratic vol (annual):  {result.annualized_idiosyncratic_volatility:>12.4f}",
        f"  Treynor ratio (per period):  {result.treynor_ratio:>12.4f}",
        "-" * 72,
        "Diagnostics",
        f"  Durbin-Watson:   {diag.durbin_watson:>8.4f}",
        f"  Jarque-Bera:     {diag.jarque_bera:>8.4f}  (p = {diag.jarque_bera_p_value:.4f})",
        f"  Breusch-Pagan:   {diag.breusch_pagan:>8.4f}  (p = {diag.breusch_pagan_p_value:.4f})",
        f"  Skewness:        {diag.skewness:>8.4f}   Excess kurtosis: {diag.excess_kurtosis:.4f}",
        f"  Cond. number:    {diag.condition_number:>8.2f}",
        "=" * 72,
        "Significance:  *** p<0.01   ** p<0.05   * p<0.10",
    ]
    return "\n".join(lines)


def _capm_coef_row(label: str, c: CoefficientEstimate) -> str:
    return (
        f"{label:<14}{c.estimate:>12.6f}{c.std_error:>12.6f}"
        f"{c.t_statistic:>10.4f}{c.p_value:>10.4f} {c.significance_stars()}"
    )
