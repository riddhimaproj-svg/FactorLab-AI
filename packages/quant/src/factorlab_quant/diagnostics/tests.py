r"""Specification hypothesis tests for fitted regressions.

Each function returns a ``(statistic, p_value)`` pair (plus any auxiliary
moments) and is written to match the conventions of ``statsmodels`` so results
are directly cross-checkable.  The tests answer three questions a reviewer will
always ask of an asset-pricing regression:

* **Are the residuals normal?**            -> Jarque--Bera
* **Are the residuals homoskedastic?**     -> Breusch--Pagan (Koenker LM form)
* **Is the model jointly significant?**    -> F-test

References
----------
Jarque, C. M., & Bera, A. K. (1980). "Efficient tests for normality,
    homoscedasticity and serial independence of regression residuals."
    *Economics Letters* 6(3), 255--259.
Breusch, T. S., & Pagan, A. R. (1979). "A Simple Test for Heteroscedasticity
    and Random Coefficient Variation." *Econometrica* 47(5), 1287--1294.
Koenker, R. (1981). "A note on studentizing a test for heteroscedasticity."
    *Journal of Econometrics* 17(1), 107--112.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from factorlab_quant.core.types import FloatArray
from factorlab_quant.diagnostics.residuals import (
    sample_excess_kurtosis,
    sample_skewness,
)

__all__ = ["breusch_pagan", "f_test", "jarque_bera"]


def jarque_bera(residuals: FloatArray) -> tuple[float, float, float, float]:
    r"""Jarque--Bera test of residual normality.

    .. math::

        JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right),

    where :math:`S` is the skewness and :math:`K` the kurtosis.  Under the null
    of normality, :math:`JB \sim \chi^2_2` asymptotically.

    Returns
    -------
    (statistic, p_value, skewness, excess_kurtosis)
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    n = residuals.shape[0]
    skew = sample_skewness(residuals)
    excess_kurt = sample_excess_kurtosis(residuals)
    statistic = (n / 6.0) * (skew**2 + (excess_kurt**2) / 4.0)
    p_value = float(stats.chi2.sf(statistic, df=2))
    return float(statistic), p_value, skew, excess_kurt


def breusch_pagan(residuals: FloatArray, design: FloatArray) -> tuple[float, float]:
    r"""Breusch--Pagan test for heteroskedasticity (Koenker's studentized LM).

    Regresses the squared residuals on the original design matrix and forms the
    Lagrange-multiplier statistic :math:`LM = n R^2_{\text{aux}}`, which under
    the null of homoskedasticity is distributed :math:`\chi^2_{p}` with
    :math:`p` equal to the number of regressors excluding the intercept.  This
    ``n R^2`` form (Koenker, 1981) is robust to non-normal residuals and
    matches ``statsmodels.stats.diagnostic.het_breuschpagan``.

    Parameters
    ----------
    residuals:
        OLS residuals, length ``n``.
    design:
        The ``n x k`` design matrix used in the original regression, including
        its intercept column.

    Returns
    -------
    (lm_statistic, p_value)
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    n, k = design.shape

    if k < 2:
        # No non-constant regressors: the test is undefined.
        return float("nan"), float("nan")

    g = residuals**2
    # Auxiliary regression of squared residuals on the design matrix.
    beta_aux, _, _, _ = np.linalg.lstsq(design, g, rcond=None)
    fitted = design @ beta_aux
    ss_res = float(np.sum((g - fitted) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    r_squared_aux = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    lm_statistic = n * r_squared_aux
    df = k - 1
    p_value = float(stats.chi2.sf(lm_statistic, df=df))
    return float(lm_statistic), p_value


def f_test(
    r_squared: float, n_obs: int, n_params: int
) -> tuple[float, float]:
    r"""Joint F-test that all slope coefficients are zero.

    .. math::

        F = \frac{R^2 / (k - 1)}{(1 - R^2) / (n - k)}
          \sim F_{k-1,\; n-k}.

    Returns ``(nan, nan)`` for an intercept-only model, where the test is
    undefined.
    """
    df_model = n_params - 1
    df_resid = n_obs - n_params
    if df_model <= 0 or df_resid <= 0:
        return float("nan"), float("nan")
    denom = (1.0 - r_squared) / df_resid
    if denom <= 0.0:
        # Perfect fit: infinite F, zero p-value.
        return float("inf"), 0.0
    statistic = (r_squared / df_model) / denom
    p_value = float(stats.f.sf(statistic, df_model, df_resid))
    return float(statistic), p_value
