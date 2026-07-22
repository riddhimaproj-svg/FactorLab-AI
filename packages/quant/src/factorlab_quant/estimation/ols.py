r"""Ordinary Least Squares estimator with robust inference.

This is the single estimation engine behind every factor model in the package.
It solves

.. math::

    y = X\beta + \varepsilon, \qquad
    \hat\beta = (X'X)^{-1}X'y,

using a numerically stable least-squares solver, then attaches a full
inferential and diagnostic apparatus:

* classical, White (HC0/HC1), or Newey--West (HAC) standard errors;
* t- or z-based p-values and confidence intervals;
* Gaussian log-likelihood, AIC and BIC;
* residual normality, autocorrelation, and heteroskedasticity diagnostics.

The estimator is a pure function of its inputs -- no global state, no I/O -- and
conforms to :class:`factorlab_quant.core.protocols.Estimator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from factorlab_quant.core.errors import CollinearityError
from factorlab_quant.core.types import (
    CoefficientEstimate,
    CovarianceType,
    FloatArray,
    RegressionDiagnostics,
    RegressionResult,
)
from factorlab_quant.diagnostics.residuals import durbin_watson
from factorlab_quant.diagnostics.tests import breusch_pagan, f_test, jarque_bera
from factorlab_quant.estimation.hac import newey_west_covariance, white_covariance
from factorlab_quant.utils.validation import (
    as_float_matrix,
    as_float_vector,
    check_finite,
    check_lengths_match,
    check_min_observations,
)

__all__ = ["OLS"]


@dataclass(frozen=True, slots=True)
class OLS:
    """Configurable OLS estimator.

    Parameters
    ----------
    condition_threshold:
        Maximum tolerated 2-norm condition number of the design matrix before a
        :class:`CollinearityError` is raised.  ``1e10`` corresponds to losing
        roughly ten significant digits and is a conservative default.
    min_observations:
        Minimum number of observations required regardless of parameter count.
        Guards against inference on samples too small to be meaningful.
    """

    condition_threshold: float = 1e10
    min_observations: int = 3

    def fit(
        self,
        y: FloatArray,
        X: FloatArray,
        *,
        param_names: tuple[str, ...] | None = None,
        covariance_type: CovarianceType = "HAC",
        conf_level: float = 0.95,
        hac_lags: int | None = None,
        small_sample_correction: bool = True,
        use_t: bool | None = None,
    ) -> RegressionResult:
        r"""Estimate ``y = X @ beta + e`` and return a full result.

        Parameters
        ----------
        y:
            Response vector, length ``n``.
        X:
            Design matrix ``n x k``.  The intercept, if desired, must already be
            included as a column -- the estimator does not add one implicitly,
            keeping the model specification explicit and auditable.
        param_names:
            Names for the ``k`` columns of ``X``.  Defaults to ``x0, x1, ...``.
        covariance_type:
            One of ``"nonrobust"``, ``"HC0"``, ``"HC1"``, ``"HAC"``.
        conf_level:
            Confidence level for the reported intervals (e.g. ``0.95``).
        hac_lags:
            Truncation lag for the HAC estimator; ``None`` selects the automatic
            Newey--West (1994) bandwidth.  Ignored unless ``covariance_type`` is
            ``"HAC"``.
        small_sample_correction:
            Apply the ``n / (n - k)`` degrees-of-freedom correction to robust
            covariances (HC1 vs HC0; corrected vs uncorrected HAC).
        use_t:
            Use the Student-t reference distribution for p-values and intervals.
            ``None`` (default) selects t for ``nonrobust`` and the standard
            normal for robust covariances, matching common econometric practice.

        Raises
        ------
        InsufficientDataError, DimensionMismatchError, NonFiniteError
            On malformed input.
        CollinearityError
            When the design matrix is numerically rank-deficient.
        """
        y = as_float_vector(y, name="y")
        X = as_float_matrix(X, name="X")
        check_lengths_match(("y", y), ("X", X))
        check_finite(y, name="y")
        check_finite(X, name="X")

        n, k = X.shape
        check_min_observations(n_obs=n, n_params=k, minimum=self.min_observations)

        if not 0.0 < conf_level < 1.0:
            raise ValueError("conf_level must lie strictly in (0, 1)")

        names = param_names if param_names is not None else tuple(f"x{i}" for i in range(k))
        if len(names) != k:
            raise ValueError(f"param_names has {len(names)} entries but X has {k} columns")

        condition_number = float(np.linalg.cond(X))
        if not np.isfinite(condition_number) or condition_number > self.condition_threshold:
            raise CollinearityError(condition_number, self.condition_threshold)

        # --- Point estimation (SVD-based least squares for stability) ---------
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        residuals = y - fitted

        dof_resid = n - k
        ss_res = float(residuals @ residuals)
        y_centered = y - y.mean()
        ss_tot = float(y_centered @ y_centered)
        r_squared = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        adj_r_squared = (
            r_squared
            if n == k
            else 1.0 - (1.0 - r_squared) * (n - 1) / dof_resid
        )
        sigma2 = ss_res / dof_resid
        residual_std_error = float(np.sqrt(sigma2))

        # --- Coefficient covariance ------------------------------------------
        xtx_inv = np.linalg.inv(X.T @ X)
        cov, cov_config = self._covariance(
            covariance_type=covariance_type,
            X=X,
            residuals=residuals,
            xtx_inv=xtx_inv,
            sigma2=sigma2,
            hac_lags=hac_lags,
            small_sample_correction=small_sample_correction,
        )
        std_errors = np.sqrt(np.diag(cov))

        # --- Inference --------------------------------------------------------
        resolved_use_t = (covariance_type == "nonrobust") if use_t is None else use_t
        coefficients = self._build_coefficients(
            names=names,
            beta=beta,
            std_errors=std_errors,
            dof_resid=dof_resid,
            conf_level=conf_level,
            use_t=resolved_use_t,
        )

        diagnostics = self._build_diagnostics(
            y=y,
            X=X,
            residuals=residuals,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            n=n,
            k=k,
            ss_res=ss_res,
            condition_number=condition_number,
        )

        return RegressionResult(
            coefficients=coefficients,
            n_observations=n,
            n_parameters=k,
            degrees_of_freedom=dof_resid,
            residual_std_error=residual_std_error,
            fitted_values=fitted,
            residuals=residuals,
            covariance_matrix=cov,
            covariance_type=covariance_type,
            diagnostics=diagnostics,
            cov_config=cov_config,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _covariance(
        self,
        *,
        covariance_type: CovarianceType,
        X: FloatArray,
        residuals: FloatArray,
        xtx_inv: FloatArray,
        sigma2: float,
        hac_lags: int | None,
        small_sample_correction: bool,
    ) -> tuple[FloatArray, dict[str, object]]:
        if covariance_type == "nonrobust":
            return sigma2 * xtx_inv, {}
        if covariance_type == "HC0":
            return white_covariance(X, residuals, xtx_inv, small_sample_correction=False), {}
        if covariance_type == "HC1":
            return white_covariance(X, residuals, xtx_inv, small_sample_correction=True), {}
        if covariance_type == "HAC":
            cov, lags_used = newey_west_covariance(
                X,
                residuals,
                xtx_inv,
                lags=hac_lags,
                small_sample_correction=small_sample_correction,
            )
            return cov, {"kernel": "bartlett", "lags": lags_used}
        raise ValueError(f"Unknown covariance_type {covariance_type!r}")

    @staticmethod
    def _build_coefficients(
        *,
        names: tuple[str, ...],
        beta: FloatArray,
        std_errors: FloatArray,
        dof_resid: int,
        conf_level: float,
        use_t: bool,
    ) -> tuple[CoefficientEstimate, ...]:
        alpha = 1.0 - conf_level
        if use_t:
            crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof_resid))
        else:
            crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

        estimates: list[CoefficientEstimate] = []
        for name, b, se in zip(names, beta, std_errors, strict=True):
            if se == 0.0 or not np.isfinite(se):
                t_stat = float("nan")
                p_value = float("nan")
                lower = upper = float("nan")
            else:
                t_stat = float(b / se)
                if use_t:
                    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=dof_resid))
                else:
                    p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
                lower = float(b - crit * se)
                upper = float(b + crit * se)
            estimates.append(
                CoefficientEstimate(
                    name=name,
                    estimate=float(b),
                    std_error=float(se),
                    t_statistic=t_stat,
                    p_value=p_value,
                    conf_int_lower=lower,
                    conf_int_upper=upper,
                    conf_level=conf_level,
                )
            )
        return tuple(estimates)

    @staticmethod
    def _build_diagnostics(
        *,
        y: FloatArray,
        X: FloatArray,
        residuals: FloatArray,
        r_squared: float,
        adj_r_squared: float,
        n: int,
        k: int,
        ss_res: float,
        condition_number: float,
    ) -> RegressionDiagnostics:
        f_stat, f_p = f_test(r_squared, n_obs=n, n_params=k)
        jb, jb_p, skew, excess_kurt = jarque_bera(residuals)
        bp, bp_p = breusch_pagan(residuals, X)
        dw = durbin_watson(residuals)

        # Gaussian log-likelihood at the MLE variance, with AIC/BIC.
        sigma2_mle = ss_res / n
        if sigma2_mle <= 0.0:
            log_likelihood = float("inf")
        else:
            log_likelihood = -0.5 * n * (np.log(2.0 * np.pi) + np.log(sigma2_mle) + 1.0)
        aic = -2.0 * log_likelihood + 2.0 * k
        bic = -2.0 * log_likelihood + k * np.log(n)

        return RegressionDiagnostics(
            r_squared=float(r_squared),
            adj_r_squared=float(adj_r_squared),
            f_statistic=float(f_stat),
            f_p_value=float(f_p),
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            durbin_watson=float(dw),
            jarque_bera=float(jb),
            jarque_bera_p_value=float(jb_p),
            breusch_pagan=float(bp),
            breusch_pagan_p_value=float(bp_p),
            skewness=float(skew),
            excess_kurtosis=float(excess_kurt),
            condition_number=float(condition_number),
        )
