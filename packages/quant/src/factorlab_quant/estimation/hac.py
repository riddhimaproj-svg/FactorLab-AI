r"""Robust covariance estimators for OLS coefficients.

This module implements the *sandwich* covariance estimators used throughout
empirical asset pricing:

.. math::

    \widehat{\operatorname{Var}}(\hat\beta)
        = (X'X)^{-1}\, \hat S \, (X'X)^{-1},

where the "bread" :math:`(X'X)^{-1}` is fixed by OLS and the "meat"
:math:`\hat S` differs by estimator.

* **White (HC0/HC1)** -- heteroskedasticity-consistent.  Meat is
  :math:`\sum_t x_t x_t' \hat e_t^2`.
* **Newey--West (HAC)** -- heteroskedasticity- *and* autocorrelation-consistent.
  Adds Bartlett-weighted autocovariance terms so that inference remains valid
  when residuals are serially correlated, which is the norm for overlapping or
  persistent financial returns.

References
----------
White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator
    and a Direct Test for Heteroskedasticity." *Econometrica* 48(4), 817--838.
Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    *Econometrica* 55(3), 703--708.
Newey, W. K., & West, K. D. (1994). "Automatic Lag Selection in Covariance
    Matrix Estimation." *Review of Economic Studies* 61(4), 631--653.
"""

from __future__ import annotations

import numpy as np

from factorlab_quant.core.types import FloatArray

__all__ = [
    "bartlett_weights",
    "default_hac_lags",
    "newey_west_covariance",
    "white_covariance",
]


def default_hac_lags(n_obs: int) -> int:
    r"""Automatic Newey--West (1994) bandwidth rule.

    Returns :math:`\lfloor 4 (n/100)^{2/9} \rfloor`, the fixed-bandwidth rule
    widely used as a sensible default when the researcher has no prior on the
    autocorrelation structure.  Always at least 0 and never exceeds ``n - 1``.
    """
    if n_obs <= 1:
        return 0
    lags = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    return int(min(max(lags, 0), n_obs - 1))


def bartlett_weights(lags: int) -> FloatArray:
    r"""Bartlett kernel weights :math:`w_\ell = 1 - \ell/(L+1)` for lags 1..L.

    These triangular weights guarantee the Newey--West estimator is positive
    semi-definite, which a naive truncated estimator is not.
    """
    if lags < 0:
        raise ValueError("lags must be non-negative")
    ell = np.arange(1, lags + 1, dtype=np.float64)
    return 1.0 - ell / (lags + 1.0)


def _score_matrix(X: FloatArray, residuals: FloatArray) -> FloatArray:
    r"""Per-observation scores :math:`u_t = x_t \hat e_t` as an ``n x k`` array."""
    return X * residuals[:, np.newaxis]


def _hac_meat(X: FloatArray, residuals: FloatArray, lags: int) -> FloatArray:
    r"""Newey--West "meat" matrix :math:`\hat S`.

    .. math::

        \hat S = \hat\Gamma_0
            + \sum_{\ell=1}^{L} w_\ell\,(\hat\Gamma_\ell + \hat\Gamma_\ell'),
        \qquad
        \hat\Gamma_\ell = \sum_{t=\ell+1}^{n} u_t u_{t-\ell}'.

    With ``lags == 0`` this reduces to the White meat
    :math:`\hat\Gamma_0 = \sum_t u_t u_t'`.
    """
    u = _score_matrix(X, residuals)
    s = u.T @ u  # Gamma_0 (k x k)
    weights = bartlett_weights(lags)
    for ell in range(1, lags + 1):
        gamma_ell = u[ell:].T @ u[:-ell]  # sum_{t=ell+1}^{n} u_t u_{t-ell}'
        s = s + weights[ell - 1] * (gamma_ell + gamma_ell.T)
    return s


def white_covariance(
    X: FloatArray,
    residuals: FloatArray,
    xtx_inv: FloatArray,
    *,
    small_sample_correction: bool = True,
) -> FloatArray:
    """Heteroskedasticity-consistent covariance (HC0, or HC1 with correction).

    Parameters
    ----------
    X:
        ``n x k`` design matrix (including the intercept column).
    residuals:
        OLS residuals, length ``n``.
    xtx_inv:
        Precomputed ``(X'X)^{-1}``; passed in to avoid recomputation.
    small_sample_correction:
        If True, multiply by ``n / (n - k)`` to obtain HC1 rather than HC0.
    """
    n, k = X.shape
    meat = _hac_meat(X, residuals, lags=0)
    cov = xtx_inv @ meat @ xtx_inv
    if small_sample_correction:
        cov = cov * (n / (n - k))
    return _symmetrize(cov)


def newey_west_covariance(
    X: FloatArray,
    residuals: FloatArray,
    xtx_inv: FloatArray,
    *,
    lags: int | None = None,
    small_sample_correction: bool = True,
) -> tuple[FloatArray, int]:
    """Newey--West HAC covariance of the OLS coefficient vector.

    Parameters
    ----------
    X, residuals, xtx_inv:
        As in :func:`white_covariance`.
    lags:
        Truncation lag ``L``.  When ``None``, the automatic
        :func:`default_hac_lags` rule is used.
    small_sample_correction:
        If True, apply the ``n / (n - k)`` degrees-of-freedom correction
        (matching ``statsmodels`` ``use_correction=True``).

    Returns
    -------
    (covariance, lags_used)
        The ``k x k`` covariance matrix and the truncation lag actually used.
    """
    n, k = X.shape
    resolved_lags = default_hac_lags(n) if lags is None else int(lags)
    if resolved_lags < 0:
        raise ValueError("lags must be non-negative")
    resolved_lags = min(resolved_lags, n - 1)

    meat = _hac_meat(X, residuals, lags=resolved_lags)
    cov = xtx_inv @ meat @ xtx_inv
    if small_sample_correction:
        cov = cov * (n / (n - k))
    return _symmetrize(cov), resolved_lags


def _symmetrize(matrix: FloatArray) -> FloatArray:
    """Remove tiny asymmetries introduced by floating-point accumulation."""
    return 0.5 * (matrix + matrix.T)
