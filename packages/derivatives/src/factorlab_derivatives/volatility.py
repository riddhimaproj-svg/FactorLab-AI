r"""Realized-volatility estimators: historical, EWMA, and GARCH(1,1).

* **Historical** — annualized sample standard deviation of (log) returns.
* **EWMA** (RiskMetrics) — :math:`\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2`,
  weighting recent observations more heavily.
* **GARCH(1,1)** — :math:`\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta\sigma_{t-1}^2`,
  fit by Gaussian maximum likelihood; captures volatility clustering and
  mean-reverts to the long-run variance :math:`\omega/(1-\alpha-\beta)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize

from factorlab_derivatives._validation import FloatArray, as_return_vector
from factorlab_derivatives.errors import ConvergenceError, DerivativesInputError

__all__ = [
    "GarchResult",
    "ewma_variance",
    "ewma_volatility",
    "fit_garch",
    "historical_volatility",
]


def _returns_from_prices(prices: object, log_returns: bool) -> FloatArray:
    px = as_return_vector(prices, name="prices")
    if px.shape[0] < 2:
        raise DerivativesInputError("need >= 2 prices to compute returns")
    if np.any(px <= 0.0):
        raise DerivativesInputError("prices must be positive")
    return np.diff(np.log(px)) if log_returns else px[1:] / px[:-1] - 1.0


def historical_volatility(
    prices: object, periods_per_year: float = 252.0, *, log_returns: bool = True
) -> float:
    """Annualized historical volatility from a price series."""
    r = _returns_from_prices(prices, log_returns)
    if r.shape[0] < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def ewma_variance(returns: object, lam: float = 0.94) -> FloatArray:
    r"""RiskMetrics EWMA conditional variance series (per period).

    Seeded with the sample variance; entry ``t`` is the variance forecast for ``t``
    made from information through ``t-1``.
    """
    r = as_return_vector(returns)
    if not 0.0 < lam < 1.0:
        raise DerivativesInputError("lambda must lie in (0, 1)")
    var = np.empty(r.shape[0], dtype=np.float64)
    var[0] = float(np.var(r, ddof=1)) if r.shape[0] > 1 else float(r[0] ** 2)
    for t in range(1, r.shape[0]):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return var


def ewma_volatility(
    returns: object, lam: float = 0.94, periods_per_year: float = 252.0
) -> float:
    """Annualized EWMA volatility (the latest conditional estimate)."""
    var = ewma_variance(returns, lam)
    return float(np.sqrt(var[-1] * periods_per_year))


@dataclass(frozen=True, slots=True)
class GarchResult:
    """A fitted GARCH(1,1) model (immutable, serializable)."""

    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    conditional_variance: FloatArray
    periods_per_year: float
    mean: float

    def __post_init__(self) -> None:
        self.conditional_variance.setflags(write=False)

    @property
    def persistence(self) -> float:
        """``alpha + beta`` — must be < 1 for a stationary, mean-reverting model."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        p = self.persistence
        return self.omega / (1.0 - p) if p < 1.0 else float("nan")

    @property
    def long_run_volatility(self) -> float:
        """Annualized long-run (unconditional) volatility."""
        return float(np.sqrt(self.long_run_variance * self.periods_per_year))

    def conditional_volatility(self, *, annualized: bool = True) -> FloatArray:
        scale = self.periods_per_year if annualized else 1.0
        return np.sqrt(self.conditional_variance * scale)

    def forecast_variance(self, horizon: int) -> FloatArray:
        r"""Multi-step variance forecast (per period), mean-reverting to long-run.

        :math:`E[\sigma_{t+h}^2] = \bar\sigma^2 + (\alpha+\beta)^{h}(\sigma_t^2 - \bar\sigma^2)`,
        for ``h = 1 .. horizon``, where :math:`\sigma_t^2` is the latest conditional
        variance.
        """
        if horizon < 1:
            raise DerivativesInputError("horizon must be >= 1")
        lrv = self.long_run_variance
        current = float(self.conditional_variance[-1])
        h = np.arange(1, horizon + 1, dtype=np.float64)
        return np.asarray(lrv + self.persistence**h * (current - lrv), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "log_likelihood": self.log_likelihood,
            "conditional_variance": self.conditional_variance.tolist(),
            "periods_per_year": self.periods_per_year,
            "mean": self.mean,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GarchResult:
        return cls(
            omega=float(data["omega"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            log_likelihood=float(data["log_likelihood"]),
            conditional_variance=np.asarray(data["conditional_variance"], dtype=np.float64),
            periods_per_year=float(data["periods_per_year"]),
            mean=float(data["mean"]),
        )


def _garch_variance(params: FloatArray, resid: FloatArray) -> FloatArray:
    omega, alpha, beta = params
    n = resid.shape[0]
    var = np.empty(n, dtype=np.float64)
    var[0] = float(np.var(resid, ddof=1))
    for t in range(1, n):
        var[t] = omega + alpha * resid[t - 1] ** 2 + beta * var[t - 1]
    return var


def _garch_neg_loglik(params: FloatArray, resid: FloatArray) -> float:
    omega, alpha, beta = params
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.9999:
        return 1e12
    var = _garch_variance(params, resid)
    if np.any(var <= 0.0):
        return 1e12
    ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(var) + resid**2 / var)
    return float(-ll)


def fit_garch(
    returns: object, periods_per_year: float = 252.0
) -> GarchResult:
    """Fit a GARCH(1,1) model by Gaussian maximum likelihood."""
    r = as_return_vector(returns)
    if r.shape[0] < 10:
        raise DerivativesInputError("need >= 10 observations to fit GARCH(1,1)")
    mean = float(np.mean(r))
    resid = r - mean
    sample_var = float(np.var(resid, ddof=1))

    x0 = np.array([sample_var * 0.1, 0.05, 0.90])
    bounds = [(1e-12, None), (0.0, 0.9999), (0.0, 0.9999)]
    result = optimize.minimize(
        _garch_neg_loglik, x0, args=(resid,), method="L-BFGS-B", bounds=bounds,
    )
    if not result.success:
        raise ConvergenceError(f"GARCH MLE failed: {result.message}")

    omega, alpha, beta = (float(v) for v in result.x)
    var = _garch_variance(result.x, resid)
    return GarchResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        log_likelihood=float(-result.fun),
        conditional_variance=var,
        periods_per_year=periods_per_year,
        mean=mean,
    )
