r"""Monte Carlo Value-at-Risk and Expected Shortfall.

Monte Carlo VaR simulates a large sample of returns from a fitted distribution,
then reads VaR/ES off the simulated distribution exactly as historical
simulation does.  It combines the flexibility of parametric modeling (any
distribution, multivariate structure) with the empirical tail estimation of
historical simulation, at the cost of sampling error (shrinking as
:math:`1/\sqrt{N}`).

* **Univariate** -- draw returns from a Normal or standardized-t fit.
* **Portfolio** -- draw asset returns from a multivariate Normal with the given
  mean vector and covariance, form portfolio returns ``sim @ w``, then estimate.

Every routine takes an explicit ``seed`` so results are reproducible.

.. note::
   This is Monte Carlo *risk estimation* from a fitted return distribution, not a
   general path-simulation or derivative-pricing engine.
"""

from __future__ import annotations

import numpy as np

from factorlab_risk._validation import (
    as_covariance,
    as_return_vector,
    as_weights,
    check_confidence,
)
from factorlab_risk.errors import DimensionMismatchError, RiskInputError
from factorlab_risk.var.historical import (
    historical_expected_shortfall,
    historical_var,
)

__all__ = [
    "monte_carlo_var",
    "monte_carlo_expected_shortfall",
    "monte_carlo_portfolio_var",
    "simulate_portfolio_returns",
]


def _simulate_univariate(
    mu: float, sigma: float, n: int, distribution: str, dof: float, rng: np.random.Generator
) -> np.ndarray:
    if distribution == "normal":
        return rng.normal(mu, sigma, size=n)
    if distribution == "t":
        if dof <= 2.0:
            raise RiskInputError("t-distribution requires dof > 2 for finite variance")
        scale = np.sqrt((dof - 2.0) / dof)
        return mu + sigma * scale * rng.standard_t(dof, size=n)
    raise RiskInputError(f"unknown distribution {distribution!r}")


def monte_carlo_var(
    returns: object | None = None,
    confidence: float = 0.95,
    horizon: int = 1,
    *,
    mean: float | None = None,
    std: float | None = None,
    n_simulations: int = 100_000,
    distribution: str = "normal",
    dof: float = 6.0,
    seed: int = 0,
) -> float:
    """Monte Carlo VaR for a single return stream (positive loss fraction)."""
    check_confidence(confidence)
    if returns is not None:
        r = as_return_vector(returns)
        mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    elif mean is not None and std is not None:
        mu, sigma = float(mean), float(std)
    else:
        raise RiskInputError("provide either returns or both mean and std")

    rng = np.random.default_rng(seed)
    sims = _simulate_univariate(mu, sigma, n_simulations, distribution, dof, rng)
    return historical_var(sims, confidence, horizon)


def monte_carlo_expected_shortfall(
    returns: object | None = None,
    confidence: float = 0.95,
    horizon: int = 1,
    *,
    mean: float | None = None,
    std: float | None = None,
    n_simulations: int = 100_000,
    distribution: str = "normal",
    dof: float = 6.0,
    seed: int = 0,
) -> float:
    """Monte Carlo Expected Shortfall for a single return stream."""
    check_confidence(confidence)
    if returns is not None:
        r = as_return_vector(returns)
        mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    elif mean is not None and std is not None:
        mu, sigma = float(mean), float(std)
    else:
        raise RiskInputError("provide either returns or both mean and std")

    rng = np.random.default_rng(seed)
    sims = _simulate_univariate(mu, sigma, n_simulations, distribution, dof, rng)
    return historical_expected_shortfall(sims, confidence, horizon)


def simulate_portfolio_returns(
    weights: object,
    mean_vector: object,
    covariance: object,
    n_simulations: int = 100_000,
    seed: int = 0,
) -> np.ndarray:
    """Simulate portfolio returns from a multivariate-Normal asset model."""
    w = as_weights(weights)
    mu = np.asarray(mean_vector, dtype=np.float64)
    cov = as_covariance(covariance)
    n = w.shape[0]
    if mu.shape != (n,) or cov.shape != (n, n):
        raise DimensionMismatchError(n, cov.shape[0], name="mean_vector/covariance")
    rng = np.random.default_rng(seed)
    asset_sims = rng.multivariate_normal(mu, cov, size=n_simulations)
    return asset_sims @ w


def monte_carlo_portfolio_var(
    weights: object,
    mean_vector: object,
    covariance: object,
    confidence: float = 0.95,
    horizon: int = 1,
    *,
    n_simulations: int = 100_000,
    seed: int = 0,
    expected_shortfall: bool = False,
) -> float:
    """Monte Carlo VaR (or ES) of a portfolio via multivariate-Normal simulation."""
    check_confidence(confidence)
    port = simulate_portfolio_returns(weights, mean_vector, covariance, n_simulations, seed)
    if expected_shortfall:
        return historical_expected_shortfall(port, confidence, horizon)
    return historical_var(port, confidence, horizon)
