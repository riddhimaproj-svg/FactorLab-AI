"""Shared fixtures and synthetic data-generating processes for the test suite.

The suite validates the engine two ways:

1. **Ground truth** -- data are simulated from a known DGP so estimates can be
   checked against the parameters that generated them.
2. **Reference cross-validation** -- results are compared against
   ``statsmodels``, an independent, mature implementation of the same
   econometrics (imported lazily so the core package never depends on it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class CapmDGP:
    """A realized CAPM sample together with the parameters that produced it."""

    asset_excess: np.ndarray
    market_excess: np.ndarray
    risk_free: np.ndarray
    alpha_true: float
    beta_true: float
    periods_per_year: int


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240521)


@pytest.fixture
def capm_dgp(rng: np.random.Generator) -> CapmDGP:
    """Monthly CAPM sample: alpha = 0.002, beta = 1.15, 15 years of data."""
    n = 180
    alpha_true = 0.002
    beta_true = 1.15
    risk_free = np.full(n, 0.0015)
    market_excess = rng.normal(loc=0.006, scale=0.043, size=n)
    noise = rng.normal(loc=0.0, scale=0.018, size=n)
    asset_excess = alpha_true + beta_true * market_excess + noise
    return CapmDGP(
        asset_excess=asset_excess,
        market_excess=market_excess,
        risk_free=risk_free,
        alpha_true=alpha_true,
        beta_true=beta_true,
        periods_per_year=12,
    )


@pytest.fixture
def design_and_response(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A generic two-regressor design (with intercept) and response."""
    n = 200
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 2.0, size=n)
    design = np.column_stack([np.ones(n), x1, x2])
    beta = np.array([0.5, -1.2, 0.8])
    y = design @ beta + rng.normal(0.0, 0.7, size=n)
    return y, design
