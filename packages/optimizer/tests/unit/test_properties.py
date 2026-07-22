"""Property-based invariants for the optimizers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_optimizer import (
    Constraint,
    MaxSharpeOptimizer,
    MinVarianceOptimizer,
    OptimizationProblem,
    RiskParityOptimizer,
    risk,
)

pytestmark = pytest.mark.property

_SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@st.composite
def problems(draw):
    n = draw(st.integers(min_value=2, max_value=5))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    L = np.tril(rng.uniform(0.05, 0.3, size=(n, n)))
    cov = L @ L.T + np.eye(n) * 1e-3
    mu = rng.uniform(0.02, 0.15, size=n)
    assets = tuple(f"A{i}" for i in range(n))
    return OptimizationProblem(assets, mu, cov, constraints=(Constraint.long_only(),))


@_SETTINGS
@given(problem=problems())
def test_weights_sum_to_one_and_nonnegative(problem) -> None:
    result = MinVarianceOptimizer().optimize(problem)
    assert result.weights.total == pytest.approx(1.0, abs=1e-5)
    assert np.all(result.weights.values >= -1e-6)


@_SETTINGS
@given(problem=problems())
def test_min_variance_is_minimal(problem) -> None:
    mvp = MinVarianceOptimizer().optimize(problem)
    rp = RiskParityOptimizer().optimize(problem)
    assert mvp.expected_volatility <= rp.expected_volatility + 1e-6


@_SETTINGS
@given(problem=problems())
def test_max_sharpe_dominates_min_variance(problem) -> None:
    ms = MaxSharpeOptimizer().optimize(problem)
    mvp = MinVarianceOptimizer().optimize(problem)
    assert ms.sharpe_ratio >= mvp.sharpe_ratio - 1e-4


@_SETTINGS
@given(problem=problems())
def test_risk_contributions_sum_to_volatility(problem) -> None:
    result = MinVarianceOptimizer().optimize(problem)
    rc = risk.risk_contributions(result.weights.values, problem.covariance)
    assert np.sum(rc) == pytest.approx(result.expected_volatility, rel=1e-6)
