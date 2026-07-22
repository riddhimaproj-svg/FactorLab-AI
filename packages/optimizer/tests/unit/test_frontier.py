"""Efficient frontier and capital allocation line tests."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import (
    Constraint,
    EfficientFrontier,
    OptimizationProblem,
    OptimizerConfig,
)


@pytest.fixture
def problem() -> OptimizationProblem:
    L = np.array([[0.20, 0.0, 0.0], [0.03, 0.25, 0.0], [0.02, 0.01, 0.18]])
    cov = L @ L.T
    mu = np.array([0.08, 0.12, 0.10])
    return OptimizationProblem(("A", "B", "C"), mu, cov, constraints=(Constraint.long_only(),))


def test_frontier_is_monotonic_in_return(problem) -> None:
    frontier = EfficientFrontier(problem).compute(n_points=12)
    assert len(frontier) >= 2
    returns = [p.expected_return for p in frontier]
    assert returns == sorted(returns)


def test_frontier_volatility_increases_with_return(problem) -> None:
    frontier = EfficientFrontier(problem).compute(n_points=12)
    vols = [p.volatility for p in frontier]
    # frontier is convex: volatility is non-decreasing along the efficient part
    assert vols[-1] >= vols[0] - 1e-9


def test_min_variance_is_frontier_minimum(problem) -> None:
    ef = EfficientFrontier(problem)
    mvp = ef.min_variance_portfolio()
    frontier = ef.compute(n_points=12)
    assert mvp.expected_volatility <= min(p.volatility for p in frontier) + 1e-6


def test_max_sharpe_on_frontier(problem) -> None:
    ef = EfficientFrontier(problem)
    tangency = ef.max_sharpe_portfolio()
    frontier = ef.compute(n_points=25)
    best_frontier_sharpe = max(p.sharpe_ratio for p in frontier)
    assert tangency.sharpe_ratio >= best_frontier_sharpe - 1e-3


def test_capital_allocation_line(problem) -> None:
    cfg = OptimizerConfig(risk_free_rate=0.02)
    ef = EfficientFrontier(problem, cfg)
    cal = ef.capital_allocation_line()
    tangency = ef.max_sharpe_portfolio()
    assert cal.slope == pytest.approx(tangency.sharpe_ratio, rel=1e-3)
    # at zero volatility, CAL return equals the risk-free rate
    assert cal.expected_return_at(0.0) == pytest.approx(0.02)
    # linearity
    pts = cal.points(np.array([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(pts, [0.02, 0.02 + cal.slope * 0.1, 0.02 + cal.slope * 0.2])


def test_compute_requires_two_points(problem) -> None:
    with pytest.raises(ValueError):
        EfficientFrontier(problem).compute(n_points=1)
