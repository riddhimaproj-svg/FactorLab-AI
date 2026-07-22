"""Supplementary tests covering remaining branches."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import (
    BlackLittermanOptimizer,
    CapitalAllocationLine,
    Constraint,
    EfficientFrontier,
    OptimizationProblem,
    OptimizerConfig,
    PortfolioWeights,
)
from factorlab_optimizer.errors import OptimizationInputError


# -- weights / problem validation ------------------------------------------ #
def test_weights_reject_2d() -> None:
    with pytest.raises(OptimizationInputError):
        PortfolioWeights(("A",), np.zeros((1, 1)))


def test_problem_nonfinite_and_prev_shape() -> None:
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    with pytest.raises(OptimizationInputError):
        OptimizationProblem(("A", "B"), np.array([0.1, np.nan]), cov)
    with pytest.raises(OptimizationInputError):
        OptimizationProblem(("A", "B"), np.array([0.1, 0.2]), cov, prev_weights=np.array([1.0]))


# -- constraint factory validation ----------------------------------------- #
def test_constraint_factory_bounds_validation() -> None:
    with pytest.raises(OptimizationInputError):
        Constraint.asset_bounds("A", 0.9, 0.1)
    with pytest.raises(OptimizationInputError):
        Constraint.cash_bounds(0.5, 0.1)


def test_budget_constraint_target() -> None:
    from factorlab_optimizer import compile_constraints

    c = compile_constraints(
        ("A", "B"), [Constraint.budget(0.9)], default_lower=0.0, default_upper=1.0,
        default_budget=None,
    )
    assert c.has_budget
    assert c.scipy_constraints[0]["fun"](np.array([0.5, 0.4])) == pytest.approx(0.0)


# -- Black-Litterman remaining paths --------------------------------------- #
def test_bl_custom_view_uncertainty() -> None:
    from factorlab_optimizer import black_litterman_posterior

    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    P = np.array([[1.0, -1.0]])
    Q = np.array([0.01])
    omega = np.array([[0.0001]])
    mu_bl, _ = black_litterman_posterior(
        cov, np.array([0.5, 0.5]), risk_aversion=2.5, tau=0.05,
        pick_matrix=P, view_returns=Q, view_uncertainty=omega,
    )
    assert mu_bl.shape == (2,)


def test_bl_bad_omega_shape() -> None:
    from factorlab_optimizer import black_litterman_posterior

    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    with pytest.raises(OptimizationInputError):
        black_litterman_posterior(
            cov, np.array([0.5, 0.5]), risk_aversion=2.5, tau=0.05,
            pick_matrix=np.array([[1.0, 0.0]]), view_returns=np.array([0.01]),
            view_uncertainty=np.eye(2),  # wrong shape (should be 1x1)
        )


def test_bl_objective_is_mean_variance_utility() -> None:
    """Exercise BL's ABC-required _objective (delegated path never calls it)."""
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    problem = OptimizationProblem(("A", "B"), np.array([0.1, 0.2]), cov)
    opt = BlackLittermanOptimizer(np.array([0.5, 0.5]), OptimizerConfig(risk_aversion=2.0))
    obj = opt._objective(problem, cov)
    w = np.array([0.5, 0.5])
    expected = 0.5 * 2.0 * (w @ cov @ w) - problem.expected_returns @ w
    assert obj(w) == pytest.approx(expected)


# -- frontier edge branches ------------------------------------------------ #
def test_frontier_equal_returns_fallback() -> None:
    # Equal expected returns trigger the r_max <= r_min fallback; the target
    # equalities become redundant with the budget so the frontier degenerates
    # (infeasible target solves are skipped rather than raising).
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    mu = np.array([0.1, 0.1])
    problem = OptimizationProblem(("A", "B"), mu, cov, constraints=(Constraint.long_only(),))
    points = EfficientFrontier(problem).compute(n_points=5)
    assert isinstance(points, tuple)


def test_cal_zero_volatility_slope_is_nan() -> None:
    cal = CapitalAllocationLine(risk_free_rate=0.02, tangency_return=0.05, tangency_volatility=0.0)
    assert np.isnan(cal.slope)
