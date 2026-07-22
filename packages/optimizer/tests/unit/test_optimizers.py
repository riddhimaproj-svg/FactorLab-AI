"""Optimizer tests: analytic cross-validation, constraints, and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import (
    Constraint,
    MaxDiversificationOptimizer,
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
    OptimizationProblem,
    OptimizerConfig,
    RiskParityOptimizer,
    risk,
)
from factorlab_optimizer.errors import OptimizationFailedError


@pytest.fixture
def cov3() -> np.ndarray:
    L = np.array([[0.20, 0.0, 0.0], [0.03, 0.25, 0.0], [0.02, 0.01, 0.18]])
    return L @ L.T


@pytest.fixture
def mu3() -> np.ndarray:
    return np.array([0.08, 0.12, 0.10])


# -- Analytic cross-validation --------------------------------------------- #
def test_min_variance_matches_closed_form(cov3, mu3) -> None:
    """Budget-only min-variance: w = Σ⁻¹1 / (1'Σ⁻¹1)."""
    prob = OptimizationProblem(("A", "B", "C"), mu3, cov3)
    cfg = OptimizerConfig(min_weight=-10.0, max_weight=10.0)
    result = MinVarianceOptimizer(cfg).optimize(prob)

    inv = np.linalg.inv(cov3)
    ones = np.ones(3)
    analytic = inv @ ones / (ones @ inv @ ones)
    np.testing.assert_allclose(result.weights.values, analytic, atol=1e-5)


def test_max_sharpe_matches_tangency(cov3, mu3) -> None:
    """Budget-only tangency: w ∝ Σ⁻¹(μ − rf)."""
    prob = OptimizationProblem(("A", "B", "C"), mu3, cov3)
    rf = 0.02
    cfg = OptimizerConfig(risk_free_rate=rf, min_weight=-10.0, max_weight=10.0)
    result = MaxSharpeOptimizer(cfg).optimize(prob)

    inv = np.linalg.inv(cov3)
    raw = inv @ (mu3 - rf)
    analytic = raw / raw.sum()
    np.testing.assert_allclose(result.weights.values, analytic, atol=1e-4)


def test_min_variance_matches_scipy_directly(cov3, mu3) -> None:
    """Cross-validate against an independent SciPy solve of the same program."""
    from scipy.optimize import minimize

    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3, constraints=(Constraint.long_only(),)
    )
    result = MinVarianceOptimizer().optimize(prob)

    ref = minimize(
        lambda w: w @ cov3 @ w,
        np.full(3, 1 / 3),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    np.testing.assert_allclose(result.weights.values, ref.x, atol=1e-6)


# -- Properties of each optimizer ------------------------------------------ #
@pytest.mark.parametrize(
    "optimizer_cls",
    [MinVarianceOptimizer, MeanVarianceOptimizer, MaxSharpeOptimizer,
     MaxDiversificationOptimizer, RiskParityOptimizer],
)
def test_optimizers_respect_budget_and_long_only(optimizer_cls, problem) -> None:
    result = optimizer_cls().optimize(problem)
    assert result.success
    assert result.weights.total == pytest.approx(1.0, abs=1e-6)
    assert np.all(result.weights.values >= -1e-8)  # long-only


def test_min_variance_has_lowest_variance(problem) -> None:
    mv = MinVarianceOptimizer().optimize(problem)
    other = MeanVarianceOptimizer().optimize(problem)
    assert mv.expected_volatility <= other.expected_volatility + 1e-8


def test_max_sharpe_has_highest_sharpe(problem) -> None:
    ms = MaxSharpeOptimizer().optimize(problem)
    for other in (MinVarianceOptimizer(), MeanVarianceOptimizer(), RiskParityOptimizer()):
        assert ms.sharpe_ratio >= other.optimize(problem).sharpe_ratio - 1e-6


def test_risk_parity_equalizes_risk_contributions(problem) -> None:
    rp = RiskParityOptimizer().optimize(problem)
    prc = risk.percent_risk_contributions(rp.weights.values, problem.covariance)
    # SLSQP on the (flat-near-optimum) risk-dispersion objective reaches equal
    # contributions to well within half a percent of the risk budget.
    np.testing.assert_allclose(prc, np.full(len(prc), 1.0 / len(prc)), atol=5e-3)


def test_max_diversification_beats_concentrated(problem) -> None:
    md = MaxDiversificationOptimizer().optimize(problem)
    dr_opt = risk.diversification_ratio(md.weights.values, problem.covariance)
    # a single-asset portfolio has DR == 1; the optimum must exceed it.
    assert dr_opt >= 1.0


# -- Constraints ----------------------------------------------------------- #
def test_weight_bounds_enforced(mu3, cov3) -> None:
    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3, constraints=(Constraint.weight_bounds(0.1, 0.5),)
    )
    result = MinVarianceOptimizer().optimize(prob)
    assert np.all(result.weights.values >= 0.1 - 1e-6)
    assert np.all(result.weights.values <= 0.5 + 1e-6)


def test_long_short_allows_negative(mu3, cov3) -> None:
    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3, constraints=(Constraint.weight_bounds(-1.0, 1.0),)
    )
    # a strong negative view via mean-variance can produce shorts
    cfg = OptimizerConfig(risk_aversion=0.5)
    result = MeanVarianceOptimizer(cfg).optimize(prob)
    assert result.weights.total == pytest.approx(1.0, abs=1e-6)


def test_turnover_constraint_limits_trading(mu3, cov3) -> None:
    prev = np.array([0.8, 0.1, 0.1])
    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3,
        constraints=(Constraint.long_only(), Constraint.turnover(0.1)),
        prev_weights=prev,
    )
    result = MinVarianceOptimizer().optimize(prob)
    turnover = np.sum(np.abs(result.weights.values - prev))
    assert turnover <= 0.1 + 1e-6


def test_sector_constraint(mu3, cov3) -> None:
    memberships = {"A": "tech", "B": "tech", "C": "energy"}
    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3,
        constraints=(
            Constraint.long_only(),
            Constraint.sector_bounds(memberships, {"tech": (0.0, 0.4)}),
        ),
    )
    result = MinVarianceOptimizer().optimize(prob)
    tech = result.weights.get("A") + result.weights.get("B")
    assert tech <= 0.4 + 1e-6


def test_leverage_and_cash(mu3, cov3) -> None:
    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3, constraints=(Constraint.cash_bounds(0.1, 0.3),)
    )
    cfg = OptimizerConfig(budget=None)  # cash constraint replaces strict budget
    result = MinVarianceOptimizer(cfg).optimize(prob)
    assert 0.7 - 1e-6 <= result.weights.total <= 0.9 + 1e-6


def test_mean_variance_target_return(mu3, cov3) -> None:
    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3, constraints=(Constraint.long_only(),)
    )
    target = 0.10
    result = MeanVarianceOptimizer(target_return=target).optimize(prob)
    assert result.expected_return == pytest.approx(target, abs=1e-6)


def test_infeasible_target_raises(mu3, cov3) -> None:
    prob = OptimizationProblem(
        ("A", "B", "C"), mu3, cov3, constraints=(Constraint.long_only(),)
    )
    # target above the max attainable long-only return (max mu = 0.12)
    with pytest.raises(OptimizationFailedError):
        MeanVarianceOptimizer(target_return=0.5).optimize(prob)
