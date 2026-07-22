"""Tests for OptimizationProblem, OptimizationResult, and OptimizerConfig."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import (
    Constraint,
    MinVarianceOptimizer,
    OptimizationProblem,
    OptimizationResult,
    OptimizerConfig,
)
from factorlab_optimizer.errors import OptimizationInputError


# -- OptimizerConfig ------------------------------------------------------- #
def test_config_default_bounds() -> None:
    assert OptimizerConfig().default_bounds() == (0.0, 1.0)
    assert OptimizerConfig(allow_short=True).default_bounds() == (-1.0, 1.0)
    assert OptimizerConfig(min_weight=-0.5, max_weight=0.5).default_bounds() == (-0.5, 0.5)


def test_config_validation() -> None:
    with pytest.raises(OptimizationInputError):
        OptimizerConfig(risk_aversion=0.0)
    with pytest.raises(OptimizationInputError):
        OptimizerConfig(covariance_regularization=-1.0)


def test_config_roundtrip() -> None:
    cfg = OptimizerConfig(risk_free_rate=0.01, risk_aversion=3.0, allow_short=True)
    assert OptimizerConfig.from_dict(cfg.to_dict()) == cfg


# -- OptimizationProblem --------------------------------------------------- #
def test_problem_validation() -> None:
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    with pytest.raises(OptimizationInputError):
        OptimizationProblem(("A", "B", "C"), mu, cov)  # dim mismatch
    with pytest.raises(OptimizationInputError):
        OptimizationProblem(("A", "A"), mu, cov)  # duplicate
    with pytest.raises(OptimizationInputError):
        OptimizationProblem(("A", "B"), mu, np.array([[0.04, 0.02], [0.01, 0.09]]))  # asymmetric


def test_problem_regularization() -> None:
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    prob = OptimizationProblem(("A", "B"), np.array([0.1, 0.2]), cov)
    reg = prob.regularized_covariance(0.01)
    np.testing.assert_allclose(np.diag(reg), [0.05, 0.10])
    assert prob.regularized_covariance(0.0) is prob.covariance


def test_problem_roundtrip() -> None:
    prob = OptimizationProblem(
        ("A", "B"), np.array([0.1, 0.2]), np.array([[0.04, 0.01], [0.01, 0.09]]),
        constraints=(Constraint.long_only(),), prev_weights=np.array([0.5, 0.5]),
    )
    restored = OptimizationProblem.from_dict(prob.to_dict())
    assert restored.assets == prob.assets
    np.testing.assert_allclose(restored.covariance, prob.covariance)
    np.testing.assert_allclose(restored.prev_weights, prob.prev_weights)
    assert len(restored.constraints) == 1


def test_problem_from_moments() -> None:
    prob = OptimizationProblem.from_moments(
        ["A", "B"], [0.1, 0.2], np.array([[0.04, 0.0], [0.0, 0.09]])
    )
    assert prob.n_assets == 2


# -- OptimizationResult ---------------------------------------------------- #
def test_result_derives_metrics(problem) -> None:
    result = MinVarianceOptimizer().optimize(problem)
    w = result.weights.values
    assert result.expected_return == pytest.approx(w @ problem.expected_returns)
    assert result.expected_volatility == pytest.approx(np.sqrt(w @ problem.covariance @ w))


def test_result_roundtrip(problem) -> None:
    result = MinVarianceOptimizer().optimize(problem)
    restored = OptimizationResult.from_dict(result.to_dict())
    np.testing.assert_allclose(restored.weights.values, result.weights.values)
    assert restored.optimizer == result.optimizer
    assert restored.sharpe_ratio == pytest.approx(result.sharpe_ratio)


def test_result_summary(problem) -> None:
    text = MinVarianceOptimizer().optimize(problem).summary()
    assert "Optimization Result" in text
    assert "min_variance" in text
    assert "Sharpe" in text
