"""Shared fixtures for the optimizer test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def assets() -> tuple[str, ...]:
    return ("A", "B", "C", "D")


@pytest.fixture
def mu() -> np.ndarray:
    return np.array([0.08, 0.12, 0.10, 0.06])


@pytest.fixture
def cov() -> np.ndarray:
    # A well-conditioned PSD covariance via L L'.
    rng = np.random.default_rng(7)
    L = np.tril(rng.uniform(0.02, 0.25, size=(4, 4)))
    return L @ L.T + np.eye(4) * 1e-4


@pytest.fixture
def problem(assets, mu, cov):
    from factorlab_optimizer import Constraint, OptimizationProblem

    return OptimizationProblem(assets, mu, cov, constraints=(Constraint.long_only(),))
