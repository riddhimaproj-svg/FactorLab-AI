"""Shared fixtures for the risk-engine test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2025)


@pytest.fixture
def returns(rng) -> np.ndarray:
    return rng.normal(0.0005, 0.012, size=1000)


@pytest.fixture
def weights() -> np.ndarray:
    return np.array([0.4, 0.35, 0.25])


@pytest.fixture
def covariance() -> np.ndarray:
    L = np.array([[0.20, 0.0, 0.0], [0.03, 0.25, 0.0], [0.02, 0.01, 0.18]])
    return L @ L.T


@pytest.fixture
def assets() -> tuple[str, ...]:
    return ("A", "B", "C")


@pytest.fixture
def returns_matrix(rng) -> np.ndarray:
    return rng.normal(0.0004, 0.01, size=(500, 3))
