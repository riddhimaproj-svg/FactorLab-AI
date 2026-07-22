"""Internal input-validation and coercion helpers shared across the engine.

Centralizes the numeric conventions so every module agrees:

* **Returns** are per-period *simple* returns in decimal units.
* **Confidence** ``c`` lies in ``(0, 1)`` (e.g. 0.95); the tail probability is
  ``alpha = 1 - c``.
* **VaR / ES** are reported as **positive loss magnitudes** (a fraction of
  portfolio value): a 95% VaR of 0.03 means "a 5% chance of losing more than 3%".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_risk.errors import DimensionMismatchError, RiskInputError

__all__ = [
    "FloatArray",
    "as_covariance",
    "as_return_matrix",
    "as_return_vector",
    "as_weights",
    "check_confidence",
    "tail_alpha",
]

FloatArray = NDArray[np.float64]


def as_return_vector(returns: object, name: str = "returns") -> FloatArray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise RiskInputError(f"{name} must be a 1-D array")
    if arr.size == 0:
        raise RiskInputError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise RiskInputError(f"{name} contains non-finite values")
    return arr


def as_return_matrix(returns: object, name: str = "returns") -> FloatArray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2:
        raise RiskInputError(f"{name} must be a 2-D (n_obs x n_assets) array")
    if arr.size == 0:
        raise RiskInputError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise RiskInputError(f"{name} contains non-finite values")
    return arr


def as_weights(weights: object, name: str = "weights") -> FloatArray:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1:
        raise RiskInputError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(arr)):
        raise RiskInputError(f"{name} contains non-finite values")
    return arr


def as_covariance(cov: object, name: str = "covariance") -> FloatArray:
    arr = np.asarray(cov, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise RiskInputError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(arr)):
        raise RiskInputError(f"{name} contains non-finite values")
    if not np.allclose(arr, arr.T, atol=1e-10):
        raise RiskInputError(f"{name} must be symmetric")
    return arr


def check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise RiskInputError(f"confidence must lie strictly in (0, 1), got {confidence}")


def tail_alpha(confidence: float) -> float:
    """Tail probability ``alpha = 1 - confidence``."""
    check_confidence(confidence)
    return 1.0 - confidence


def check_lengths_match(a: FloatArray, b: FloatArray, name_b: str = "benchmark") -> None:
    if a.shape[0] != b.shape[0]:
        raise DimensionMismatchError(a.shape[0], b.shape[0], name=name_b)
