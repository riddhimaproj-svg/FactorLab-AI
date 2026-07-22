"""Internal input validation shared across the engine.

Conventions
-----------
* All rates, yields, and volatilities are **annualized decimals** (``0.05`` = 5%).
* Maturity ``T`` is in **years**.
* Prices are per unit of the underlying.
* Greeks are the raw partial derivatives (per unit): vega is ``dV/dsigma`` (a 1%
  vol move is ``vega * 0.01``); rho is ``dV/dr``; theta is per **year** (a
  one-day decay is roughly ``theta / 365``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_derivatives.errors import DerivativesInputError

__all__ = ["FloatArray", "as_return_vector", "check_non_negative", "check_positive"]

FloatArray = NDArray[np.float64]


def check_positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise DerivativesInputError(f"{name} must be finite and > 0, got {value}")
    return float(value)


def check_non_negative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise DerivativesInputError(f"{name} must be finite and >= 0, got {value}")
    return float(value)


def as_return_vector(returns: object, name: str = "returns") -> FloatArray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise DerivativesInputError(f"{name} must be a 1-D array")
    if arr.size == 0:
        raise DerivativesInputError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise DerivativesInputError(f"{name} contains non-finite values")
    return arr
