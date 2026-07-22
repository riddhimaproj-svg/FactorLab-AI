"""Input guards shared by estimators and models.

Validation runs *before* any numerical work.  Its job is to turn malformed
inputs into precise, typed exceptions (never a cryptic NumPy broadcast error
three call frames deep).  Every public entry point of the engine funnels its
inputs through these helpers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from factorlab_quant.core.errors import (
    DimensionMismatchError,
    InsufficientDataError,
    NonFiniteError,
)
from factorlab_quant.core.types import FloatArray

__all__ = [
    "as_float_matrix",
    "as_float_vector",
    "check_finite",
    "check_lengths_match",
    "check_min_observations",
]


def as_float_vector(x: Sequence[float] | FloatArray, name: str = "input") -> FloatArray:
    """Coerce ``x`` to a contiguous 1-D ``float64`` array.

    Accepts Python sequences, NumPy arrays, and array-likes (anything with an
    ``__array__``, e.g. a pandas Series).  Rejects multi-dimensional inputs so
    that shape bugs surface immediately.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise DimensionMismatchError(expected=1, received=arr.ndim, name=f"{name} (ndim)")
    return np.ascontiguousarray(arr)


def as_float_matrix(x: FloatArray, name: str = "design matrix") -> FloatArray:
    """Coerce ``x`` to a contiguous 2-D ``float64`` array.

    A 1-D input is promoted to a single-column matrix, which is the common case
    of a one-regressor model before the intercept is prepended.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DimensionMismatchError(expected=2, received=arr.ndim, name=f"{name} (ndim)")
    return np.ascontiguousarray(arr)


def check_finite(arr: FloatArray, name: str = "input") -> None:
    """Raise :class:`NonFiniteError` if ``arr`` contains NaN or infinity.

    The engine's contract is that *cleaning* (dropping or imputing missing
    observations) is the caller's responsibility -- typically the data
    pipeline.  The engine refuses to silently propagate non-finite values into
    a regression, where they would poison every downstream statistic.
    """
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise NonFiniteError(
            f"{name} contains {n_bad} non-finite value(s) (NaN or inf). "
            f"Align and clean the series before estimation."
        )


def check_lengths_match(*arrays: tuple[str, FloatArray]) -> None:
    """Ensure every ``(name, array)`` pair shares the leading dimension."""
    if not arrays:
        return
    _, ref = arrays[0]
    ref_len = ref.shape[0]
    for name, arr in arrays[1:]:
        if arr.shape[0] != ref_len:
            raise DimensionMismatchError(expected=ref_len, received=arr.shape[0], name=name)


def check_min_observations(n_obs: int, n_params: int, minimum: int | None = None) -> None:
    """Guard against under-identified or statistically unreliable regressions.

    Requires at least ``n_params + 1`` observations for positive residual
    degrees of freedom.  A caller-supplied ``minimum`` raises the bar to a
    level where inference (t-tests, HAC covariance) is meaningful.
    """
    hard_floor = n_params + 1
    required = max(hard_floor, minimum) if minimum is not None else hard_floor
    if n_obs < required:
        detail = (
            f"Model has {n_params} parameter(s), so at least {hard_floor} "
            f"observations are needed for a non-degenerate fit."
        )
        raise InsufficientDataError(n_obs=n_obs, minimum=required, detail=detail)
