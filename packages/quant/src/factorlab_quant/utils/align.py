"""Observation-alignment helpers.

Time-series regressions require that every series share the same index and
have no gaps.  In the full platform this is the data pipeline's job, but the
engine still offers a minimal, dependency-light alignment primitive so it is
usable standalone (a notebook user handing in three raw arrays) without pulling
in pandas.
"""

from __future__ import annotations

import numpy as np

from factorlab_quant.core.types import FloatArray

__all__ = ["apply_mask", "complete_case_mask"]


def complete_case_mask(*arrays: FloatArray) -> FloatArray:
    """Boolean mask selecting rows where *every* input is finite.

    This is listwise deletion (complete-case analysis).  It is the correct
    default for asset-pricing regressions: an observation is only usable if the
    asset return, the market return, and the risk-free rate are all present for
    that period.
    """
    if not arrays:
        raise ValueError("complete_case_mask requires at least one array")
    n = arrays[0].shape[0]
    mask = np.ones(n, dtype=bool)
    for arr in arrays:
        if arr.shape[0] != n:
            raise ValueError("all arrays must share the leading dimension")
        finite = np.isfinite(arr)
        # Collapse any trailing dimensions (e.g. a design matrix) to per-row.
        if finite.ndim > 1:
            finite = finite.all(axis=tuple(range(1, finite.ndim)))
        mask &= finite
    return mask


def apply_mask(mask: FloatArray, *arrays: FloatArray) -> tuple[FloatArray, ...]:
    """Apply a row mask to each array, returning the filtered copies."""
    return tuple(np.ascontiguousarray(arr[mask]) for arr in arrays)
