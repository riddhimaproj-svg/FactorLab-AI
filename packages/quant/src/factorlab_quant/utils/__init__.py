"""Shared, dependency-light utilities: validation guards and alignment."""

from __future__ import annotations

from factorlab_quant.utils.align import apply_mask, complete_case_mask
from factorlab_quant.utils.validation import (
    as_float_matrix,
    as_float_vector,
    check_finite,
    check_lengths_match,
    check_min_observations,
)

__all__ = [
    "apply_mask",
    "as_float_matrix",
    "as_float_vector",
    "check_finite",
    "check_lengths_match",
    "check_min_observations",
    "complete_case_mask",
]
