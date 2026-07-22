"""Unit tests for the input-validation guards."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant.core.errors import (
    DimensionMismatchError,
    InsufficientDataError,
    NonFiniteError,
)
from factorlab_quant.utils.align import apply_mask, complete_case_mask
from factorlab_quant.utils.validation import (
    as_float_matrix,
    as_float_vector,
    check_finite,
    check_lengths_match,
    check_min_observations,
)


def test_as_float_vector_accepts_sequence() -> None:
    out = as_float_vector([1, 2, 3])
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_as_float_vector_rejects_2d() -> None:
    with pytest.raises(DimensionMismatchError):
        as_float_vector(np.ones((3, 2)))  # type: ignore[arg-type]


def test_as_float_matrix_promotes_1d_to_column() -> None:
    out = as_float_matrix(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (3, 1)


def test_check_finite_flags_nan() -> None:
    with pytest.raises(NonFiniteError):
        check_finite(np.array([1.0, np.nan, 3.0]))


def test_check_finite_flags_inf() -> None:
    with pytest.raises(NonFiniteError):
        check_finite(np.array([1.0, np.inf]))


def test_check_lengths_match_raises_on_mismatch() -> None:
    with pytest.raises(DimensionMismatchError):
        check_lengths_match(("a", np.ones(3)), ("b", np.ones(4)))


def test_check_min_observations_enforces_hard_floor() -> None:
    # 2 params -> need at least 3 observations.
    with pytest.raises(InsufficientDataError):
        check_min_observations(n_obs=2, n_params=2)


def test_check_min_observations_respects_caller_minimum() -> None:
    with pytest.raises(InsufficientDataError):
        check_min_observations(n_obs=5, n_params=2, minimum=30)


def test_complete_case_mask_drops_rows_with_any_nan() -> None:
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([1.0, 2.0, np.nan, 4.0])
    mask = complete_case_mask(a, b)
    assert mask.tolist() == [True, False, False, True]
    a2, b2 = apply_mask(mask, a, b)
    assert a2.tolist() == [1.0, 4.0]
    assert b2.tolist() == [1.0, 4.0]


def test_complete_case_mask_handles_2d_design() -> None:
    design = np.array([[1.0, 2.0], [1.0, np.nan], [1.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0])
    mask = complete_case_mask(y, design)
    assert mask.tolist() == [True, False, True]
