"""Analytical validation of benchmark-relative metrics."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_portfolio.analytics import relative as R
from factorlab_portfolio.errors import DimensionMismatchError


def test_beta_self_is_one() -> None:
    b = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    assert R.beta(b, b) == pytest.approx(1.0)


def test_beta_scaled() -> None:
    b = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    assert R.beta(2.0 * b, b) == pytest.approx(2.0)
    assert R.beta(-b, b) == pytest.approx(-1.0)


def test_beta_constant_benchmark_is_nan() -> None:
    assert np.isnan(R.beta(np.array([0.01, 0.02, 0.03]), np.full(3, 0.01)))
    assert np.isnan(R.beta(np.array([0.01]), np.array([0.01])))  # n<2


def test_active_return() -> None:
    r = np.array([0.02, 0.03, 0.01])
    b = np.array([0.01, 0.01, 0.01])
    assert R.active_return(r, b, 12.0) == pytest.approx(np.mean(r - b) * 12.0)


def test_tracking_error() -> None:
    r = np.array([0.02, 0.03, 0.01, 0.04])
    b = np.array([0.01, 0.02, 0.00, 0.02])
    expected = np.std(r - b, ddof=1) * np.sqrt(252)
    assert R.tracking_error(r, b, 252) == pytest.approx(expected)


def test_information_ratio() -> None:
    r = np.array([0.02, 0.03, 0.01, 0.04])
    b = np.array([0.01, 0.02, 0.00, 0.02])
    active = r - b
    expected = np.mean(active) / np.std(active, ddof=1) * np.sqrt(252)
    assert R.information_ratio(r, b, 252) == pytest.approx(expected)


def test_information_ratio_zero_te_is_nan() -> None:
    r = np.array([0.02, 0.03, 0.04])
    b = r - 0.01  # constant active return -> zero tracking error
    assert np.isnan(R.information_ratio(r, b, 252))


def test_treynor_ratio() -> None:
    b = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    r = 1.5 * b + 0.001
    beta = R.beta(r, b)
    expected = np.mean(r - 0.0) * 252 / beta
    assert R.treynor_ratio(r, b, 0.0, 252) == pytest.approx(expected)


def test_treynor_zero_beta_is_nan() -> None:
    r = np.array([0.02, 0.03, 0.01])
    b = np.full(3, 0.01)  # zero-variance benchmark -> beta nan
    assert np.isnan(R.treynor_ratio(r, b, 0.0, 252))


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(DimensionMismatchError):
        R.beta(np.array([0.01, 0.02]), np.array([0.01, 0.02, 0.03]))
