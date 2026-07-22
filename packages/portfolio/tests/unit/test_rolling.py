"""Validation of rolling analytics."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_portfolio.analytics import rolling as RO
from factorlab_portfolio.errors import DimensionMismatchError


def test_rolling_return() -> None:
    out = RO.rolling_return(np.array([0.1, 0.1, 0.1]), window=2)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [0.21, 0.21])


def test_rolling_return_window_one() -> None:
    out = RO.rolling_return(np.array([0.1, 0.2]), window=1)
    np.testing.assert_allclose(out, [0.1, 0.2])


def test_rolling_volatility() -> None:
    r = np.array([0.0, 0.02, 0.0, 0.02])
    out = RO.rolling_volatility(r, window=2, periods_per_year=1.0)
    assert np.isnan(out[0])
    expected = np.std(r[0:2], ddof=1)
    assert out[1] == pytest.approx(expected)


def test_rolling_sharpe_length_and_nan_prefix() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 50)
    out = RO.rolling_sharpe(r, window=10)
    assert out.shape == (50,)
    assert np.all(np.isnan(out[:9]))
    assert np.all(np.isfinite(out[9:]))


def test_rolling_beta() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(0.0, 0.01, 40)
    r = 1.3 * b + rng.normal(0.0, 0.001, 40)
    out = RO.rolling_beta(r, b, window=20)
    assert np.all(np.isnan(out[:19]))
    # beta should be near 1.3 in the complete windows
    assert out[-1] == pytest.approx(1.3, abs=0.15)


def test_rolling_beta_dimension_mismatch() -> None:
    with pytest.raises(DimensionMismatchError):
        RO.rolling_beta(np.zeros(10), np.zeros(9), window=3)


def test_window_validation() -> None:
    with pytest.raises(ValueError):
        RO.rolling_return(np.zeros(5), window=0)
    with pytest.raises(ValueError):
        RO.rolling_return(np.zeros(5), window=6)
