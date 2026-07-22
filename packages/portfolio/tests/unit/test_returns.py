"""Tests for the ReturnSeries object."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_portfolio.analytics import drawdown as DD
from factorlab_portfolio.analytics import performance as P
from factorlab_portfolio.errors import DimensionMismatchError, PortfolioValidationError
from factorlab_portfolio.returns import ReturnSeries


def _series(values, dates=None, ppy=252.0) -> ReturnSeries:
    return ReturnSeries(np.asarray(values, dtype=float), dates, ppy)


# -- Validation ------------------------------------------------------------ #
def test_rejects_non_finite() -> None:
    with pytest.raises(PortfolioValidationError):
        _series([0.1, np.nan])


def test_rejects_2d() -> None:
    with pytest.raises(PortfolioValidationError):
        ReturnSeries(np.zeros((2, 2)))


def test_rejects_nonpositive_ppy() -> None:
    with pytest.raises(PortfolioValidationError):
        _series([0.1, 0.2], ppy=0.0)


def test_dates_length_must_match() -> None:
    with pytest.raises(DimensionMismatchError):
        ReturnSeries(np.array([0.1, 0.2]), np.array(["2024-01-01"], dtype="datetime64[D]"))


def test_values_read_only() -> None:
    s = _series([0.1, 0.2])
    with pytest.raises(ValueError):
        s.values[0] = 0.0


# -- Delegation to analytics ---------------------------------------------- #
def test_methods_match_free_functions() -> None:
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    s = _series(r, ppy=252.0)
    assert s.total_return() == pytest.approx(P.cumulative_return(r))
    assert s.cagr() == pytest.approx(P.cagr(r, 252.0))
    assert s.volatility() == pytest.approx(P.annualized_volatility(r, 252.0))
    assert s.sharpe(0.0) == pytest.approx(P.sharpe_ratio(r, 0.0, 252.0))
    assert s.max_drawdown() == pytest.approx(DD.max_drawdown(r))


def test_excess() -> None:
    s = _series([0.02, 0.03])
    ex = s.excess(0.01)
    np.testing.assert_allclose(ex.values, [0.01, 0.02])


def test_from_prices() -> None:
    s = ReturnSeries.from_prices([100.0, 110.0, 121.0])
    np.testing.assert_allclose(s.values, [0.1, 0.1])


def test_from_prices_too_short() -> None:
    with pytest.raises(PortfolioValidationError):
        ReturnSeries.from_prices([100.0])


def test_from_prices_with_dates() -> None:
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    s = ReturnSeries.from_prices([100.0, 110.0, 121.0], dates=dates)
    assert s.dates is not None and s.dates.shape[0] == 2  # one fewer than prices


# -- Benchmark alignment --------------------------------------------------- #
def test_relative_requires_equal_length_without_dates() -> None:
    s = _series([0.01, 0.02, 0.03])
    bench = _series([0.01, 0.02])
    with pytest.raises(DimensionMismatchError):
        s.beta(bench)


def test_relative_aligns_on_common_dates() -> None:
    d1 = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    d2 = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    s = _series([0.01, 0.02, 0.03], dates=d1)
    bench = _series([0.02, 0.03, 0.04], dates=d2)
    r, b = s.aligned_pair(bench)
    np.testing.assert_allclose(r, [0.02, 0.03])
    np.testing.assert_allclose(b, [0.02, 0.03])
    assert s.beta(bench) == pytest.approx(1.0)


def test_no_common_dates_raises() -> None:
    d1 = np.array(["2024-01-01"], dtype="datetime64[D]")
    d2 = np.array(["2025-01-01"], dtype="datetime64[D]")
    with pytest.raises(DimensionMismatchError):
        _series([0.01], dates=d1).aligned_pair(_series([0.02], dates=d2))


# -- Rolling --------------------------------------------------------------- #
def test_rolling_methods() -> None:
    rng = np.random.default_rng(0)
    s = _series(rng.normal(0.001, 0.01, 60))
    assert s.rolling_return(10).shape == (60,)
    assert s.rolling_volatility(10).shape == (60,)
    assert s.rolling_sharpe(10).shape == (60,)
    bench = _series(rng.normal(0.001, 0.01, 60))
    assert s.rolling_beta(bench, 10).shape == (60,)


# -- Serialization --------------------------------------------------------- #
def test_roundtrip_without_dates() -> None:
    s = _series([0.01, -0.02, 0.03], ppy=12.0)
    restored = ReturnSeries.from_dict(s.to_dict())
    np.testing.assert_allclose(restored.values, s.values)
    assert restored.periods_per_year == 12.0
    assert restored.dates is None


def test_roundtrip_with_dates() -> None:
    dates = np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[D]")
    s = ReturnSeries(np.array([0.01, 0.02]), dates, 12.0, "fund")
    restored = ReturnSeries.from_dict(s.to_dict())
    assert (restored.dates == s.dates).all()
    assert restored.name == "fund"


def test_performance_report_helper() -> None:
    s = _series([0.01, -0.02, 0.03, 0.0, 0.015])
    report = s.performance_report(risk_free=0.0)
    assert report.n_observations == 5
    assert not report.has_benchmark
