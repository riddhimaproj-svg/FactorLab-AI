"""Tests for PerformanceReport."""

from __future__ import annotations

import json

import numpy as np
import pytest

from factorlab_portfolio.report import PerformanceReport
from factorlab_portfolio.returns import ReturnSeries


@pytest.fixture
def series() -> ReturnSeries:
    rng = np.random.default_rng(3)
    return ReturnSeries(rng.normal(0.0005, 0.01, 300), periods_per_year=252.0, name="fund")


@pytest.fixture
def benchmark() -> ReturnSeries:
    rng = np.random.default_rng(4)
    return ReturnSeries(rng.normal(0.0003, 0.009, 300), periods_per_year=252.0, name="bench")


def test_report_without_benchmark(series) -> None:
    rep = PerformanceReport.from_series(series, risk_free=0.0)
    assert not rep.has_benchmark
    assert np.isnan(rep.beta)
    assert np.isnan(rep.information_ratio)
    assert rep.n_observations == 300
    assert np.isfinite(rep.sharpe_ratio)
    assert rep.max_drawdown <= 0.0


def test_report_with_benchmark(series, benchmark) -> None:
    rep = PerformanceReport.from_series(series, benchmark=benchmark, risk_free=0.0)
    assert rep.has_benchmark
    assert np.isfinite(rep.beta)
    assert np.isfinite(rep.tracking_error)
    assert np.isfinite(rep.information_ratio)


def test_report_matches_series_methods(series, benchmark) -> None:
    rep = PerformanceReport.from_series(series, benchmark=benchmark, risk_free=0.0)
    assert rep.total_return == pytest.approx(series.total_return())
    assert rep.sharpe_ratio == pytest.approx(series.sharpe(0.0))
    assert rep.beta == pytest.approx(series.beta(benchmark))
    assert rep.information_ratio == pytest.approx(series.information_ratio(benchmark))


def test_report_roundtrip(series, benchmark) -> None:
    rep = PerformanceReport.from_series(series, benchmark=benchmark, risk_free=0.001)
    restored = PerformanceReport.from_dict(rep.to_dict())
    assert restored == rep


def test_report_json_valid(series) -> None:
    rep = PerformanceReport.from_series(series)
    payload = json.dumps(rep.to_dict())
    assert json.loads(payload)["name"] == "fund"


def test_summary_sections(series, benchmark) -> None:
    text = PerformanceReport.from_series(series, benchmark=benchmark).summary()
    assert "Performance Report" in text
    assert "Sharpe ratio" in text
    assert "Max drawdown" in text
    assert "Benchmark-relative" in text
    assert "Information ratio" in text


def test_summary_omits_relative_without_benchmark(series) -> None:
    text = PerformanceReport.from_series(series).summary()
    assert "Benchmark-relative" not in text
