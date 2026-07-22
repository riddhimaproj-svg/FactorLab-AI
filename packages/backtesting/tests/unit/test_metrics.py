"""Tests for backtest-specific metrics."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import metrics
from factorlab_backtesting.errors import BacktestInputError


def test_alpha_beta_recovers_construction() -> None:
    rng = np.random.default_rng(0)
    bench = rng.normal(0.0, 0.01, 500)
    beta_true, alpha_true = 1.5, 0.0004
    returns = alpha_true + beta_true * bench  # exact linear relation
    alpha, beta = metrics.alpha_beta(returns, bench, risk_free=0.0, periods_per_year=252)
    assert beta == pytest.approx(beta_true, rel=1e-6)
    assert alpha == pytest.approx(alpha_true * 252, rel=1e-6)


def test_alpha_beta_length_mismatch() -> None:
    with pytest.raises(BacktestInputError):
        metrics.alpha_beta(np.zeros(3), np.zeros(4))


def test_win_rate() -> None:
    assert metrics.win_rate(np.array([0.1, -0.1, 0.2, 0.0])) == pytest.approx(0.5)
    assert np.isnan(metrics.win_rate(np.array([])))


def test_hit_ratio() -> None:
    r = np.array([0.02, 0.01, 0.03])
    b = np.array([0.01, 0.02, 0.01])
    assert metrics.hit_ratio(r, b) == pytest.approx(2 / 3)


def test_hit_ratio_length_mismatch() -> None:
    with pytest.raises(BacktestInputError):
        metrics.hit_ratio(np.zeros(2), np.zeros(3))


def test_annualized_turnover() -> None:
    assert metrics.annualized_turnover(0.1, 12.0) == pytest.approx(1.2)


def test_alpha_beta_constant_benchmark_nan() -> None:
    a, b = metrics.alpha_beta(np.array([0.01, 0.02, 0.03]), np.full(3, 0.01))
    assert np.isnan(a) and np.isnan(b)
