"""Analytical validation of absolute performance metrics."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_portfolio.analytics import performance as P


def test_wealth_index() -> None:
    np.testing.assert_allclose(P.wealth_index(np.array([0.1, 0.1])), [1.1, 1.21])
    np.testing.assert_allclose(P.wealth_index(np.array([0.1]), initial=100.0), [110.0])


def test_cumulative_return() -> None:
    assert P.cumulative_return(np.array([0.1, 0.1])) == pytest.approx(0.21)
    assert P.cumulative_return(np.array([0.5, -0.5])) == pytest.approx(-0.25)
    assert np.isnan(P.cumulative_return(np.array([])))


def test_mean_return() -> None:
    assert P.mean_return(np.array([0.1, 0.2, 0.3])) == pytest.approx(0.2)
    assert np.isnan(P.mean_return(np.array([])))


def test_cagr_single_period() -> None:
    # ppy=1, one period: CAGR == the period return.
    assert P.cagr(np.array([0.5]), 1.0) == pytest.approx(0.5)


def test_cagr_two_periods_one_year() -> None:
    # ppy=2, two periods == one year: CAGR == total return.
    assert P.cagr(np.array([0.1, 0.1]), 2.0) == pytest.approx(0.21)


def test_cagr_wipeout_is_nan() -> None:
    assert np.isnan(P.cagr(np.array([-1.0]), 1.0))
    assert np.isnan(P.cagr(np.array([]), 12.0))


def test_annualized_volatility() -> None:
    r = np.array([0.0, 0.02])
    expected = np.std(r, ddof=1) * np.sqrt(252)
    assert P.annualized_volatility(r, 252) == pytest.approx(expected)
    assert np.isnan(P.annualized_volatility(np.array([0.01]), 252))  # n<2


def test_constant_returns_zero_volatility() -> None:
    # Numerically zero up to floating-point precision.
    assert P.annualized_volatility(np.full(10, 0.01), 252) == pytest.approx(0.0, abs=1e-12)


def test_downside_deviation() -> None:
    r = np.array([0.1, -0.2, -0.1, 0.05])
    # shortfalls vs 0: [0, -0.2, -0.1, 0]; mean sq = (0.04+0.01)/4 = 0.0125
    assert P.downside_deviation(r, target=0.0, periods_per_year=1.0) == pytest.approx(
        np.sqrt(0.0125)
    )
    # no downside -> 0
    assert P.downside_deviation(np.array([0.1, 0.2]), 0.0, 1.0) == 0.0


def test_sharpe_ratio() -> None:
    r = np.array([0.01, 0.02, 0.03])
    expected = np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)
    assert P.sharpe_ratio(r, 0.0, 252) == pytest.approx(expected)


def test_sharpe_zero_vol_is_nan() -> None:
    assert np.isnan(P.sharpe_ratio(np.full(5, 0.01), 0.0, 252))
    assert np.isnan(P.sharpe_ratio(np.array([0.01]), 0.0, 252))  # n<2


def test_sharpe_risk_free_reduces_ratio() -> None:
    r = np.array([0.02, 0.03, 0.01, 0.04])
    assert P.sharpe_ratio(r, 0.0, 252) > P.sharpe_ratio(r, 0.01, 252)


def test_sortino_ratio() -> None:
    r = np.array([0.02, -0.01, 0.03, -0.02])
    dd = P.downside_deviation(r, 0.0, 1.0)
    expected = np.mean(r) / dd * np.sqrt(252)
    assert P.sortino_ratio(r, 0.0, 0.0, 252) == pytest.approx(expected)


def test_sortino_no_downside_is_nan() -> None:
    assert np.isnan(P.sortino_ratio(np.array([0.01, 0.02]), 0.0, 0.0, 252))


def test_omega_ratio() -> None:
    r = np.array([0.1, -0.05, 0.02, -0.03])
    # gains = 0.12, losses = 0.08 -> 1.5
    assert P.omega_ratio(r, 0.0) == pytest.approx(1.5)


def test_omega_all_gains_is_inf() -> None:
    assert P.omega_ratio(np.array([0.01, 0.02])) == float("inf")
    assert np.isnan(P.omega_ratio(np.array([])))


def test_calmar_ratio() -> None:
    r = np.array([0.1, -0.5, 0.2, 0.1])  # max dd = -0.5
    from factorlab_portfolio.analytics.drawdown import max_drawdown

    expected = P.cagr(r, 12.0) / abs(max_drawdown(r))
    assert P.calmar_ratio(r, 12.0) == pytest.approx(expected)


def test_calmar_no_drawdown_is_nan() -> None:
    assert np.isnan(P.calmar_ratio(np.array([0.1, 0.1, 0.1]), 12.0))


def test_rejects_2d_input() -> None:
    with pytest.raises(ValueError):
        P.cumulative_return(np.zeros((2, 2)))
