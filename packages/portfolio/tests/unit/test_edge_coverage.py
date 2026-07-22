"""Edge-case coverage: empty/short series, aliases, and remaining accessors."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_portfolio.analytics import performance as P
from factorlab_portfolio.analytics import relative as R
from factorlab_portfolio.errors import InsufficientDataError
from factorlab_portfolio.returns import ReturnSeries

_EMPTY = np.array([])
_ONE = np.array([0.01])


# -- errors ---------------------------------------------------------------- #
def test_insufficient_data_error_message() -> None:
    err = InsufficientDataError(1, 2, statistic="volatility")
    assert err.n_obs == 1 and err.minimum == 2
    assert "volatility" in str(err)


# -- performance edges ----------------------------------------------------- #
def test_annualized_return_alias() -> None:
    r = np.array([0.1, 0.1])
    assert P.annualized_return(r, 2.0) == pytest.approx(P.cagr(r, 2.0))


def test_downside_deviation_empty() -> None:
    assert np.isnan(P.downside_deviation(_EMPTY))


def test_sortino_empty() -> None:
    assert np.isnan(P.sortino_ratio(_EMPTY))


def test_calmar_wipeout_growth_nan() -> None:
    # Wealth goes to zero mid-series -> CAGR nan -> Calmar nan.
    assert np.isnan(P.calmar_ratio(np.array([-1.0, 0.5]), 12.0))


# -- relative edges -------------------------------------------------------- #
def test_active_return_empty() -> None:
    assert np.isnan(R.active_return(_EMPTY, _EMPTY, 252.0))


def test_tracking_error_single_obs() -> None:
    assert np.isnan(R.tracking_error(_ONE, _ONE, 252.0))


def test_information_ratio_single_obs() -> None:
    assert np.isnan(R.information_ratio(_ONE, _ONE, 252.0))


def test_treynor_single_obs() -> None:
    assert np.isnan(R.treynor_ratio(_ONE, _ONE, 0.0, 252.0))


# -- ReturnSeries convenience accessors ------------------------------------ #
def test_return_series_accessors() -> None:
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.015, 0.0])
    s = ReturnSeries(r, periods_per_year=12.0)
    bench = ReturnSeries(np.array([0.01, 0.0, 0.02, -0.01, 0.01, 0.0]), periods_per_year=12.0)

    assert s.mean() == pytest.approx(np.mean(r))
    assert s.annualized_return() == pytest.approx(P.cagr(r, 12.0))
    assert np.isfinite(s.sortino())
    assert np.isfinite(s.downside_deviation())
    assert s.omega() >= 0.0
    assert s.max_drawdown() <= 0.0
    assert isinstance(s.max_drawdown_duration(), int)
    assert s.drawdown_series().shape == (6,)
    assert s.time_to_recovery() is None or isinstance(s.time_to_recovery(), int)
    assert np.isfinite(s.calmar()) or np.isnan(s.calmar())
    assert np.isfinite(s.wealth_index()[-1])

    # relative accessors
    assert np.isfinite(s.beta(bench))
    assert np.isfinite(s.active_return(bench))
    assert np.isfinite(s.tracking_error(bench))
    assert np.isfinite(s.information_ratio(bench))
    assert np.isfinite(s.treynor(bench))
