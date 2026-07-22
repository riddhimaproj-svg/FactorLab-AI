"""Tests for MarketData."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import MarketData
from factorlab_backtesting.errors import BacktestInputError


def test_shape_and_accessors(market_data) -> None:
    assert market_data.n_periods == 400
    assert market_data.n_assets == 3
    assert market_data.prices_at(0).shape == (3,)


def test_simple_returns(flat_market_data) -> None:
    r = flat_market_data.simple_returns()
    assert r.shape == (59, 2)
    np.testing.assert_allclose(r[:, 0], 0.001, atol=1e-9)


def test_returns_window_no_lookahead(flat_market_data) -> None:
    # window ending at index 10 uses only prices[..10]
    w = flat_market_data.returns_window(10, lookback=5)
    assert w.shape[0] <= 5
    # equals the manual trailing returns
    expected = flat_market_data.prices[6:11] / flat_market_data.prices[5:10] - 1.0
    np.testing.assert_allclose(w, expected)


def test_validation() -> None:
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    with pytest.raises(BacktestInputError):
        MarketData(dates, ("A",), np.array([[1.0], [2.0], [3.0]]))  # shape mismatch
    with pytest.raises(BacktestInputError):
        MarketData(dates, ("A",), np.array([[1.0], [-1.0]]))  # non-positive price
    with pytest.raises(BacktestInputError):
        MarketData(np.array(["2020-01-02", "2020-01-01"], dtype="datetime64[D]"),
                   ("A",), np.array([[1.0], [2.0]]))  # unsorted dates


def test_from_prices() -> None:
    md = MarketData.from_prices(["2020-01-01", "2020-01-02"], ["A"], np.array([[10.0], [11.0]]))
    assert md.n_periods == 2
