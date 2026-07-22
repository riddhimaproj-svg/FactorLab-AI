"""Shared fixtures for the backtesting test suite."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import MarketData


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def market_data(rng) -> MarketData:
    n, k = 400, 3
    dates = np.datetime64("2020-01-01") + np.arange(n)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return MarketData(dates, ("A", "B", "C"), prices)


@pytest.fixture
def flat_market_data() -> MarketData:
    """Deterministic slowly-rising prices for analytic checks."""
    n = 60
    dates = np.datetime64("2021-01-01") + np.arange(n)
    prices = np.column_stack([
        100.0 * (1.001 ** np.arange(n)),
        100.0 * (1.0005 ** np.arange(n)),
    ])
    return MarketData(dates, ("X", "Y"), prices)
