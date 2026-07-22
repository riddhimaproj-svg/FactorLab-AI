"""Shared fixtures and reference values for the derivatives test suite."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_derivatives import MarketData, Option, OptionType


@pytest.fixture
def atm_market() -> MarketData:
    """A canonical at-the-money market: S=100, r=5%, sigma=20%."""
    return MarketData(spot=100.0, rate=0.05, volatility=0.2)


@pytest.fixture
def atm_call() -> Option:
    return Option(OptionType.CALL, strike=100.0, maturity=1.0)


@pytest.fixture
def atm_put() -> Option:
    return Option(OptionType.PUT, strike=100.0, maturity=1.0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
