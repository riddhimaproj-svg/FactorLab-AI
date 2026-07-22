"""Property-based invariants for the analytics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from factorlab_portfolio.analytics import drawdown as DD
from factorlab_portfolio.analytics import performance as P
from factorlab_portfolio.analytics import relative as R

pytestmark = pytest.mark.property

_SETTINGS = settings(max_examples=100, deadline=None)

# Returns bounded away from -1 so wealth stays positive.
_returns = st.lists(
    st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=200,
).map(np.array)


@_SETTINGS
@given(returns=_returns)
def test_cumulative_return_matches_wealth(returns) -> None:
    cum = P.cumulative_return(returns)
    assert cum == pytest.approx(P.wealth_index(returns)[-1] - 1.0)


@_SETTINGS
@given(returns=_returns)
def test_drawdown_bounded(returns) -> None:
    dd = DD.drawdown_series(returns)
    assert dd.max() <= 1e-9
    assert dd.min() >= -1.0 - 1e-9
    assert DD.max_drawdown(returns) <= 1e-9


@_SETTINGS
@given(returns=_returns)
def test_volatility_nonnegative(returns) -> None:
    vol = P.annualized_volatility(returns, 252.0)
    assert vol >= 0.0 or np.isnan(vol)


@_SETTINGS
@given(returns=_returns)
def test_omega_nonnegative(returns) -> None:
    omega = P.omega_ratio(returns)
    assert omega >= 0.0 or np.isinf(omega) or np.isnan(omega)


@_SETTINGS
@given(returns=_returns, scale=st.floats(min_value=0.1, max_value=10.0))
def test_sharpe_scale_invariant(returns, scale) -> None:
    """Sharpe (rf=0) is invariant to positive scaling of returns."""
    base = P.sharpe_ratio(returns, 0.0, 252.0)
    scaled = P.sharpe_ratio(scale * returns, 0.0, 252.0)
    if np.isnan(base):
        assert np.isnan(scaled)
    else:
        assert scaled == pytest.approx(base, rel=1e-9)


@_SETTINGS
@given(returns=_returns)
def test_beta_self_is_one(returns) -> None:
    b = R.beta(returns, returns)
    if not np.isnan(b):
        assert b == pytest.approx(1.0, abs=1e-9)


@_SETTINGS
@given(returns=_returns)
def test_sortino_ge_zero_when_mean_positive(returns) -> None:
    """When mean return is positive, Sortino (if defined) is positive."""
    if np.mean(returns) > 0:
        s = P.sortino_ratio(returns, 0.0, 0.0, 252.0)
        assert np.isnan(s) or s > 0.0
