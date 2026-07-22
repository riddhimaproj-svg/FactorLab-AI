"""Property-based invariants and numerical-stability tests for the risk engine."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_risk import attribution as A
from factorlab_risk.var import decomposition as D
from factorlab_risk.var import historical as H
from factorlab_risk.var import parametric as P

pytestmark = pytest.mark.property

_SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def return_arrays(draw):
    n = draw(st.integers(min_value=20, max_value=500))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, size=n)


@st.composite
def portfolios(draw):
    n = draw(st.integers(min_value=2, max_value=6))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    L = np.tril(rng.uniform(0.05, 0.3, size=(n, n)))
    cov = L @ L.T + np.eye(n) * 1e-4
    w = rng.uniform(0.0, 1.0, size=n)
    w = w / w.sum()
    return w, cov


@_SETTINGS
@given(r=return_arrays(), c=st.floats(min_value=0.90, max_value=0.99))
def test_es_ge_var(r, c) -> None:
    assert H.historical_expected_shortfall(r, c) >= H.historical_var(r, c) - 1e-12


@_SETTINGS
@given(c=st.floats(min_value=0.90, max_value=0.999), sigma=st.floats(0.001, 0.1))
def test_parametric_es_ge_var(c, sigma) -> None:
    assert P.parametric_expected_shortfall(mean=0.0, std=sigma, confidence=c) >= (
        P.parametric_var(mean=0.0, std=sigma, confidence=c) - 1e-12
    )


@_SETTINGS
@given(c=st.floats(min_value=0.91, max_value=0.99), sigma=st.floats(0.005, 0.05))
def test_var_monotone_in_confidence(c, sigma) -> None:
    lower = P.parametric_var(mean=0.0, std=sigma, confidence=c)
    higher = P.parametric_var(mean=0.0, std=sigma, confidence=min(c + 0.005, 0.999))
    assert higher >= lower - 1e-12  # higher confidence -> larger loss


@_SETTINGS
@given(p=portfolios(), c=st.floats(min_value=0.90, max_value=0.99))
def test_component_var_sums_to_total(p, c) -> None:
    w, cov = p
    total = D.portfolio_var(w, cov, c)
    comp = D.component_var(w, cov, c)
    assert np.sum(comp) == pytest.approx(total, rel=1e-8, abs=1e-12)


@_SETTINGS
@given(p=portfolios())
def test_risk_contributions_sum_to_volatility(p) -> None:
    w, cov = p
    ccr = A.component_contribution_to_risk(w, cov)
    assert np.sum(ccr) == pytest.approx(A.portfolio_volatility(w, cov), rel=1e-8)


@_SETTINGS
@given(p=portfolios())
def test_percentage_contributions_sum_to_one(p) -> None:
    w, cov = p
    pct = A.percentage_contribution_to_risk(w, cov)
    assert np.sum(pct) == pytest.approx(1.0, abs=1e-8)


def test_numerical_stability_tiny_returns() -> None:
    r = np.full(500, 1e-8)
    r[0] = -1e-7
    # near-constant series should not blow up
    assert np.isfinite(H.historical_var(r, 0.95))
    assert np.isfinite(P.parametric_var(r, confidence=0.95))


def test_numerical_stability_large_scale() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1000.0, 500)  # huge scale
    assert np.isfinite(H.historical_expected_shortfall(r, 0.99))
