"""Property-based invariants for the estimation engine.

These tests assert mathematical properties that must hold for *any* valid input,
not just the specific fixtures.  They are the kind of guarantees a quant desk
relies on when trusting a library with real capital.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_quant.models.capm import CAPM

pytestmark = pytest.mark.property

_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def capm_samples(draw) -> tuple[np.ndarray, np.ndarray, int]:
    n = draw(st.integers(min_value=40, max_value=250))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    market = rng.normal(0.005, 0.04, size=n)
    beta = draw(st.floats(min_value=-2.5, max_value=2.5))
    alpha = draw(st.floats(min_value=-0.02, max_value=0.02))
    asset = alpha + beta * market + rng.normal(0.0, 0.02, size=n)
    return asset, market, n


@_SETTINGS
@given(sample=capm_samples())
def test_r_squared_in_unit_interval(sample) -> None:
    asset, market, _ = sample
    res = CAPM().fit(asset, market, returns_are_excess=True)
    assert -1e-9 <= res.systematic_variance_ratio <= 1.0 + 1e-9


@_SETTINGS
@given(sample=capm_samples())
def test_covariance_is_symmetric_psd(sample) -> None:
    asset, market, _ = sample
    res = CAPM().fit(asset, market, returns_are_excess=True)
    cov = res.regression.covariance_matrix
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    assert np.linalg.eigvalsh(cov).min() > -1e-8


@_SETTINGS
@given(sample=capm_samples(), scale=st.floats(min_value=0.1, max_value=10.0))
def test_beta_scales_linearly_with_asset(sample, scale) -> None:
    """Scaling the asset return by c scales alpha and beta by c (linearity)."""
    asset, market, _ = sample
    base = CAPM().fit(asset, market, returns_are_excess=True)
    scaled = CAPM().fit(scale * asset, market, returns_are_excess=True)
    assert scaled.beta.estimate == pytest.approx(scale * base.beta.estimate, rel=1e-6)
    assert scaled.alpha.estimate == pytest.approx(scale * base.alpha.estimate, rel=1e-6, abs=1e-9)


@_SETTINGS
@given(sample=capm_samples(), shift=st.floats(min_value=-0.05, max_value=0.05))
def test_alpha_shifts_with_intercept_translation(sample, shift) -> None:
    """Adding a constant to the asset return shifts alpha by the same constant,
    leaving beta unchanged."""
    asset, market, _ = sample
    base = CAPM().fit(asset, market, returns_are_excess=True)
    shifted = CAPM().fit(asset + shift, market, returns_are_excess=True)
    assert shifted.beta.estimate == pytest.approx(base.beta.estimate, rel=1e-6, abs=1e-9)
    assert shifted.alpha.estimate == pytest.approx(
        base.alpha.estimate + shift, rel=1e-6, abs=1e-9
    )
