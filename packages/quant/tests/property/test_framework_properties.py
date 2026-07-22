"""Property-based invariants for the generic linear factor framework.

These assert mathematical guarantees for *any* valid factor set: prediction
consistency, interval ordering, serialization fidelity, and the equivalence of
the generic engine with the CAPM special case.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_quant.models import CAPM, LinearFactorModel
from factorlab_quant.models.factors import Factor, FactorSet
from factorlab_quant.models.linear_factor_model import FactorModelResult

pytestmark = pytest.mark.property

_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def factor_problem(draw):
    n = draw(st.integers(min_value=40, max_value=200))
    k = draw(st.integers(min_value=1, max_value=4))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    factor_matrix = rng.normal(0.0, 0.04, size=(n, k))
    betas = rng.uniform(-2.0, 2.0, size=k)
    alpha = draw(st.floats(min_value=-0.02, max_value=0.02))
    y = alpha + factor_matrix @ betas + rng.normal(0.0, 0.02, size=n)
    names = tuple(f"F{i}" for i in range(k))
    fs = FactorSet([Factor(names[i], factor_matrix[:, i]) for i in range(k)])
    return y, fs, names


def _fit(y, fs, names) -> FactorModelResult:
    return LinearFactorModel("P", factor_names=names).fit(y, fs)


@_SETTINGS
@given(problem=factor_problem())
def test_r_squared_in_unit_interval(problem) -> None:
    y, fs, names = problem
    result = _fit(y, fs, names)
    assert -1e-9 <= result.r_squared <= 1.0 + 1e-9


@_SETTINGS
@given(problem=factor_problem())
def test_prediction_interval_contains_confidence_interval(problem) -> None:
    y, fs, names = problem
    result = _fit(y, fs, names)
    scenario = dict.fromkeys(names, 0.01)
    ci_lo, ci_hi = result.confidence_interval(scenario)
    pi_lo, pi_hi = result.prediction_interval(scenario)
    assert pi_lo <= ci_lo + 1e-12
    assert pi_hi >= ci_hi - 1e-12


@_SETTINGS
@given(problem=factor_problem())
def test_predict_on_training_factors_equals_fitted_values(problem) -> None:
    """Predicting at the in-sample factor rows reproduces the fitted values."""
    y, fs, names = problem
    result = _fit(y, fs, names)
    predicted = result.predict(fs)
    np.testing.assert_allclose(predicted, result.fitted_values, rtol=1e-8, atol=1e-10)


@_SETTINGS
@given(problem=factor_problem())
def test_dict_roundtrip_preserves_estimates(problem) -> None:
    y, fs, names = problem
    result = _fit(y, fs, names)
    restored = FactorModelResult.from_dict(result.to_dict())
    np.testing.assert_allclose(restored.params, result.params, rtol=1e-12)
    np.testing.assert_allclose(
        restored.covariance_matrix, result.covariance_matrix, rtol=1e-12
    )


@_SETTINGS
@given(
    n=st.integers(min_value=40, max_value=200),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_capm_equals_generic_one_factor_model(n, seed) -> None:
    """CAPM must be numerically identical to the generic one-factor model."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.005, 0.04, size=n)
    asset = 0.001 + 1.2 * mkt + rng.normal(0.0, 0.02, size=n)

    capm = CAPM().fit(asset, mkt, returns_are_excess=True)
    generic = LinearFactorModel(
        "G", factor_names=("Mkt-RF",), intercept_name="alpha"
    ).fit(asset, FactorSet([Factor("Mkt-RF", mkt)]))

    np.testing.assert_allclose(capm.params, generic.params, rtol=1e-12)
    np.testing.assert_allclose(
        capm.standard_errors, generic.standard_errors, rtol=1e-12
    )
