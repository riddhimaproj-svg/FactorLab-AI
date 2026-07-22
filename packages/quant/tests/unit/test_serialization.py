"""Serialization tests: to_dict/from_dict, JSON, pickle, and versioning."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from factorlab_quant.core.types import RegressionResult
from factorlab_quant.models import CAPM, LinearFactorModel
from factorlab_quant.models.factors import Factor, FactorSet
from factorlab_quant.models.linear_factor_model import (
    SCHEMA_VERSION,
    FactorModelResult,
)


@pytest.fixture
def capm_result(capm_dgp):
    return CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )


@pytest.fixture
def generic_result(rng):
    n = 150
    f1 = rng.normal(0.005, 0.04, n)
    f2 = rng.normal(0.0, 0.02, n)
    y = 0.001 + 0.9 * f1 + 0.3 * f2 + rng.normal(0, 0.01, n)
    fs = FactorSet([Factor("F1", f1), Factor("F2", f2)])
    return LinearFactorModel("Two", factor_names=("F1", "F2")).fit(y, fs)


def _assert_results_close(a: FactorModelResult, b: FactorModelResult) -> None:
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=1e-12)
    np.testing.assert_allclose(a.design_matrix, b.design_matrix, rtol=1e-12)
    np.testing.assert_allclose(a.response, b.response, rtol=1e-12)
    assert a.param_names == b.param_names
    assert a.factor_names == b.factor_names
    assert a.r_squared == pytest.approx(b.r_squared)


# -- to_dict / from_dict --------------------------------------------------- #
def test_generic_dict_roundtrip(generic_result) -> None:
    restored = FactorModelResult.from_dict(generic_result.to_dict())
    assert type(restored) is FactorModelResult
    _assert_results_close(generic_result, restored)


def test_schema_version_present(generic_result) -> None:
    d = generic_result.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["result_type"] == "FactorModelResult"


def test_capm_dict_roundtrip_dispatches_to_subclass(capm_result) -> None:
    restored = FactorModelResult.from_dict(capm_result.to_dict())
    assert type(restored).__name__ == "CAPMResult"
    _assert_results_close(capm_result, restored)
    # CAPM-specific fields survive.
    assert restored.beta_t_vs_one == pytest.approx(capm_result.beta_t_vs_one)
    assert restored.mean_market_excess == pytest.approx(capm_result.mean_market_excess)


# -- JSON ------------------------------------------------------------------ #
def test_json_is_valid_and_roundtrips(generic_result) -> None:
    payload = generic_result.to_json()
    parsed = json.loads(payload)  # must be valid JSON structure
    assert parsed["model_name"] == "Two"
    restored = FactorModelResult.from_json(payload)
    _assert_results_close(generic_result, restored)


def test_capm_json_roundtrip(capm_result) -> None:
    restored = FactorModelResult.from_json(capm_result.to_json())
    assert restored.beta.estimate == pytest.approx(capm_result.beta.estimate)


# -- Pickle ---------------------------------------------------------------- #
def test_pickle_roundtrip_generic(generic_result) -> None:
    restored = pickle.loads(pickle.dumps(generic_result))
    _assert_results_close(generic_result, restored)


def test_pickle_roundtrip_capm(capm_result) -> None:
    restored = pickle.loads(pickle.dumps(capm_result))
    assert restored.alpha.estimate == pytest.approx(capm_result.alpha.estimate)
    assert restored.beta_p_vs_one == pytest.approx(capm_result.beta_p_vs_one)


# -- Core-type serialization ---------------------------------------------- #
def test_regression_result_roundtrip(capm_result) -> None:
    reg = capm_result.regression
    restored = RegressionResult.from_dict(reg.to_dict())
    np.testing.assert_allclose(restored.params, reg.params, rtol=1e-12)
    np.testing.assert_allclose(
        restored.covariance_matrix, reg.covariance_matrix, rtol=1e-12
    )
    assert restored.covariance_type == reg.covariance_type
    assert restored.cov_config == reg.cov_config
    assert restored.diagnostics.r_squared == pytest.approx(reg.diagnostics.r_squared)


def test_factor_and_factorset_roundtrip(rng) -> None:
    fs = FactorSet(
        [
            Factor("A", rng.normal(size=20), display_name="Alpha", frequency="monthly",
                   source="unit-test", description="a factor"),
            Factor("B", rng.normal(size=20), frequency="monthly"),
        ]
    )
    restored = FactorSet.from_dict(fs.to_dict())
    assert restored.names == fs.names
    np.testing.assert_allclose(restored.matrix(), fs.matrix())
    assert restored["A"].display_name == "Alpha"
    assert restored["A"].source == "unit-test"
