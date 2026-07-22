"""Unit tests for the generic LinearFactorModel and the prediction API."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant.core.errors import ConstantFactorError, DuplicateObservationError
from factorlab_quant.models.factors import Factor, FactorSet
from factorlab_quant.models.linear_factor_model import FactorModelResult, LinearFactorModel


@pytest.fixture
def multi_factor_data(rng):
    """A known 3-factor DGP: y = 0.001 + 1.1 f1 - 0.4 f2 + 0.6 f3 + noise."""
    n = 240
    f1 = rng.normal(0.005, 0.04, n)
    f2 = rng.normal(0.0, 0.02, n)
    f3 = rng.normal(0.0, 0.03, n)
    betas = (1.1, -0.4, 0.6)
    alpha = 0.001
    y = alpha + betas[0] * f1 + betas[1] * f2 + betas[2] * f3 + rng.normal(0, 0.01, n)
    fs = FactorSet(
        [
            Factor("F1", f1, frequency="monthly"),
            Factor("F2", f2, frequency="monthly"),
            Factor("F3", f3, frequency="monthly"),
        ]
    )
    return y, fs, alpha, betas


def test_generic_model_recovers_parameters(multi_factor_data) -> None:
    y, fs, alpha, betas = multi_factor_data
    model = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3"))
    result = model.fit(y, fs)
    assert isinstance(result, FactorModelResult)
    assert result.intercept.estimate == pytest.approx(alpha, abs=0.003)
    # Tolerance is ~2.5 standard errors; the low-variance factor (F2) has the
    # largest loading SE (~0.03) given the DGP.
    for name, true_beta in zip(("F1", "F2", "F3"), betas, strict=True):
        assert result.factor_loading(name).estimate == pytest.approx(true_beta, abs=0.08)


def test_model_enforces_factor_specification_order(multi_factor_data) -> None:
    """A model declared with a factor order selects/reorders the incoming set."""
    y, fs, _, _ = multi_factor_data
    reordered = fs.select(["F3", "F1", "F2"])
    model = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3"))
    result = model.fit(y, reordered)
    assert result.factor_names == ("F1", "F2", "F3")
    assert result.param_names == ("alpha", "F1", "F2", "F3")


def test_no_intercept_model(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    model = LinearFactorModel("NoAlpha", factor_names=("F1", "F2", "F3"), intercept=False)
    result = model.fit(y, fs)
    assert not result.has_intercept
    assert result.param_names == ("F1", "F2", "F3")
    with pytest.raises(AttributeError):
        _ = result.intercept


def test_metadata_captured(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    model = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3"))
    result = model.fit(y, fs, covariance_type="HAC", extra_metadata={"tag": "unit-test"})
    assert result.metadata["model"] == "Test3F"
    assert result.metadata["covariance_type"] == "HAC"
    assert result.metadata["frequency"] == "monthly"
    assert result.metadata["tag"] == "unit-test"


def test_constant_factor_rejected(rng) -> None:
    n = 60
    y = rng.normal(size=n)
    fs = FactorSet([Factor("F1", rng.normal(size=n)), Factor("C", np.ones(n))])
    model = LinearFactorModel("Bad", factor_names=("F1", "C"))
    with pytest.raises(ConstantFactorError):
        model.fit(y, fs)


def test_duplicate_observation_rejection_opt_in() -> None:
    y = np.array([1.0, 1.0, 2.0, 3.0])
    fs = FactorSet([Factor("F1", [0.1, 0.1, 0.2, 0.3])])
    model = LinearFactorModel("Dup", factor_names=("F1",))
    # Off by default:
    model.fit(y, fs)
    # On when requested:
    with pytest.raises(DuplicateObservationError):
        model.fit(y, fs, reject_duplicate_observations=True)


def test_nonpositive_periods_per_year_rejected(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    model = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3"))
    with pytest.raises(ValueError, match="periods_per_year"):
        model.fit(y, fs, periods_per_year=0)


def test_accepts_mapping_and_matrix_inputs(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    mapping = {name: fs[name].values for name in fs.names}
    model = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3"))
    from_set = model.fit(y, fs)
    from_map = model.fit(y, mapping)
    np.testing.assert_allclose(from_set.params, from_map.params, rtol=1e-10)


# -- Prediction API -------------------------------------------------------- #
def test_predict_matches_manual_dot_product(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(y, fs)
    scenario = {"F1": 0.02, "F2": -0.01, "F3": 0.03}
    manual = (
        result.intercept.estimate
        + result.factor_loading("F1").estimate * 0.02
        + result.factor_loading("F2").estimate * -0.01
        + result.factor_loading("F3").estimate * 0.03
    )
    assert result.predict(scenario) == pytest.approx(manual)


def test_predict_accepts_array_and_batches(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(y, fs)
    single = result.predict(np.array([0.02, -0.01, 0.03]))
    assert isinstance(single, float)
    batch = result.predict(np.array([[0.02, -0.01, 0.03], [0.0, 0.0, 0.0]]))
    assert isinstance(batch, np.ndarray) and batch.shape == (2,)


def test_expected_return_adds_risk_free(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(y, fs)
    scenario = {"F1": 0.02, "F2": -0.01, "F3": 0.03}
    excess = result.predict(scenario)
    assert result.expected_return(scenario, risk_free=0.002) == pytest.approx(excess + 0.002)


def test_prediction_interval_wider_than_confidence_interval(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(y, fs)
    scenario = {"F1": 0.02, "F2": -0.01, "F3": 0.03}
    ci_lo, ci_hi = result.confidence_interval(scenario, level=0.95)
    pi_lo, pi_hi = result.prediction_interval(scenario, level=0.95)
    assert (pi_hi - pi_lo) > (ci_hi - ci_lo)
    # Both centered on the point prediction.
    point = result.predict(scenario)
    assert (ci_lo + ci_hi) / 2 == pytest.approx(point, abs=1e-9)
    assert (pi_lo + pi_hi) / 2 == pytest.approx(point, abs=1e-9)


def test_confidence_interval_invalid_level(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(y, fs)
    with pytest.raises(ValueError):
        result.confidence_interval({"F1": 0.0, "F2": 0.0, "F3": 0.0}, level=1.5)


def test_predict_missing_factor_raises(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(y, fs)
    with pytest.raises(KeyError):
        result.predict({"F1": 0.0, "F2": 0.0})  # missing F3


def test_generic_summary_renders(multi_factor_data) -> None:
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(y, fs)
    text = result.summary()
    assert "Linear Factor Model" in text
    assert "Test3F" in text
    assert "F1" in text and "F2" in text and "F3" in text


@pytest.mark.validation
def test_multifactor_matches_statsmodels(multi_factor_data) -> None:
    sm = pytest.importorskip("statsmodels.api")
    y, fs, _, _ = multi_factor_data
    result = LinearFactorModel("Test3F", factor_names=("F1", "F2", "F3")).fit(
        y, fs, hac_lags=5, small_sample_correction=False
    )
    X = sm.add_constant(fs.matrix())
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    np.testing.assert_allclose(result.params, ref.params, rtol=1e-9)
    np.testing.assert_allclose(result.standard_errors, ref.bse, rtol=1e-8, atol=1e-11)
    assert result.r_squared == pytest.approx(ref.rsquared, rel=1e-10)
