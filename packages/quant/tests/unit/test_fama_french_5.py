"""Exhaustive tests for the Fama-French five-factor model.

FF5 consumes factor data through the data layer via a duck-typed
``to_factor_set()``.  These quant-package tests exercise that contract with a
lightweight stub provider plus FactorSet/mapping inputs, so they need no
dependency on ``factorlab_data`` (the true end-to-end integration lives in the
data package's suite).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_quant import FamaFrench5Model, FamaFrench5Result
from factorlab_quant.core.errors import CollinearityError, ConstantFactorError
from factorlab_quant.models import Factor, FactorModelResult, FactorSet, get_model
from factorlab_quant.models.linear_factor_model import LinearFactorModel

_NAMES = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")


class _PanelStub:
    """Minimal stand-in for a data-layer panel: exposes only to_factor_set()."""

    def __init__(self, factor_set: FactorSet) -> None:
        self._fs = factor_set

    def to_factor_set(self) -> FactorSet:
        return self._fs


@dataclass(frozen=True)
class FF5DGP:
    asset_excess: np.ndarray
    factor_set: FactorSet
    mapping: dict[str, np.ndarray]
    alpha_true: float
    betas_true: dict[str, float]


@pytest.fixture
def ff5_dgp(rng) -> FF5DGP:
    n = 360
    alpha = 0.001
    betas = {"Mkt-RF": 1.05, "SMB": -0.20, "HML": 0.30, "RMW": 0.40, "CMA": 0.25}
    cols = {name: rng.normal(0.0, 0.03, n) for name in _NAMES}
    noise = rng.normal(0.0, 0.006, n)
    asset = alpha + noise
    for name, b in betas.items():
        asset = asset + b * cols[name]
    fs = FactorSet([Factor(name, cols[name], frequency="monthly") for name in _NAMES])
    return FF5DGP(asset, fs, cols, alpha, betas)


def _fit(dgp: FF5DGP, factors=None, **kw) -> FamaFrench5Result:
    return FamaFrench5Model().fit(
        dgp.asset_excess, factors if factors is not None else dgp.factor_set,
        returns_are_excess=True, **kw,
    )


# ----------------------------------------------------------------------- #
# Identity & registration                                                 #
# ----------------------------------------------------------------------- #
def test_model_identity() -> None:
    model = FamaFrench5Model()
    assert model.name == "Fama-French 5-Factor Model"
    assert model.factor_names == _NAMES


def test_registered_under_both_keys() -> None:
    assert get_model("FF5") is FamaFrench5Model
    assert get_model("FamaFrench5") is FamaFrench5Model


def test_is_linear_factor_model() -> None:
    assert isinstance(FamaFrench5Model(), LinearFactorModel)


def test_result_type(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    assert isinstance(res, FamaFrench5Result)
    assert isinstance(res, FactorModelResult)


# ----------------------------------------------------------------------- #
# Factor-source resolution (data-layer consumption)                       #
# ----------------------------------------------------------------------- #
def test_consumes_factor_set(ff5_dgp) -> None:
    res = _fit(ff5_dgp, factors=ff5_dgp.factor_set)
    assert res.param_names == ("alpha", *_NAMES)


def test_consumes_mapping(ff5_dgp) -> None:
    res = _fit(ff5_dgp, factors=ff5_dgp.mapping)
    for name, b in ff5_dgp.betas_true.items():
        assert res.factor_loading(name).estimate == pytest.approx(b, abs=0.03)


def test_consumes_panel_like_via_to_factor_set(ff5_dgp) -> None:
    """The duck-typed data-layer path: any object with to_factor_set()."""
    stub = _PanelStub(ff5_dgp.factor_set)
    res_panel = _fit(ff5_dgp, factors=stub)
    res_set = _fit(ff5_dgp, factors=ff5_dgp.factor_set)
    np.testing.assert_allclose(res_panel.params, res_set.params, rtol=1e-12)


def test_extra_factors_ignored(ff5_dgp) -> None:
    """A source with extra columns (e.g. RF, Mom) still fits FF5's five."""
    augmented = ff5_dgp.factor_set.add(Factor("RF", np.full(360, 0.002)))
    res = _fit(ff5_dgp, factors=augmented)
    assert res.factor_names == _NAMES


def test_invalid_factor_source_raises(ff5_dgp) -> None:
    with pytest.raises(TypeError):
        FamaFrench5Model().fit(ff5_dgp.asset_excess, 12345, returns_are_excess=True)


def test_bad_to_factor_set_return_raises(ff5_dgp) -> None:
    class _BadStub:
        def to_factor_set(self):  # type: ignore[no-untyped-def]
            return "not a factor set"

    with pytest.raises(TypeError):
        FamaFrench5Model().fit(ff5_dgp.asset_excess, _BadStub(), returns_are_excess=True)


def test_missing_factor_raises(ff5_dgp) -> None:
    incomplete = ff5_dgp.factor_set.select(["Mkt-RF", "SMB", "HML", "RMW"])  # no CMA
    with pytest.raises(KeyError):
        FamaFrench5Model().fit(ff5_dgp.asset_excess, incomplete, returns_are_excess=True)


# ----------------------------------------------------------------------- #
# Parameter recovery & statistics                                         #
# ----------------------------------------------------------------------- #
def test_recovers_parameters(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    assert res.alpha.estimate == pytest.approx(ff5_dgp.alpha_true, abs=0.002)
    for name, b in ff5_dgp.betas_true.items():
        assert res.factor_loading(name).estimate == pytest.approx(b, abs=0.03)


def test_named_accessors(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    assert res.alpha is res.intercept
    assert res.market_beta.name == "Mkt-RF"
    assert res.rmw_loading.name == "RMW"
    assert res.cma_loading.name == "CMA"


def test_style_tilts(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    assert res.size_tilt == "large-cap"        # SMB < 0
    assert res.value_tilt == "value"           # HML > 0
    assert res.profitability_tilt == "robust"  # RMW > 0
    assert res.investment_tilt == "conservative"  # CMA > 0


def test_full_statistical_surface(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    assert res.standard_errors.shape == (6,)
    assert res.confidence_intervals.shape == (6, 2)
    assert res.covariance_matrix.shape == (6, 6)
    assert res.adj_r_squared <= res.r_squared
    assert np.isfinite([res.aic, res.bic, res.log_likelihood]).all()


def test_robust_covariance_options(ff5_dgp) -> None:
    for cov in ("nonrobust", "HC0", "HC1", "HAC"):
        res = _fit(ff5_dgp, covariance_type=cov)
        assert res.regression.covariance_type == cov


def test_default_is_hac(ff5_dgp) -> None:
    assert _fit(ff5_dgp).regression.covariance_type == "HAC"


def test_annualized_alpha(ff5_dgp) -> None:
    res = _fit(ff5_dgp, periods_per_year=12)
    assert res.annualized_alpha == pytest.approx((1.0 + res.alpha.estimate) ** 12 - 1.0)


# ----------------------------------------------------------------------- #
# Excess handling & edge cases                                            #
# ----------------------------------------------------------------------- #
def test_excess_flag_with_risk_free_rejected(ff5_dgp) -> None:
    with pytest.raises(ValueError, match="risk_free must be None"):
        FamaFrench5Model().fit(
            ff5_dgp.asset_excess, ff5_dgp.factor_set, risk_free=0.001, returns_are_excess=True
        )


def test_scalar_risk_free(ff5_dgp) -> None:
    res = FamaFrench5Model().fit(ff5_dgp.asset_excess + 0.001, ff5_dgp.factor_set, risk_free=0.001)
    assert res.market_beta.estimate == pytest.approx(ff5_dgp.betas_true["Mkt-RF"], abs=0.03)


def test_constant_factor_rejected(ff5_dgp) -> None:
    cols = dict(ff5_dgp.mapping)
    cols["CMA"] = np.zeros(360)
    with pytest.raises(ConstantFactorError):
        FamaFrench5Model().fit(ff5_dgp.asset_excess, cols, returns_are_excess=True)


def test_collinear_factors_rejected(ff5_dgp) -> None:
    cols = dict(ff5_dgp.mapping)
    cols["CMA"] = 2.0 * cols["Mkt-RF"] + 3.0  # exact linear combo
    with pytest.raises(CollinearityError):
        FamaFrench5Model().fit(ff5_dgp.asset_excess, cols, returns_are_excess=True)


def test_nan_listwise_deletion(ff5_dgp) -> None:
    cols = {k: v.copy() for k, v in ff5_dgp.mapping.items()}
    cols["RMW"][7] = np.nan
    res = FamaFrench5Model().fit(ff5_dgp.asset_excess, cols, returns_are_excess=True)
    assert res.n_observations == 360 - 1


# ----------------------------------------------------------------------- #
# Prediction                                                              #
# ----------------------------------------------------------------------- #
def test_predict_matches_manual(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    sc = {"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.01, "RMW": 0.015, "CMA": 0.005}
    manual = res.alpha.estimate + sum(res.factor_loading(k).estimate * v for k, v in sc.items())
    assert res.predict(sc) == pytest.approx(manual)


def test_prediction_interval_wider_than_confidence(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    sc = {"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.01, "RMW": 0.015, "CMA": 0.005}
    ci_lo, ci_hi = res.confidence_interval(sc)
    pi_lo, pi_hi = res.prediction_interval(sc)
    assert (pi_hi - pi_lo) > (ci_hi - ci_lo)


def test_expected_return_adds_rf(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    sc = {"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.01, "RMW": 0.015, "CMA": 0.005}
    assert res.expected_return(sc, risk_free=0.002) == pytest.approx(res.predict(sc) + 0.002)


# ----------------------------------------------------------------------- #
# Serialization                                                           #
# ----------------------------------------------------------------------- #
def test_dict_roundtrip_dispatches_to_ff5(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    restored = FactorModelResult.from_dict(res.to_dict())
    assert type(restored) is FamaFrench5Result
    np.testing.assert_allclose(restored.params, res.params, rtol=1e-12)


def test_json_roundtrip(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    restored = FactorModelResult.from_json(res.to_json())
    assert restored.rmw_loading.estimate == pytest.approx(res.rmw_loading.estimate)


def test_pickle_roundtrip(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    restored = pickle.loads(pickle.dumps(res))
    assert restored.cma_loading.estimate == pytest.approx(res.cma_loading.estimate)


def test_metadata(ff5_dgp) -> None:
    res = _fit(ff5_dgp)
    assert res.metadata["reference"] == "Fama & French (2015)"
    assert res.metadata["n_factors"] == 5
    assert res.metadata["specification"] == "Fama-French (2015) five-factor"


def test_summary(ff5_dgp) -> None:
    text = _fit(ff5_dgp).summary()
    assert "Fama-French 5-Factor Model" in text
    assert "RMW" in text and "CMA" in text
    assert "Profitability" in text and "Investment" in text


# ----------------------------------------------------------------------- #
# statsmodels cross-validation                                            #
# ----------------------------------------------------------------------- #
@pytest.mark.validation
def test_matches_statsmodels_nonrobust(ff5_dgp) -> None:
    sm = pytest.importorskip("statsmodels.api")
    X = sm.add_constant(np.column_stack([ff5_dgp.mapping[n] for n in _NAMES]))
    ref = sm.OLS(ff5_dgp.asset_excess, X).fit()
    res = _fit(ff5_dgp, covariance_type="nonrobust")
    np.testing.assert_allclose(res.params, ref.params, rtol=1e-9)
    np.testing.assert_allclose(res.standard_errors, ref.bse, rtol=1e-9)
    assert res.r_squared == pytest.approx(ref.rsquared, rel=1e-10)
    assert res.aic == pytest.approx(ref.aic, rel=1e-9)


@pytest.mark.validation
def test_matches_statsmodels_hac(ff5_dgp) -> None:
    sm = pytest.importorskip("statsmodels.api")
    X = sm.add_constant(np.column_stack([ff5_dgp.mapping[n] for n in _NAMES]))
    ref = sm.OLS(ff5_dgp.asset_excess, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False}
    )
    res = _fit(ff5_dgp, hac_lags=6, small_sample_correction=False)
    np.testing.assert_allclose(res.params, ref.params, rtol=1e-9)
    np.testing.assert_allclose(res.standard_errors, ref.bse, rtol=1e-8, atol=1e-11)


# ----------------------------------------------------------------------- #
# Property tests                                                          #
# ----------------------------------------------------------------------- #
_PROP = settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@st.composite
def _ff5_samples(draw):
    n = draw(st.integers(min_value=50, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    cols = {name: rng.normal(0.0, 0.03, n) for name in _NAMES}
    betas = draw(
        st.lists(st.floats(-2.0, 2.0), min_size=5, max_size=5)
    )
    asset = 0.001 + rng.normal(0, 0.01, n)
    for name, b in zip(_NAMES, betas, strict=True):
        asset = asset + b * cols[name]
    return asset, cols


@pytest.mark.property
@_PROP
@given(sample=_ff5_samples())
def test_property_r2_and_psd(sample) -> None:
    asset, cols = sample
    res = FamaFrench5Model().fit(asset, cols, returns_are_excess=True)
    assert -1e-9 <= res.r_squared <= 1.0 + 1e-9
    cov = res.covariance_matrix
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    assert np.linalg.eigvalsh(cov).min() > -1e-8
