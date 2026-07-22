"""Exhaustive tests for the Carhart four-factor model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_quant import CarhartModel, CarhartResult
from factorlab_quant.core.errors import CollinearityError, ConstantFactorError
from factorlab_quant.models import Factor, FactorModelResult, FactorSet, get_model
from factorlab_quant.models.linear_factor_model import LinearFactorModel

_NAMES = ("Mkt-RF", "SMB", "HML", "MOM")


@dataclass(frozen=True)
class CarhartDGP:
    asset_excess: np.ndarray
    mapping: dict[str, np.ndarray]  # uses 'MOM'
    alpha_true: float
    betas_true: dict[str, float]


@pytest.fixture
def dgp(rng) -> CarhartDGP:
    n = 360
    alpha = 0.001
    betas = {"Mkt-RF": 1.02, "SMB": -0.20, "HML": 0.35, "MOM": 0.50}
    cols = {name: rng.normal(0.0, 0.03, n) for name in _NAMES}
    asset = alpha + rng.normal(0.0, 0.008, n)
    for name, b in betas.items():
        asset = asset + b * cols[name]
    return CarhartDGP(asset, cols, alpha, betas)


def _fit(dgp: CarhartDGP, factors=None, **kw) -> CarhartResult:
    return CarhartModel().fit(
        dgp.asset_excess, factors if factors is not None else dgp.mapping,
        returns_are_excess=True, **kw,
    )


# -- Identity & registration ---------------------------------------------- #
def test_identity() -> None:
    model = CarhartModel()
    assert model.name == "Carhart 4-Factor Model"
    assert model.factor_names == _NAMES
    assert isinstance(model, LinearFactorModel)


@pytest.mark.parametrize("key", ["Carhart4", "Carhart", "MomentumFactorModel"])
def test_registered_under_all_keys(key) -> None:
    assert get_model(key) is CarhartModel


def test_result_type(dgp) -> None:
    res = _fit(dgp)
    assert isinstance(res, CarhartResult)
    assert isinstance(res, FactorModelResult)


# -- Momentum alias normalization (data-layer names) ---------------------- #
@pytest.mark.parametrize("alias", ["Mom", "UMD", "WML", "Momentum"])
def test_momentum_alias_normalized(dgp, alias) -> None:
    cols = {
        "Mkt-RF": dgp.mapping["Mkt-RF"],
        "SMB": dgp.mapping["SMB"],
        "HML": dgp.mapping["HML"],
        alias: dgp.mapping["MOM"],
    }
    res = _fit(dgp, factors=cols)
    assert res.factor_names == _NAMES  # normalized to MOM
    assert res.momentum_loading.estimate == pytest.approx(dgp.betas_true["MOM"], abs=0.03)


def test_consumes_factor_set_with_mom_alias(dgp) -> None:
    fs = FactorSet(
        [
            Factor("Mkt-RF", dgp.mapping["Mkt-RF"]),
            Factor("SMB", dgp.mapping["SMB"]),
            Factor("HML", dgp.mapping["HML"]),
            Factor("Mom", dgp.mapping["MOM"]),  # library name
        ]
    )
    res = _fit(dgp, factors=fs)
    assert res.factor_names == _NAMES


def test_consumes_panel_like(dgp) -> None:
    class _Stub:
        def to_factor_set(self) -> FactorSet:
            return FactorSet(
                [Factor(n, dgp.mapping["MOM" if n == "MOM" else n]) for n in _NAMES]
            )

    res_stub = _fit(dgp, factors=_Stub())
    res_map = _fit(dgp)
    np.testing.assert_allclose(res_stub.params, res_map.params, rtol=1e-12)


def test_missing_momentum_raises(dgp) -> None:
    cols = {k: dgp.mapping[k] for k in ("Mkt-RF", "SMB", "HML")}  # no momentum
    with pytest.raises(KeyError):
        _fit(dgp, factors=cols)


def test_invalid_source_raises(dgp) -> None:
    with pytest.raises(TypeError):
        CarhartModel().fit(dgp.asset_excess, 42, returns_are_excess=True)


# -- Parameter recovery & accessors --------------------------------------- #
def test_recovers_parameters(dgp) -> None:
    res = _fit(dgp)
    assert res.alpha.estimate == pytest.approx(dgp.alpha_true, abs=0.002)
    for name, b in dgp.betas_true.items():
        assert res.factor_loading(name).estimate == pytest.approx(b, abs=0.03)


def test_named_accessors(dgp) -> None:
    res = _fit(dgp)
    assert res.market_beta.name == "Mkt-RF"
    assert res.size_loading.name == "SMB"
    assert res.value_loading.name == "HML"
    assert res.momentum_loading.name == "MOM"
    assert res.alpha is res.intercept


def test_style_tilts(dgp) -> None:
    res = _fit(dgp)
    assert res.size_tilt == "large-cap"
    assert res.value_tilt == "value"
    assert res.momentum_tilt == "winner"


def test_annualized_alpha(dgp) -> None:
    res = _fit(dgp, periods_per_year=12)
    assert res.annualized_alpha == pytest.approx((1.0 + res.alpha.estimate) ** 12 - 1.0)


def test_statistical_surface(dgp) -> None:
    res = _fit(dgp)
    assert res.standard_errors.shape == (5,)
    assert res.confidence_intervals.shape == (5, 2)
    assert res.adj_r_squared <= res.r_squared
    assert np.isfinite([res.aic, res.bic]).all()


def test_robust_covariance_options(dgp) -> None:
    for cov in ("nonrobust", "HC0", "HC1", "HAC"):
        assert _fit(dgp, covariance_type=cov).regression.covariance_type == cov


# -- Edge cases ------------------------------------------------------------ #
def test_excess_flag_with_rf_rejected(dgp) -> None:
    with pytest.raises(ValueError, match="risk_free must be None"):
        CarhartModel().fit(dgp.asset_excess, dgp.mapping, risk_free=0.001, returns_are_excess=True)


def test_constant_factor_rejected(dgp) -> None:
    cols = dict(dgp.mapping)
    cols["MOM"] = np.zeros(360)
    with pytest.raises(ConstantFactorError):
        _fit(dgp, factors=cols)


def test_collinear_rejected(dgp) -> None:
    cols = dict(dgp.mapping)
    cols["MOM"] = 2.0 * cols["Mkt-RF"] + 1.0
    with pytest.raises(CollinearityError):
        _fit(dgp, factors=cols)


def test_nan_listwise_deletion(dgp) -> None:
    cols = {k: v.copy() for k, v in dgp.mapping.items()}
    cols["MOM"][3] = np.nan
    res = _fit(dgp, factors=cols)
    assert res.n_observations == 359


# -- Prediction ------------------------------------------------------------ #
def test_prediction_and_intervals(dgp) -> None:
    res = _fit(dgp)
    sc = {"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.01, "MOM": 0.03}
    manual = res.alpha.estimate + sum(res.factor_loading(k).estimate * v for k, v in sc.items())
    assert res.predict(sc) == pytest.approx(manual)
    ci_lo, ci_hi = res.confidence_interval(sc)
    pi_lo, pi_hi = res.prediction_interval(sc)
    assert (pi_hi - pi_lo) > (ci_hi - ci_lo)


# -- Serialization --------------------------------------------------------- #
def test_dict_roundtrip_dispatches(dgp) -> None:
    res = _fit(dgp)
    restored = FactorModelResult.from_dict(res.to_dict())
    assert type(restored) is CarhartResult
    np.testing.assert_allclose(restored.params, res.params, rtol=1e-12)


def test_pickle_roundtrip(dgp) -> None:
    res = _fit(dgp)
    restored = pickle.loads(pickle.dumps(res))
    assert restored.momentum_loading.estimate == pytest.approx(res.momentum_loading.estimate)


def test_summary(dgp) -> None:
    text = _fit(dgp).summary()
    assert "Carhart 4-Factor Model" in text
    assert "MOM" in text and "Momentum" in text


def test_metadata(dgp) -> None:
    res = _fit(dgp)
    assert res.metadata["reference"] == "Carhart (1997)"
    assert res.metadata["n_factors"] == 4


# -- statsmodels cross-validation ----------------------------------------- #
@pytest.mark.validation
def test_matches_statsmodels_nonrobust(dgp) -> None:
    sm = pytest.importorskip("statsmodels.api")
    X = sm.add_constant(np.column_stack([dgp.mapping[n] for n in _NAMES]))
    ref = sm.OLS(dgp.asset_excess, X).fit()
    res = _fit(dgp, covariance_type="nonrobust")
    np.testing.assert_allclose(res.params, ref.params, rtol=1e-9)
    np.testing.assert_allclose(res.standard_errors, ref.bse, rtol=1e-9)
    assert res.r_squared == pytest.approx(ref.rsquared, rel=1e-10)
    assert res.bic == pytest.approx(ref.bic, rel=1e-9)


@pytest.mark.validation
def test_matches_statsmodels_hac(dgp) -> None:
    sm = pytest.importorskip("statsmodels.api")
    X = sm.add_constant(np.column_stack([dgp.mapping[n] for n in _NAMES]))
    ref = sm.OLS(dgp.asset_excess, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False}
    )
    res = _fit(dgp, hac_lags=6, small_sample_correction=False)
    np.testing.assert_allclose(res.standard_errors, ref.bse, rtol=1e-8, atol=1e-11)


# -- Property -------------------------------------------------------------- #
_PROP = settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@st.composite
def _samples(draw):
    n = draw(st.integers(min_value=50, max_value=200))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    cols = {name: rng.normal(0.0, 0.03, n) for name in _NAMES}
    betas = draw(st.lists(st.floats(-2.0, 2.0), min_size=4, max_size=4))
    asset = 0.001 + rng.normal(0, 0.01, n)
    for name, b in zip(_NAMES, betas, strict=True):
        asset = asset + b * cols[name]
    return asset, cols


@pytest.mark.property
@_PROP
@given(sample=_samples())
def test_property_r2_and_psd(sample) -> None:
    asset, cols = sample
    res = CarhartModel().fit(asset, cols, returns_are_excess=True)
    assert -1e-9 <= res.r_squared <= 1.0 + 1e-9
    cov = res.covariance_matrix
    assert np.linalg.eigvalsh(cov).min() > -1e-8
