"""Exhaustive tests for the Fama-French three-factor model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from factorlab_quant import FamaFrench3Model, FamaFrench3Result
from factorlab_quant.core.errors import (
    CollinearityError,
    ConstantFactorError,
    DimensionMismatchError,
    DuplicateFactorError,
    InsufficientDataError,
    QuantError,
)
from factorlab_quant.models import FactorModelResult, get_model
from factorlab_quant.models.linear_factor_model import LinearFactorModel


@dataclass(frozen=True)
class FF3DGP:
    asset_excess: np.ndarray
    mkt: np.ndarray
    smb: np.ndarray
    hml: np.ndarray
    risk_free: np.ndarray
    alpha_true: float
    betas_true: tuple[float, float, float]


@pytest.fixture
def ff3_dgp(rng) -> FF3DGP:
    """Monthly FF3 sample: alpha=0.001, betas=(1.05,-0.30,0.55), 25 years."""
    n = 300
    alpha = 0.001
    betas = (1.05, -0.30, 0.55)
    rf = np.full(n, 0.0015)
    mkt = rng.normal(0.005, 0.04, n)
    smb = rng.normal(0.001, 0.02, n)
    hml = rng.normal(0.002, 0.03, n)
    noise = rng.normal(0.0, 0.008, n)
    asset_excess = alpha + betas[0] * mkt + betas[1] * smb + betas[2] * hml + noise
    return FF3DGP(asset_excess, mkt, smb, hml, rf, alpha, betas)


def _fit(dgp: FF3DGP, **kwargs) -> FamaFrench3Result:
    return FamaFrench3Model().fit(
        dgp.asset_excess, dgp.mkt, dgp.smb, dgp.hml, returns_are_excess=True, **kwargs
    )


# ----------------------------------------------------------------------- #
# Identity & registration                                                 #
# ----------------------------------------------------------------------- #
def test_model_identity() -> None:
    model = FamaFrench3Model()
    assert model.name == "Fama-French 3-Factor Model"
    assert model.factor_names == ("Mkt-RF", "SMB", "HML")


def test_registered_under_both_keys() -> None:
    assert get_model("FF3") is FamaFrench3Model
    assert get_model("FamaFrench3") is FamaFrench3Model


def test_result_type(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    assert isinstance(res, FamaFrench3Result)
    assert isinstance(res, FactorModelResult)


def test_reuses_framework_not_custom_estimation() -> None:
    """FF3 must not reimplement fitting -- it delegates to the generic engine."""
    model = FamaFrench3Model()
    assert isinstance(model, LinearFactorModel)


# ----------------------------------------------------------------------- #
# Parameter recovery                                                      #
# ----------------------------------------------------------------------- #
def test_recovers_true_parameters(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    assert res.alpha.estimate == pytest.approx(ff3_dgp.alpha_true, abs=0.002)
    assert res.market_beta.estimate == pytest.approx(ff3_dgp.betas_true[0], abs=0.03)
    assert res.smb_loading.estimate == pytest.approx(ff3_dgp.betas_true[1], abs=0.06)
    assert res.hml_loading.estimate == pytest.approx(ff3_dgp.betas_true[2], abs=0.05)


def test_named_accessors_match_generic_lookup(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    assert res.alpha is res.intercept
    assert res.market_beta.name == "Mkt-RF"
    assert res.smb_loading.name == "SMB"
    assert res.hml_loading.name == "HML"


def test_style_tilt_interpretation(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    # DGP has negative SMB loading (large-cap) and positive HML (value).
    assert res.size_tilt == "large-cap"
    assert res.value_tilt == "value"


# ----------------------------------------------------------------------- #
# Statistical outputs                                                     #
# ----------------------------------------------------------------------- #
def test_full_statistical_surface_present(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    assert res.param_names == ("alpha", "Mkt-RF", "SMB", "HML")
    assert res.standard_errors.shape == (4,)
    assert res.t_statistics.shape == (4,)
    assert res.p_values.shape == (4,)
    assert res.confidence_intervals.shape == (4, 2)
    assert res.covariance_matrix.shape == (4, 4)
    assert 0.0 <= res.r_squared <= 1.0
    assert res.adj_r_squared <= res.r_squared
    assert np.isfinite(res.aic) and np.isfinite(res.bic)
    assert np.isfinite(res.log_likelihood)
    assert np.isfinite(res.diagnostics.f_statistic)
    assert res.residuals.shape == (ff3_dgp.asset_excess.shape[0],)
    assert res.fitted_values.shape == (ff3_dgp.asset_excess.shape[0],)


def test_annualized_alpha_formula(ff3_dgp) -> None:
    res = _fit(ff3_dgp, periods_per_year=12)
    assert res.annualized_alpha == pytest.approx((1.0 + res.alpha.estimate) ** 12 - 1.0)


def test_default_covariance_is_hac(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    assert res.regression.covariance_type == "HAC"
    assert res.regression.cov_config["kernel"] == "bartlett"


def test_robust_covariance_options(ff3_dgp) -> None:
    for cov in ("nonrobust", "HC0", "HC1", "HAC"):
        res = _fit(ff3_dgp, covariance_type=cov)
        assert res.regression.covariance_type == cov
        assert np.all(res.standard_errors > 0)


# ----------------------------------------------------------------------- #
# Excess-return handling                                                  #
# ----------------------------------------------------------------------- #
def test_excess_flag_matches_explicit_risk_free(ff3_dgp) -> None:
    rf = ff3_dgp.risk_free
    via_rf = FamaFrench3Model().fit(
        ff3_dgp.asset_excess + rf, ff3_dgp.mkt, ff3_dgp.smb, ff3_dgp.hml, risk_free=rf
    )
    via_excess = _fit(ff3_dgp)
    assert via_rf.alpha.estimate == pytest.approx(via_excess.alpha.estimate, rel=1e-10)
    for name in ("Mkt-RF", "SMB", "HML"):
        assert via_rf.factor_loading(name).estimate == pytest.approx(
            via_excess.factor_loading(name).estimate, rel=1e-10
        )


def test_scalar_risk_free(ff3_dgp) -> None:
    res = FamaFrench3Model().fit(
        ff3_dgp.asset_excess + 0.001, ff3_dgp.mkt, ff3_dgp.smb, ff3_dgp.hml, risk_free=0.001
    )
    assert res.market_beta.estimate == pytest.approx(ff3_dgp.betas_true[0], abs=0.03)


def test_factors_never_excess_adjusted(ff3_dgp) -> None:
    """risk_free must adjust only the asset, never the (already-excess) factors."""
    rf = 0.002
    res_rf = FamaFrench3Model().fit(
        ff3_dgp.asset_excess + rf, ff3_dgp.mkt, ff3_dgp.smb, ff3_dgp.hml, risk_free=rf
    )
    res_ex = _fit(ff3_dgp)
    # Market factor column in the design must be identical (un-adjusted).
    np.testing.assert_allclose(
        res_rf.design_matrix[:, 1], res_ex.design_matrix[:, 1], rtol=1e-12
    )


def test_excess_flag_with_risk_free_rejected(ff3_dgp) -> None:
    with pytest.raises(ValueError, match="risk_free must be None"):
        FamaFrench3Model().fit(
            ff3_dgp.asset_excess,
            ff3_dgp.mkt,
            ff3_dgp.smb,
            ff3_dgp.hml,
            risk_free=0.001,
            returns_are_excess=True,
        )


# ----------------------------------------------------------------------- #
# Data validation                                                         #
# ----------------------------------------------------------------------- #
def test_length_mismatch_rejected(ff3_dgp) -> None:
    with pytest.raises(DimensionMismatchError):
        FamaFrench3Model().fit(
            ff3_dgp.asset_excess, ff3_dgp.mkt[:-1], ff3_dgp.smb, ff3_dgp.hml,
            returns_are_excess=True,
        )


def test_constant_factor_rejected(ff3_dgp) -> None:
    const_hml = np.zeros_like(ff3_dgp.hml)
    with pytest.raises(ConstantFactorError):
        FamaFrench3Model().fit(
            ff3_dgp.asset_excess, ff3_dgp.mkt, ff3_dgp.smb, const_hml,
            returns_are_excess=True,
        )


def test_duplicate_factor_columns_rejected(ff3_dgp) -> None:
    # HML identical to SMB -> singular design, caught as a duplicate column.
    with pytest.raises(DuplicateFactorError):
        FamaFrench3Model().fit(
            ff3_dgp.asset_excess, ff3_dgp.mkt, ff3_dgp.smb, ff3_dgp.smb.copy(),
            returns_are_excess=True,
        )


def test_rank_deficient_design_rejected(ff3_dgp) -> None:
    # SMB an exact linear combination of MKT -> rank-deficient (not identical
    # columns, so it slips past the duplicate check and hits the estimator's
    # condition-number guard).
    collinear_smb = 2.0 * ff3_dgp.mkt + 3.0
    with pytest.raises(CollinearityError):
        FamaFrench3Model().fit(
            ff3_dgp.asset_excess, ff3_dgp.mkt, collinear_smb, ff3_dgp.hml,
            returns_are_excess=True,
        )


def test_nan_values_listwise_deleted(ff3_dgp) -> None:
    mkt = ff3_dgp.mkt.copy()
    mkt[5] = np.nan
    hml = ff3_dgp.hml.copy()
    hml[9] = np.nan
    res = FamaFrench3Model().fit(
        ff3_dgp.asset_excess, mkt, ff3_dgp.smb, hml, returns_are_excess=True
    )
    assert res.n_observations == ff3_dgp.asset_excess.shape[0] - 2


def test_all_nan_factor_raises(ff3_dgp) -> None:
    # An all-NaN factor leaves zero usable observations after listwise deletion;
    # the framework rejects it with a typed QuantError (the empty, degenerate
    # design surfaces as a data-validation failure rather than a silent result).
    with pytest.raises(QuantError):
        FamaFrench3Model().fit(
            ff3_dgp.asset_excess, np.full_like(ff3_dgp.mkt, np.nan), ff3_dgp.smb,
            ff3_dgp.hml, returns_are_excess=True,
        )


def test_insufficient_observations_rejected() -> None:
    # 4 params (alpha + 3 factors) need >= 5 observations.
    with pytest.raises(InsufficientDataError):
        FamaFrench3Model().fit(
            np.array([0.01, 0.02, 0.03]),
            np.array([0.01, -0.01, 0.02]),
            np.array([0.0, 0.01, -0.01]),
            np.array([0.01, 0.0, 0.02]),
            returns_are_excess=True,
        )


def test_nonpositive_periods_per_year_rejected(ff3_dgp) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        _fit(ff3_dgp, periods_per_year=0)


def test_duplicate_observations_opt_in(ff3_dgp) -> None:
    from factorlab_quant.core.errors import DuplicateObservationError

    # Force two identical rows.
    asset = ff3_dgp.asset_excess.copy()
    mkt = ff3_dgp.mkt.copy()
    smb = ff3_dgp.smb.copy()
    hml = ff3_dgp.hml.copy()
    for arr in (asset, mkt, smb, hml):
        arr[1] = arr[0]
    # Off by default:
    FamaFrench3Model().fit(asset, mkt, smb, hml, returns_are_excess=True)
    # On when requested:
    with pytest.raises(DuplicateObservationError):
        FamaFrench3Model().fit(
            asset, mkt, smb, hml, returns_are_excess=True,
            reject_duplicate_observations=True,
        )


# ----------------------------------------------------------------------- #
# Prediction                                                              #
# ----------------------------------------------------------------------- #
def test_predict_matches_manual(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    sc = {"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.03}
    manual = (
        res.alpha.estimate
        + res.market_beta.estimate * 0.02
        + res.smb_loading.estimate * -0.01
        + res.hml_loading.estimate * 0.03
    )
    assert res.predict(sc) == pytest.approx(manual)


def test_expected_return_adds_risk_free(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    sc = {"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.03}
    assert res.expected_return(sc, risk_free=0.002) == pytest.approx(res.predict(sc) + 0.002)


def test_prediction_interval_wider_than_confidence(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    sc = {"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.03}
    ci_lo, ci_hi = res.confidence_interval(sc)
    pi_lo, pi_hi = res.prediction_interval(sc)
    assert (pi_hi - pi_lo) > (ci_hi - ci_lo)


def test_predict_batch(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    batch = res.predict(np.array([[0.02, -0.01, 0.03], [0.0, 0.0, 0.0]]))
    assert isinstance(batch, np.ndarray) and batch.shape == (2,)


# ----------------------------------------------------------------------- #
# Serialization                                                           #
# ----------------------------------------------------------------------- #
def test_dict_roundtrip_dispatches_to_ff3(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    restored = FactorModelResult.from_dict(res.to_dict())
    assert type(restored) is FamaFrench3Result
    np.testing.assert_allclose(restored.params, res.params, rtol=1e-12)
    assert restored.hml_loading.estimate == pytest.approx(res.hml_loading.estimate)


def test_json_roundtrip(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    restored = FactorModelResult.from_json(res.to_json())
    assert type(restored) is FamaFrench3Result
    np.testing.assert_allclose(restored.covariance_matrix, res.covariance_matrix, rtol=1e-12)


def test_pickle_roundtrip(ff3_dgp) -> None:
    res = _fit(ff3_dgp)
    restored = pickle.loads(pickle.dumps(res))
    assert restored.smb_loading.estimate == pytest.approx(res.smb_loading.estimate)


def test_metadata_captured(ff3_dgp) -> None:
    res = _fit(ff3_dgp, frequency="monthly")
    assert res.metadata["reference"] == "Fama & French (1993)"
    assert res.metadata["n_factors"] == 3
    assert res.metadata["frequency"] == "monthly"
    assert res.metadata["specification"] == "Fama-French (1993) three-factor"
    factor_names = [f["name"] for f in res.metadata["factors"]]
    assert factor_names == ["Mkt-RF", "SMB", "HML"]


def test_summary_sections(ff3_dgp) -> None:
    text = _fit(ff3_dgp).summary()
    assert "Fama-French 3-Factor Model" in text
    assert "SMB" in text and "HML" in text
    assert "Size tilt" in text and "Value tilt" in text
    assert "Jarque-Bera" in text


# ----------------------------------------------------------------------- #
# statsmodels cross-validation                                            #
# ----------------------------------------------------------------------- #
@pytest.mark.validation
def test_matches_statsmodels_nonrobust(ff3_dgp) -> None:
    sm = pytest.importorskip("statsmodels.api")
    X = sm.add_constant(np.column_stack([ff3_dgp.mkt, ff3_dgp.smb, ff3_dgp.hml]))
    ref = sm.OLS(ff3_dgp.asset_excess, X).fit()
    res = _fit(ff3_dgp, covariance_type="nonrobust")
    np.testing.assert_allclose(res.params, ref.params, rtol=1e-9)
    np.testing.assert_allclose(res.standard_errors, ref.bse, rtol=1e-9)
    np.testing.assert_allclose(res.t_statistics, ref.tvalues, rtol=1e-8)
    np.testing.assert_allclose(res.p_values, ref.pvalues, rtol=1e-7, atol=1e-12)
    assert res.r_squared == pytest.approx(ref.rsquared, rel=1e-10)
    assert res.adj_r_squared == pytest.approx(ref.rsquared_adj, rel=1e-10)
    assert res.aic == pytest.approx(ref.aic, rel=1e-9)
    assert res.bic == pytest.approx(ref.bic, rel=1e-9)
    assert res.diagnostics.f_statistic == pytest.approx(ref.fvalue, rel=1e-8)


@pytest.mark.validation
def test_matches_statsmodels_hac(ff3_dgp) -> None:
    sm = pytest.importorskip("statsmodels.api")
    X = sm.add_constant(np.column_stack([ff3_dgp.mkt, ff3_dgp.smb, ff3_dgp.hml]))
    ref = sm.OLS(ff3_dgp.asset_excess, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False}
    )
    res = _fit(ff3_dgp, hac_lags=6, small_sample_correction=False)
    np.testing.assert_allclose(res.params, ref.params, rtol=1e-9)
    np.testing.assert_allclose(res.standard_errors, ref.bse, rtol=1e-8, atol=1e-11)


# ----------------------------------------------------------------------- #
# Numerical stability                                                     #
# ----------------------------------------------------------------------- #
def test_scale_invariance_of_loadings(ff3_dgp) -> None:
    """Scaling the asset by c scales alpha and all loadings by c."""
    base = _fit(ff3_dgp)
    scaled = FamaFrench3Model().fit(
        10.0 * ff3_dgp.asset_excess, ff3_dgp.mkt, ff3_dgp.smb, ff3_dgp.hml,
        returns_are_excess=True,
    )
    np.testing.assert_allclose(scaled.params, 10.0 * base.params, rtol=1e-6)


def test_tiny_and_large_magnitudes_stable(rng) -> None:
    n = 200
    mkt = rng.normal(0.0, 1e-4, n)
    smb = rng.normal(0.0, 1e-4, n)
    hml = rng.normal(0.0, 1e-4, n)
    asset = 1e-6 + 0.8 * mkt - 0.2 * smb + 0.4 * hml + rng.normal(0, 1e-5, n)
    res = FamaFrench3Model().fit(asset, mkt, smb, hml, returns_are_excess=True)
    assert np.all(np.isfinite(res.params))
    assert np.all(np.isfinite(res.covariance_matrix))
    assert res.market_beta.estimate == pytest.approx(0.8, abs=0.1)


# ----------------------------------------------------------------------- #
# Property tests                                                          #
# ----------------------------------------------------------------------- #
_PROP_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def _ff3_samples(draw):
    n = draw(st.integers(min_value=40, max_value=240))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.005, 0.04, n)
    smb = rng.normal(0.0, 0.02, n)
    hml = rng.normal(0.0, 0.03, n)
    b = draw(
        st.tuples(
            st.floats(-2.0, 2.0), st.floats(-2.0, 2.0), st.floats(-2.0, 2.0)
        )
    )
    a = draw(st.floats(-0.02, 0.02))
    asset = a + b[0] * mkt + b[1] * smb + b[2] * hml + rng.normal(0, 0.015, n)
    return asset, mkt, smb, hml


@pytest.mark.property
@_PROP_SETTINGS
@given(sample=_ff3_samples())
def test_property_r_squared_unit_interval(sample) -> None:
    asset, mkt, smb, hml = sample
    res = FamaFrench3Model().fit(asset, mkt, smb, hml, returns_are_excess=True)
    assert -1e-9 <= res.r_squared <= 1.0 + 1e-9


@pytest.mark.property
@_PROP_SETTINGS
@given(sample=_ff3_samples())
def test_property_covariance_psd(sample) -> None:
    asset, mkt, smb, hml = sample
    res = FamaFrench3Model().fit(asset, mkt, smb, hml, returns_are_excess=True)
    cov = res.covariance_matrix
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    assert np.linalg.eigvalsh(cov).min() > -1e-8


@pytest.mark.property
@_PROP_SETTINGS
@given(sample=_ff3_samples())
def test_property_predict_reproduces_fitted(sample) -> None:
    asset, mkt, smb, hml = sample
    res = FamaFrench3Model().fit(asset, mkt, smb, hml, returns_are_excess=True)
    from factorlab_quant.models.factors import Factor, FactorSet

    fs = FactorSet([Factor("Mkt-RF", mkt), Factor("SMB", smb), Factor("HML", hml)])
    # align to whatever the fit used (no NaNs here, so full sample)
    np.testing.assert_allclose(res.predict(fs), res.fitted_values, rtol=1e-8, atol=1e-10)
