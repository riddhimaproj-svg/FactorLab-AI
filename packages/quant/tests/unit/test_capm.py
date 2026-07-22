"""Unit tests for the CAPM model."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant.core.errors import NonFiniteError
from factorlab_quant.models.capm import CAPM, CAPMResult


def test_recovers_true_alpha_and_beta(capm_dgp) -> None:
    res = CAPM().fit(
        capm_dgp.asset_excess,
        capm_dgp.market_excess,
        returns_are_excess=True,
        periods_per_year=capm_dgp.periods_per_year,
    )
    # Tolerances are ~3 standard errors of the respective estimator given the
    # DGP's sample size and noise (SE_beta ~ 0.03, SE_alpha ~ 0.0013).
    assert res.beta.estimate == pytest.approx(capm_dgp.beta_true, abs=0.10)
    assert res.alpha.estimate == pytest.approx(capm_dgp.alpha_true, abs=0.004)


def test_excess_flag_matches_explicit_risk_free(capm_dgp) -> None:
    """Passing raw returns with a risk-free series must equal the excess form."""
    rf = capm_dgp.risk_free
    asset_raw = capm_dgp.asset_excess + rf
    market_raw = capm_dgp.market_excess + rf

    via_rf = CAPM().fit(asset_raw, market_raw, risk_free=rf)
    via_excess = CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )
    assert via_rf.alpha.estimate == pytest.approx(via_excess.alpha.estimate, rel=1e-10)
    assert via_rf.beta.estimate == pytest.approx(via_excess.beta.estimate, rel=1e-10)


def test_scalar_risk_free_supported(capm_dgp) -> None:
    asset_raw = capm_dgp.asset_excess + 0.001
    market_raw = capm_dgp.market_excess + 0.001
    res = CAPM().fit(asset_raw, market_raw, risk_free=0.001)
    assert res.beta.estimate == pytest.approx(capm_dgp.beta_true, abs=0.10)


def test_variance_ratios_sum_to_one(capm_dgp) -> None:
    res = CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )
    assert (
        res.systematic_variance_ratio + res.idiosyncratic_variance_ratio
        == pytest.approx(1.0)
    )
    assert 0.0 <= res.systematic_variance_ratio <= 1.0


def test_annualized_alpha_formula(capm_dgp) -> None:
    res = CAPM().fit(
        capm_dgp.asset_excess,
        capm_dgp.market_excess,
        returns_are_excess=True,
        periods_per_year=12,
    )
    expected = (1.0 + res.alpha.estimate) ** 12 - 1.0
    assert res.annualized_alpha == pytest.approx(expected)


def test_beta_equals_one_test_populated(capm_dgp) -> None:
    res = CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )
    # With true beta = 1.15 and low noise, beta != 1 should be detectable.
    assert np.isfinite(res.beta_t_vs_one)
    assert np.isfinite(res.beta_p_vs_one)
    # Internal consistency: the t vs 1 equals (beta - 1)/se.
    expected_t = (res.beta.estimate - 1.0) / res.beta.std_error
    assert res.beta_t_vs_one == pytest.approx(expected_t)


def test_beta_one_dgp_not_rejected(rng) -> None:
    n = 400
    mkt = rng.normal(0.005, 0.04, size=n)
    asset = mkt + rng.normal(0.0, 0.02, size=n)  # alpha=0, beta=1
    res = CAPM().fit(asset, mkt, returns_are_excess=True)
    assert res.beta_p_vs_one > 0.05  # fail to reject H0: beta = 1


def test_treynor_ratio(capm_dgp) -> None:
    res = CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )
    expected = res.mean_asset_excess / res.beta.estimate
    assert res.treynor_ratio == pytest.approx(expected)


def test_listwise_deletion_of_missing_periods(capm_dgp) -> None:
    asset = capm_dgp.asset_excess.copy()
    market = capm_dgp.market_excess.copy()
    asset[10] = np.nan
    market[20] = np.nan
    res = CAPM().fit(asset, market, returns_are_excess=True)
    # Two periods dropped by complete-case analysis.
    assert res.regression.n_observations == len(asset) - 2


def test_all_nan_after_excess_raises(capm_dgp) -> None:
    asset = np.full_like(capm_dgp.asset_excess, np.nan)
    with pytest.raises((NonFiniteError, Exception)):
        CAPM().fit(asset, capm_dgp.market_excess, returns_are_excess=True)


def test_excess_flag_with_risk_free_rejected(capm_dgp) -> None:
    with pytest.raises(ValueError, match="risk_free must be None"):
        CAPM().fit(
            capm_dgp.asset_excess,
            capm_dgp.market_excess,
            risk_free=0.001,
            returns_are_excess=True,
        )


def test_nonpositive_periods_per_year_rejected(capm_dgp) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        CAPM().fit(
            capm_dgp.asset_excess,
            capm_dgp.market_excess,
            returns_are_excess=True,
            periods_per_year=0,
        )


def test_summary_contains_key_sections(capm_dgp) -> None:
    res = CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )
    text = res.summary()
    assert "Capital Asset Pricing Model" in text
    assert "alpha" in text and "beta" in text
    assert "Jarque-Bera" in text
    assert "H0: beta  = 1" in text


def test_model_identity() -> None:
    model = CAPM()
    assert model.name == "Capital Asset Pricing Model"
    assert model.factor_names == ("Mkt-RF",)


def test_result_type(capm_dgp) -> None:
    res = CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )
    assert isinstance(res, CAPMResult)


@pytest.mark.validation
def test_capm_matches_statsmodels(capm_dgp) -> None:
    sm = pytest.importorskip("statsmodels.api")
    design = sm.add_constant(capm_dgp.market_excess)
    ref = sm.OLS(capm_dgp.asset_excess, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False}
    )
    res = CAPM().fit(
        capm_dgp.asset_excess,
        capm_dgp.market_excess,
        returns_are_excess=True,
        hac_lags=4,
        small_sample_correction=False,
    )
    np.testing.assert_allclose(
        [res.alpha.estimate, res.beta.estimate], ref.params, rtol=1e-9
    )
    np.testing.assert_allclose(
        [res.alpha.std_error, res.beta.std_error], ref.bse, rtol=1e-8, atol=1e-11
    )
