"""Backward-compatibility contract for the CAPM public API.

The refactor introduced the generic framework beneath CAPM.  These tests pin the
exact surface that existed before, so any future change that breaks a caller is
caught here rather than in a downstream project.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant import CAPM, CAPMResult


@pytest.fixture
def result(capm_dgp) -> CAPMResult:
    return CAPM().fit(
        capm_dgp.asset_excess,
        capm_dgp.market_excess,
        returns_are_excess=True,
        periods_per_year=capm_dgp.periods_per_year,
    )


def test_constructor_and_identity() -> None:
    model = CAPM()
    assert model.name == "Capital Asset Pricing Model"
    assert model.factor_names == ("Mkt-RF",)


def test_result_is_capmresult(result) -> None:
    assert isinstance(result, CAPMResult)


def test_all_legacy_attributes_present(result) -> None:
    # Coefficient accessors
    assert result.alpha.estimate is not None
    assert result.beta.estimate is not None
    assert result.alpha.std_error >= 0.0
    assert result.beta.p_value >= 0.0
    # Risk decomposition
    assert 0.0 <= result.systematic_variance_ratio <= 1.0
    assert result.idiosyncratic_variance_ratio == pytest.approx(
        1.0 - result.systematic_variance_ratio
    )
    assert result.idiosyncratic_volatility >= 0.0
    assert result.annualized_idiosyncratic_volatility >= 0.0
    # Annualization & performance
    assert np.isfinite(result.annualized_alpha)
    assert np.isfinite(result.treynor_ratio)
    # Hypothesis test fields
    assert np.isfinite(result.beta_t_vs_one)
    assert np.isfinite(result.beta_p_vs_one)
    # Means
    assert np.isfinite(result.mean_asset_excess)
    assert np.isfinite(result.mean_market_excess)
    # Underlying regression + metadata
    assert result.regression.n_observations == len(result.response)
    assert result.periods_per_year == 12


def test_summary_contains_legacy_sections(result) -> None:
    text = result.summary()
    assert "Capital Asset Pricing Model" in text
    assert "alpha" in text and "beta" in text
    assert "Jarque-Bera" in text
    assert "H0: beta  = 1" in text


def test_excess_flag_matches_explicit_risk_free(capm_dgp) -> None:
    rf = capm_dgp.risk_free
    via_rf = CAPM().fit(
        capm_dgp.asset_excess + rf, capm_dgp.market_excess + rf, risk_free=rf
    )
    via_excess = CAPM().fit(
        capm_dgp.asset_excess, capm_dgp.market_excess, returns_are_excess=True
    )
    assert via_rf.beta.estimate == pytest.approx(via_excess.beta.estimate, rel=1e-10)


def test_excess_flag_with_risk_free_still_rejected(capm_dgp) -> None:
    with pytest.raises(ValueError, match="risk_free must be None"):
        CAPM().fit(
            capm_dgp.asset_excess,
            capm_dgp.market_excess,
            risk_free=0.001,
            returns_are_excess=True,
        )


def test_covariance_matrix_still_read_only(result) -> None:
    with pytest.raises(ValueError):
        result.regression.covariance_matrix[0, 0] = 1.0


def test_hac_default_and_lag_config(result) -> None:
    assert result.regression.covariance_type == "HAC"
    assert result.regression.cov_config["kernel"] == "bartlett"
