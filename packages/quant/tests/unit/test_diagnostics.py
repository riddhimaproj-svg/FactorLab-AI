"""Unit tests for regression diagnostics, cross-validated vs statsmodels."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant.diagnostics.residuals import (
    durbin_watson,
    sample_excess_kurtosis,
    sample_skewness,
)
from factorlab_quant.diagnostics.tests import breusch_pagan, f_test, jarque_bera


@pytest.fixture
def residuals_and_design(design_and_response):
    y, design = design_and_response
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return resid, design


def test_skewness_matches_scipy(rng) -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    x = rng.gamma(2.0, 1.0, size=500)
    assert sample_skewness(x) == pytest.approx(scipy_stats.skew(x), rel=1e-10)


def test_excess_kurtosis_matches_scipy(rng) -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    x = rng.standard_t(5, size=500)
    # scipy.stats.kurtosis default is Fisher (excess) and biased -> matches ours.
    assert sample_excess_kurtosis(x) == pytest.approx(
        scipy_stats.kurtosis(x), rel=1e-10
    )


def test_moments_constant_series_are_zero() -> None:
    x = np.full(10, 3.0)
    assert sample_skewness(x) == 0.0
    assert sample_excess_kurtosis(x) == 0.0


@pytest.mark.validation
def test_durbin_watson_matches_statsmodels(residuals_and_design) -> None:
    sm_tools = pytest.importorskip("statsmodels.stats.stattools")
    resid, _ = residuals_and_design
    assert durbin_watson(resid) == pytest.approx(
        sm_tools.durbin_watson(resid), rel=1e-12
    )


@pytest.mark.validation
def test_jarque_bera_matches_statsmodels(residuals_and_design) -> None:
    sm_tools = pytest.importorskip("statsmodels.stats.stattools")
    resid, _ = residuals_and_design
    jb, jb_p, skew, _ = jarque_bera(resid)
    ref_jb, ref_p, ref_skew, _ = sm_tools.jarque_bera(resid)
    assert jb == pytest.approx(ref_jb, rel=1e-9)
    assert jb_p == pytest.approx(ref_p, rel=1e-7, abs=1e-14)
    assert skew == pytest.approx(ref_skew, rel=1e-9)


@pytest.mark.validation
def test_breusch_pagan_matches_statsmodels(residuals_and_design) -> None:
    sm_diag = pytest.importorskip("statsmodels.stats.diagnostic")
    resid, design = residuals_and_design
    lm, lm_p = breusch_pagan(resid, design)
    ref_lm, ref_lm_p, _, _ = sm_diag.het_breuschpagan(resid, design)
    assert lm == pytest.approx(ref_lm, rel=1e-9)
    assert lm_p == pytest.approx(ref_lm_p, rel=1e-7, abs=1e-14)


def test_f_test_undefined_for_intercept_only() -> None:
    stat, p = f_test(r_squared=0.0, n_obs=100, n_params=1)
    assert np.isnan(stat) and np.isnan(p)


def test_durbin_watson_near_two_for_white_noise(rng) -> None:
    noise = rng.normal(size=5000)
    assert durbin_watson(noise) == pytest.approx(2.0, abs=0.1)
