"""Unit tests for the robust covariance estimators."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant.estimation.hac import (
    bartlett_weights,
    default_hac_lags,
    newey_west_covariance,
    white_covariance,
)


def test_default_hac_lags_known_value() -> None:
    # floor(4 * (100/100)^(2/9)) = floor(4) = 4
    assert default_hac_lags(100) == 4


def test_default_hac_lags_is_nondecreasing() -> None:
    values = [default_hac_lags(n) for n in (10, 50, 100, 500, 5000)]
    assert values == sorted(values)


def test_default_hac_lags_bounds() -> None:
    assert default_hac_lags(1) == 0
    assert default_hac_lags(2) <= 1


def test_bartlett_weights_triangular() -> None:
    w = bartlett_weights(3)
    np.testing.assert_allclose(w, [0.75, 0.5, 0.25])


def test_bartlett_weights_empty_for_zero_lags() -> None:
    assert bartlett_weights(0).size == 0


def _bread(design: np.ndarray) -> np.ndarray:
    return np.linalg.inv(design.T @ design)


def test_newey_west_is_symmetric_and_psd(design_and_response) -> None:
    y, design = design_and_response
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    cov, lags = newey_west_covariance(design, resid, _bread(design))
    assert lags == default_hac_lags(len(y))
    np.testing.assert_allclose(cov, cov.T, atol=1e-15)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > -1e-10  # positive semi-definite


def test_hac_with_zero_lags_equals_white_hc0(design_and_response) -> None:
    y, design = design_and_response
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    bread = _bread(design)
    hac0, _ = newey_west_covariance(
        design, resid, bread, lags=0, small_sample_correction=False
    )
    hc0 = white_covariance(design, resid, bread, small_sample_correction=False)
    np.testing.assert_allclose(hac0, hc0, atol=1e-12)


@pytest.mark.validation
def test_newey_west_matches_statsmodels(design_and_response) -> None:
    sm = pytest.importorskip("statsmodels.api")
    y, design = design_and_response
    lags = 5
    ref = sm.OLS(y, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    cov, _ = newey_west_covariance(
        design, resid, _bread(design), lags=lags, small_sample_correction=False
    )
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), ref.bse, rtol=1e-8, atol=1e-10)


@pytest.mark.validation
def test_white_hc0_matches_statsmodels(design_and_response) -> None:
    sm = pytest.importorskip("statsmodels.api")
    y, design = design_and_response
    ref = sm.OLS(y, design).fit(cov_type="HC0")
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    cov = white_covariance(design, resid, _bread(design), small_sample_correction=False)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), ref.bse, rtol=1e-9, atol=1e-11)
