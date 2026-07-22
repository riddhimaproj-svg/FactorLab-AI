"""Unit tests for the OLS estimator, including cross-validation vs statsmodels."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant.core.errors import (
    CollinearityError,
    InsufficientDataError,
    NonFiniteError,
)
from factorlab_quant.estimation.ols import OLS


def test_recovers_exact_coefficients_on_noiseless_data() -> None:
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    design = np.column_stack([np.ones(n), x])
    true = np.array([0.3, 2.5])
    y = design @ true  # no noise -> perfect fit
    res = OLS().fit(y, design, param_names=("const", "x"), covariance_type="nonrobust")
    np.testing.assert_allclose(res.params, true, atol=1e-12)
    assert res.diagnostics.r_squared == pytest.approx(1.0, abs=1e-12)


def test_rejects_nonfinite_inputs(design_and_response) -> None:
    y, design = design_and_response
    y_bad = y.copy()
    y_bad[3] = np.nan
    with pytest.raises(NonFiniteError):
        OLS().fit(y_bad, design)


def test_rejects_insufficient_observations() -> None:
    design = np.array([[1.0, 0.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises(InsufficientDataError):
        OLS().fit(y, design)


def test_rejects_collinear_design() -> None:
    n = 40
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    # Third column is an exact linear combination of the first two.
    design = np.column_stack([np.ones(n), x, 2.0 * x + 3.0])
    y = rng.normal(size=n)
    with pytest.raises(CollinearityError):
        OLS().fit(y, design)


def test_unknown_covariance_type_raises(design_and_response) -> None:
    y, design = design_and_response
    with pytest.raises(ValueError, match="covariance_type"):
        OLS().fit(y, design, covariance_type="bogus")  # type: ignore[arg-type]


def test_param_names_length_validated(design_and_response) -> None:
    y, design = design_and_response
    with pytest.raises(ValueError, match="param_names"):
        OLS().fit(y, design, param_names=("only_one",))


@pytest.mark.validation
def test_nonrobust_matches_statsmodels(design_and_response) -> None:
    sm = pytest.importorskip("statsmodels.api")
    y, design = design_and_response
    ref = sm.OLS(y, design).fit()
    res = OLS().fit(y, design, covariance_type="nonrobust")

    np.testing.assert_allclose(res.params, ref.params, rtol=1e-9)
    np.testing.assert_allclose(
        [c.std_error for c in res.coefficients], ref.bse, rtol=1e-9
    )
    np.testing.assert_allclose(
        [c.t_statistic for c in res.coefficients], ref.tvalues, rtol=1e-8
    )
    np.testing.assert_allclose(
        [c.p_value for c in res.coefficients], ref.pvalues, rtol=1e-7, atol=1e-12
    )
    d = res.diagnostics
    assert d.r_squared == pytest.approx(ref.rsquared, rel=1e-10)
    assert d.adj_r_squared == pytest.approx(ref.rsquared_adj, rel=1e-10)
    assert d.f_statistic == pytest.approx(ref.fvalue, rel=1e-8)
    assert d.f_p_value == pytest.approx(ref.f_pvalue, rel=1e-7, abs=1e-14)
    assert d.log_likelihood == pytest.approx(ref.llf, rel=1e-9)
    assert d.aic == pytest.approx(ref.aic, rel=1e-9)
    assert d.bic == pytest.approx(ref.bic, rel=1e-9)


@pytest.mark.validation
def test_hac_standard_errors_match_statsmodels(design_and_response) -> None:
    sm = pytest.importorskip("statsmodels.api")
    y, design = design_and_response
    lags = 6
    ref = sm.OLS(y, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    res = OLS().fit(
        y,
        design,
        covariance_type="HAC",
        hac_lags=lags,
        small_sample_correction=False,
    )
    np.testing.assert_allclose(
        [c.std_error for c in res.coefficients], ref.bse, rtol=1e-8, atol=1e-10
    )
    assert res.cov_config["lags"] == lags
    assert res.cov_config["kernel"] == "bartlett"


def test_covariance_matrix_is_read_only(design_and_response) -> None:
    y, design = design_and_response
    res = OLS().fit(y, design)
    with pytest.raises(ValueError):
        res.covariance_matrix[0, 0] = 999.0


def test_coefficient_lookup_by_name(design_and_response) -> None:
    y, design = design_and_response
    res = OLS().fit(y, design, param_names=("a", "b", "c"))
    assert res.coefficient("b").name == "b"
    with pytest.raises(KeyError):
        res.coefficient("missing")
