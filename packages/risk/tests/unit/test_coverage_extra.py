"""Edge-case, validation, and branch-coverage tests."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_risk import _validation as VAL
from factorlab_risk import attribution as A
from factorlab_risk import portfolio_risk as PR
from factorlab_risk.attribution import FactorRiskAttribution
from factorlab_risk.errors import (
    DimensionMismatchError,
    InsufficientDataError,
    RiskInputError,
)
from factorlab_risk.scenario import ScenarioEngine, SensitivityResult
from factorlab_risk.var import monte_carlo as MC
from factorlab_risk.var import parametric as P
from factorlab_risk.var import rolling as ROLL


# -- _validation ----------------------------------------------------------- #
def test_validation_return_vector() -> None:
    with pytest.raises(RiskInputError):
        VAL.as_return_vector(np.zeros((2, 2)))  # 2-D
    with pytest.raises(RiskInputError):
        VAL.as_return_vector([])  # empty
    with pytest.raises(RiskInputError):
        VAL.as_return_vector([0.1, np.inf])  # non-finite


def test_validation_return_matrix() -> None:
    with pytest.raises(RiskInputError):
        VAL.as_return_matrix(np.zeros(5))  # 1-D
    with pytest.raises(RiskInputError):
        VAL.as_return_matrix(np.empty((0, 3)))  # empty
    with pytest.raises(RiskInputError):
        VAL.as_return_matrix([[0.1, np.nan]])  # non-finite


def test_validation_weights_and_covariance() -> None:
    with pytest.raises(RiskInputError):
        VAL.as_weights(np.zeros((2, 2)))
    with pytest.raises(RiskInputError):
        VAL.as_weights([0.5, np.nan])
    with pytest.raises(RiskInputError):
        VAL.as_covariance(np.zeros((2, 3)))  # non-square
    with pytest.raises(RiskInputError):
        VAL.as_covariance([[0.1, np.nan], [np.nan, 0.1]])  # non-finite
    with pytest.raises(RiskInputError):
        VAL.as_covariance([[0.04, 0.02], [0.01, 0.09]])  # asymmetric


def test_validation_confidence_and_lengths() -> None:
    with pytest.raises(RiskInputError):
        VAL.check_confidence(0.0)
    assert VAL.tail_alpha(0.95) == pytest.approx(0.05)
    with pytest.raises(DimensionMismatchError):
        VAL.check_lengths_match(np.zeros(3), np.zeros(4))


# -- errors ---------------------------------------------------------------- #
def test_insufficient_data_error() -> None:
    err = InsufficientDataError(1, 30, statistic="VaR")
    assert err.n_obs == 1 and err.minimum == 30
    assert "VaR" in str(err)


# -- attribution edge branches --------------------------------------------- #
def test_zero_vol_contributions() -> None:
    w = np.array([0.5, 0.5])
    cov = np.zeros((2, 2))
    np.testing.assert_allclose(A.marginal_contribution_to_risk(w, cov), [0.0, 0.0])
    np.testing.assert_allclose(A.percentage_contribution_to_risk(w, cov), [0.0, 0.0])


def test_attribution_dim_mismatches() -> None:
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    with pytest.raises(DimensionMismatchError):
        A.portfolio_volatility(np.array([0.4, 0.3, 0.3]), cov)
    with pytest.raises(DimensionMismatchError):
        A.risk_budget_deviation(np.array([0.5, 0.5]), cov, np.array([1.0]))


def test_factor_attribution_zero_variance_fractions() -> None:
    fa = FactorRiskAttribution(0.0, 0.0, 0.0, np.zeros(2), np.zeros(2))
    assert np.isnan(fa.systematic_fraction)
    assert np.isnan(fa.specific_fraction)
    assert fa.total_volatility == 0.0


# -- portfolio_risk edge branches ------------------------------------------ #
def test_pr_nan_on_single_obs() -> None:
    assert np.isnan(PR.volatility(np.array([0.01])))
    assert np.isnan(PR.tracking_error(np.array([0.01]), np.array([0.0])))
    assert np.isnan(PR.information_ratio(np.array([0.01]), np.array([0.0])))
    assert np.isnan(PR.beta(np.array([0.01]), np.array([0.0])))


def test_pr_degenerate_cases() -> None:
    # constant benchmark -> beta nan
    assert np.isnan(PR.beta(np.array([0.01, 0.02, 0.03]), np.full(3, 0.01)))
    # zero active -> IR nan
    r = np.array([0.02, 0.03, 0.04])
    assert np.isnan(PR.information_ratio(r, r))
    # zero-vol portfolio -> diversification nan
    assert np.isnan(PR.diversification_ratio([0.5, 0.5], np.zeros((2, 2))))
    # all-zero weights -> herfindahl / concentration nan
    assert np.isnan(PR.herfindahl_index([0.0, 0.0]))
    assert np.isnan(PR.effective_number_of_assets([0.0, 0.0]))
    assert np.isnan(PR.concentration_ratio([0.0, 0.0]))


def test_pr_window_validation() -> None:
    with pytest.raises(RiskInputError):
        PR.rolling_volatility(np.array([0.01, 0.02, 0.03]), window=1)
    with pytest.raises(RiskInputError):
        PR.rolling_volatility(np.array([0.01, 0.02, 0.03]), window=10)


# -- scenario edge branches ------------------------------------------------ #
def test_sensitivity_single_value_delta() -> None:
    sr = SensitivityResult("A", "asset", np.array([0.1]), np.array([0.04]), np.array([0.04]))
    assert np.isnan(sr.delta)


def test_scenario_engine_bad_exposures() -> None:
    with pytest.raises(DimensionMismatchError):
        ScenarioEngine(("A", "B"), exposures=np.ones((3, 2)), factor_names=("f1", "f2"))


def test_sensitivity_bad_kind(assets, weights) -> None:
    eng = ScenarioEngine(assets)
    with pytest.raises(RiskInputError):
        eng.sensitivity(weights, "A", np.array([0.1, 0.2]), kind="bogus")
    with pytest.raises(RiskInputError):
        eng.sensitivity(weights, "A", np.array([]))  # empty


def test_sensitivity_factor_kind(assets, weights) -> None:
    B = np.array([[1.0, 0.2], [0.9, -0.1], [1.1, 0.3]])
    eng = ScenarioEngine(assets, exposures=B, factor_names=("MKT", "SMB"))
    sens = eng.sensitivity(weights, "MKT", np.linspace(-0.1, 0.1, 11), kind="factor")
    # slope = portfolio exposure to MKT = w @ B[:, 0]
    assert sens.delta == pytest.approx(float(weights @ B[:, 0]), abs=1e-9)


# -- stress validation branches -------------------------------------------- #
def test_stress_builder_validation(assets, weights) -> None:
    from factorlab_risk import stress as S

    with pytest.raises(DimensionMismatchError):
        S.market_crash_scenario(assets, -0.3, betas=[1.0, 0.9])  # wrong length
    with pytest.raises(DimensionMismatchError):
        S.interest_rate_shock_scenario(assets, 0.01, rate_betas=[-2.0])
    with pytest.raises(DimensionMismatchError):
        S.sector_shock_scenario(assets, ["tech", "tech"], "tech", -0.1)  # sectors mismatch
    with pytest.raises(DimensionMismatchError):
        S.historical_scenario("h", np.zeros((5, 2)), assets)  # asset count mismatch
    # factor_shock_scenario builds a valid scenario
    sc = S.factor_shock_scenario("MKT", -0.1)
    assert sc.factor_shocks == {"MKT": -0.1}


# -- integration (no peers needed) ----------------------------------------- #
def test_integration_extract_plain() -> None:
    from factorlab_risk import integration as I

    np.testing.assert_allclose(I.extract_returns([0.01, -0.02, 0.03]), [0.01, -0.02, 0.03])
    assets, w = I.extract_weights({"A": 0.6, "B": 0.4})
    assert assets == ("A", "B")
    np.testing.assert_allclose(w, [0.6, 0.4])
    with pytest.raises(RiskInputError):
        I.extract_weights(object())


# -- parametric edge branches ---------------------------------------------- #
def test_parametric_std_negative() -> None:
    with pytest.raises(RiskInputError):
        P.parametric_var(mean=0.0, std=-0.01, confidence=0.95)


def test_parametric_es_student_t() -> None:
    es_t = P.parametric_expected_shortfall(
        mean=0.0, std=0.02, confidence=0.99, distribution="t", dof=5
    )
    es_n = P.parametric_expected_shortfall(
        mean=0.0, std=0.02, confidence=0.99, distribution="normal"
    )
    assert es_t > es_n  # fatter tail
    with pytest.raises(RiskInputError):
        P.parametric_expected_shortfall(mean=0.0, std=0.02, distribution="cauchy")
    with pytest.raises(RiskInputError):
        P.parametric_expected_shortfall(mean=0.0, std=0.02, distribution="t", dof=2.0)


# -- monte carlo edge branches --------------------------------------------- #
def test_mc_es_from_mean_std() -> None:
    es = MC.monte_carlo_expected_shortfall(
        mean=0.0, std=0.02, confidence=0.95, n_simulations=50_000, seed=1
    )
    assert es > 0


def test_mc_es_requires_inputs() -> None:
    with pytest.raises(RiskInputError):
        MC.monte_carlo_expected_shortfall(confidence=0.95)


def test_mc_portfolio_dim_mismatch(weights, covariance) -> None:
    with pytest.raises(DimensionMismatchError):
        MC.simulate_portfolio_returns(weights, np.zeros(2), covariance)


# -- rolling parametric ES ------------------------------------------------- #
def test_rolling_es_parametric(returns) -> None:
    out = ROLL.rolling_expected_shortfall(returns, window=60, method="parametric")
    assert np.all(np.isfinite(out[59:]))
    with pytest.raises(RiskInputError):
        ROLL.rolling_expected_shortfall(returns, window=60, method="bogus")
