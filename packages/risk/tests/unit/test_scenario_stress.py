"""Tests for scenario analysis and stress testing."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_risk import stress as S
from factorlab_risk.errors import DimensionMismatchError, RiskInputError
from factorlab_risk.scenario import Scenario, ScenarioEngine


@pytest.fixture
def engine(assets) -> ScenarioEngine:
    B = np.array([[1.0, 0.2], [0.9, -0.1], [1.1, 0.3]])
    return ScenarioEngine(assets, exposures=B, factor_names=("MKT", "SMB"))


# -- Scenario engine ------------------------------------------------------- #
def test_revalue_asset_shocks(engine, weights) -> None:
    sc = Scenario("s", asset_shocks={"A": -0.1, "B": -0.05, "C": 0.0})
    out = engine.revalue(weights, sc, portfolio_value=1_000_000)
    expected_ret = 0.4 * -0.1 + 0.35 * -0.05 + 0.25 * 0.0
    assert out.portfolio_return == pytest.approx(expected_ret)
    assert out.pnl == pytest.approx(expected_ret * 1_000_000)


def test_revalue_factor_shocks(engine, weights) -> None:
    sc = Scenario("f", factor_shocks={"MKT": -0.1})
    out = engine.revalue(weights, sc)
    # asset shock_i = exposures[i, MKT] * -0.1 ; MKT betas [1.0, 0.9, 1.1]
    expected = weights @ (np.array([1.0, 0.9, 1.1]) * -0.1)
    assert out.portfolio_return == pytest.approx(expected)


def test_compare_sorted_worst_first(engine, weights) -> None:
    scenarios = [
        Scenario("mild", asset_shocks={"A": -0.02}),
        Scenario("severe", asset_shocks={"A": -0.20}),
    ]
    ranked = engine.compare(weights, scenarios)
    assert ranked[0].scenario_name == "severe"


def test_sensitivity(engine, weights) -> None:
    values = np.linspace(-0.1, 0.1, 21)
    sens = engine.sensitivity(weights, "A", values, kind="asset")
    assert sens.pnls.shape == values.shape
    # dP&L/dshockA = w_A = 0.4
    assert sens.delta == pytest.approx(0.4, abs=1e-9)


def test_scenario_validation(engine, weights) -> None:
    with pytest.raises(RiskInputError):
        engine.revalue(weights, Scenario("x", asset_shocks={"Z": -0.1}))  # unknown asset
    with pytest.raises(RiskInputError):
        engine.revalue(weights, Scenario("x", factor_shocks={"NOPE": -0.1}))  # unknown factor
    with pytest.raises(DimensionMismatchError):
        engine.revalue(np.array([0.5, 0.5]), Scenario("x"))  # wrong weights length


def test_factor_shock_without_exposures(assets, weights) -> None:
    eng = ScenarioEngine(assets)  # no exposures
    with pytest.raises(RiskInputError):
        eng.revalue(weights, Scenario("f", factor_shocks={"MKT": -0.1}))


def test_scenario_roundtrip() -> None:
    sc = Scenario("s", asset_shocks={"A": -0.1}, factor_shocks={"MKT": -0.05}, description="d")
    assert Scenario.from_dict(sc.to_dict()) == sc


# -- Stress builders ------------------------------------------------------- #
def test_market_crash(assets, weights) -> None:
    sc = S.market_crash_scenario(assets, -0.3, betas=[1.0, 0.9, 1.1])
    eng = ScenarioEngine(assets)
    out = eng.revalue(weights, sc)
    expected = weights @ (np.array([1.0, 0.9, 1.1]) * -0.3)
    assert out.portfolio_return == pytest.approx(expected)


def test_market_crash_uniform(assets, weights) -> None:
    sc = S.market_crash_scenario(assets, -0.2)  # no betas -> uniform
    eng = ScenarioEngine(assets)
    assert eng.revalue(weights, sc).portfolio_return == pytest.approx(-0.2)


def test_interest_rate_shock(assets, weights) -> None:
    sc = S.interest_rate_shock_scenario(assets, 0.01, rate_betas=[-2.0, -5.0, -1.0])
    eng = ScenarioEngine(assets)
    expected = weights @ (np.array([-2.0, -5.0, -1.0]) * 0.01)
    assert eng.revalue(weights, sc).portfolio_return == pytest.approx(expected)


def test_sector_shock(assets, weights) -> None:
    sc = S.sector_shock_scenario(assets, ["tech", "tech", "energy"], "tech", -0.15)
    eng = ScenarioEngine(assets)
    expected = (0.4 + 0.35) * -0.15
    assert eng.revalue(weights, sc).portfolio_return == pytest.approx(expected)


def test_sector_shock_unknown_sector(assets) -> None:
    with pytest.raises(RiskInputError):
        S.sector_shock_scenario(assets, ["tech", "tech", "energy"], "healthcare", -0.1)


def test_historical_scenario(assets, weights) -> None:
    window = np.array([[-0.02, -0.01, 0.0], [-0.03, -0.02, -0.01]])
    sc = S.historical_scenario("mini_crash", window, assets)
    # cumulative per asset
    cum = np.prod(1 + window, axis=0) - 1
    eng = ScenarioEngine(assets)
    assert eng.revalue(weights, sc).portfolio_return == pytest.approx(weights @ cum)


def test_volatility_shock(weights, covariance) -> None:
    res = S.volatility_shock(weights, covariance, vol_multiplier=2.0, confidence=0.95)
    # VaR scales linearly with vol multiplier
    assert res.shocked_var == pytest.approx(2.0 * res.base_var)
    assert res.var_increase == pytest.approx(res.base_var)
    assert "var_increase" in res.to_dict()


def test_volatility_shock_validation(weights, covariance) -> None:
    with pytest.raises(RiskInputError):
        S.volatility_shock(weights, covariance, vol_multiplier=-1.0)


def test_run_stress_test(assets, weights) -> None:
    eng = ScenarioEngine(assets)
    scenarios = [
        S.market_crash_scenario(assets, -0.3),
        S.sector_shock_scenario(assets, ["tech", "tech", "energy"], "energy", -0.2),
    ]
    report = S.run_stress_test(eng, weights, scenarios, portfolio_value=1_000_000)
    assert report.worst_case is not None
    assert report.worst_case.pnl <= report.best_case.pnl
    assert len(report.as_table()) == 2
