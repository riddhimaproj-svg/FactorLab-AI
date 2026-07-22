"""Tests for the risk data models / reports."""

from __future__ import annotations

import json

import pytest

from factorlab_risk.reports import (
    RiskContribution,
    RiskDecomposition,
    RiskReport,
    RiskSnapshot,
    ScenarioReport,
    StressTestReport,
    VaRReport,
)
from factorlab_risk.scenario import ScenarioOutcome


def test_risk_contribution_roundtrip() -> None:
    c = RiskContribution("A", 0.4, 0.1, 0.04, 0.3)
    assert RiskContribution.from_dict(c.to_dict()) == c


def test_risk_decomposition(assets, weights, covariance) -> None:
    decomp = RiskDecomposition.from_weights_covariance(assets, weights, covariance)
    # component contributions sum to total volatility
    assert sum(c.component_contribution for c in decomp.contributions) == pytest.approx(
        decomp.total_volatility
    )
    assert sum(c.percentage_contribution for c in decomp.contributions) == pytest.approx(1.0)
    restored = RiskDecomposition.from_dict(decomp.to_dict())
    assert restored.total_volatility == pytest.approx(decomp.total_volatility)
    assert "Risk decomposition" in decomp.summary()


def test_var_report_from_returns(returns) -> None:
    hist = VaRReport.from_returns(returns, 0.95, method="historical")
    param = VaRReport.from_returns(returns, 0.95, method="parametric")
    assert hist.var > 0 and param.var > 0
    assert VaRReport.from_dict(hist.to_dict()) == hist
    assert "VaR" in hist.summary()


def test_var_report_unknown_method(returns) -> None:
    from factorlab_risk.errors import RiskInputError

    with pytest.raises(RiskInputError):
        VaRReport.from_returns(returns, method="bogus")


def test_risk_report(assets, weights, covariance) -> None:
    report = RiskReport.from_portfolio(assets, weights, covariance, confidence=0.95)
    assert report.volatility > 0
    assert report.parametric_var is not None
    # serialization (JSON) round-trip
    payload = json.dumps(report.to_dict())
    restored = RiskReport.from_dict(json.loads(payload))
    assert restored.volatility == pytest.approx(report.volatility)
    assert restored.assets == report.assets
    assert "Portfolio Risk Report" in report.summary()


def test_risk_snapshot(assets, weights, covariance) -> None:
    report = RiskReport.from_portfolio(assets, weights, covariance)
    snap = report.snapshot(as_of="2024-12-31")
    assert snap.as_of == "2024-12-31"
    assert snap.volatility == pytest.approx(report.volatility)
    assert isinstance(snap, RiskSnapshot)
    assert RiskSnapshot.from_dict(snap.to_dict()).volatility == pytest.approx(snap.volatility)


def test_stress_and_scenario_reports() -> None:
    outcomes = (
        ScenarioOutcome("crash", -0.3, -300000.0, {"A": -0.3}),
        ScenarioOutcome("mild", -0.05, -50000.0, {"A": -0.05}),
    )
    st = StressTestReport(outcomes, portfolio_value=1_000_000)
    assert st.worst_case.scenario_name == "crash"
    assert st.best_case.scenario_name == "mild"
    assert st.as_table()[0][0] == "crash"  # worst first
    assert "Stress Test Report" in st.summary()

    sr = ScenarioReport(outcomes, portfolio_value=1_000_000)
    assert sr.ranked()[0].scenario_name == "crash"
    assert "Scenario comparison" in sr.summary()
    assert json.dumps(st.to_dict()) and json.dumps(sr.to_dict())


def test_empty_stress_report() -> None:
    st = StressTestReport((), portfolio_value=1.0)
    assert st.worst_case is None and st.best_case is None
