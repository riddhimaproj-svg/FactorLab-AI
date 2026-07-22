"""FactorLab risk engine — institutional portfolio & market risk analytics.

A pure, typed computational library (numpy + scipy) for quantitative risk:

* **VaR / ES** (:mod:`factorlab_risk.var`) — historical, parametric
  (Normal / Student-t), and Monte Carlo VaR and Expected Shortfall; rolling
  variants; and the portfolio VaR decomposition (marginal / component /
  incremental VaR).
* **Risk attribution** (:mod:`factorlab_risk.attribution`) — marginal, component,
  and percentage contributions; risk budgeting; factor and sector attribution.
* **Portfolio risk** (:mod:`factorlab_risk.portfolio_risk`) — volatility, tracking
  error, beta, correlation/covariance (and rolling), diversification ratio, and
  concentration (Herfindahl) measures.
* **Stress testing** (:mod:`factorlab_risk.stress`) & **scenario analysis**
  (:mod:`factorlab_risk.scenario`) — named/custom/historical shocks, factor and
  sector shocks, volatility shocks, and portfolio revaluation.
* **Reports** (:mod:`factorlab_risk.reports`) — immutable, serializable
  ``RiskReport``, ``VaRReport``, ``StressTestReport``, ``ScenarioReport``,
  ``RiskDecomposition``, ``RiskContribution``, ``RiskSnapshot``.
* **Integration** (:mod:`factorlab_risk.integration`) — duck-typed bridges to the
  portfolio / optimizer / backtesting packages.

Conventions: returns are per-period simple returns (decimal); confidence ``c`` in
``(0, 1)`` with tail ``alpha = 1 - c``; VaR and ES are positive loss magnitudes.

>>> import numpy as np
>>> from factorlab_risk import historical_var, portfolio_var
>>> r = np.array([-0.03, 0.01, -0.05, 0.02, 0.00, -0.01, 0.03, -0.02])
>>> round(historical_var(r, confidence=0.95), 4)  # doctest: +SKIP
"""

from __future__ import annotations

from factorlab_risk import attribution, integration, portfolio_risk, scenario, stress, var
from factorlab_risk.attribution import (
    FactorRiskAttribution,
    component_contribution_to_risk,
    factor_risk_attribution,
    marginal_contribution_to_risk,
    percentage_contribution_to_risk,
    risk_budget,
    risk_budget_deviation,
    sector_risk_attribution,
)
from factorlab_risk.errors import (
    DimensionMismatchError,
    InsufficientDataError,
    RiskError,
    RiskInputError,
)
from factorlab_risk.portfolio_risk import (
    beta,
    concentration_metrics,
    correlation_matrix,
    covariance_matrix,
    diversification_ratio,
    effective_number_of_assets,
    herfindahl_index,
    information_ratio,
    portfolio_volatility,
    rolling_beta,
    rolling_correlation,
    rolling_covariance,
    rolling_volatility,
    tracking_error,
    volatility,
)
from factorlab_risk.reports import (
    RiskContribution,
    RiskDecomposition,
    RiskReport,
    RiskSnapshot,
    ScenarioReport,
    StressTestReport,
    VaRReport,
)
from factorlab_risk.scenario import Scenario, ScenarioEngine, ScenarioOutcome, SensitivityResult
from factorlab_risk.stress import (
    VolatilityShockResult,
    factor_shock_scenario,
    historical_scenario,
    interest_rate_shock_scenario,
    market_crash_scenario,
    run_stress_test,
    sector_shock_scenario,
    volatility_shock,
)
from factorlab_risk.var import (
    component_var,
    historical_expected_shortfall,
    historical_var,
    incremental_var,
    marginal_var,
    monte_carlo_expected_shortfall,
    monte_carlo_portfolio_var,
    monte_carlo_var,
    parametric_expected_shortfall,
    parametric_var,
    percent_contribution_var,
    portfolio_var,
    rolling_expected_shortfall,
    rolling_var,
    tail_loss,
    worst_loss,
)

__version__ = "0.1.0"

__all__ = [
    "DimensionMismatchError",
    "FactorRiskAttribution",
    "InsufficientDataError",
    "RiskContribution",
    "RiskDecomposition",
    "RiskError",
    "RiskInputError",
    "RiskReport",
    "RiskSnapshot",
    "Scenario",
    "ScenarioEngine",
    "ScenarioOutcome",
    "ScenarioReport",
    "SensitivityResult",
    "StressTestReport",
    "VaRReport",
    "VolatilityShockResult",
    "__version__",
    "attribution",
    "beta",
    "component_contribution_to_risk",
    "component_var",
    "concentration_metrics",
    "correlation_matrix",
    "covariance_matrix",
    "diversification_ratio",
    "effective_number_of_assets",
    "factor_risk_attribution",
    "factor_shock_scenario",
    "herfindahl_index",
    "historical_expected_shortfall",
    "historical_scenario",
    "historical_var",
    "incremental_var",
    "information_ratio",
    "integration",
    "interest_rate_shock_scenario",
    "marginal_contribution_to_risk",
    "marginal_var",
    "market_crash_scenario",
    "monte_carlo_expected_shortfall",
    "monte_carlo_portfolio_var",
    "monte_carlo_var",
    "parametric_expected_shortfall",
    "parametric_var",
    "percent_contribution_var",
    "percentage_contribution_to_risk",
    "portfolio_risk",
    "portfolio_var",
    "portfolio_volatility",
    "risk_budget",
    "risk_budget_deviation",
    "rolling_beta",
    "rolling_correlation",
    "rolling_covariance",
    "rolling_expected_shortfall",
    "rolling_var",
    "rolling_volatility",
    "run_stress_test",
    "scenario",
    "sector_risk_attribution",
    "sector_shock_scenario",
    "stress",
    "tail_loss",
    "tracking_error",
    "var",
    "volatility",
    "volatility_shock",
    "worst_loss",
]
