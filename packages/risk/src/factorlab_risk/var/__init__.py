"""Value-at-Risk and Expected Shortfall: historical, parametric, Monte Carlo."""

from __future__ import annotations

from factorlab_risk.var.decomposition import (
    component_var,
    incremental_var,
    marginal_var,
    percent_contribution_var,
    portfolio_var,
    portfolio_volatility,
)
from factorlab_risk.var.historical import (
    historical_expected_shortfall,
    historical_var,
    tail_loss,
    worst_loss,
)
from factorlab_risk.var.monte_carlo import (
    monte_carlo_expected_shortfall,
    monte_carlo_portfolio_var,
    monte_carlo_var,
    simulate_portfolio_returns,
)
from factorlab_risk.var.parametric import (
    parametric_expected_shortfall,
    parametric_var,
)
from factorlab_risk.var.rolling import rolling_expected_shortfall, rolling_var

__all__ = [
    "component_var",
    "historical_expected_shortfall",
    "historical_var",
    "incremental_var",
    "marginal_var",
    "monte_carlo_expected_shortfall",
    "monte_carlo_portfolio_var",
    "monte_carlo_var",
    "parametric_expected_shortfall",
    "parametric_var",
    "percent_contribution_var",
    "portfolio_var",
    "portfolio_volatility",
    "rolling_expected_shortfall",
    "rolling_var",
    "simulate_portfolio_returns",
    "tail_loss",
    "worst_loss",
]
