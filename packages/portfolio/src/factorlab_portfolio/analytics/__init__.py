"""Pure performance and risk analytics (operate on return arrays)."""

from __future__ import annotations

from factorlab_portfolio.analytics.drawdown import (
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    time_to_recovery,
)
from factorlab_portfolio.analytics.performance import (
    annualized_return,
    annualized_volatility,
    cagr,
    calmar_ratio,
    cumulative_return,
    downside_deviation,
    mean_return,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    wealth_index,
)
from factorlab_portfolio.analytics.relative import (
    active_return,
    beta,
    information_ratio,
    tracking_error,
    treynor_ratio,
)
from factorlab_portfolio.analytics.rolling import (
    rolling_beta,
    rolling_return,
    rolling_sharpe,
    rolling_volatility,
)

__all__ = [
    "active_return",
    "annualized_return",
    "annualized_volatility",
    "beta",
    "cagr",
    "calmar_ratio",
    "cumulative_return",
    "downside_deviation",
    "drawdown_series",
    "information_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "mean_return",
    "omega_ratio",
    "rolling_beta",
    "rolling_return",
    "rolling_sharpe",
    "rolling_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "time_to_recovery",
    "tracking_error",
    "treynor_ratio",
    "wealth_index",
]
