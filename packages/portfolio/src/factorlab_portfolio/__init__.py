"""FactorLab portfolio analytics — immutable models + reusable performance/risk metrics.

A pure, framework-free library.  It provides:

* **Models** -- :class:`Position`, :class:`Holding`, :class:`Trade`,
  :class:`Portfolio`, :class:`PortfolioSnapshot` (all immutable and serializable).
* **ReturnSeries** -- the central date-aware analytics object.
* **Analytics** -- absolute (return, volatility, Sharpe, Sortino, Calmar, Omega),
  drawdown (max drawdown, duration), benchmark-relative (beta, tracking error,
  active return, information ratio, Treynor), and rolling variants.
* **PerformanceReport** -- a serializable bundle of every headline metric.

It performs no optimization, no backtesting, and no I/O.  The immutable models
and pure analytics are the substrate those future layers will build on.

Quick start
-----------
>>> import numpy as np
>>> from factorlab_portfolio import ReturnSeries
>>> rng = np.random.default_rng(0)
>>> series = ReturnSeries(rng.normal(0.0004, 0.01, 756), periods_per_year=252, name="fund")
>>> report = series.performance_report(risk_free=0.0)
>>> print(report.summary())  # doctest: +SKIP
"""

from __future__ import annotations

from factorlab_portfolio import analytics
from factorlab_portfolio.errors import (
    DimensionMismatchError,
    InsufficientDataError,
    PortfolioError,
    PortfolioValidationError,
)
from factorlab_portfolio.holdings import Holding, Position, Trade
from factorlab_portfolio.portfolio import Portfolio, PortfolioSnapshot
from factorlab_portfolio.report import PerformanceReport
from factorlab_portfolio.returns import ReturnSeries

__version__ = "0.1.0"

__all__ = [
    "DimensionMismatchError",
    "Holding",
    "InsufficientDataError",
    "PerformanceReport",
    "Portfolio",
    "PortfolioError",
    "PortfolioSnapshot",
    "PortfolioValidationError",
    "Position",
    "ReturnSeries",
    "Trade",
    "__version__",
    "analytics",
]
