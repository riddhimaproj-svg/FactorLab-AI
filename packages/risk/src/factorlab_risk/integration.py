"""Integration helpers bridging the risk engine to the rest of the platform.

These adapters accept objects from the approved packages -- a
``factorlab_portfolio.ReturnSeries``, a ``factorlab_optimizer.OptimizationResult``
/ ``PortfolioWeights``, or a ``factorlab_backtesting.BacktestResult`` -- and feed
their numeric content into the risk computations.  They use **duck typing** (and
no imports of the peer packages), so the risk core stays dependency-light while
the platform workflow

    market data -> factor model -> optimizer -> backtest -> risk report

composes end to end.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from factorlab_risk._validation import FloatArray
from factorlab_risk.errors import RiskInputError
from factorlab_risk.reports import RiskReport, RiskSnapshot, VaRReport
from factorlab_risk.var import (
    historical_expected_shortfall,
    historical_var,
)

__all__ = [
    "extract_returns",
    "extract_weights",
    "risk_report_from_weights",
    "snapshot_from_returns_and_weights",
    "var_report_from_returns",
]


def extract_returns(source: object) -> FloatArray:
    """Pull a 1-D return array from a ReturnSeries / BacktestResult / array.

    Recognizes any object exposing ``.returns`` (BacktestResult) or ``.values``
    (ReturnSeries); otherwise coerces ``source`` directly.
    """
    for attr in ("returns", "values"):
        candidate = getattr(source, attr, None)
        if candidate is not None and not callable(candidate):
            arr = np.asarray(candidate, dtype=np.float64)
            if arr.ndim == 1:
                return arr
    arr = np.asarray(source, dtype=np.float64)
    if arr.ndim != 1:
        raise RiskInputError("could not extract a 1-D return series from source")
    return arr


def extract_weights(source: object) -> tuple[tuple[str, ...], FloatArray]:
    """Pull ``(assets, weights)`` from PortfolioWeights / OptimizationResult / mapping."""
    # OptimizationResult exposes `.weights` (a PortfolioWeights).
    inner = cast(Any, getattr(source, "weights", None))
    if inner is not None and hasattr(inner, "assets") and hasattr(inner, "values"):
        return tuple(inner.assets), np.asarray(inner.values, dtype=np.float64)
    # PortfolioWeights exposes `.assets` and `.values`.
    if hasattr(source, "assets") and hasattr(source, "values"):
        pw = cast(Any, source)
        return tuple(pw.assets), np.asarray(pw.values, dtype=np.float64)
    # A plain mapping {asset: weight}.
    if isinstance(source, dict):
        return tuple(source.keys()), np.asarray(list(source.values()), dtype=np.float64)
    raise RiskInputError("could not extract (assets, weights) from source")


def var_report_from_returns(
    source: object, confidence: float = 0.95, horizon: int = 1, method: str = "historical"
) -> VaRReport:
    """Build a :class:`VaRReport` from any returns-like object."""
    returns = extract_returns(source)
    return VaRReport.from_returns(returns, confidence, horizon, method)


def risk_report_from_weights(
    weights_source: object,
    covariance: object,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    periods_per_year: float = 252.0,
) -> RiskReport:
    """Build a :class:`RiskReport` from an optimizer result / weights + covariance."""
    assets, weights = extract_weights(weights_source)
    return RiskReport.from_portfolio(
        assets, weights, covariance,
        confidence=confidence, horizon=horizon, periods_per_year=periods_per_year,
    )


def snapshot_from_returns_and_weights(
    returns_source: object,
    weights_source: object,
    covariance: object,
    *,
    confidence: float = 0.95,
    as_of: str | None = None,
) -> RiskSnapshot:
    """A point-in-time :class:`RiskSnapshot` combining realized and model risk."""
    report = risk_report_from_weights(weights_source, covariance, confidence=confidence)
    returns = extract_returns(returns_source)
    # Overlay realized (historical) VaR/ES onto the snapshot.
    snap = report.snapshot(as_of=as_of)
    hist_var = historical_var(returns, confidence)
    hist_es = historical_expected_shortfall(returns, confidence)
    from dataclasses import replace

    return replace(snap, var_95=hist_var, expected_shortfall_95=hist_es)
