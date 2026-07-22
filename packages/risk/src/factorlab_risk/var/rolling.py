"""Rolling VaR and Expected Shortfall over a trailing window.

Rolling risk reveals how tail risk evolves through time -- rising into crises and
falling in calm regimes -- rather than collapsing the sample to one number.  Each
returned array matches the input length; the first ``window - 1`` entries are
``nan`` (an incomplete window), and entry ``i`` is computed from the trailing
window ``returns[i - window + 1 : i + 1]``.
"""

from __future__ import annotations

import numpy as np

from factorlab_risk._validation import FloatArray, as_return_vector, check_confidence
from factorlab_risk.errors import RiskInputError
from factorlab_risk.var.historical import historical_expected_shortfall, historical_var
from factorlab_risk.var.parametric import parametric_expected_shortfall, parametric_var

__all__ = ["rolling_var", "rolling_expected_shortfall"]


def _validate_window(n: int, window: int) -> None:
    if window < 2:
        raise RiskInputError("window must be >= 2")
    if window > n:
        raise RiskInputError(f"window ({window}) exceeds series length ({n})")


def rolling_var(
    returns: object,
    window: int,
    confidence: float = 0.95,
    horizon: int = 1,
    method: str = "historical",
) -> FloatArray:
    """Rolling VaR (``method`` = ``"historical"`` or ``"parametric"``)."""
    check_confidence(confidence)
    r = as_return_vector(returns)
    _validate_window(r.shape[0], window)
    out = np.full(r.shape[0], np.nan)
    for i in range(window - 1, r.shape[0]):
        block = r[i - window + 1 : i + 1]
        if method == "historical":
            out[i] = historical_var(block, confidence, horizon)
        elif method == "parametric":
            out[i] = parametric_var(block, confidence, horizon)
        else:
            raise RiskInputError(f"unknown method {method!r}")
    return out


def rolling_expected_shortfall(
    returns: object,
    window: int,
    confidence: float = 0.95,
    horizon: int = 1,
    method: str = "historical",
) -> FloatArray:
    """Rolling Expected Shortfall / CVaR."""
    check_confidence(confidence)
    r = as_return_vector(returns)
    _validate_window(r.shape[0], window)
    out = np.full(r.shape[0], np.nan)
    for i in range(window - 1, r.shape[0]):
        block = r[i - window + 1 : i + 1]
        if method == "historical":
            out[i] = historical_expected_shortfall(block, confidence, horizon)
        elif method == "parametric":
            out[i] = parametric_expected_shortfall(block, confidence, horizon)
        else:
            raise RiskInputError(f"unknown method {method!r}")
    return out
