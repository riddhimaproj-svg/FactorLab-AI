"""Tests for historical VaR / ES."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_risk.errors import RiskInputError
from factorlab_risk.var import historical as H


def test_historical_var_matches_quantile() -> None:
    r = np.linspace(-0.10, 0.10, 201)  # symmetric grid
    # 5th percentile of this grid; VaR = -quantile
    expected = -float(np.quantile(r, 0.05, method="linear"))
    assert H.historical_var(r, 0.95) == pytest.approx(expected)


def test_var_is_positive_loss() -> None:
    r = np.array([-0.05, -0.03, 0.01, 0.02, 0.04, -0.02, 0.03, -0.06])
    assert H.historical_var(r, 0.9) > 0


def test_es_ge_var() -> None:
    r = np.array([-0.08, -0.05, -0.03, 0.01, 0.02, 0.04, -0.02, 0.03, -0.06, 0.05])
    var = H.historical_var(r, 0.9)
    es = H.historical_expected_shortfall(r, 0.9)
    assert es >= var  # ES is the mean beyond VaR, always at least as large


def test_es_analytic() -> None:
    # returns where tail is clear: worst 10% of 10 obs = the single worst
    r = np.array([-0.10, -0.05, -0.03, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    es = H.historical_expected_shortfall(r, 0.90)
    # tail (r <= 10th percentile) -> the worst value(s); mean of them
    q = np.quantile(r, 0.10, method="linear")
    expected = -np.mean(r[r <= q])
    assert es == pytest.approx(expected)


def test_horizon_scaling() -> None:
    r = np.array([-0.05, -0.03, 0.01, 0.02, 0.04, -0.02, 0.03, -0.06])
    assert H.historical_var(r, 0.9, horizon=4) == pytest.approx(2.0 * H.historical_var(r, 0.9))


def test_tail_loss_and_worst_loss() -> None:
    r = np.array([-0.08, -0.05, 0.01, 0.02, 0.03])
    assert H.worst_loss(r) == pytest.approx(0.08)
    assert H.tail_loss(r, 0.8) == H.historical_expected_shortfall(r, 0.8, horizon=1)


def test_validation() -> None:
    with pytest.raises(RiskInputError):
        H.historical_var(np.array([]), 0.95)
    with pytest.raises(RiskInputError):
        H.historical_var(np.array([0.01, np.nan]), 0.95)
    with pytest.raises(RiskInputError):
        H.historical_var(np.array([0.01, 0.02]), 1.5)  # bad confidence
