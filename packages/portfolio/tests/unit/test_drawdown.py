"""Analytical validation of drawdown analytics."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_portfolio.analytics import drawdown as D


def test_drawdown_series() -> None:
    dd = D.drawdown_series(np.array([0.1, -0.5]))
    # wealth [1.1, 0.55]; peak [1.1, 1.1]; dd [0, -0.5]
    np.testing.assert_allclose(dd, [0.0, -0.5])


def test_drawdown_series_empty() -> None:
    assert D.drawdown_series(np.array([])).size == 0


def test_max_drawdown() -> None:
    assert D.max_drawdown(np.array([0.1, -0.5])) == pytest.approx(-0.5)
    assert D.max_drawdown(np.array([0.1, 0.1, 0.1])) == pytest.approx(0.0)
    assert np.isnan(D.max_drawdown(np.array([])))


def test_max_drawdown_duration() -> None:
    # wealth 0.9,0.81,0.729,1.0935 ; dd 0,-0.1,-0.19,0 -> underwater run length 2
    r = np.array([-0.1, -0.1, -0.1, 0.5])
    assert D.max_drawdown_duration(r) == 2
    assert D.max_drawdown_duration(np.array([0.1, 0.1])) == 0


def test_time_to_recovery() -> None:
    r = np.array([-0.1, -0.1, -0.1, 0.5])  # trough at idx 2, recovers idx 3
    assert D.time_to_recovery(r) == 1


def test_time_to_recovery_never_recovers() -> None:
    assert D.time_to_recovery(np.array([0.1, -0.5])) is None


def test_time_to_recovery_no_drawdown() -> None:
    assert D.time_to_recovery(np.array([0.1, 0.1])) == 0


def test_time_to_recovery_empty() -> None:
    assert D.time_to_recovery(np.array([])) is None


def test_drawdown_bounded() -> None:
    rng = np.random.default_rng(0)
    r = rng.uniform(-0.3, 0.3, 500)
    dd = D.drawdown_series(r)
    assert dd.max() <= 1e-12
    assert dd.min() >= -1.0
