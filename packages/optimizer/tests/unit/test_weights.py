"""Tests for PortfolioWeights."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import PortfolioWeights
from factorlab_optimizer.errors import OptimizationInputError


def test_basic_accessors() -> None:
    w = PortfolioWeights(("A", "B", "C"), np.array([0.5, 0.3, 0.2]))
    assert len(w) == 3
    assert w.get("B") == pytest.approx(0.3)
    assert w.total == pytest.approx(1.0)
    assert w.as_dict() == {"A": 0.5, "B": 0.3, "C": 0.2}


def test_exposures_long_short() -> None:
    w = PortfolioWeights(("A", "B"), np.array([1.5, -0.5]))
    assert w.gross_exposure == pytest.approx(2.0)
    assert w.net_exposure == pytest.approx(1.0)
    assert w.long_exposure == pytest.approx(1.5)
    assert w.short_exposure == pytest.approx(-0.5)
    assert w.cash == pytest.approx(0.0)


def test_cash_residual() -> None:
    w = PortfolioWeights(("A",), np.array([0.7]))
    assert w.cash == pytest.approx(0.3)


def test_nonzero_filter() -> None:
    w = PortfolioWeights(("A", "B", "C"), np.array([0.6, 0.0, 0.4]))
    assert set(w.nonzero()) == {"A", "C"}


def test_validation() -> None:
    with pytest.raises(OptimizationInputError):
        PortfolioWeights(("A", "B"), np.array([1.0]))  # length mismatch
    with pytest.raises(OptimizationInputError):
        PortfolioWeights(("A", "A"), np.array([0.5, 0.5]))  # duplicate
    with pytest.raises(OptimizationInputError):
        PortfolioWeights(("A",), np.array([np.nan]))  # non-finite


def test_get_missing() -> None:
    with pytest.raises(KeyError):
        PortfolioWeights(("A",), np.array([1.0])).get("Z")


def test_constructors() -> None:
    eq = PortfolioWeights.equal_weight(["A", "B", "C", "D"])
    np.testing.assert_allclose(eq.values, 0.25)
    m = PortfolioWeights.from_mapping({"A": 0.6, "B": 0.4})
    assert m.as_dict() == {"A": 0.6, "B": 0.4}


def test_roundtrip() -> None:
    w = PortfolioWeights(("A", "B"), np.array([0.7, 0.3]))
    restored = PortfolioWeights.from_dict(w.to_dict())
    assert restored.assets == w.assets
    np.testing.assert_allclose(restored.values, w.values)


def test_values_read_only() -> None:
    w = PortfolioWeights(("A",), np.array([1.0]))
    with pytest.raises(ValueError):
        w.values[0] = 2.0
