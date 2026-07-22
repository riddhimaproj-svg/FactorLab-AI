"""Tests for strategies."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_backtesting import EqualWeightStrategy, OptimizerStrategy, StaticWeightStrategy
from factorlab_backtesting.errors import BacktestInputError
from factorlab_backtesting.strategy import StrategyContext


def _context(assets=("A", "B", "C"), window=None):
    if window is None:
        window = np.zeros((10, len(assets)))
    return StrategyContext(np.datetime64("2020-06-01"), assets, window, dict.fromkeys(assets, 0.0))


def test_equal_weight() -> None:
    w = EqualWeightStrategy().compute_weights(_context())
    assert w == {"A": pytest.approx(1 / 3), "B": pytest.approx(1 / 3), "C": pytest.approx(1 / 3)}


def test_static_weight() -> None:
    strat = StaticWeightStrategy({"A": 0.7, "B": 0.3})
    assert strat.compute_weights(_context(("A", "B"))) == {"A": 0.7, "B": 0.3}


def test_static_weight_requires_weights() -> None:
    with pytest.raises(BacktestInputError):
        StaticWeightStrategy({})


def test_optimizer_strategy_lookback_validation() -> None:
    from factorlab_optimizer import MinVarianceOptimizer

    with pytest.raises(BacktestInputError):
        OptimizerStrategy(MinVarianceOptimizer(), lookback=1)


def test_optimizer_strategy_produces_weights(rng) -> None:
    from factorlab_optimizer import Constraint, MinVarianceOptimizer

    assets = ("A", "B", "C")
    window = rng.normal(0.001, 0.01, size=(60, 3))
    strat = OptimizerStrategy(
        MinVarianceOptimizer(), lookback=60, constraints=(Constraint.long_only(),)
    )
    w = strat.compute_weights(_context(assets, window))
    assert set(w) == set(assets)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(v >= -1e-8 for v in w.values())


def test_optimizer_strategy_custom_estimators(rng) -> None:
    from factorlab_optimizer import MinVarianceOptimizer

    window = rng.normal(0.001, 0.01, size=(60, 3))
    strat = OptimizerStrategy(
        MinVarianceOptimizer(),
        lookback=60,
        mean_estimator=lambda r: r.mean(axis=0),
        cov_estimator=lambda r: np.cov(r, rowvar=False, ddof=1),
    )
    w = strat.compute_weights(_context(("A", "B", "C"), window))
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)


def test_optimizer_strategy_insufficient_data_falls_back(rng) -> None:
    from factorlab_optimizer import MinVarianceOptimizer

    window = rng.normal(0, 0.01, size=(1, 3))  # only 1 row
    strat = OptimizerStrategy(MinVarianceOptimizer(), lookback=60)
    w = strat.compute_weights(_context(("A", "B", "C"), window))
    assert w == {"A": pytest.approx(1 / 3), "B": pytest.approx(1 / 3), "C": pytest.approx(1 / 3)}
