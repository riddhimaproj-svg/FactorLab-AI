"""Strategies: rules that map trailing data to target weights.

A :class:`Strategy` sees only data available *at* the rebalance date -- its
``StrategyContext`` carries trailing returns computed strictly from past prices --
so strategies built on this interface are structurally free of **look-ahead
bias** (using information that would not have been known at decision time).

Concrete strategies:

* :class:`StaticWeightStrategy` -- fixed target weights.
* :class:`EqualWeightStrategy` -- 1/N over the universe.
* :class:`OptimizerStrategy` -- estimate moments from the trailing window and feed
  them to a ``factorlab_optimizer`` optimizer.  This is the integration point of
  the workflow *expected returns -> optimizer -> weights*.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from factorlab_backtesting.errors import BacktestInputError

__all__ = [
    "EqualWeightStrategy",
    "OptimizerStrategy",
    "StaticWeightStrategy",
    "Strategy",
    "StrategyContext",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy may use to decide weights at one rebalance date.

    ``trailing_returns`` (shape ``window x n_assets``) contains only realized
    past returns, guaranteeing no look-ahead.
    """

    as_of: np.datetime64
    assets: tuple[str, ...]
    trailing_returns: FloatArray
    current_weights: dict[str, float]


class Strategy(abc.ABC):
    """A rule producing target weights from a :class:`StrategyContext`."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def lookback(self) -> int:
        """Minimum trailing periods required before the first rebalance."""
        return 0

    @abc.abstractmethod
    def compute_weights(self, context: StrategyContext) -> dict[str, float]: ...


class StaticWeightStrategy(Strategy):
    """Hold fixed target weights, rebalancing back to them each period."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        if not weights:
            raise BacktestInputError("StaticWeightStrategy requires non-empty weights")
        self._weights = dict(weights)

    @property
    def name(self) -> str:
        return "static_weight"

    def compute_weights(self, context: StrategyContext) -> dict[str, float]:
        return dict(self._weights)


class EqualWeightStrategy(Strategy):
    """Equal-weight (1/N) the universe."""

    @property
    def name(self) -> str:
        return "equal_weight"

    def compute_weights(self, context: StrategyContext) -> dict[str, float]:
        n = len(context.assets)
        return dict.fromkeys(context.assets, 1.0 / n)


class OptimizerStrategy(Strategy):
    """Estimate moments from the trailing window and optimize.

    Parameters
    ----------
    optimizer:
        Any ``factorlab_optimizer`` optimizer (exposes ``.optimize(problem)``).
    lookback:
        Trailing periods used to estimate the mean vector and covariance.
    constraints:
        Constraints passed to the optimization problem.
    mean_estimator, cov_estimator:
        Optional overrides mapping a ``window x n_assets`` return block to a mean
        vector / covariance matrix.  Default to the sample mean and covariance.
    """

    def __init__(
        self,
        optimizer: object,
        lookback: int,
        constraints: Sequence[object] = (),
        *,
        mean_estimator: object | None = None,
        cov_estimator: object | None = None,
    ) -> None:
        if lookback < 2:
            raise BacktestInputError("OptimizerStrategy lookback must be >= 2")
        self._optimizer = optimizer
        self._lookback = lookback
        self._constraints = tuple(constraints)
        self._mean_estimator = mean_estimator
        self._cov_estimator = cov_estimator

    @property
    def name(self) -> str:
        return "optimizer"

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_weights(self, context: StrategyContext) -> dict[str, float]:
        from factorlab_optimizer import OptimizationProblem  # lazy peer import

        returns = context.trailing_returns
        if returns.shape[0] < 2:
            # Not enough data yet: fall back to equal weight.
            n = len(context.assets)
            return dict.fromkeys(context.assets, 1.0 / n)

        mu = (
            self._mean_estimator(returns)  # type: ignore[operator]
            if self._mean_estimator is not None
            else returns.mean(axis=0)
        )
        cov = (
            self._cov_estimator(returns)  # type: ignore[operator]
            if self._cov_estimator is not None
            else np.cov(returns, rowvar=False, ddof=1)
        )
        problem = OptimizationProblem(
            assets=context.assets,
            expected_returns=np.asarray(mu, dtype=np.float64),
            covariance=np.atleast_2d(np.asarray(cov, dtype=np.float64)),
            constraints=tuple(self._constraints),
        )
        result = self._optimizer.optimize(problem)  # type: ignore[attr-defined]
        return dict(result.weights.as_dict())
