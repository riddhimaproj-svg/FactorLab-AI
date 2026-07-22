r"""Efficient frontier and capital allocation line.

The **efficient frontier** is the set of portfolios that minimize variance for
each level of expected return -- the upper boundary of what is achievable in
mean-variance space (Markowitz, 1952).  It is traced by solving

.. math:: \min_w w'\Sigma w \quad \text{s.t.}\quad \mu'w = r^\*,

for a grid of target returns :math:`r^\*` between the global minimum-variance
return and the maximum attainable return.

The **capital allocation line (CAL)** is the straight line from the risk-free
asset through the tangency (maximum-Sharpe) portfolio.  Combining the risk-free
asset with the tangency portfolio dominates every portfolio on the risky
frontier, so the CAL is the efficient frontier *with* a risk-free asset; its
slope is the tangency Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from factorlab_optimizer.config import OptimizerConfig
from factorlab_optimizer.errors import OptimizationFailedError
from factorlab_optimizer.optimizers.max_sharpe import MaxSharpeOptimizer
from factorlab_optimizer.optimizers.mean_variance import MeanVarianceOptimizer
from factorlab_optimizer.optimizers.min_variance import MinVarianceOptimizer
from factorlab_optimizer.problem import OptimizationProblem
from factorlab_optimizer.result import OptimizationResult
from factorlab_optimizer.weights import PortfolioWeights

__all__ = ["CapitalAllocationLine", "EfficientFrontier", "FrontierPoint"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One point on the efficient frontier."""

    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: PortfolioWeights


@dataclass(frozen=True, slots=True)
class CapitalAllocationLine:
    """The line ``return = rf + slope * volatility`` through the tangency portfolio."""

    risk_free_rate: float
    tangency_return: float
    tangency_volatility: float

    @property
    def slope(self) -> float:
        """The CAL slope = the tangency portfolio's Sharpe ratio."""
        if self.tangency_volatility == 0.0:
            return float("nan")
        return (self.tangency_return - self.risk_free_rate) / self.tangency_volatility

    def expected_return_at(self, volatility: float) -> float:
        """Expected return of the rf+tangency mix at a given total volatility."""
        return float(self.risk_free_rate + self.slope * volatility)

    def points(self, volatilities: FloatArray) -> FloatArray:
        vols = np.asarray(volatilities, dtype=np.float64)
        return self.risk_free_rate + self.slope * vols


class EfficientFrontier:
    """Computes the efficient frontier and CAL for an optimization problem."""

    def __init__(self, problem: OptimizationProblem, config: OptimizerConfig | None = None) -> None:
        self.problem = problem
        self.config = config if config is not None else OptimizerConfig()

    def min_variance_portfolio(self) -> OptimizationResult:
        return MinVarianceOptimizer(self.config).optimize(self.problem)

    def max_sharpe_portfolio(self) -> OptimizationResult:
        """The tangency portfolio (maximum Sharpe ratio)."""
        return MaxSharpeOptimizer(self.config).optimize(self.problem)

    def compute(self, n_points: int = 20) -> tuple[FrontierPoint, ...]:
        """Trace the frontier with ``n_points`` target-return solves.

        Target returns span the global-minimum-variance return up to the maximum
        attainable expected return.  Infeasible targets are skipped, so the
        returned tuple may contain fewer than ``n_points`` points.
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2")

        r_min = self.min_variance_portfolio().expected_return
        r_max = float(np.max(self.problem.expected_returns))
        if r_max <= r_min:
            r_max = r_min + abs(r_min) * 0.5 + 1e-6

        points: list[FrontierPoint] = []
        for target in np.linspace(r_min, r_max, n_points):
            try:
                result = MeanVarianceOptimizer(self.config, target_return=float(target)).optimize(
                    self.problem
                )
            except OptimizationFailedError:
                continue
            points.append(
                FrontierPoint(
                    expected_return=result.expected_return,
                    volatility=result.expected_volatility,
                    sharpe_ratio=result.sharpe_ratio,
                    weights=result.weights,
                )
            )
        return tuple(points)

    def capital_allocation_line(self) -> CapitalAllocationLine:
        """The CAL through the tangency portfolio."""
        tangency = self.max_sharpe_portfolio()
        return CapitalAllocationLine(
            risk_free_rate=self.config.risk_free_rate,
            tangency_return=tangency.expected_return,
            tangency_volatility=tangency.expected_volatility,
        )
