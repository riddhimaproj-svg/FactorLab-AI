r"""Minimum-variance optimizer.

Solves :math:`\min_w w'\Sigma w` subject to the budget/box/other constraints.
Independent of expected returns, so it is robust to the notoriously noisy mean
estimates that plague mean-variance portfolios.  The unconstrained (budget-only)
solution is the closed form :math:`w \propto \Sigma^{-1}\mathbf{1}`.
"""

from __future__ import annotations

from factorlab_optimizer.optimizers.base import BaseOptimizer, FloatArray, Objective
from factorlab_optimizer.problem import OptimizationProblem

__all__ = ["MinVarianceOptimizer"]


class MinVarianceOptimizer(BaseOptimizer):
    @property
    def name(self) -> str:
        return "min_variance"

    def _objective(self, problem: OptimizationProblem, covariance: FloatArray) -> Objective:
        def objective(w: FloatArray) -> float:
            return float(w @ covariance @ w)

        return objective
