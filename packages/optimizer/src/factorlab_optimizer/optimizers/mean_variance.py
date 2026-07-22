r"""Mean-variance (Markowitz) optimizer.

Two modes:

* **Utility** (default): maximize the mean-variance utility
  :math:`\mu'w - \tfrac{\gamma}{2} w'\Sigma w` (minimize its negative), where
  ``gamma`` is ``config.risk_aversion``.
* **Target return**: minimize variance :math:`w'\Sigma w` subject to
  :math:`\mu'w = r^\*` (pass ``target_return``).  This is the constrained
  Markowitz problem whose locus over ``r*`` traces the efficient frontier.
"""

from __future__ import annotations

from typing import Any

from factorlab_optimizer.config import OptimizerConfig
from factorlab_optimizer.optimizers.base import BaseOptimizer, FloatArray, Objective
from factorlab_optimizer.problem import OptimizationProblem

__all__ = ["MeanVarianceOptimizer"]


class MeanVarianceOptimizer(BaseOptimizer):
    """Markowitz mean-variance optimizer (utility or target-return mode)."""

    def __init__(
        self, config: OptimizerConfig | None = None, target_return: float | None = None
    ) -> None:
        super().__init__(config)
        self.target_return = target_return

    @property
    def name(self) -> str:
        return "mean_variance"

    def _objective(self, problem: OptimizationProblem, covariance: FloatArray) -> Objective:
        mu = problem.expected_returns
        if self.target_return is not None:
            # Minimize variance; the target return is imposed as a constraint.
            def variance(w: FloatArray) -> float:
                return float(w @ covariance @ w)

            return variance

        gamma = self.config.risk_aversion

        def neg_utility(w: FloatArray) -> float:
            return float(0.5 * gamma * (w @ covariance @ w) - mu @ w)

        return neg_utility

    def _extra_constraints(
        self, problem: OptimizationProblem, covariance: FloatArray
    ) -> list[dict[str, Any]]:
        if self.target_return is None:
            return []
        mu = problem.expected_returns
        target = self.target_return
        return [{"type": "eq", "fun": lambda w, m=mu, t=target: float(m @ w - t)}]
