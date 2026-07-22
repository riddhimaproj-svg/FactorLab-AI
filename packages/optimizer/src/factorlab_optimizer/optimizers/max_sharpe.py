r"""Maximum-Sharpe (tangency) optimizer.

Maximizes the ex-ante Sharpe ratio :math:`(\mu'w - r_f)/\sqrt{w'\Sigma w}`
(minimizing its negative).  The solution is the **tangency portfolio**: the point
where the capital allocation line from the risk-free asset is tangent to the
efficient frontier.  The unconstrained, budget-only closed form is
:math:`w \propto \Sigma^{-1}(\mu - r_f\mathbf{1})`.
"""

from __future__ import annotations

import numpy as np

from factorlab_optimizer.optimizers.base import BaseOptimizer, FloatArray, Objective
from factorlab_optimizer.problem import OptimizationProblem

__all__ = ["MaxSharpeOptimizer"]


class MaxSharpeOptimizer(BaseOptimizer):
    @property
    def name(self) -> str:
        return "max_sharpe"

    def _objective(self, problem: OptimizationProblem, covariance: FloatArray) -> Objective:
        mu = problem.expected_returns
        rf = self.config.risk_free_rate

        def neg_sharpe(w: FloatArray) -> float:
            excess = float(mu @ w - rf)
            vol = float(np.sqrt(max(float(w @ covariance @ w), 1e-18)))
            return -excess / vol

        return neg_sharpe
