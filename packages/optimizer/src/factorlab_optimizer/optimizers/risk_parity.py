r"""Risk-parity (equal risk contribution) optimizer.

Finds weights whose **risk contributions are equal** across assets:
:math:`\mathrm{RC}_i = w_i (\Sigma w)_i = \tfrac{1}{n} w'\Sigma w` for all ``i``.
Rather than equalizing capital (as 1/N does) or minimizing variance, risk parity
equalizes each asset's share of portfolio risk, producing a portfolio that is not
dominated by the most volatile or most correlated holdings.

The objective minimizes the dispersion of risk contributions,
:math:`\sum_i (\mathrm{RC}_i - \overline{\mathrm{RC}})^2`.  Risk parity is defined
for a long-only, fully-invested portfolio, so pair it with ``long_only`` bounds.
"""

from __future__ import annotations

import numpy as np

from factorlab_optimizer.optimizers.base import BaseOptimizer, FloatArray, Objective
from factorlab_optimizer.problem import OptimizationProblem

__all__ = ["RiskParityOptimizer"]


class RiskParityOptimizer(BaseOptimizer):
    @property
    def name(self) -> str:
        return "risk_parity"

    def _objective(self, problem: OptimizationProblem, covariance: FloatArray) -> Objective:
        def dispersion(w: FloatArray) -> float:
            marginal = covariance @ w
            contributions = w * marginal  # RC_i = w_i (Sigma w)_i
            target = np.mean(contributions)
            return float(np.sum((contributions - target) ** 2))

        return dispersion
