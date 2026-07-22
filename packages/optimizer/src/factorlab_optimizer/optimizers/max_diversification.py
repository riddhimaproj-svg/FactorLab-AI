r"""Maximum-diversification optimizer.

Maximizes the **diversification ratio**
:math:`\mathrm{DR}(w) = (w'\sigma)/\sqrt{w'\Sigma w}`, the weighted-average asset
volatility divided by the portfolio volatility (Choueifaty & Coignard, 2008).
Portfolios that maximize DR spread risk across weakly-correlated bets rather than
concentrating it, which is why the objective favors diversification without
reference to expected returns.
"""

from __future__ import annotations

import numpy as np

from factorlab_optimizer.optimizers.base import BaseOptimizer, FloatArray, Objective
from factorlab_optimizer.problem import OptimizationProblem

__all__ = ["MaxDiversificationOptimizer"]


class MaxDiversificationOptimizer(BaseOptimizer):
    @property
    def name(self) -> str:
        return "max_diversification"

    def _objective(self, problem: OptimizationProblem, covariance: FloatArray) -> Objective:
        asset_vols = np.sqrt(np.diag(covariance))

        def neg_diversification(w: FloatArray) -> float:
            weighted_vol = float(w @ asset_vols)
            portfolio_vol = float(np.sqrt(max(float(w @ covariance @ w), 1e-18)))
            return -weighted_vol / portfolio_vol

        return neg_diversification
