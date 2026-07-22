"""FactorLab optimizer — portfolio optimization engine.

A pure, typed optimization library built on SciPy's SLSQP solver.  It turns a set
of expected returns and a covariance matrix (an :class:`OptimizationProblem`)
plus declarative :class:`Constraint`\\ s into optimal :class:`PortfolioWeights`,
via six optimizers:

* :class:`MeanVarianceOptimizer` -- Markowitz utility / target-return
* :class:`MinVarianceOptimizer` -- global minimum variance
* :class:`MaxSharpeOptimizer` -- tangency portfolio
* :class:`MaxDiversificationOptimizer` -- maximum diversification ratio
* :class:`RiskParityOptimizer` -- equal risk contribution
* :class:`BlackLittermanOptimizer` -- equilibrium returns blended with views

Plus the :class:`EfficientFrontier` (and capital allocation line) and a full risk
decomposition (:mod:`factorlab_optimizer.risk`).

It is independent of the rest of the platform; the optimizer's output weights
feed the portfolio/backtesting layers in the standard workflow
(factor model -> expected returns -> optimizer -> portfolio -> backtester).

Quick start
-----------
>>> import numpy as np
>>> from factorlab_optimizer import OptimizationProblem, MinVarianceOptimizer, Constraint
>>> cov = np.array([[0.04, 0.006], [0.006, 0.09]])
>>> problem = OptimizationProblem(("A", "B"), np.array([0.08, 0.12]), cov,
...                               constraints=(Constraint.long_only(),))
>>> result = MinVarianceOptimizer().optimize(problem)
>>> round(result.weights.total, 6)
1.0
"""

from __future__ import annotations

from factorlab_optimizer import risk
from factorlab_optimizer.config import OptimizerConfig
from factorlab_optimizer.constraints import Constraint, ConstraintKind, compile_constraints
from factorlab_optimizer.errors import (
    InfeasibleProblemError,
    OptimizationFailedError,
    OptimizationInputError,
    OptimizerError,
)
from factorlab_optimizer.frontier import (
    CapitalAllocationLine,
    EfficientFrontier,
    FrontierPoint,
)
from factorlab_optimizer.optimizers import (
    BaseOptimizer,
    BlackLittermanOptimizer,
    MaxDiversificationOptimizer,
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    black_litterman_posterior,
)
from factorlab_optimizer.problem import OptimizationProblem
from factorlab_optimizer.result import OptimizationResult
from factorlab_optimizer.weights import PortfolioWeights

__version__ = "0.1.0"

__all__ = [
    "BaseOptimizer",
    "BlackLittermanOptimizer",
    "CapitalAllocationLine",
    "Constraint",
    "ConstraintKind",
    "EfficientFrontier",
    "FrontierPoint",
    "InfeasibleProblemError",
    "MaxDiversificationOptimizer",
    "MaxSharpeOptimizer",
    "MeanVarianceOptimizer",
    "MinVarianceOptimizer",
    "OptimizationFailedError",
    "OptimizationInputError",
    "OptimizationProblem",
    "OptimizationResult",
    "OptimizerConfig",
    "OptimizerError",
    "PortfolioWeights",
    "RiskParityOptimizer",
    "__version__",
    "black_litterman_posterior",
    "compile_constraints",
    "risk",
]
