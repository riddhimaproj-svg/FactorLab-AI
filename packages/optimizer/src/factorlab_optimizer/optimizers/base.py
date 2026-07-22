"""BaseOptimizer: the shared SLSQP driver.

Concrete optimizers supply only an objective function (and, optionally, extra
constraints or a custom starting point).  Everything else -- constraint
compilation, bounds, the solve, and result assembly -- lives here, so a new
optimizer is a few lines.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from factorlab_optimizer.config import OptimizerConfig
from factorlab_optimizer.constraints import CompiledConstraints, compile_constraints
from factorlab_optimizer.errors import OptimizationFailedError
from factorlab_optimizer.problem import OptimizationProblem
from factorlab_optimizer.result import OptimizationResult
from factorlab_optimizer.weights import PortfolioWeights

__all__ = ["BaseOptimizer"]

FloatArray = NDArray[np.float64]
Objective = Callable[[FloatArray], float]


class BaseOptimizer(abc.ABC):
    """Abstract base for weight optimizers."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config if config is not None else OptimizerConfig()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short optimizer identifier (e.g. ``"min_variance"``)."""

    # ------------------------------------------------------------------ #
    # Hooks for subclasses                                                #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _objective(self, problem: OptimizationProblem, covariance: FloatArray) -> Objective:
        """Return the scalar objective to *minimize*."""

    def _extra_constraints(
        self, problem: OptimizationProblem, covariance: FloatArray
    ) -> list[dict[str, Any]]:
        """Optimizer-specific SciPy constraints (e.g. a target-return equality)."""
        return []

    def _initial_guess(
        self, problem: OptimizationProblem, compiled: CompiledConstraints
    ) -> FloatArray:
        n = problem.n_assets
        lower = np.array([b[0] for b in compiled.bounds])
        upper = np.array([b[1] for b in compiled.bounds])
        x0 = np.clip(np.full(n, 1.0 / n), lower, upper)
        budget = self.config.budget
        if budget is not None and np.sum(x0) > 0:
            x0 = x0 * (budget / np.sum(x0))
            x0 = np.clip(x0, lower, upper)
        return np.asarray(x0, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Solve                                                               #
    # ------------------------------------------------------------------ #
    def optimize(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve the problem and return an :class:`OptimizationResult`.

        Raises :class:`OptimizationFailedError` if the solver does not converge.
        """
        covariance = problem.regularized_covariance(self.config.covariance_regularization)
        lower, upper = self.config.default_bounds()
        compiled = compile_constraints(
            problem.assets,
            problem.constraints,
            default_lower=lower,
            default_upper=upper,
            default_budget=self.config.budget,
            prev_weights=problem.prev_weights,
        )
        constraints: Sequence[dict[str, Any]] = [
            *compiled.scipy_constraints,
            *self._extra_constraints(problem, covariance),
        ]
        objective = self._objective(problem, covariance)
        x0 = self._initial_guess(problem, compiled)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=compiled.bounds,
            constraints=constraints,
            options={"maxiter": self.config.max_iterations, "ftol": self.config.tolerance},
        )
        if not result.success:
            raise OptimizationFailedError(str(result.message), optimizer=self.name)

        weights = PortfolioWeights(problem.assets, np.asarray(result.x, dtype=np.float64))
        return OptimizationResult.build(
            weights=weights,
            optimizer=self.name,
            success=bool(result.success),
            message=str(result.message),
            expected_returns=problem.expected_returns,
            covariance=covariance,
            risk_free_rate=self.config.risk_free_rate,
            objective_value=float(result.fun),
            n_iterations=int(getattr(result, "nit", 0)),
        )
