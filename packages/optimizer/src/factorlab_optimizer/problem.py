"""OptimizationProblem: the immutable inputs to an optimizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from factorlab_optimizer.constraints import Constraint
from factorlab_optimizer.errors import OptimizationInputError

__all__ = ["OptimizationProblem"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    """Expected returns, covariance, assets, and constraints for an optimization.

    Immutable and self-validating.  The optimizer is agnostic to the frequency of
    the moments: expected returns, covariance, and the config's risk-free rate
    must simply share consistent units (all per-period, or all annualized).
    """

    assets: tuple[str, ...]
    expected_returns: FloatArray
    covariance: FloatArray
    constraints: tuple[Constraint, ...] = ()
    prev_weights: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mu = np.asarray(self.expected_returns, dtype=np.float64)
        cov = np.asarray(self.covariance, dtype=np.float64)
        n = len(self.assets)
        if len(set(self.assets)) != n:
            raise OptimizationInputError("duplicate asset names")
        if mu.shape != (n,):
            raise OptimizationInputError(f"expected_returns must have shape ({n},)")
        if cov.shape != (n, n):
            raise OptimizationInputError(f"covariance must have shape ({n}, {n})")
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(cov)):
            raise OptimizationInputError("expected_returns/covariance contain non-finite values")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise OptimizationInputError("covariance matrix must be symmetric")
        mu.setflags(write=False)
        cov.setflags(write=False)
        object.__setattr__(self, "expected_returns", mu)
        object.__setattr__(self, "covariance", cov)
        if self.prev_weights is not None:
            pw = np.asarray(self.prev_weights, dtype=np.float64)
            if pw.shape != (n,):
                raise OptimizationInputError(f"prev_weights must have shape ({n},)")
            pw.setflags(write=False)
            object.__setattr__(self, "prev_weights", pw)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def regularized_covariance(self, ridge: float) -> FloatArray:
        """Covariance with ``ridge`` added to the diagonal (numerical stability)."""
        if ridge <= 0.0:
            return self.covariance
        return self.covariance + ridge * np.eye(self.n_assets)

    @classmethod
    def from_moments(
        cls,
        assets: Sequence[str],
        expected_returns: Sequence[float] | FloatArray,
        covariance: FloatArray,
        constraints: Sequence[Constraint] = (),
        prev_weights: FloatArray | None = None,
    ) -> OptimizationProblem:
        return cls(
            assets=tuple(assets),
            expected_returns=np.asarray(expected_returns, dtype=np.float64),
            covariance=np.asarray(covariance, dtype=np.float64),
            constraints=tuple(constraints),
            prev_weights=prev_weights,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": list(self.assets),
            "expected_returns": self.expected_returns.tolist(),
            "covariance": self.covariance.tolist(),
            "constraints": [c.to_dict() for c in self.constraints],
            "prev_weights": None if self.prev_weights is None else self.prev_weights.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationProblem:
        pw = data.get("prev_weights")
        return cls(
            assets=tuple(data["assets"]),
            expected_returns=np.asarray(data["expected_returns"], dtype=np.float64),
            covariance=np.asarray(data["covariance"], dtype=np.float64),
            constraints=tuple(Constraint.from_dict(c) for c in data.get("constraints", [])),
            prev_weights=None if pw is None else np.asarray(pw, dtype=np.float64),
            metadata=dict(data.get("metadata", {})),
        )
