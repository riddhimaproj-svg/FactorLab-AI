"""Portfolio constraints and their compilation to a solver representation.

A :class:`Constraint` is a declarative, immutable description of a restriction on
the weight vector.  :func:`compile_constraints` turns a list of them (plus the
config defaults) into the concrete ``(bounds, scipy_constraints)`` a SciPy SLSQP
solve consumes.  Keeping the declaration separate from the compilation lets the
same constraint set be inspected, serialized, and reused across optimizers.

Supported restrictions
----------------------
* per-asset weight **bounds** (global or asset-specific) -> box bounds
* **long-only** / **long-short** -> box bounds
* **full investment** / **budget** (sum of weights = target) -> equality
* **cash allocation** (cash = 1 - sum(w) within a band) -> inequalities
* **leverage limit** (sum |w| <= L) -> inequality
* **sector** bounds (group weight within a band) -> inequalities
* **turnover** (sum |w - w_prev| <= tau) -> inequality
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from factorlab_optimizer.errors import OptimizationInputError

__all__ = ["CompiledConstraints", "Constraint", "ConstraintKind", "compile_constraints"]

FloatArray = NDArray[np.float64]


class ConstraintKind:
    """String tags for the supported constraint kinds."""

    LONG_ONLY = "long_only"
    WEIGHT_BOUNDS = "weight_bounds"
    ASSET_BOUNDS = "asset_bounds"
    FULL_INVESTMENT = "full_investment"
    BUDGET = "budget"
    CASH_BOUNDS = "cash_bounds"
    LEVERAGE_LIMIT = "leverage_limit"
    SECTOR_BOUNDS = "sector_bounds"
    TURNOVER = "turnover"


@dataclass(frozen=True, slots=True)
class Constraint:
    """A declarative portfolio constraint (build via the factory classmethods)."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    # -- Factories --------------------------------------------------------
    @classmethod
    def long_only(cls) -> Constraint:
        return cls(ConstraintKind.LONG_ONLY)

    @classmethod
    def weight_bounds(cls, lower: float, upper: float) -> Constraint:
        if lower > upper:
            raise OptimizationInputError("weight_bounds: lower > upper")
        return cls(ConstraintKind.WEIGHT_BOUNDS, {"lower": lower, "upper": upper})

    @classmethod
    def asset_bounds(cls, asset: str, lower: float, upper: float) -> Constraint:
        if lower > upper:
            raise OptimizationInputError("asset_bounds: lower > upper")
        return cls(ConstraintKind.ASSET_BOUNDS, {"asset": asset, "lower": lower, "upper": upper})

    @classmethod
    def full_investment(cls) -> Constraint:
        return cls(ConstraintKind.FULL_INVESTMENT)

    @classmethod
    def budget(cls, target: float) -> Constraint:
        return cls(ConstraintKind.BUDGET, {"target": target})

    @classmethod
    def cash_bounds(cls, lower: float, upper: float) -> Constraint:
        if lower > upper:
            raise OptimizationInputError("cash_bounds: lower > upper")
        return cls(ConstraintKind.CASH_BOUNDS, {"lower": lower, "upper": upper})

    @classmethod
    def leverage_limit(cls, max_leverage: float) -> Constraint:
        if max_leverage <= 0:
            raise OptimizationInputError("leverage_limit must be positive")
        return cls(ConstraintKind.LEVERAGE_LIMIT, {"max_leverage": max_leverage})

    @classmethod
    def sector_bounds(
        cls,
        memberships: Mapping[str, str],
        bounds: Mapping[str, tuple[float, float]],
    ) -> Constraint:
        return cls(
            ConstraintKind.SECTOR_BOUNDS,
            {"memberships": dict(memberships), "bounds": {k: tuple(v) for k, v in bounds.items()}},
        )

    @classmethod
    def turnover(cls, max_turnover: float) -> Constraint:
        if max_turnover < 0:
            raise OptimizationInputError("turnover must be non-negative")
        return cls(ConstraintKind.TURNOVER, {"max_turnover": max_turnover})

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Constraint:
        return cls(str(data["kind"]), dict(data.get("params", {})))


@dataclass(frozen=True, slots=True)
class CompiledConstraints:
    """Solver-ready constraints: box bounds plus SciPy constraint dicts."""

    bounds: tuple[tuple[float, float], ...]
    scipy_constraints: tuple[dict[str, Any], ...]
    has_budget: bool


def compile_constraints(
    assets: Sequence[str],
    constraints: Sequence[Constraint],
    *,
    default_lower: float,
    default_upper: float,
    default_budget: float | None,
    prev_weights: FloatArray | None = None,
) -> CompiledConstraints:
    """Compile declarative constraints into SciPy bounds and constraint dicts."""
    n = len(assets)
    index = {a: i for i, a in enumerate(assets)}
    lower = np.full(n, default_lower, dtype=np.float64)
    upper = np.full(n, default_upper, dtype=np.float64)
    scipy_cons: list[dict[str, Any]] = []
    has_sum_constraint = False

    for c in constraints:
        if c.kind == ConstraintKind.LONG_ONLY:
            lower = np.maximum(lower, 0.0)
        elif c.kind == ConstraintKind.WEIGHT_BOUNDS:
            lower = np.full(n, c.params["lower"])
            upper = np.full(n, c.params["upper"])
        elif c.kind == ConstraintKind.ASSET_BOUNDS:
            asset = c.params["asset"]
            if asset not in index:
                raise OptimizationInputError(f"asset_bounds references unknown asset {asset!r}")
            i = index[asset]
            lower[i] = c.params["lower"]
            upper[i] = c.params["upper"]
        elif c.kind in (ConstraintKind.FULL_INVESTMENT, ConstraintKind.BUDGET):
            target = 1.0 if c.kind == ConstraintKind.FULL_INVESTMENT else c.params["target"]
            scipy_cons.append(_sum_equals(target))
            has_sum_constraint = True
        elif c.kind == ConstraintKind.CASH_BOUNDS:
            # cash = 1 - sum(w) in [lo, hi]  <=>  sum(w) in [1-hi, 1-lo]
            lo, hi = c.params["lower"], c.params["upper"]
            scipy_cons.append(_sum_geq(1.0 - hi))
            scipy_cons.append(_sum_leq(1.0 - lo))
            has_sum_constraint = True
        elif c.kind == ConstraintKind.LEVERAGE_LIMIT:
            scipy_cons.append(_leverage_leq(c.params["max_leverage"]))
        elif c.kind == ConstraintKind.SECTOR_BOUNDS:
            scipy_cons.extend(
                _sector_constraints(assets, c.params["memberships"], c.params["bounds"])
            )
        elif c.kind == ConstraintKind.TURNOVER:
            if prev_weights is None:
                raise OptimizationInputError("turnover constraint requires prev_weights")
            scipy_cons.append(_turnover_leq(c.params["max_turnover"], np.asarray(prev_weights)))
        else:  # pragma: no cover - guarded by factories
            raise OptimizationInputError(f"unknown constraint kind {c.kind!r}")

    if np.any(lower > upper):
        raise OptimizationInputError("infeasible box bounds (lower > upper for some asset)")

    if not has_sum_constraint and default_budget is not None:
        scipy_cons.append(_sum_equals(default_budget))
        has_sum_constraint = True

    bounds = tuple((float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True))
    return CompiledConstraints(bounds, tuple(scipy_cons), has_sum_constraint)


# --------------------------------------------------------------------------- #
# SciPy constraint builders                                                   #
# --------------------------------------------------------------------------- #
def _sum_equals(target: float) -> dict[str, Any]:
    return {"type": "eq", "fun": lambda w, t=target: float(np.sum(w) - t)}


def _sum_geq(bound: float) -> dict[str, Any]:
    return {"type": "ineq", "fun": lambda w, b=bound: float(np.sum(w) - b)}


def _sum_leq(bound: float) -> dict[str, Any]:
    return {"type": "ineq", "fun": lambda w, b=bound: float(b - np.sum(w))}


def _leverage_leq(max_leverage: float) -> dict[str, Any]:
    return {"type": "ineq", "fun": lambda w, L=max_leverage: float(L - np.sum(np.abs(w)))}


def _turnover_leq(max_turnover: float, prev: FloatArray) -> dict[str, Any]:
    return {
        "type": "ineq",
        "fun": lambda w, tau=max_turnover, p=prev: float(tau - np.sum(np.abs(w - p))),
    }


def _sector_constraints(
    assets: Sequence[str],
    memberships: Mapping[str, str],
    bounds: Mapping[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    cons: list[dict[str, Any]] = []
    for sector, (lo, hi) in bounds.items():
        mask = np.array([1.0 if memberships.get(a) == sector else 0.0 for a in assets])
        cons.append({"type": "ineq", "fun": lambda w, m=mask, b=lo: float(np.dot(m, w) - b)})
        cons.append({"type": "ineq", "fun": lambda w, m=mask, b=hi: float(b - np.dot(m, w))})
    return cons
