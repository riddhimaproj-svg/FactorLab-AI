"""Tests for Constraint declaration and compilation."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import Constraint, compile_constraints
from factorlab_optimizer.errors import OptimizationInputError

_ASSETS = ("A", "B", "C")


def _compile(constraints, **kw):
    defaults = {"default_lower": 0.0, "default_upper": 1.0, "default_budget": 1.0}
    defaults.update(kw)
    return compile_constraints(_ASSETS, constraints, **defaults)


def test_long_only_sets_lower_zero() -> None:
    c = _compile([Constraint.long_only()], default_lower=-1.0)
    assert all(lo == 0.0 for lo, _ in c.bounds)


def test_weight_bounds() -> None:
    c = _compile([Constraint.weight_bounds(0.1, 0.5)])
    assert all(b == (0.1, 0.5) for b in c.bounds)


def test_asset_bounds_override() -> None:
    c = _compile([Constraint.asset_bounds("B", 0.2, 0.3)])
    assert c.bounds[1] == (0.2, 0.3)


def test_asset_bounds_unknown_asset() -> None:
    with pytest.raises(OptimizationInputError):
        _compile([Constraint.asset_bounds("Z", 0.0, 1.0)])


def test_full_investment_adds_budget() -> None:
    c = _compile([Constraint.full_investment()], default_budget=None)
    assert c.has_budget
    # the equality constraint enforces sum(w) == 1
    con = c.scipy_constraints[0]
    assert con["type"] == "eq"
    assert con["fun"](np.array([0.5, 0.3, 0.2])) == pytest.approx(0.0)


def test_default_budget_applied_when_absent() -> None:
    c = _compile([])
    assert c.has_budget  # auto full-investment from default_budget=1.0


def test_no_budget_when_none_and_absent() -> None:
    c = _compile([], default_budget=None)
    assert not c.has_budget
    assert c.scipy_constraints == ()


def test_leverage_limit() -> None:
    c = _compile([Constraint.leverage_limit(1.5)], default_budget=None)
    fun = c.scipy_constraints[0]["fun"]
    assert fun(np.array([1.0, -0.4, 0.0])) == pytest.approx(1.5 - 1.4)


def test_cash_bounds() -> None:
    # cash in [0.1, 0.3] -> sum(w) in [0.7, 0.9]
    c = _compile([Constraint.cash_bounds(0.1, 0.3)], default_budget=None)
    kinds = [con["type"] for con in c.scipy_constraints]
    assert kinds == ["ineq", "ineq"]
    # sum = 0.8 satisfies both
    for con in c.scipy_constraints:
        assert con["fun"](np.array([0.4, 0.3, 0.1])) >= -1e-12


def test_sector_bounds() -> None:
    memberships = {"A": "tech", "B": "tech", "C": "energy"}
    c = _compile(
        [Constraint.sector_bounds(memberships, {"tech": (0.2, 0.6)})], default_budget=None
    )
    # two inequalities for the tech sector
    assert len(c.scipy_constraints) == 2
    w = np.array([0.25, 0.25, 0.5])  # tech = 0.5 within [0.2, 0.6]
    for con in c.scipy_constraints:
        assert con["fun"](w) >= -1e-12


def test_turnover_requires_prev_weights() -> None:
    with pytest.raises(OptimizationInputError):
        _compile([Constraint.turnover(0.2)])


def test_turnover_constraint() -> None:
    prev = np.array([0.4, 0.3, 0.3])
    c = _compile([Constraint.turnover(0.2)], prev_weights=prev, default_budget=None)
    fun = c.scipy_constraints[0]["fun"]
    w = np.array([0.45, 0.30, 0.25])  # turnover = 0.05+0+0.05 = 0.1 <= 0.2
    assert fun(w) == pytest.approx(0.2 - 0.1)


def test_infeasible_bounds() -> None:
    # weight_bounds forces every upper < 0, then long_only forces lower = 0,
    # leaving lower(0) > upper(<0) for every asset -> infeasible box.
    with pytest.raises(OptimizationInputError):
        _compile([Constraint.weight_bounds(-0.5, -0.2), Constraint.long_only()])


def test_factory_validation() -> None:
    with pytest.raises(OptimizationInputError):
        Constraint.weight_bounds(0.5, 0.1)
    with pytest.raises(OptimizationInputError):
        Constraint.leverage_limit(-1.0)
    with pytest.raises(OptimizationInputError):
        Constraint.turnover(-0.1)


def test_roundtrip() -> None:
    c = Constraint.sector_bounds({"A": "x"}, {"x": (0.1, 0.5)})
    restored = Constraint.from_dict(c.to_dict())
    assert restored.kind == c.kind
