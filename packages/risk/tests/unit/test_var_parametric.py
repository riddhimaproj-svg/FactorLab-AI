"""Tests for parametric VaR / ES, cross-validated against SciPy closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from factorlab_risk.errors import RiskInputError
from factorlab_risk.var import parametric as P


def test_normal_var_closed_form() -> None:
    # standard normal, 95% -> VaR = -Phi^{-1}(0.05) = 1.6449
    assert P.parametric_var(mean=0.0, std=1.0, confidence=0.95) == pytest.approx(
        -float(stats.norm.ppf(0.05)), rel=1e-12
    )


def test_normal_es_closed_form() -> None:
    # ES_95 for standard normal = phi(z)/alpha = 2.0627
    alpha = 0.05
    z = stats.norm.ppf(alpha)
    expected = float(stats.norm.pdf(z) / alpha)
    assert P.parametric_expected_shortfall(mean=0.0, std=1.0, confidence=0.95) == pytest.approx(
        expected, rel=1e-12
    )


def test_var_scales_with_sigma_and_mean() -> None:
    base = P.parametric_var(mean=0.0, std=0.02, confidence=0.99)
    assert P.parametric_var(mean=0.0, std=0.04, confidence=0.99) == pytest.approx(2 * base)
    # positive mean reduces loss VaR
    assert P.parametric_var(mean=0.01, std=0.02, confidence=0.99) < base


def test_es_ge_var_normal() -> None:
    v = P.parametric_var(mean=0.0, std=0.02, confidence=0.95)
    es = P.parametric_expected_shortfall(mean=0.0, std=0.02, confidence=0.95)
    assert es > v


def test_student_t_fatter_tails() -> None:
    # t VaR (low dof) exceeds normal VaR at the same vol at high confidence
    normal = P.parametric_var(mean=0.0, std=0.02, confidence=0.99, distribution="normal")
    t = P.parametric_var(mean=0.0, std=0.02, confidence=0.99, distribution="t", dof=4)
    assert t > normal


def test_estimates_moments_from_returns(rng) -> None:
    r = rng.normal(0.001, 0.02, 5000)
    v = P.parametric_var(r, confidence=0.95)
    manual = -(np.mean(r) + np.std(r, ddof=1) * stats.norm.ppf(0.05))
    assert v == pytest.approx(manual, rel=1e-9)


def test_horizon_scaling() -> None:
    v1 = P.parametric_var(mean=0.0, std=0.02, confidence=0.95, horizon=1)
    v4 = P.parametric_var(mean=0.0, std=0.02, confidence=0.95, horizon=4)
    assert v4 == pytest.approx(2 * v1)  # sqrt(4) = 2, mean = 0


def test_validation() -> None:
    with pytest.raises(RiskInputError):
        P.parametric_var(mean=0.0)  # missing std
    with pytest.raises(RiskInputError):
        P.parametric_var(mean=0.0, std=0.02, distribution="cauchy")
    with pytest.raises(RiskInputError):
        P.parametric_var(mean=0.0, std=0.02, distribution="t", dof=2.0)
    with pytest.raises(RiskInputError):
        P.parametric_var(np.array([0.01]), confidence=0.95)  # < 2 obs
