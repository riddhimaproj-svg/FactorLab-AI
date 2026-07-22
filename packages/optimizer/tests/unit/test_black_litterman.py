"""Black-Litterman posterior and optimizer tests."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_optimizer import (
    BlackLittermanOptimizer,
    OptimizationProblem,
    OptimizerConfig,
    black_litterman_posterior,
)
from factorlab_optimizer.errors import OptimizationInputError


@pytest.fixture
def cov() -> np.ndarray:
    L = np.array([[0.18, 0.0, 0.0], [0.02, 0.20, 0.0], [0.01, 0.02, 0.16]])
    return L @ L.T


def test_no_views_returns_equilibrium(cov) -> None:
    w_mkt = np.array([0.4, 0.35, 0.25])
    delta = 2.5
    mu_bl, sigma_bl = black_litterman_posterior(cov, w_mkt, risk_aversion=delta, tau=0.05)
    np.testing.assert_allclose(mu_bl, delta * cov @ w_mkt)
    np.testing.assert_allclose(sigma_bl, cov)


def test_posterior_matches_manual_formula(cov) -> None:
    w_mkt = np.array([0.4, 0.35, 0.25])
    delta, tau = 2.5, 0.05
    P = np.array([[1.0, -1.0, 0.0]])  # asset A outperforms B
    Q = np.array([0.02])
    mu_bl, sigma_bl = black_litterman_posterior(
        cov, w_mkt, risk_aversion=delta, tau=tau, pick_matrix=P, view_returns=Q
    )

    pi = delta * cov @ w_mkt
    tau_sigma = tau * cov
    omega = np.diag(np.diag(P @ tau_sigma @ P.T)) + np.eye(1) * 1e-12
    ts_inv = np.linalg.inv(tau_sigma)
    om_inv = np.linalg.inv(omega)
    m = np.linalg.inv(ts_inv + P.T @ om_inv @ P)
    expected_mu = m @ (ts_inv @ pi + P.T @ om_inv @ Q)
    np.testing.assert_allclose(mu_bl, expected_mu, atol=1e-12)
    np.testing.assert_allclose(sigma_bl, cov + m, atol=1e-12)


def test_view_tilts_posterior(cov) -> None:
    w_mkt = np.array([1 / 3, 1 / 3, 1 / 3])
    pi = 2.5 * cov @ w_mkt
    # Bullish view on asset A well above its equilibrium return.
    P = np.array([[1.0, 0.0, 0.0]])
    Q = np.array([pi[0] + 0.05])
    mu_bl, _ = black_litterman_posterior(
        cov, w_mkt, risk_aversion=2.5, tau=0.05, pick_matrix=P, view_returns=Q
    )
    assert mu_bl[0] > pi[0]  # posterior tilts up toward the bullish view


def test_shape_validation(cov) -> None:
    with pytest.raises(OptimizationInputError):
        black_litterman_posterior(cov, np.array([0.5, 0.5]), risk_aversion=2.5, tau=0.05)
    with pytest.raises(OptimizationInputError):
        black_litterman_posterior(cov, np.ones(3), risk_aversion=2.5, tau=-0.1)
    with pytest.raises(OptimizationInputError):
        black_litterman_posterior(
            cov, np.ones(3), risk_aversion=2.5, tau=0.05,
            pick_matrix=np.array([[1.0, 0.0]]), view_returns=np.array([0.01]),
        )


def test_optimizer_no_views_equilibrium(cov) -> None:
    prob = OptimizationProblem(("A", "B", "C"), np.zeros(3), cov)
    w_mkt = np.array([1 / 3, 1 / 3, 1 / 3])
    result = BlackLittermanOptimizer(w_mkt, OptimizerConfig()).optimize(prob)
    assert result.optimizer == "black_litterman"
    assert result.weights.total == pytest.approx(1.0, abs=1e-6)
    assert not result.metadata["has_views"]


def test_optimizer_with_views(cov) -> None:
    prob = OptimizationProblem(("A", "B", "C"), np.zeros(3), cov)
    w_mkt = np.array([1 / 3, 1 / 3, 1 / 3])
    P = np.array([[1.0, 0.0, 0.0]])
    Q = np.array([0.15])
    opt = BlackLittermanOptimizer(
        w_mkt, OptimizerConfig(), pick_matrix=P, view_returns=Q, tau=0.05
    )
    result = opt.optimize(prob)
    assert result.metadata["has_views"]
    # bullish view on A raises its weight relative to the no-view case
    no_view = BlackLittermanOptimizer(w_mkt, OptimizerConfig()).optimize(prob)
    assert result.weights.get("A") > no_view.weights.get("A")
