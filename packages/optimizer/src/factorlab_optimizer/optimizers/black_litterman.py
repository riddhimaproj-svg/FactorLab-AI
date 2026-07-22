r"""Black-Litterman optimizer.

The Black-Litterman (1992) model blends the market's *implied equilibrium*
expected returns with an investor's subjective *views*, producing a posterior set
of expected returns that is far better behaved than raw sample means -- the usual
cause of the wild, concentrated portfolios plain Markowitz produces.

Steps
-----
1. **Reverse-optimize** the market portfolio to back out the equilibrium excess
   returns implied by holding it:
   :math:`\pi = \delta\,\Sigma\,w_{\mathrm{mkt}}` (``delta`` is market risk aversion).
2. Express views as :math:`P\mu = Q + \varepsilon`, :math:`\varepsilon\sim N(0,\Omega)`
   (``P`` = pick matrix, ``Q`` = view returns, ``Omega`` = view uncertainty).
   With no ``Omega`` supplied, the He-Litterman default
   :math:`\Omega = \mathrm{diag}(P(\tau\Sigma)P')` is used.
3. **Combine** prior and views into the posterior mean

   .. math::
      \mu_{BL} = \left[(\tau\Sigma)^{-1} + P'\Omega^{-1}P\right]^{-1}
                 \left[(\tau\Sigma)^{-1}\pi + P'\Omega^{-1}Q\right],

   with posterior covariance of returns :math:`\Sigma_{BL} = \Sigma + M`,
   :math:`M = \left[(\tau\Sigma)^{-1} + P'\Omega^{-1}P\right]^{-1}`.
4. Optimize mean-variance utility with :math:`(\mu_{BL}, \Sigma_{BL})`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from factorlab_optimizer.config import OptimizerConfig
from factorlab_optimizer.errors import OptimizationInputError
from factorlab_optimizer.optimizers.base import BaseOptimizer, FloatArray, Objective
from factorlab_optimizer.optimizers.mean_variance import MeanVarianceOptimizer
from factorlab_optimizer.problem import OptimizationProblem
from factorlab_optimizer.result import OptimizationResult

__all__ = ["BlackLittermanOptimizer", "black_litterman_posterior"]


def black_litterman_posterior(
    covariance: FloatArray,
    market_weights: FloatArray,
    *,
    risk_aversion: float,
    tau: float,
    pick_matrix: NDArray[np.float64] | None = None,
    view_returns: NDArray[np.float64] | None = None,
    view_uncertainty: NDArray[np.float64] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute the Black-Litterman posterior ``(mu_BL, Sigma_BL)``.

    With no views (``pick_matrix``/``view_returns`` omitted), returns the
    equilibrium prior ``(pi, Sigma)``.
    """
    sigma = np.asarray(covariance, dtype=np.float64)
    w_mkt = np.asarray(market_weights, dtype=np.float64)
    n = sigma.shape[0]
    if w_mkt.shape != (n,):
        raise OptimizationInputError("market_weights shape does not match covariance")
    if tau <= 0.0:
        raise OptimizationInputError("tau must be positive")

    pi = risk_aversion * (sigma @ w_mkt)

    if pick_matrix is None or view_returns is None:
        return pi.copy(), sigma.copy()

    P = np.atleast_2d(np.asarray(pick_matrix, dtype=np.float64))
    Q = np.asarray(view_returns, dtype=np.float64).ravel()
    if P.shape[1] != n:
        raise OptimizationInputError("pick_matrix columns must equal number of assets")
    if Q.shape[0] != P.shape[0]:
        raise OptimizationInputError("view_returns length must equal number of views")

    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)

    if view_uncertainty is None:
        omega = np.diag(np.diag(P @ tau_sigma @ P.T))
        omega = omega + np.eye(P.shape[0]) * 1e-12  # guard singularity
    else:
        omega = np.atleast_2d(np.asarray(view_uncertainty, dtype=np.float64))
        if omega.shape != (P.shape[0], P.shape[0]):
            raise OptimizationInputError("view_uncertainty must be (n_views, n_views)")

    omega_inv = np.linalg.inv(omega)
    m_inv = tau_sigma_inv + P.T @ omega_inv @ P
    m = np.linalg.inv(m_inv)
    mu_bl = m @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    posterior_cov = sigma + m
    return mu_bl, posterior_cov


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman: blend equilibrium returns with views, then mean-variance."""

    def __init__(
        self,
        market_weights: FloatArray,
        config: OptimizerConfig | None = None,
        *,
        tau: float = 0.05,
        market_risk_aversion: float | None = None,
        pick_matrix: NDArray[np.float64] | None = None,
        view_returns: NDArray[np.float64] | None = None,
        view_uncertainty: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(config)
        self.market_weights = np.asarray(market_weights, dtype=np.float64)
        self.tau = tau
        self.market_risk_aversion = (
            market_risk_aversion if market_risk_aversion is not None else self.config.risk_aversion
        )
        self.pick_matrix = pick_matrix
        self.view_returns = view_returns
        self.view_uncertainty = view_uncertainty

    @property
    def name(self) -> str:
        return "black_litterman"

    def posterior(self, problem: OptimizationProblem) -> tuple[FloatArray, FloatArray]:
        """Return the posterior ``(mu_BL, Sigma_BL)`` for ``problem``'s covariance."""
        return black_litterman_posterior(
            problem.covariance,
            self.market_weights,
            risk_aversion=self.market_risk_aversion,
            tau=self.tau,
            pick_matrix=self.pick_matrix,
            view_returns=self.view_returns,
            view_uncertainty=self.view_uncertainty,
        )

    # Required by the ABC; BL delegates the actual objective to mean-variance.
    def _objective(self, problem: OptimizationProblem, covariance: FloatArray) -> Objective:
        gamma = self.config.risk_aversion
        mu = problem.expected_returns

        def neg_utility(w: FloatArray) -> float:
            return float(0.5 * gamma * (w @ covariance @ w) - mu @ w)

        return neg_utility

    def optimize(self, problem: OptimizationProblem) -> OptimizationResult:
        mu_bl, sigma_bl = self.posterior(problem)
        posterior_problem = OptimizationProblem(
            assets=problem.assets,
            expected_returns=mu_bl,
            covariance=sigma_bl,
            constraints=problem.constraints,
            prev_weights=problem.prev_weights,
        )
        mv = MeanVarianceOptimizer(self.config)
        result = mv.optimize(posterior_problem)
        return replace(
            result,
            optimizer=self.name,
            metadata={
                "posterior_returns": mu_bl.tolist(),
                "tau": self.tau,
                "market_risk_aversion": self.market_risk_aversion,
                "has_views": self.pick_matrix is not None,
            },
        )
