# factorlab-optimizer

**Portfolio optimization engine.** Pure, typed, SciPy-backed. Turns expected returns and a
covariance matrix into optimal portfolio weights under realistic constraints.

Independent of the frontend and of the rest of the platform; its output weights feed the
portfolio and backtesting layers in the standard workflow:

```
Factor Model → Expected Returns → Optimizer → Portfolio → Backtester → Performance Report
```

---

## Architecture

```
        OptimizationProblem                 OptimizerConfig
        (mu, Sigma, assets, constraints)    (rf, gamma, bounds, budget, ridge)
                    |                                |
                    v                                v
        Constraint[]  --compile_constraints-->  bounds + SciPy constraint dicts
                    |                                |
                    +----------------+---------------+
                                     v
                             BaseOptimizer  (SLSQP driver)
             +---------------+-----------+-----------+----------------+
             v               v           v           v                v
      MinVariance     MeanVariance   MaxSharpe   MaxDiversification  RiskParity
             +---------------+-----------+-----------+----------------+
                                         |                    BlackLitterman
                                         v                (posterior -> mean-variance)
                                OptimizationResult
                          (PortfolioWeights, return, vol, Sharpe)
                                         |
              +--------------------------+---------------------------+
              v                          v                           v
      EfficientFrontier         CapitalAllocationLine        risk.* decomposition
      (frontier points)         (rf -> tangency line)   (marginal / risk contribution,
                                                        variance decomposition, DR)
```

Every model object (`OptimizationProblem`, `OptimizerConfig`, `Constraint`, `PortfolioWeights`,
`OptimizationResult`) is an **immutable, serializable** frozen dataclass. Constraint declaration
is separated from compilation, so a constraint set can be inspected, serialized, and reused.

---

## Modern Portfolio Theory

Markowitz (1952) casts portfolio choice as a trade-off between expected return
`mu'w` and variance `w'Sigma w`. For a given risk aversion `gamma`, the investor maximizes the
**mean-variance utility** `mu'w - (gamma/2) w'Sigma w`. Sweeping the risk/return trade-off traces
the **efficient frontier**: the set of portfolios with minimum variance for each attainable level
of expected return. Key results the package implements and verifies:

- **Global minimum-variance portfolio**: `w ~ Sigma^-1 1` (budget-only, closed form).
- **Tangency (maximum-Sharpe) portfolio**: `w ~ Sigma^-1 (mu - rf*1)` — the risky portfolio with
  the highest Sharpe ratio.
- **Capital allocation line (CAL)**: mixing the risk-free asset with the tangency portfolio
  produces a straight line in (sigma, return) space whose slope is the tangency Sharpe ratio.
  With a risk-free asset the CAL dominates the risky frontier (two-fund separation).

## The six optimizers

| Optimizer | Objective | Notes |
|-----------|-----------|-------|
| **Mean-Variance** | max `mu'w - (gamma/2)w'Sigma w`, or min variance s.t. `mu'w = r*` | Markowitz; target-return mode traces the frontier |
| **Minimum Variance** | min `w'Sigma w` | ignores mu -> robust to noisy means |
| **Maximum Sharpe** | max `(mu'w - rf)/sqrt(w'Sigma w)` | tangency portfolio |
| **Maximum Diversification** | max `(w'sigma)/sqrt(w'Sigma w)` | Choueifaty-Coignard diversification ratio |
| **Risk Parity** | equalize risk contributions `w_i(Sigma w)_i` | see below |
| **Black-Litterman** | posterior returns -> mean-variance | see below |

### Risk parity

Risk parity allocates so that **each asset contributes equally to portfolio risk**, rather than
equal capital (1/N) or minimum variance. By Euler's theorem the volatility decomposes exactly
into per-asset risk contributions `RC_i = w_i (Sigma w)_i / sigma_p` that sum to `sigma_p`; risk
parity solves for `RC_i = sigma_p/n` for all `i`. The result is a portfolio not dominated by the
most volatile or most correlated holdings. The full decomposition (marginal contribution, risk
contribution, percent contribution, variance decomposition, diversification ratio) lives in
`factorlab_optimizer.risk`.

### Black-Litterman

Plain Markowitz is notoriously sensitive to expected-return estimates: tiny changes in `mu`
produce wildly different, concentrated portfolios. Black-Litterman (1992) fixes this by starting
from a sensible prior and blending in views:

1. **Reverse-optimize** the market portfolio to recover the *implied equilibrium* excess
   returns `pi = delta * Sigma * w_mkt`.
2. Express subjective **views** as `P mu = Q + eps`, `eps ~ N(0, Omega)` (`P` = pick matrix,
   `Q` = view returns, `Omega` = view uncertainty; default `Omega = diag(P(tau Sigma)P')`).
3. Combine prior and views into the **posterior** mean

   `mu_BL = [(tau Sigma)^-1 + P'Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P'Omega^-1 Q]`,
   posterior covariance `Sigma_BL = Sigma + M`.

4. Optimize mean-variance utility with `(mu_BL, Sigma_BL)`.

With no views the posterior collapses to the equilibrium `pi`, so BL degrades gracefully to
holding the market. Views tilt the posterior — and hence the weights — toward the expressed
opinion in proportion to its confidence.

---

## Constraints

Declarative, composable, and compiled to SciPy bounds/constraints:

- **long-only** / **long-short** (via `allow_short` or `weight_bounds`)
- per-asset and global **weight bounds**
- **full investment** / **budget** (`sum(w) = target`)
- **cash allocation** (`cash = 1 - sum(w)` within a band)
- **leverage limit** (`sum|w| <= L`)
- **sector bounds** (group weight within a band)
- **turnover** (`sum|w - w_prev| <= tau`, relative to a previous portfolio)

```python
from factorlab_optimizer import Constraint
constraints = (
    Constraint.long_only(),
    Constraint.weight_bounds(0.0, 0.25),
    Constraint.sector_bounds({"AAPL": "tech", "XOM": "energy"}, {"tech": (0.0, 0.4)}),
    Constraint.turnover(0.20),
)
```

---

## Quick start

```python
import numpy as np
from factorlab_optimizer import (
    OptimizationProblem, OptimizerConfig, Constraint,
    MinVarianceOptimizer, MaxSharpeOptimizer, EfficientFrontier, risk,
)

mu = np.array([0.08, 0.12, 0.10])
cov = np.array([[0.040, 0.006, 0.004],
                [0.006, 0.090, 0.002],
                [0.004, 0.002, 0.070]])
problem = OptimizationProblem(("A", "B", "C"), mu, cov,
                              constraints=(Constraint.long_only(),))

mv = MinVarianceOptimizer().optimize(problem)
print(mv.summary())

ms = MaxSharpeOptimizer(OptimizerConfig(risk_free_rate=0.02)).optimize(problem)
print(risk.percent_risk_contributions(ms.weights.values, cov))

ef = EfficientFrontier(problem, OptimizerConfig(risk_free_rate=0.02))
frontier = ef.compute(n_points=25)
cal = ef.capital_allocation_line()      # slope == tangency Sharpe
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=factorlab_optimizer     # 99% coverage
ruff check src tests ; mypy src
```

Min-variance and tangency solutions are cross-validated against their **analytic closed forms**
and against an **independent SciPy solve**; the Black-Litterman posterior is checked against the
hand-computed matrix formula; risk parity is verified to equalize risk contributions; the
efficient frontier is verified monotonic with the CAL slope equal to the tangency Sharpe ratio.

---

## References

- Markowitz, H. (1952). *Portfolio Selection.* Journal of Finance.
- Black, F., & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- Choueifaty, Y., & Coignard, Y. (2008). *Toward Maximum Diversification.* J. Portfolio Management.
- Maillard, Roncalli & Teiletche (2010). *Properties of Equally-Weighted Risk Contribution Portfolios.*
