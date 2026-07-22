# factorlab-risk

**Institutional portfolio & market risk engine.** Pure, typed, dependency-light (numpy + scipy).
Value-at-Risk and Expected Shortfall, risk attribution, portfolio risk statistics, stress
testing, and scenario analysis — the risk stage of the platform workflow:

```
Market Data -> Factor Model -> Expected Returns -> Optimizer -> Backtester -> Risk Engine -> Risk Report
```

---

## Architecture

Hexagonal, modular, immutable public data models. The pure computational core depends only on
numpy/scipy; a thin **integration** layer bridges (via duck typing) to the approved
portfolio / optimizer / backtesting packages without importing them.

```
        returns / weights / covariance / exposures  (plain numpy)
                              |
   +----------------+---------+-----------+-------------------+--------------------+
   v                v                     v                   v                    v
  var/          attribution        portfolio_risk         scenario              stress
  historical    MCR / CCR / %       volatility, TE,        Scenario,            crash / rate /
  parametric    risk budgeting      beta, corr/cov,        ScenarioEngine,      factor / sector
  monte_carlo   factor & sector     rolling *, div.        revalue / compare /  shocks, vol shock,
  rolling       attribution         ratio, HHI             sensitivity          run_stress_test
  decomposition
   |                |                     |                   |                    |
   +----------------+----------+----------+-------------------+--------------------+
                               v
                   reports (immutable, serializable)
       RiskReport · VaRReport · RiskDecomposition · RiskContribution
       RiskSnapshot · StressTestReport · ScenarioReport
                               ^
                               |  integration (duck-typed adapters)
        factorlab_portfolio.ReturnSeries · factorlab_optimizer.OptimizationResult
        factorlab_backtesting.BacktestResult
```

**Conventions.** Returns are per-period *simple* returns (decimal). Confidence `c` in `(0, 1)`
with tail `alpha = 1 - c`. **VaR and ES are positive loss magnitudes** (fractions of value): a
95% VaR of 0.03 means "a 5% chance of losing more than 3%". Horizon scaling uses the
square-root-of-time rule.

---

## Value-at-Risk methodologies

VaR estimates the loss that is exceeded only with probability `alpha` over a horizon. Three
complementary estimators, each with trade-offs:

- **Historical** (`var.historical`) — reads VaR straight off the empirical distribution
  (`-Q_alpha(r)`). No distributional assumption, captures real fat tails and skew; limited to
  losses already seen in the window.
- **Parametric / variance-covariance** (`var.parametric`) — closed-form from a fitted Normal
  (`-(mu + sigma·z_alpha)`) or standardized Student-t (fatter tails). Fast and smooth, but a
  Normal model *understates* tail risk when returns are fat-tailed.
- **Monte Carlo** (`var.monte_carlo`) — simulate a large sample from a fitted (uni- or
  multivariate) distribution, then estimate empirically. Flexible; sampling error shrinks as
  `1/sqrt(N)`. Every routine takes a `seed` for reproducibility.
  *(This is Monte Carlo risk estimation from a fitted return distribution — not path simulation
  or derivative pricing.)*

**Rolling** VaR/ES (`var.rolling`) track how tail risk evolves through time.

## Expected Shortfall (CVaR)

Expected Shortfall is the **mean loss conditional on breaching VaR**:
`ES_c = -E[r | r <= Q_alpha(r)]`. Unlike VaR it is *coherent* (sub-additive) and describes the
severity of the tail, not just its threshold — always `>= VaR`. Historical, parametric
(Normal and t, with the closed-form `sigma·phi(z)/alpha`), and Monte Carlo variants are provided,
plus `tail_loss` and `worst_loss`.

## Portfolio VaR decomposition

For a Normal model, portfolio VaR `= z_c·sqrt(w'Sigma w)·sqrt(h)` decomposes exactly (Euler):

- **Marginal VaR** `= z_c·sqrt(h)·(Sigma w)_i / sigma_p` — risk of a marginal unit of asset `i`.
- **Component VaR** `= w_i·MVaR_i` — each asset's share, **summing exactly to total VaR**.
- **Incremental VaR** — the change in portfolio VaR from a finite trade `Delta w`.

## Risk attribution

`attribution` decomposes portfolio *volatility* into marginal (MCR), component (CCR), and
percentage contributions (the realized **risk budget**), and offers:

- **Factor risk attribution** — split variance into systematic (`w'BFB'w`) and specific (`w'Dw`)
  parts with per-factor contributions from the portfolio exposure `b_p = B'w`. Consumes factor
  loadings from the quant package's models.
- **Sector risk attribution** — group asset contributions by sector.

## Portfolio risk statistics

`portfolio_risk`: volatility, tracking error / active risk, information ratio, beta (and rolling
volatility/beta), covariance and correlation matrices (and rolling), diversification ratio, and
concentration measures — **Herfindahl index**, effective number of holdings, top-N concentration.

## Stress testing

Stress tests apply severe, named shocks and measure impact — complementing VaR's "typical" tail
with specific adverse episodes.

- Builders: **market crash** (beta-scaled), **interest-rate shock** (per-asset rate sensitivities),
  **factor shock**, **sector shock**, and a **historical scenario** reconstructed from a past
  return window ("what if that episode repeated").
- **Volatility shock** — scale the covariance and re-measure VaR (a vol shock changes *risk*).
- `run_stress_test` revalues a portfolio across a battery of scenarios into a `StressTestReport`
  (worst/best case, ranked table).

## Scenario analysis

The `ScenarioEngine` revalues a portfolio under a `Scenario` (per-asset and/or per-factor shocks,
applied through exposures): `r_i = a_i + sum_j B_ij f_j`, portfolio `Δ = w·r`. It supports
`revalue`, `compare` (ranked), and one-dimensional `sensitivity` (P&L as a single shock sweeps,
with a first-order `delta`).

---

## Data models (immutable & serializable)

`RiskReport`, `VaRReport`, `RiskDecomposition`, `RiskContribution`, `RiskSnapshot`,
`StressTestReport`, `ScenarioReport` — all frozen dataclasses with `to_dict` / `from_dict` and a
`summary()`.

---

## Usage

```python
import numpy as np
import factorlab_risk as risk

# --- Market VaR / ES on a return series ---
r = np.random.default_rng(0).normal(0.0005, 0.012, 1000)
risk.historical_var(r, confidence=0.95)                 # empirical
risk.parametric_expected_shortfall(r, confidence=0.99)  # Normal closed form
risk.monte_carlo_var(r, confidence=0.95, n_simulations=100_000, seed=1)
risk.rolling_var(r, window=250, confidence=0.95)        # time-varying

# --- Portfolio VaR + decomposition ---
w = np.array([0.4, 0.35, 0.25])
cov = np.array([[0.040, 0.006, 0.004],
                [0.006, 0.090, 0.002],
                [0.004, 0.002, 0.070]])
risk.portfolio_var(w, cov, confidence=0.99)
risk.component_var(w, cov, confidence=0.99)             # sums to portfolio VaR
risk.marginal_var(w, cov); risk.percent_contribution_var(w, cov)

# --- Risk attribution ---
risk.percentage_contribution_to_risk(w, cov)            # realized risk budget
B = np.array([[1.0, 0.2], [0.9, -0.1], [1.1, 0.3]])
F = np.diag([0.04, 0.02]); d = np.array([0.01, 0.015, 0.008])
fa = risk.factor_risk_attribution(w, B, F, d)           # systematic vs specific

# --- Concentration ---
risk.herfindahl_index(w); risk.diversification_ratio(w, cov)

# --- Stress & scenarios ---
eng = risk.ScenarioEngine(("A", "B", "C"), exposures=B, factor_names=("MKT", "SMB"))
crash = risk.market_crash_scenario(("A", "B", "C"), -0.30, betas=[1.0, 0.9, 1.1])
eng.revalue(w, crash, portfolio_value=1_000_000).pnl
report = risk.run_stress_test(eng, w, [crash], portfolio_value=1_000_000)
print(report.summary())

# --- Full report + serialization ---
rr = risk.RiskReport.from_portfolio(("A", "B", "C"), w, cov, confidence=0.95)
print(rr.summary())
rr2 = risk.RiskReport.from_dict(rr.to_dict())           # round-trips
```

### Integration (workflow tail)

```python
from factorlab_optimizer import Constraint, MinVarianceOptimizer, OptimizationProblem
from factorlab_risk.integration import risk_report_from_weights, var_report_from_returns

opt = MinVarianceOptimizer().optimize(
    OptimizationProblem(("A", "B", "C"), mu, cov, constraints=(Constraint.long_only(),))
)
risk_report_from_weights(opt, cov)          # optimizer result -> RiskReport
var_report_from_returns(backtest_result)    # BacktestResult / ReturnSeries -> VaRReport
```

---

## Validation methodology

- **Closed-form / published formulas**: parametric Normal VaR checked against `-Phi^{-1}(alpha)`;
  Normal ES against `sigma·phi(z)/alpha`; component VaR and risk contributions verified to satisfy
  the Euler additivity identity; factor attribution verified `systematic + specific = total`.
- **Independent SciPy**: parametric quantiles/pdf validated against `scipy.stats`.
- **Convergence**: Monte Carlo VaR/ES verified to converge to the parametric values.
- **Property-based** (Hypothesis): `ES >= VaR`, VaR monotone in confidence, contributions sum to
  the total, percentage contributions sum to 1.
- **Numerical stability**: tiny/large-scale returns produce finite results.

## Testing

```bash
pip install -e ".[dev]"        # plus peers for integration tests: ../portfolio ../optimizer ../backtesting
pytest --cov=factorlab_risk    # 98% coverage
ruff check src tests ; mypy src
```

---

## References

- Jorion, P. *Value at Risk* (VaR, marginal/component VaR).
- Artzner et al. (1999). *Coherent Measures of Risk* (Expected Shortfall).
- Litterman (1996). *Hot Spots and Hedges* (risk decomposition / contributions).
