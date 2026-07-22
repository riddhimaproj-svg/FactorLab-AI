# factorlab-derivatives

**Institutional Derivatives Engine** — a pure, fully-typed computational core for
option pricing, Greeks, implied and realized volatility, a volatility surface, and
Monte Carlo pricing.

It is an *independent* package: it depends only on `numpy` and `scipy`, does no I/O,
opens no network connections, and holds no global state. Every public result is an
**immutable, serializable** dataclass (`to_dict` / `from_dict`). This makes it safe to
embed inside a service, a research notebook, or a batch job without surprises.

---

## Architecture

The engine is layered strictly from the inside out — the pure math core knows nothing
about the convenience façade, and nothing reaches back into calling code (hexagonal /
ports-and-adapters, dependencies point inward only):

```
                        ┌───────────────────────────────────────┐
   public API           │            factorlab_derivatives       │
   (engine façade)      │   price_option()  ·  PricingResult      │
                        └───────────────┬───────────────────────┘
                                        │ dispatches to
             ┌──────────────────────────┼──────────────────────────────┐
             ▼                          ▼                              ▼
   ┌───────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
   │  pricing/          │   │  volatility / surface   │   │  monte_carlo           │
   │  black_scholes     │   │  implied_volatility     │   │  monte_carlo_european  │
   │  black76           │   │  historical / ewma      │   │  (antithetic + control │
   │  binomial (CRR)    │   │  fit_garch (MLE)        │   │   variates)            │
   │  digital, barrier  │   │  VolatilitySurface      │   │                        │
   └─────────┬──────────┘   └───────────┬────────────┘   └───────────┬────────────┘
             │                          │                            │
             └──────────────────────────┼────────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │  core primitives (no dependencies)        │
                    │  instruments · reports · greeks           │
                    │  _validation · errors                     │
                    └──────────────────────────────────────────┘
```

* **`instruments`** — immutable contracts (`Option`, `DigitalOption`, `BarrierOption`)
  and market state (`MarketData`), plus the `OptionType` / `ExerciseStyle` /
  `BarrierType` / `DigitalKind` enums.
* **`reports`** — the return "currency": `Greeks`, `PricingResult`, `MonteCarloResult`,
  `ImpliedVolatilityResult`.
* **`pricing/`** — the closed-form and lattice models.
* **`greeks`** — a generic central finite-difference Greeks calculator, used both to
  *validate* the analytical Greeks and to *produce* Greeks for models without a closed
  form (American options).
* **`volatility`, `surface`** — realized-vol estimators and the interpolated surface.
* **`monte_carlo`** — GBM European pricing with variance reduction.
* **`engine`** — the ergonomic front door (`price_option`).
* **`_validation`, `errors`** — shared input checks and the typed exception hierarchy.

### Conventions

* Rates, dividend yields and volatilities are **annualized decimals** (`0.05` = 5%).
* Maturity `T` is in **years**.
* Greeks are raw partial derivatives: `vega = dV/dσ` (a 1% vol move ≈ `vega·0.01`),
  `rho = dV/dr`, `theta = dV/dt` **per year** (one day ≈ `theta/365`).

---

## Models

### Pricing

| Function | Model | Notes |
|---|---|---|
| `black_scholes_price` / `black_scholes_greeks` | Black-Scholes-Merton | European, continuous dividend yield `q`; closed-form Greeks |
| `black76_price` / `black76_greeks` | Black-76 | Options on forwards/futures; `rho = -T·price` |
| `binomial_price` | Cox-Ross-Rubinstein tree | European **and** American (early exercise); converges to BS as `steps→∞` |
| `digital_price` | Cash- / asset-or-nothing binaries | Closed form |
| `barrier_price` | Reiner-Rubinstein | Single continuous barrier; knock-out via in/out parity |

### Volatility

| Function | Purpose |
|---|---|
| `implied_volatility` | Invert BS for σ — Newton-Raphson with a bracketed Brent fallback; enforces no-arbitrage bounds |
| `historical_volatility` | Annualized sample std-dev of returns |
| `ewma_variance` / `ewma_volatility` | RiskMetrics exponentially-weighted variance |
| `fit_garch` | GARCH(1,1) by Gaussian MLE → `GarchResult` (persistence, long-run vol, forecasts) |
| `VolatilitySurface` | Strike × maturity grid with bilinear interpolation and edge clamping |

### Monte Carlo

`monte_carlo_european` simulates the exact GBM terminal price and prices any European
payoff, with optional **antithetic** and **control-variate** variance reduction. The
result reports a standard error and 95% confidence interval.

---

## Usage

```python
from factorlab_derivatives import (
    Option, OptionType, MarketData, price_option,
    implied_volatility, monte_carlo_european,
)

# Vanilla European call via the engine façade
option = Option(OptionType.CALL, strike=100.0, maturity=1.0)
market = MarketData(spot=100.0, rate=0.05, volatility=0.2)
result = price_option(option, market)
print(result.summary())
# Call option (black_scholes)
#   Price: 10.450584
#   Delta +0.6368  Gamma +0.0188  Vega +37.5240  Theta -6.4140  Rho +53.2325

# American put — priced on a CRR tree, Greeks by finite differences
from factorlab_derivatives import ExerciseStyle
amer = Option(OptionType.PUT, strike=100.0, maturity=1.0, exercise=ExerciseStyle.AMERICAN)
print(price_option(amer, market).price)

# Implied volatility from a quoted price
iv = implied_volatility(10.4506, OptionType.CALL, spot=100, strike=100,
                        maturity=1.0, rate=0.05)
print(iv.implied_volatility, iv.converged)   # ≈ 0.2  True

# Monte Carlo cross-check with variance reduction
mc = monte_carlo_european(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2,
                          n_paths=200_000, seed=0)
print(mc.summary())
```

Every result is serializable:

```python
payload = result.to_dict()
restored = type(result).from_dict(payload)   # round-trips exactly
```

---

## Public API

Instruments & market: `OptionType`, `ExerciseStyle`, `BarrierType`, `DigitalKind`,
`Option`, `DigitalOption`, `BarrierOption`, `MarketData`.

Results: `Greeks`, `PricingResult`, `MonteCarloResult`, `ImpliedVolatilityResult`.

Pricing: `black_scholes_price`, `black_scholes_greeks`, `black76_price`,
`black76_greeks`, `binomial_price`, `digital_price`, `barrier_price`, `d1_d2`.

Engine: `price_option`, `PricingMethod`.

Greeks: `finite_difference_greeks`.

Volatility: `implied_volatility`, `historical_volatility`, `ewma_variance`,
`ewma_volatility`, `fit_garch`, `GarchResult`, `VolatilitySurface`.

Monte Carlo: `monte_carlo_european`.

Errors: `DerivativesError`, `DerivativesInputError`, `ConvergenceError`,
`NoArbitrageError`.

---

## Validation & testing

The suite (`pytest`) validates the engine against independent references:

* **Black-Scholes** against textbook analytical values and **put-call parity**.
* **Greeks** against the generic central **finite-difference** calculator.
* **Binomial** convergence to Black-Scholes as `steps → ∞`.
* **Implied vol** recovers known input volatilities (round-trip).
* **Digital** options cross-checked against BS `N(d₂)` relationships.
* **Barrier** in/out parity (`in + out = vanilla`) and Monte Carlo sanity.
* **GARCH** stationarity (`α + β < 1`) and likelihood improvement over the seed.
* **Monte Carlo** convergence to BS within the reported standard error, and that
  antithetic + control variates **reduce** variance.
* **Serialization** round-trips for every public model.

Property-based tests (Hypothesis) assert structural invariants: non-negative prices,
monotonicity in spot/strike, parity, and bounds.

### Quality gates

```bash
PKG=packages/derivatives
packages/quant/.venv/bin/ruff check $PKG
packages/quant/.venv/bin/mypy --strict $PKG/src
packages/quant/.venv/bin/pytest $PKG --cov=factorlab_derivatives --cov-report=term-missing
```

Targets: Ruff clean, `mypy --strict` clean, **> 95%** line+branch coverage.
