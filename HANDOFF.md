# FactorLab AI — Developer Handoff

**Institutional Multi-Factor Research & Alpha Discovery Platform**

This document is the single source of truth for an engineer taking over the project. It
describes what exists, how it is built, the conventions that must be preserved, and what to build
next. Read it end to end before writing code.

> **Status at handoff:** seven pure-Python analytics packages are complete, reviewed, and frozen
> (or pending review) — the six-package research substrate plus the standalone **Derivatives
> Engine** (Package 8). No API, no frontend, and no AI layer have been built yet — those are
> deliberate future deliverables. Everything so far is a **pure computational library**.

---

## 1. Overall project architecture

The end-state vision (from the original architecture brief) is a full platform:

```
Frontend (Next.js 15 / TS / Tailwind / Plotly)   ← NOT built yet
        │  HTTPS / typed OpenAPI client
API gateway (FastAPI, Python 3.12)               ← NOT built yet
        │
        ├── Quant engine        (factorlab_quant)        ✅ built
        ├── Data layer          (factorlab_data)         ✅ built
        ├── Portfolio analytics (factorlab_portfolio)    ✅ built
        ├── Optimizer           (factorlab_optimizer)    ✅ built
        ├── Backtesting         (factorlab_backtesting)  ✅ built
        ├── Risk engine         (factorlab_risk)         ✅ built
        ├── Derivatives engine  (factorlab_derivatives)  ✅ built  (standalone, Package 8)
        └── AI layer            (grounded LLM explain)   ← NOT built yet
```

**What is actually implemented today** is the analytics substrate: seven independently installable,
fully-typed Python packages under `packages/`. Six compose into the canonical research workflow;
the seventh (`factorlab_derivatives`) is a **standalone options/derivatives pricing core** that
does not participate in the factor-research pipeline:

```
Market Data → Factor Model → Expected Returns → Optimizer → Portfolio → Backtester → Risk Engine → Reports

Derivatives Engine (standalone):  Instruments + Market → Pricing / Greeks / Vol / Monte Carlo → Reports
```

### Guiding architectural principles (respected throughout)

- **Hexagonal / ports-and-adapters.** Business logic depends on interfaces (`FactorDataPort`,
  `Estimator`, `FactorModel`), never on concrete providers. Adding a data source or an estimator
  is an additive change, not a refactor.
- **The quant engine is a library, not a service.** Models are pure, side-effect-free functions
  of typed inputs → typed, immutable results. No HTTP, DB, or provider knowledge anywhere in the
  computational core.
- **Dependency direction points inward.** Outer packages may depend on inner ones; inner packages
  never import outward. Concrete wiring lives only at composition roots / integration layers.
- **Immutability & serialization everywhere.** Public results are frozen dataclasses with
  `to_dict` / `from_dict` and read-only NumPy arrays.

### Package dependency graph (current)

```
factorlab_quant        (numpy, scipy)                 — no internal deps
factorlab_data         (numpy) ───lazy──▶ factorlab_quant   (FactorPanel.to_factor_set)
factorlab_portfolio    (numpy)                          — standalone
factorlab_optimizer    (numpy, scipy)                   — standalone
factorlab_backtesting  (numpy, factorlab_portfolio; optimizer optional/lazy)
factorlab_risk         (numpy, scipy; portfolio/optimizer/backtesting via lazy duck-typed bridges)
factorlab_derivatives  (numpy, scipy)                   — standalone, ZERO internal deps
```

**Key rule:** cross-package integration is done by **lazy imports and/or duck typing**, so each
package's pure core stays dependency-light and independently testable. The quant engine imports
*nothing* from the other packages.

---

## 2. Folder structure

```
factorlab-ai/
├── HANDOFF.md                      ← this document
├── .git/
└── packages/
    ├── quant/                      factorlab_quant  (asset-pricing models + estimation engine)
    │   ├── pyproject.toml
    │   ├── README.md
    │   ├── .venv/                  ← SHARED dev virtualenv (all packages installed editable here)
    │   ├── src/factorlab_quant/
    │   │   ├── core/               types.py (RegressionResult, CoefficientEstimate, …), protocols.py, errors.py
    │   │   ├── estimation/         ols.py (OLS + inference), hac.py (Newey–West / White covariance)
    │   │   ├── diagnostics/        tests.py (JB, Breusch–Pagan, F), residuals.py (skew/kurtosis/DW)
    │   │   ├── models/             base.py, linear_factor_model.py (engine + FactorModelResult),
    │   │   │                       factors.py (Factor/FactorSet), registry.py,
    │   │   │                       capm.py, fama_french_3.py, fama_french_5.py, carhart.py
    │   │   └── utils/              validation.py, align.py
    │   └── tests/  (unit/ + property/)
    ├── data/                       factorlab_data  (factor-data infrastructure)
    │   └── src/factorlab_data/     ports.py, adapters/kenneth_french.py, panel.py, metadata.py,
    │                               validation.py, alignment.py, cache.py, registry.py, loader.py, errors.py
    ├── portfolio/                  factorlab_portfolio  (immutable models + performance/risk analytics)
    │   └── src/factorlab_portfolio/ holdings.py, portfolio.py, returns.py, report.py, errors.py,
    │                                analytics/{performance,drawdown,relative,rolling}.py
    ├── optimizer/                  factorlab_optimizer  (portfolio optimization)
    │   └── src/factorlab_optimizer/ problem.py, config.py, constraints.py, weights.py, result.py,
    │                                risk.py, frontier.py, optimizers/{base,min_variance,mean_variance,
    │                                max_sharpe,max_diversification,risk_parity,black_litterman}.py
    ├── backtesting/                factorlab_backtesting  (event-driven backtester)
    │   └── src/factorlab_backtesting/ market_data.py, schedule.py, orders.py, costs.py, execution.py,
    │                                  strategy.py, benchmark.py, metrics.py, report.py, backtest.py,
    │                                  walkforward.py, errors.py
    ├── risk/                       factorlab_risk  (institutional risk engine)
    │   └── src/factorlab_risk/     var/{historical,parametric,monte_carlo,rolling,decomposition}.py,
    │                               attribution.py, portfolio_risk.py, scenario.py, stress.py,
    │                               reports.py, integration.py, _validation.py, errors.py
    └── derivatives/                factorlab_derivatives  (standalone options/derivatives engine)
        └── src/factorlab_derivatives/ instruments.py, reports.py, greeks.py, implied_vol.py,
                                    volatility.py, surface.py, monte_carlo.py, engine.py,
                                    _validation.py, errors.py,
                                    pricing/{black_scholes,black76,binomial,digital,barrier}.py
```

Each package is a standard `src/`-layout project with its own `pyproject.toml`, `README.md`, and
`tests/` (unit tests plus, where relevant, property tests). There is **no root-level build config
yet** (no `turbo.json`, no workspace `pyproject`); packages are managed individually.

---

## 3. Completed packages

All seven are ruff-clean, mypy-strict-clean, and exceed 95% coverage (data is 92%, see note).

| Package | Purpose | Public highlights | Tests | Cov | Status |
|---|---|---|---|---|---|
| **factorlab_quant** | Asset-pricing models + OLS/HAC engine | `LinearFactorModel`, `FactorModelResult`, `Factor`/`FactorSet`, model registry; **CAPM, FF3, FF5, Carhart** | 232 | ~95% | **Approved (frozen framework, CAPM, FF3, FF5, Carhart)** |
| **factorlab_data** | Factor-data infrastructure | `FactorDataPort`, `KennethFrenchAdapter`, `FactorPanel`/`FactorDataset`, `FactorLoader`, `FactorValidator`, `FactorAlignment`, `FactorCache`, `FactorRegistry` | 71 | 92% | **Approved (frozen)** |
| **factorlab_portfolio** | Immutable portfolio models + analytics | `Position`, `Holding`, `Trade`, `Portfolio`, `PortfolioSnapshot`, `ReturnSeries`, `PerformanceReport`; analytics (Sharpe/Sortino/Calmar/Omega/drawdown/rolling/relative) | 111 | 99% | **Approved (frozen)** |
| **factorlab_optimizer** | Portfolio optimization | `OptimizationProblem`, `OptimizerConfig`, `Constraint`, `PortfolioWeights`, `OptimizationResult`, `EfficientFrontier`; 6 optimizers (MinVar, MeanVar, MaxSharpe, MaxDiversification, RiskParity, BlackLitterman) | 85 | 99% | **Approved (frozen)** |
| **factorlab_backtesting** | Event-driven backtester | `Backtest`, `Strategy` (Static/Equal/Optimizer), `RebalanceSchedule`, `ExecutionEngine`, `Order`/`Fill`/`OrderBook`, cost/slippage/broker models, `Benchmark`, `BacktestReport`, `WalkForward` | 86 | 99% | **Approved (frozen)** |
| **factorlab_risk** | Market & portfolio risk | VaR/ES (historical/parametric/MC/rolling), VaR decomposition, attribution, portfolio risk, stress, scenarios, reports, integration | 121 | 98% | **Approved (frozen)** |
| **factorlab_derivatives** | Options / derivatives pricing engine (Package 8, standalone) | `price_option`/`PricingMethod`; Black-Scholes, Black-76, CRR binomial (Euro+American), digital, barrier; analytical + finite-diff `Greeks`; `implied_volatility` (Newton+Brent); historical/EWMA/GARCH(1,1) vol; `VolatilitySurface`; `monte_carlo_european` (antithetic+control variates) | 149 | 99% | **Complete — pending review** |

> **factorlab_data coverage note:** 92% is the approved baseline; its integration surface
> (`load()` network path) is intentionally exercised only through injected fetchers, not live I/O.

### Frozen packages
`factorlab_optimizer` and `factorlab_backtesting` are explicitly **frozen** — do not modify unless
a critical bug or integration issue is found. The approved models/infra in `factorlab_quant`,
`factorlab_data`, `factorlab_portfolio`, and `factorlab_risk` are likewise frozen.
`factorlab_derivatives` is the most recent deliverable and is **awaiting review** (treat as
complete-but-not-yet-frozen; do not build on top of it until approved).

### 3.1 Derivatives Engine (Package 8) — architecture & public API

**Independent by design:** `factorlab_derivatives` depends only on numpy + scipy and imports
**nothing** from the other six packages (and they import nothing from it). Same house rules apply:
hexagonal layering with dependencies pointing inward, validation-first, immutable + serializable
public results (`frozen=True, slots=True`, `to_dict`/`from_dict`).

**Internal layering (inner ← outer, dependencies point inward only):**

```
core primitives     instruments · reports · greeks · _validation · errors     (numpy/scipy only)
models              pricing/{black_scholes,black76,binomial,digital,barrier} · volatility ·
                    surface · monte_carlo · implied_vol                         (compose the core)
façade              engine.price_option  →  dispatches Euro→Black-Scholes, American→binomial
```

**Conventions (documented in `_validation.py`):** rates/yields/vols are annualized decimals;
maturity `T` in years; Greeks are raw partials (`vega = ∂V/∂σ`, `rho = ∂V/∂r`, `theta = ∂V/∂t` per
**year**). Enums are `StrEnum` (Python 3.11+) for clean string serialization.

**Public API surface (`from factorlab_derivatives import …`):**

| Group | Exports |
|---|---|
| Instruments / market | `OptionType`, `ExerciseStyle`, `BarrierType`, `DigitalKind`, `Option`, `DigitalOption`, `BarrierOption`, `MarketData` |
| Results | `Greeks`, `PricingResult`, `MonteCarloResult`, `ImpliedVolatilityResult` |
| Pricing | `black_scholes_price` / `black_scholes_greeks`, `black76_price` / `black76_greeks`, `binomial_price`, `digital_price`, `barrier_price`, `d1_d2` |
| Engine (façade) | `price_option`, `PricingMethod` |
| Greeks | `finite_difference_greeks` |
| Volatility | `implied_volatility`, `historical_volatility`, `ewma_variance`, `ewma_volatility`, `fit_garch`, `GarchResult`, `VolatilitySurface` |
| Monte Carlo | `monte_carlo_european` |
| Errors | `DerivativesError`, `DerivativesInputError`, `ConvergenceError`, `NoArbitrageError` |

**Model notes / gotchas future work must preserve:**
- **Black-76 `rho = −T · price`** (the forward is observed directly and does not move with `r`) —
  intentionally different from Black-Scholes rho.
- **Barrier options** are priced by Reiner–Rubinstein closed forms assuming **continuous
  monitoring, no rebate**; knock-outs use exact in/out parity (`in + out = vanilla`).
- **American options** have no closed-form Greeks — the engine returns finite-difference Greeks over
  the CRR binomial price.
- **Implied vol** solver is Newton-Raphson with a **bracketed Brent fallback**, guarded by static
  no-arbitrage bounds (raises `NoArbitrageError` for unreachable prices).
- **Monte Carlo** simulates the exact GBM terminal price with optional antithetic + control variates
  and reports a standard error / 95% CI.

---

## 4. Remaining packages / deliverables (NOT built)

In rough dependency order:

1. **Additional factor models** (inside `factorlab_quant`): **Hou–Xue–Zhang q-factor** and
   **APT**. Each is a thin `LinearFactorModel` subclass (see §10).
2. **Additional data adapters** (inside `factorlab_data`): the architecture designed ports for
   Polygon, Tiingo, AlphaVantage, Yahoo (market); FMP, SEC EDGAR (fundamentals); FRED, World Bank,
   OECD, IMF SDMX, BIS, ECB (macro); Finnhub, Marketaux, NewsAPI (news). **Only
   `KennethFrenchAdapter` exists.** New adapters must satisfy `FactorDataPort` and pass the shared
   adapter contract tests.
3. **API service** (`apps/api`, FastAPI, Python 3.12): routers → controllers → services →
   repositories, Pydantic DTOs, DI container, structured logging, error middleware, caching.
   **Not started.**
4. **Frontend** (`apps/web`, Next.js 15 / TS / Tailwind / shadcn / Plotly / React Query / Zustand).
   **Not started.**
5. **AI layer** (grounded LLM explanation with citations). **Not started.**
6. ~~Derivatives / options pricing (Black–Scholes), general Monte-Carlo simulation framework.~~
   ✅ **DONE** — delivered as the standalone `factorlab_derivatives` package (Package 8); see §3 and
   §3.1. (Note: `factorlab_derivatives.monte_carlo` is an option **pricing** path-sim engine; the
   *Monte Carlo VaR* in `factorlab_risk` is a distinct risk estimator from a fitted return
   distribution — the two do not overlap or depend on each other.)

Do **not** begin the API, frontend, or AI layer without explicit instruction.

---

## 5. Important engineering decisions

- **Pure library first, platform later.** Every quantitative capability is a dependency-light,
  independently testable library. The API/frontend/AI are thin layers to be added on top.
- **The quant engine depends on nothing internal.** It is the crown jewel and stays pure. Data
  flows *into* it as `FactorSet`; results flow out as `FactorModelResult`.
- **`LinearFactorModel` is the shared engine.** CAPM/FF3/FF5/Carhart are ~thin subclasses that
  specify only their factors (and, for CAPM/FF3/FF5/Carhart, their excess-return construction and
  a richer result subclass). Adding a linear factor model is additive, never a rewrite.
- **Data → quant coupling via lazy `to_factor_set()`.** `factorlab_data` produces
  `factorlab_quant` `FactorSet`s through a lazy import inside `FactorPanel.to_factor_set()`, so the
  quant engine never imports data.
- **Models consume factors by duck typing.** FF5/Carhart accept anything exposing
  `to_factor_set()` (a data-layer panel), a `FactorSet`, or a mapping — so the quant package has no
  hard dependency on the data package.
- **Risk/backtesting integrate by duck typing too.** `factorlab_risk.integration` and
  `factorlab_backtesting.strategy.OptimizerStrategy` bridge to peers via `getattr`/lazy import.
- **Robust inference by default.** OLS defaults to Newey–West (HAC) standard errors because
  financial residuals are heteroskedastic and autocorrelated.
- **Numerical-safety conventions.** Dispersion guards use a small tolerance (`_ZERO_TOL = 1e-13`)
  rather than exact `== 0` so near-constant series yield `nan`/`0` instead of `1e17` artifacts.
  Quadratic forms are wrapped `float(w @ Σ @ w)` before `np.sqrt(max(..., 0.0))` (mypy + stability).
- **Registry pattern for models.** `factorlab_quant` models self-register via `@register_model`;
  `get_model(name)` / `list_models()` enable config-driven instantiation.
- **VaR/ES sign convention:** returned as **positive loss magnitudes**; confidence `c ∈ (0,1)`,
  tail `α = 1 − c`; horizon scaling by √h. This convention is documented in
  `factorlab_risk/_validation.py` and must be preserved.

---

## 6. Coding conventions

- **Python ≥ 3.11** target (runtime here is 3.13); mypy `python_version = "3.12"`.
- **`from __future__ import annotations`** at the top of every module.
- **Typing is strict and complete.** `mypy --strict` must pass. Prefer precise types;
  `NDArray[np.float64]` (aliased `FloatArray` per package) for arrays. Duck-typed peer objects are
  taken as `object` and narrowed with `cast(Any, …)` at the boundary.
- **Immutable public models:** `@dataclass(frozen=True, slots=True)`; NumPy arrays set read-only in
  `__post_init__`. Every public model has `to_dict()` / `from_dict()`; JSON-serializable.
- **Validation-first:** coerce and validate inputs at the top of public functions, raising a typed
  package error (`RiskInputError`, `OptimizationInputError`, `BacktestInputError`, …) before any
  math. Each package has an `errors.py` with a base `*Error` and specific subclasses.
- **Docstrings carry the math and the economics.** Module and function docstrings include the
  formula (reStructuredText `.. math::`), interpretation, and academic citations. Match the density
  of the surrounding code.
- **Naming:** `X` for design matrices and `y`/`b` for vectors is intentional (ruff `N803/N806`
  ignored). Short module aliases in tests (`import performance as P`) are allowed (`N812` ignored).
- **Line length 100.** Ruff rule set: `E,F,I,N,UP,B,C4,SIM,RUF`. Math-heavy packages also ignore
  `RUF001/2/3` (ambiguous-unicode in docstrings) — see each `pyproject.toml`.
- **No dead code / unused imports.** Prefer removing over `# noqa`. Use `# pragma: no cover` only
  for genuinely defensive-unreachable branches, with a comment saying why.
- **`__all__`** is maintained (isort-sorted) in every module; the package `__init__` re-exports the
  public surface.

---

## 7. Testing conventions

- **pytest**, `src`-layout, tests under `tests/unit/` (+ `tests/property/` where used). Markers:
  `property`, `integration`, `validation` (declared per `pyproject.toml`, `--strict-markers`).
- **Validate against ground truth, three ways:**
  1. **Known DGP** — simulate from a known data-generating process and recover the parameters.
  2. **Independent reference** — cross-validate against `statsmodels` (quant regressions/diagnostics)
     and `scipy` (risk quantiles/pdf; optimizer solves), to machine tolerance.
  3. **Closed-form / analytic** — e.g. min-variance `Σ⁻¹𝟙/(𝟙ᵀΣ⁻¹𝟙)`, tangency `Σ⁻¹(μ−rf)`,
     Normal VaR `−Φ⁻¹(α)`, Normal ES `σφ(z)/α`, Euler additivity of risk contributions.
- **Property-based tests (Hypothesis):** assert invariants over random inputs (R² ∈ [0,1], PSD
  covariance, component contributions sum to total, ES ≥ VaR, VaR monotone in confidence, prediction
  interval ⊇ confidence interval, serialization round-trips). Use
  `settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)`.
- **Serialization round-trip tests** for every public model (`to_dict`→`from_dict`, JSON, pickle).
- **Edge/validation tests** for every raised error and every degenerate branch (empty, single-obs,
  zero-variance, dimension mismatch, unknown method).
- **Coverage target > 95%** per package (enforced by review, not CI yet). Tolerances in
  parameter-recovery tests are set to ~2–3 standard errors and documented inline.

---

## 8. Current quality gates

Run per package (from the package directory), using the shared venv:

```bash
V=/Users/riddhimaohri/factorlab-ai/packages/quant/.venv/bin
$V/python -m ruff check src tests      # must be clean
$V/python -m mypy src                  # --strict, must be clean
$V/python -m pytest -q                 # all pass
$V/python -m pytest --cov=<pkg> --cov-report=term-missing   # > 95%
```

**Definition of done** for a package/feature: ruff clean · mypy-strict clean · all tests pass ·
coverage > 95% · public APIs documented (README + docstrings) · README examples execute · no
regression in previously approved packages.

**Dev environment:** a single shared virtualenv at `packages/quant/.venv` has all six packages
installed editable (`pip install -e`) plus dev deps (pytest, pytest-cov, hypothesis, mypy, ruff,
statsmodels). New packages should be `pip install -e`'d into this same venv so cross-package
integration tests resolve. There is no CI pipeline yet — gates are run manually.

---

## 9. Current roadmap

1. ✅ Quant framework + CAPM → FF3 → FF5 → Carhart (each reviewed one at a time).
2. ✅ Factor Data Layer (Kenneth French).
3. ✅ Portfolio analytics.
4. ✅ Optimizer.
5. ✅ Backtesting.
6. ✅ Risk engine *(approved / frozen).*
7. ✅ **Derivatives engine** (`factorlab_derivatives`, Package 8) — *complete, pending review.*
8. ⏭ **Next:** see §10.
9. ⏳ Remaining factor models (q-factor, APT).
10. ⏳ Additional data adapters (market/fundamentals/macro/news).
11. ⏳ API service (FastAPI).
12. ⏳ Frontend (Next.js).
13. ⏳ AI explanation layer.

The project has been built **strictly one component at a time, stopping for review before the
next.** Preserve this cadence — do not batch multiple deliverables.

---

## 10. What to build next

> **Immediate gate:** the Derivatives Engine (Package 8) is complete and **awaiting review**. Do not
> start the next unit until it is reviewed/approved and frozen. Nothing below has been started.

**Recommended next unit: the Hou–Xue–Zhang q-factor model** (Carhart-style momentum is done;
q-factor and APT remain) inside `factorlab_quant`. It is the smallest, highest-value increment and
exercises the frozen framework exactly as intended.

- Factors: `Mkt-RF`, `ME` (size), `I/A` (investment), `ROE` (profitability).
- Implement as `packages/quant/src/factorlab_quant/models/hou_xue_zhang_q.py`:
  a `HouXueZhangQModel(LinearFactorModel)` + `HouXueZhangQResult(FactorModelResult)`, mirroring
  `fama_french_5.py` / `carhart.py` (thin subclass; specify factor names, excess-return handling,
  named accessors, `_make_result`, `@register_model("HXZ"/"q-Factor")`).
- Consume factors through the data layer via the duck-typed `to_factor_set()` path (like FF5).
- Tests: parameter recovery on a known DGP, `statsmodels` cross-validation (nonrobust + HAC),
  serialization dispatch, prediction, validation edges, property tests. Target > 95% coverage.
- Then **APT** (`arbitrage_pricing.py`): a generic multi-factor `LinearFactorModel` accepting an
  arbitrary user-supplied `FactorSet` (this is essentially `LinearFactorModel` with an APT identity
  and result wrapper).

Alternatively, if the priority shifts to the platform, the **API service** is the next big rock —
but confirm with the product owner first; the pattern so far has been to finish the analytics
substrate before the transport/UX layers.

---

## 11. Known TODOs / issues

- **⚠ Flaky Hypothesis test in the frozen `factorlab_portfolio` package:**
  `tests/unit/test_properties.py::test_sharpe_scale_invariant`. A near-constant return series whose
  sample std sits right at the `_ZERO_TOL = 1e-13` guard flips between `nan` and finite when the
  series is scaled, so the scale-invariance assertion fails at that boundary. It is **pre-existing**
  (surfaced by Hypothesis's persisted example DB), **not caused by the risk engine**, and was left
  unfixed because the portfolio package is frozen. **Suggested fix (needs approval to touch frozen
  code):** in the property test, skip series whose std is below the tolerance (or assert with a
  tolerance-aware equality). It fails deterministically now because Hypothesis cached the
  counterexample under `packages/portfolio/.hypothesis/`.
- **`factorlab_derivatives` (Package 8) — two intentionally-uncovered defensive lines.** Coverage is
  99% (line+branch); the only misses are two guards **inside the SciPy optimizer callback** in
  `volatility.py` (`_garch_neg_loglik`): the `variance ≤ 0` mid-search rejection and the MLE
  non-convergence `ConvergenceError`. They are not deterministically reachable without mocking
  optimizer internals, so they were left uncovered rather than tested via fragile mocks. Not a bug.
- **`factorlab_derivatives` barrier pricing assumes continuous monitoring, no rebate.** Discrete-
  monitoring correction and rebates are out of scope; documented in `pricing/barrier.py`. A path-sim
  sanity test uses a loose tolerance because discrete monitoring slightly over-prices knock-outs.
- **`factorlab_data` has only one adapter** (Kenneth French). The `FactorDataPort` contract and the
  shared adapter-contract test are in place for adding more; nothing else is wired.
- **No CI.** Quality gates are run manually. A future TODO is a CI workflow running ruff/mypy/pytest
  + coverage per package, plus a cross-package regression job.
- **No root workspace tooling** (`turbo.json`, workspace `pyproject`, unified `make test`). Consider
  adding once a second consumer (API) appears.
- **Horizon scaling for VaR** uses the √h square-root-of-time approximation; if a future requirement
  needs h-period overlapping-return VaR, extend `factorlab_risk.var` explicitly (documented as an
  approximation today).

---

## 12. Assumptions future development must preserve

1. **Returns are per-period simple returns in decimal** unless a function explicitly says otherwise.
2. **VaR/ES are positive loss magnitudes; confidence `c ∈ (0,1)`, tail `α = 1 − c`.** Do not flip
   the sign convention.
3. **Sample statistics use `ddof = 1`; annualization uses `√(periods_per_year)`** — consistent
   across portfolio and risk so numbers reconcile.
4. **The quant engine imports nothing internal.** Keep it pure. Never add a data/API dependency to
   `factorlab_quant`.
5. **Cross-package integration is lazy + duck-typed.** New integrations must not create hard import
   cycles or force a heavy dependency into a pure core.
6. **Public models stay immutable and serializable** (`frozen=True, slots=True`, read-only arrays,
   `to_dict`/`from_dict`). Downstream code (and the future API/DTO layer) relies on this.
7. **New factor models are thin `LinearFactorModel` subclasses** that specify only their factors +
   metadata + (optionally) a result subclass — never new estimation logic. The estimation engine
   (`OLS`, HAC) is frozen.
8. **New data providers implement `FactorDataPort`** and pass the shared adapter-contract tests;
   parsing stays pure, network I/O stays behind an injected fetcher (no network in tests).
9. **Frozen packages stay frozen.** `factorlab_optimizer`, `factorlab_backtesting`, and the approved
   parts of quant/data/portfolio must not be modified without explicit approval, even to fix a flaky
   test — surface it instead (see §11).
10. **One component at a time, stop for review.** This is the established working agreement; keep it.
11. **The derivatives engine stays standalone.** `factorlab_derivatives` imports nothing internal and
    nothing internal imports it. Keep it dependency-light (numpy/scipy only). Its Black-76 rho sign
    (`−T·price`), per-year theta, annualized-decimal conventions, and continuous-monitoring barrier
    assumption must be preserved for backwards compatibility.

---

*End of handoff. Start by reading each package's `README.md` (they contain architecture diagrams,
math, and runnable examples), then the module docstrings, then the tests — the tests are the
executable specification.*
