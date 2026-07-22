# factorlab-portfolio

**Immutable portfolio models + reusable performance and risk analytics.**

A pure, framework-free library: immutable value objects and pure numerical
analytics. **No optimization, no backtesting, no I/O.** It is the substrate those
future layers will build on.

---

## Architecture

```
                          ┌────────────────────────────┐
   prices / returns  ────▶│        ReturnSeries         │  central analytics object
                          │  (immutable, date-aware)    │  delegates to ↓
                          └──────────────┬──────────────┘
                                         │
        ┌──────────────────────┬─────────┴───────────┬────────────────────┐
        ▼                      ▼                       ▼                    ▼
   analytics/performance   analytics/drawdown   analytics/relative   analytics/rolling
   return, vol, Sharpe,    max DD, DD duration, beta, tracking err,   rolling return,
   Sortino, Calmar, Omega  time-to-recovery     active, IR, Treynor   vol, Sharpe, beta
        └──────────────────────┴───────────┬───────────┴────────────────────┘
                                           ▼
                                  ┌──────────────────┐
                                  │ PerformanceReport │  serializable metric bundle
                                  └──────────────────┘

   Models (immutable, serializable):  Position → Holding → Portfolio → PortfolioSnapshot
                                       Trade (pure state transition: Portfolio.apply_trade)
```

**Design principles**

- **Immutable value objects.** Every model is a frozen dataclass; state changes
  (e.g. `Portfolio.apply_trade`) return a *new* object. History stays
  reconstructable and the models are safe to share across future layers.
- **Pure analytics.** Metrics are free functions over `numpy` arrays of
  per-period returns; `ReturnSeries` is a thin, ergonomic wrapper. This keeps the
  math independently testable and reusable (a backtester or optimizer can call
  the same functions).
- **Graceful degradation.** Dispersion-based metrics return `nan` (not
  exceptions) when undefined (n < 2, zero denominator), so a report never
  crashes on a degenerate series.
- **Plug-in ready.** Optimization consumes `Portfolio` + `ReturnSeries`;
  backtesting produces a sequence of `PortfolioSnapshot`s whose values form a
  `ReturnSeries`. Both slot into this package without changing it.

---

## Models

| Type | Meaning |
|------|---------|
| `Position` | Held `quantity` of a `symbol` at a `price` (short if negative); optional `cost_basis` → `market_value`, `unrealized_pnl`. |
| `Holding` | Portfolio-level view of a position: `market_value` + `weight`. |
| `Trade` | A signed transaction (`+` buy / `−` sell) with `price` and `fees`; exposes `side`, `notional`, `cash_flow`. |
| `Portfolio` | Positions + cash; `total_value`, `weights()`, `holdings()`, pure `apply_trade()`. |
| `PortfolioSnapshot` | Timestamped valuation (total value + weighted holdings). |
| `ReturnSeries` | Date-aware per-period returns; the entry point to all analytics. |
| `PerformanceReport` | Serializable bundle of every headline metric. |

---

## Metric interpretation

Returns are per-period simple returns in decimal; annualization uses
`periods_per_year` (252 daily, 12 monthly, …). Volatility uses the sample stdev
(`ddof=1`).

**Return & growth**
- **Cumulative return** — total compounded gain, `∏(1+rₜ) − 1`.
- **CAGR** — geometric annualized growth rate; the constant annual rate that
  reproduces the terminal wealth.

**Risk**
- **Annualized volatility** — dispersion of returns; higher = wider outcome
  distribution.
- **Downside deviation** — like volatility but counts only shortfalls below a
  target; the "bad" volatility investors actually dislike.
- **Maximum drawdown** — worst peak-to-trough loss; the deepest hole the strategy
  dug (e.g. −0.30 = down 30% from a high-water mark).
- **Drawdown duration** — longest underwater stretch (periods below a prior
  peak); how long recovery took in the worst case.

**Risk-adjusted (higher is better)**
- **Sharpe** — excess return per unit of *total* volatility. Reward per unit of
  overall variability.
- **Sortino** — excess return per unit of *downside* deviation; does not penalize
  upside variability.
- **Calmar** — CAGR ÷ |max drawdown|; growth per unit of worst-case pain.
- **Omega** — probability-weighted gains ÷ losses about a threshold; uses the
  whole return distribution (all moments), not just mean/variance. `>1` favorable.

**Benchmark-relative**
- **Beta** — sensitivity to the benchmark (`Cov/Var`); 1 = moves one-for-one.
- **Active return** — annualized mean out/under-performance vs the benchmark.
- **Tracking error** — volatility of active returns; how far the portfolio strays
  from the benchmark.
- **Information ratio** — active return per unit of tracking error; the "Sharpe of
  active returns" — consistency of outperformance.
- **Treynor** — excess return per unit of *beta* (systematic risk); apt for
  diversified portfolios where idiosyncratic risk is negligible.

**Rolling** variants (`rolling_return/volatility/sharpe/beta`) recompute a metric
over a trailing window so you can see how risk/performance evolve over time.

---

## Quick start

```python
import numpy as np
from factorlab_portfolio import ReturnSeries

fund = ReturnSeries(np.array([...]), periods_per_year=252, name="fund")
bench = ReturnSeries(np.array([...]), periods_per_year=252, name="benchmark")

fund.cagr(); fund.volatility(); fund.sharpe(risk_free=0.0)
fund.max_drawdown(); fund.max_drawdown_duration()
fund.information_ratio(bench); fund.beta(bench)
fund.rolling_sharpe(window=63)

report = fund.performance_report(benchmark=bench, risk_free=0.0)
print(report.summary())
report.to_dict()   # JSON-serializable
```

```python
from factorlab_portfolio import Portfolio, Position, Trade
p = Portfolio([Position("AAPL", 10, 150.0)], cash=1_000.0, as_of="2024-01-01")
p2 = p.apply_trade(Trade("AAPL", 5, 160.0))   # immutable → new portfolio
p2.weights(); p2.holdings()
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=factorlab_portfolio     # target 95%+
ruff check src tests
mypy src
```

Every metric is validated against closed-form analytical examples, plus
property-based invariants and serialization round-trips.
