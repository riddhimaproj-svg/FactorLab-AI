# factorlab-backtesting

**Event-driven backtesting engine.** Typed, dependency-light, cost-aware. Turns a strategy and
a price history into a realistic track record and a full performance report — completing the
platform workflow:

```
Factor Model -> Expected Returns -> Optimizer -> Portfolio -> Backtester -> Performance Report
```

---

## Architecture

```
   MarketData (prices)          Strategy                RebalanceSchedule
   (dates x assets)      (trailing data -> weights)   (daily/weekly/monthly/
        |                        |                      quarterly/custom)
        |                        |  StrategyContext           |
        |                        |  (trailing returns only    |
        |                        |   -> no look-ahead)        |
        +-----------+------------+----------------------------+
                    v
            +-----------------+        ExecutionEngine
            |    Backtest     |------> Order -> BrokerModel -> Fill
            |  (simulation    |            (SlippageModel +
            |   loop)         |             TransactionCostModel)
            +--------+--------+
                     v
              BacktestResult  --------> ReturnSeries  (factorlab_portfolio)
                     |                       |
                     v                       v
              BacktestReport  <-----  PerformanceReport
        (alpha, beta, turnover,   (return, vol, Sharpe, Sortino,
         hit ratio, win rate,      Calmar, IR, Treynor, tracking
         costs) + Benchmark        error, beta, max drawdown)

   WalkForward (rolling / expanding windows) --> stitched out-of-sample ReturnSeries
```

All value objects (`MarketData`, `Order`, `Fill`, `OrderBook`, `BacktestResult`,
`BacktestReport`) are immutable. The engine is **pure**: `ExecutionEngine.rebalance` returns a
new outcome and the loop never mutates shared state.

## Integrations

- **factorlab_portfolio** (approved): `BacktestResult.to_return_series()` yields a
  `ReturnSeries`; `BacktestReport` wraps a `PerformanceReport`.
- **factorlab_optimizer**: `OptimizerStrategy` estimates moments from the trailing window and
  calls any optimizer — the *expected returns -> optimizer -> weights* seam.
- **Factor models**: a factor model's expected-return signal plugs in as the
  `mean_estimator` of `OptimizerStrategy` (a plain `window -> mu` callable), so the full
  factor-model -> optimizer -> backtest chain works without coupling the packages.

---

## Rebalancing

A `RebalanceSchedule` decides *when* to re-weight: `daily`, `weekly`, `monthly`, `quarterly`
(first trading day of each period), or `custom` (an explicit date set or a date predicate).
Rebalancing more often tracks the target more tightly but pays more transaction costs; the
schedule is the main lever on that trade-off, and turnover is reported so its cost is visible.

## Transaction costs

Costs are the main reason paper strategies disappoint live, and they bias a naive backtest
toward high-turnover strategies that look great gross. The engine models them explicitly:

- **Fixed commission** — a flat fee per trade.
- **Percentage commission** — proportional to traded notional.
- **Bid-ask spread** — crossing half the quoted spread on each side.
- **Slippage** — a basis-point gap between the mid price and the fill price (adverse to the
  trade direction).
- **Cash drag** — uninvested cash earns `cash_rate` (0 by default), so a partially-invested
  strategy lags a fully-invested one in a rising market.

`BrokerModel` combines a `SlippageModel` and a `TransactionCostModel`; the resulting `Fill`
carries the realized price and commission, and every cost reduces portfolio value at the
rebalance, exactly as it would live.

## Walk-forward validation

A single backtest over the whole history invites **overfitting** — parameters and researcher
choices get tuned, implicitly or explicitly, to that one path. **Walk-forward** validation
splits the timeline into consecutive folds, each with an in-sample *train* window (history
only) and an out-of-sample *test* window on which the strategy trades and is scored, then
stitches the test-window returns into one continuous out-of-sample track record.

- **Rolling window** — a fixed-length train block precedes each test block (adapts to regime
  change; discards distant history).
- **Expanding window** — the train block grows from the start (uses all history; adapts more
  slowly).

Out-of-sample results are the honest measure of a strategy; they are what `WalkForward.run`
returns.

## Look-ahead bias

**Look-ahead bias** is using information at decision time that would not actually have been
known then (tomorrow's price, a restated fundamental, a full-sample estimate). It is the most
common way a backtest lies. This engine is *structurally* resistant: a strategy only ever sees a
`StrategyContext` whose `trailing_returns` are computed strictly from prices **up to and
including** the rebalance date, and weights take effect from that date forward. There is no API
by which a strategy can read the future.

## Survivorship bias

**Survivorship bias** is testing only the assets that survived to today (delisted, merged, or
bankrupt names silently dropped), which inflates returns and hides risk. The engine itself is
neutral — it backtests exactly the `MarketData` it is given — so avoiding survivorship bias is
a **data** responsibility: supply a point-in-time universe that includes assets *as they existed
historically*, delistings and all. (The factor-data layer's provenance metadata is where that
discipline lives.)

---

## Performance metrics

From `PerformanceReport` (via the portfolio package): annualized return, volatility, Sharpe,
Sortino, Calmar, information ratio, Treynor, tracking error, beta, maximum drawdown.
Added by `BacktestReport`: **Jensen's alpha**, **turnover** (average + annualized),
**hit ratio** (vs benchmark), **win rate**, and realized **transaction costs**.

---

## Quick start

```python
import numpy as np
from factorlab_backtesting import (
    MarketData, RebalanceSchedule, EqualWeightStrategy, OptimizerStrategy,
    Backtest, Benchmark, BrokerModel, ExecutionEngine,
    PercentageCommission, FixedBpsSlippage, WalkForward, rolling_windows,
)

md = MarketData(dates, ("A", "B", "C"), prices)      # dates x assets
broker = BrokerModel(PercentageCommission(0.0005), FixedBpsSlippage(5))
engine = ExecutionEngine(broker)

# Equal-weight, monthly, with costs.
result = Backtest(md, EqualWeightStrategy(), RebalanceSchedule.monthly(), engine).run()
report = result.report(benchmark=Benchmark.from_prices("EW", md.prices.mean(axis=1)))
print(report.summary())

# Optimizer-driven strategy (expected returns -> optimizer -> weights).
from factorlab_optimizer import MinVarianceOptimizer, Constraint
strat = OptimizerStrategy(MinVarianceOptimizer(), lookback=90,
                          constraints=(Constraint.long_only(),))
opt_result = Backtest(md, strat, RebalanceSchedule.monthly(), engine).run()

# Out-of-sample walk-forward.
wf = WalkForward(md, EqualWeightStrategy(), RebalanceSchedule.monthly(), engine)
oos = wf.run(rolling_windows(md.n_periods, train_size=252, test_size=63))
print(oos.performance_report().summary())
```

---

## Testing

```bash
pip install -e ".[dev]"        # plus: pip install -e ../portfolio ../optimizer  (peers)
pytest --cov=factorlab_backtesting     # 99% coverage
ruff check src tests ; mypy src
```

Transaction costs are verified to reduce value, rebalancing to hit targets and conserve value
when frictionless, alpha/beta against a constructed linear relation, rolling/expanding windows
for structure, and buy-and-hold to track its asset exactly — plus property tests (value stays
positive, costs never increase final value) and serialization round-trips.
