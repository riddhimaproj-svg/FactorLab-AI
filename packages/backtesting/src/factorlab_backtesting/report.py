"""BacktestReport: performance + strategy diagnostics for a backtest.

Wraps the portfolio package's :class:`~factorlab_portfolio.PerformanceReport`
(return, volatility, Sharpe, Sortino, Calmar, information ratio, Treynor, tracking
error, beta, max drawdown) and adds strategy-specific figures: Jensen's alpha,
turnover, hit ratio, win rate, and realized transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from factorlab_backtesting import metrics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_portfolio import PerformanceReport

    from factorlab_backtesting.backtest import BacktestResult
    from factorlab_backtesting.benchmark import Benchmark

__all__ = ["BacktestReport"]


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """A backtest's full performance and diagnostic summary."""

    strategy_name: str
    performance: PerformanceReport
    alpha: float
    beta: float
    average_turnover: float
    annualized_turnover: float
    hit_ratio: float
    win_rate: float
    total_costs: float
    n_rebalances: int
    initial_value: float
    final_value: float
    total_return: float

    @classmethod
    def from_result(
        cls,
        result: BacktestResult,
        benchmark: Benchmark | None = None,
        risk_free: float = 0.0,
    ) -> BacktestReport:
        from factorlab_portfolio import PerformanceReport

        series = result.to_return_series()
        ppy = result.periods_per_year
        bench_series = None
        alpha = beta = hit = float("nan")

        if benchmark is not None:
            bench_returns = _align(benchmark.returns, result.returns.shape[0])
            bench_series = benchmark.to_return_series(ppy)
            # rebuild bench series at the aligned length
            from factorlab_portfolio import ReturnSeries

            bench_series = ReturnSeries(bench_returns, periods_per_year=ppy, name=benchmark.name)
            alpha, beta = metrics.alpha_beta(result.returns, bench_returns, risk_free, ppy)
            hit = metrics.hit_ratio(result.returns, bench_returns)

        performance = PerformanceReport.from_series(
            series, benchmark=bench_series, risk_free=risk_free
        )

        n_return_periods = result.returns.shape[0]
        years = n_return_periods / ppy if ppy > 0 else float("nan")
        rebalances_per_year = result.n_rebalances / years if years and years > 0 else 0.0

        return cls(
            strategy_name=result.strategy_name,
            performance=performance,
            alpha=alpha,
            beta=beta,
            average_turnover=result.average_turnover,
            annualized_turnover=metrics.annualized_turnover(
                result.average_turnover, rebalances_per_year
            ),
            hit_ratio=hit,
            win_rate=metrics.win_rate(result.returns),
            total_costs=result.total_costs,
            n_rebalances=result.n_rebalances,
            initial_value=result.initial_value,
            final_value=result.final_value,
            total_return=result.final_value / result.initial_value - 1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "performance": self.performance.to_dict(),
            "alpha": self.alpha,
            "beta": self.beta,
            "average_turnover": self.average_turnover,
            "annualized_turnover": self.annualized_turnover,
            "hit_ratio": self.hit_ratio,
            "win_rate": self.win_rate,
            "total_costs": self.total_costs,
            "n_rebalances": self.n_rebalances,
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_return": self.total_return,
        }

    def summary(self) -> str:
        p = self.performance
        lines = [
            "=" * 62,
            f"Backtest Report — {self.strategy_name}",
            "=" * 62,
            f"Total return:        {self.total_return:>12.4%}   "
            f"CAGR: {p.cagr:>10.4%}",
            f"Volatility:          {p.annualized_volatility:>12.4%}   "
            f"MaxDD: {p.max_drawdown:>9.4%}",
            f"Sharpe:              {p.sharpe_ratio:>12.4f}   "
            f"Sortino: {p.sortino_ratio:>8.4f}",
            f"Calmar:              {p.calmar_ratio:>12.4f}",
            "-" * 62,
            "Benchmark-relative",
            f"  Alpha (ann.):      {self.alpha:>12.4%}   Beta: {self.beta:>10.4f}",
            f"  Information ratio: {p.information_ratio:>12.4f}   "
            f"Treynor: {p.treynor_ratio:>8.4f}",
            f"  Tracking error:    {p.tracking_error:>12.4%}",
            f"  Hit ratio:         {self.hit_ratio:>12.4%}   "
            f"Win rate: {self.win_rate:>8.4%}",
            "-" * 62,
            "Trading",
            f"  Rebalances:        {self.n_rebalances:>12d}",
            f"  Avg turnover:      {self.average_turnover:>12.4%}   "
            f"Annualized: {self.annualized_turnover:>8.4%}",
            f"  Total costs:       {self.total_costs:>12.2f}",
            f"  Final value:       {self.final_value:>12.2f}",
            "=" * 62,
        ]
        return "\n".join(lines)


def _align(benchmark_returns: Any, length: int) -> Any:
    arr = np.asarray(benchmark_returns, dtype=np.float64)
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] > length:
        return arr[-length:]  # align to the tail
    from factorlab_backtesting.errors import BacktestInputError

    raise BacktestInputError(
        f"benchmark has {arr.shape[0]} returns but backtest produced {length}"
    )
