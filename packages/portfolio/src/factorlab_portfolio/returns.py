"""ReturnSeries: an immutable, date-aware series of periodic returns.

This is the central analytics object.  It holds per-period simple returns and a
sampling frequency, and exposes the full metric suite (delegating to the pure
functions in :mod:`factorlab_portfolio.analytics`) as convenient methods.  It
never mutates; every derived quantity is computed on demand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from factorlab_portfolio.analytics import (
    active_return,
    annualized_volatility,
    beta,
    cagr,
    calmar_ratio,
    cumulative_return,
    downside_deviation,
    drawdown_series,
    information_ratio,
    max_drawdown,
    max_drawdown_duration,
    mean_return,
    omega_ratio,
    rolling_beta,
    rolling_return,
    rolling_sharpe,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    time_to_recovery,
    tracking_error,
    treynor_ratio,
    wealth_index,
)
from factorlab_portfolio.errors import DimensionMismatchError, PortfolioValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_portfolio.report import PerformanceReport

__all__ = ["ReturnSeries"]

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """Immutable per-period simple returns with a sampling frequency.

    Parameters
    ----------
    values:
        Per-period simple returns in decimal units.  Must be finite.
    dates:
        Optional ``datetime64[D]`` index aligned with ``values``.
    periods_per_year:
        Observations per year, used to annualize (252 daily, 12 monthly, ...).
    name:
        Label for reports.
    """

    values: FloatArray
    dates: DateArray | None = None
    periods_per_year: float = 252.0
    name: str = "series"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise PortfolioValidationError("values must be a 1-D array")
        if not np.all(np.isfinite(values)):
            raise PortfolioValidationError("values contain non-finite entries (NaN/inf)")
        if self.periods_per_year <= 0:
            raise PortfolioValidationError("periods_per_year must be positive")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        if self.dates is not None:
            dates = np.asarray(self.dates, dtype="datetime64[D]")
            if dates.shape[0] != values.shape[0]:
                raise DimensionMismatchError(values.shape[0], dates.shape[0], name="dates")
            dates.setflags(write=False)
            object.__setattr__(self, "dates", dates)

    # ------------------------------------------------------------------ #
    # Basics                                                              #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_observations(self) -> int:
        return int(self.values.shape[0])

    def mean(self) -> float:
        return mean_return(self.values)

    def wealth_index(self, initial: float = 1.0) -> FloatArray:
        return wealth_index(self.values, initial=initial)

    def excess(self, risk_free: float) -> ReturnSeries:
        """Return a new series of excess returns over a per-period risk-free rate."""
        return ReturnSeries(
            self.values - risk_free, self.dates, self.periods_per_year, f"{self.name}-excess"
        )

    # ------------------------------------------------------------------ #
    # Absolute metrics                                                    #
    # ------------------------------------------------------------------ #
    def total_return(self) -> float:
        return cumulative_return(self.values)

    def cagr(self) -> float:
        return cagr(self.values, self.periods_per_year)

    def annualized_return(self) -> float:
        return cagr(self.values, self.periods_per_year)

    def volatility(self) -> float:
        return annualized_volatility(self.values, self.periods_per_year)

    def downside_deviation(self, target: float = 0.0) -> float:
        return downside_deviation(
            self.values, target=target, periods_per_year=self.periods_per_year
        )

    def sharpe(self, risk_free: float = 0.0) -> float:
        return sharpe_ratio(
            self.values, risk_free=risk_free, periods_per_year=self.periods_per_year
        )

    def sortino(self, risk_free: float = 0.0, target: float = 0.0) -> float:
        return sortino_ratio(
            self.values, risk_free=risk_free, target=target, periods_per_year=self.periods_per_year
        )

    def calmar(self) -> float:
        return calmar_ratio(self.values, self.periods_per_year)

    def omega(self, threshold: float = 0.0) -> float:
        return omega_ratio(self.values, threshold=threshold)

    # ------------------------------------------------------------------ #
    # Drawdown                                                            #
    # ------------------------------------------------------------------ #
    def drawdown_series(self) -> FloatArray:
        return drawdown_series(self.values)

    def max_drawdown(self) -> float:
        return max_drawdown(self.values)

    def max_drawdown_duration(self) -> int:
        return max_drawdown_duration(self.values)

    def time_to_recovery(self) -> int | None:
        return time_to_recovery(self.values)

    # ------------------------------------------------------------------ #
    # Benchmark-relative metrics                                          #
    # ------------------------------------------------------------------ #
    def beta(self, benchmark: ReturnSeries) -> float:
        r, b = self.aligned_pair(benchmark)
        return beta(r, b)

    def active_return(self, benchmark: ReturnSeries) -> float:
        r, b = self.aligned_pair(benchmark)
        return active_return(r, b, self.periods_per_year)

    def tracking_error(self, benchmark: ReturnSeries) -> float:
        r, b = self.aligned_pair(benchmark)
        return tracking_error(r, b, self.periods_per_year)

    def information_ratio(self, benchmark: ReturnSeries) -> float:
        r, b = self.aligned_pair(benchmark)
        return information_ratio(r, b, self.periods_per_year)

    def treynor(self, benchmark: ReturnSeries, risk_free: float = 0.0) -> float:
        r, b = self.aligned_pair(benchmark)
        return treynor_ratio(r, b, risk_free=risk_free, periods_per_year=self.periods_per_year)

    # ------------------------------------------------------------------ #
    # Rolling metrics                                                     #
    # ------------------------------------------------------------------ #
    def rolling_return(self, window: int) -> FloatArray:
        return rolling_return(self.values, window)

    def rolling_volatility(self, window: int) -> FloatArray:
        return rolling_volatility(self.values, window, self.periods_per_year)

    def rolling_sharpe(self, window: int, risk_free: float = 0.0) -> FloatArray:
        return rolling_sharpe(self.values, window, risk_free, self.periods_per_year)

    def rolling_beta(self, benchmark: ReturnSeries, window: int) -> FloatArray:
        r, b = self.aligned_pair(benchmark)
        return rolling_beta(r, b, window)

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #
    def performance_report(
        self, benchmark: ReturnSeries | None = None, risk_free: float = 0.0
    ) -> PerformanceReport:
        """Build a :class:`~factorlab_portfolio.report.PerformanceReport`."""
        from factorlab_portfolio.report import PerformanceReport

        return PerformanceReport.from_series(self, benchmark=benchmark, risk_free=risk_free)

    # ------------------------------------------------------------------ #
    # Alignment helper                                                    #
    # ------------------------------------------------------------------ #
    def aligned_pair(self, benchmark: ReturnSeries) -> tuple[FloatArray, FloatArray]:
        """Return ``(self_values, benchmark_values)`` aligned on common dates."""
        if self.dates is not None and benchmark.dates is not None:
            common = np.intersect1d(self.dates, benchmark.dates)
            if common.size == 0:
                raise DimensionMismatchError(self.n_observations, 0, name="benchmark")
            return self.values[_index_of(self.dates, common)], benchmark.values[
                _index_of(benchmark.dates, common)
            ]
        if benchmark.n_observations != self.n_observations:
            raise DimensionMismatchError(
                self.n_observations, benchmark.n_observations, name="benchmark"
            )
        return self.values, benchmark.values

    # ------------------------------------------------------------------ #
    # Construction & serialization                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_prices(
        cls,
        prices: Sequence[float] | FloatArray,
        dates: DateArray | None = None,
        periods_per_year: float = 252.0,
        name: str = "series",
    ) -> ReturnSeries:
        """Construct from a price level series via simple returns."""
        px = np.asarray(prices, dtype=np.float64)
        if px.size < 2:
            raise PortfolioValidationError("need at least two prices to compute returns")
        returns = px[1:] / px[:-1] - 1.0
        ret_dates = None if dates is None else np.asarray(dates, dtype="datetime64[D]")[1:]
        return cls(returns, ret_dates, periods_per_year, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "dates": None
            if self.dates is None
            else [np.datetime_as_string(d, unit="D") for d in self.dates],
            "periods_per_year": self.periods_per_year,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReturnSeries:
        dates = data.get("dates")
        return cls(
            values=np.asarray(data["values"], dtype=np.float64),
            dates=None if dates is None else np.asarray(dates, dtype="datetime64[D]"),
            periods_per_year=float(data.get("periods_per_year", 252.0)),
            name=str(data.get("name", "series")),
        )


def _index_of(dates: DateArray, targets: DateArray) -> NDArray[np.intp]:
    order = np.argsort(dates)
    positions = np.searchsorted(dates[order], targets)
    return order[positions]
