"""FactorLab Institutional Derivatives Engine.

A pure, fully-typed computational core for option pricing, Greeks, implied and
realized volatility, a volatility surface, and Monte Carlo pricing.  No I/O, no
network, no global state — every public model is an immutable, serializable
dataclass.

Quick start
-----------
>>> from factorlab_derivatives import Option, OptionType, MarketData, price_option
>>> option = Option(OptionType.CALL, strike=100.0, maturity=1.0)
>>> market = MarketData(spot=100.0, rate=0.05, volatility=0.2)
>>> result = price_option(option, market)
>>> round(result.price, 4)
10.4506
"""

from __future__ import annotations

from factorlab_derivatives.engine import PricingMethod, price_option
from factorlab_derivatives.errors import (
    ConvergenceError,
    DerivativesError,
    DerivativesInputError,
    NoArbitrageError,
)
from factorlab_derivatives.greeks import finite_difference_greeks
from factorlab_derivatives.implied_vol import implied_volatility
from factorlab_derivatives.instruments import (
    BarrierOption,
    BarrierType,
    DigitalKind,
    DigitalOption,
    ExerciseStyle,
    MarketData,
    Option,
    OptionType,
)
from factorlab_derivatives.monte_carlo import monte_carlo_european
from factorlab_derivatives.pricing import (
    barrier_price,
    binomial_price,
    black76_greeks,
    black76_price,
    black_scholes_greeks,
    black_scholes_price,
    d1_d2,
    digital_price,
)
from factorlab_derivatives.reports import (
    Greeks,
    ImpliedVolatilityResult,
    MonteCarloResult,
    PricingResult,
)
from factorlab_derivatives.surface import VolatilitySurface
from factorlab_derivatives.volatility import (
    GarchResult,
    ewma_variance,
    ewma_volatility,
    fit_garch,
    historical_volatility,
)

__version__ = "0.1.0"

__all__ = [
    "BarrierOption",
    "BarrierType",
    "ConvergenceError",
    "DerivativesError",
    "DerivativesInputError",
    "DigitalKind",
    "DigitalOption",
    "ExerciseStyle",
    "GarchResult",
    "Greeks",
    "ImpliedVolatilityResult",
    "MarketData",
    "MonteCarloResult",
    "NoArbitrageError",
    "Option",
    "OptionType",
    "PricingMethod",
    "PricingResult",
    "VolatilitySurface",
    "__version__",
    "barrier_price",
    "binomial_price",
    "black76_greeks",
    "black76_price",
    "black_scholes_greeks",
    "black_scholes_price",
    "d1_d2",
    "digital_price",
    "ewma_variance",
    "ewma_volatility",
    "finite_difference_greeks",
    "fit_garch",
    "historical_volatility",
    "implied_volatility",
    "monte_carlo_european",
    "price_option",
]
