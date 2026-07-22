"""Immutable instrument and market-data models.

Instruments describe the *contract* (option type, strike, maturity, exercise
style); :class:`MarketData` describes the *state of the world* (spot, rate,
volatility, dividend yield).  Pricing functions accept either explicit floats
(the pure core) or these objects (convenience) — the two are always consistent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from factorlab_derivatives._validation import check_non_negative, check_positive
from factorlab_derivatives.errors import DerivativesInputError

__all__ = [
    "BarrierOption",
    "BarrierType",
    "DigitalKind",
    "DigitalOption",
    "ExerciseStyle",
    "MarketData",
    "Option",
    "OptionType",
]


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> float:
        """+1 for a call, -1 for a put (the ``phi`` in option formulas)."""
        return 1.0 if self is OptionType.CALL else -1.0


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


class BarrierType(StrEnum):
    DOWN_AND_IN = "down_and_in"
    DOWN_AND_OUT = "down_and_out"
    UP_AND_IN = "up_and_in"
    UP_AND_OUT = "up_and_out"

    @property
    def is_knock_in(self) -> bool:
        return self in (BarrierType.DOWN_AND_IN, BarrierType.UP_AND_IN)

    @property
    def is_down(self) -> bool:
        return self in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT)


class DigitalKind(StrEnum):
    CASH_OR_NOTHING = "cash_or_nothing"
    ASSET_OR_NOTHING = "asset_or_nothing"


@dataclass(frozen=True, slots=True)
class Option:
    """A vanilla European or American call/put."""

    option_type: OptionType
    strike: float
    maturity: float
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        check_positive(self.strike, "strike")
        check_non_negative(self.maturity, "maturity")

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    def intrinsic_value(self, spot: float) -> float:
        """Payoff if exercised now at ``spot``."""
        return max(self.option_type.sign * (spot - self.strike), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_type": self.option_type.value,
            "strike": self.strike,
            "maturity": self.maturity,
            "exercise": self.exercise.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Option:
        return cls(
            option_type=OptionType(data["option_type"]),
            strike=float(data["strike"]),
            maturity=float(data["maturity"]),
            exercise=ExerciseStyle(data.get("exercise", "european")),
        )


@dataclass(frozen=True, slots=True)
class DigitalOption:
    """A binary option paying a fixed cash amount, or the asset, if in-the-money."""

    option_type: OptionType
    strike: float
    maturity: float
    payout: float = 1.0
    kind: DigitalKind = DigitalKind.CASH_OR_NOTHING

    def __post_init__(self) -> None:
        check_positive(self.strike, "strike")
        check_non_negative(self.maturity, "maturity")
        check_non_negative(self.payout, "payout")

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_type": self.option_type.value,
            "strike": self.strike,
            "maturity": self.maturity,
            "payout": self.payout,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DigitalOption:
        return cls(
            option_type=OptionType(data["option_type"]),
            strike=float(data["strike"]),
            maturity=float(data["maturity"]),
            payout=float(data.get("payout", 1.0)),
            kind=DigitalKind(data.get("kind", "cash_or_nothing")),
        )


@dataclass(frozen=True, slots=True)
class BarrierOption:
    """A single-barrier knock-in/out option (continuous monitoring, no rebate)."""

    option_type: OptionType
    strike: float
    maturity: float
    barrier: float
    barrier_type: BarrierType

    def __post_init__(self) -> None:
        check_positive(self.strike, "strike")
        check_non_negative(self.maturity, "maturity")
        check_positive(self.barrier, "barrier")

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_type": self.option_type.value,
            "strike": self.strike,
            "maturity": self.maturity,
            "barrier": self.barrier,
            "barrier_type": self.barrier_type.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BarrierOption:
        return cls(
            option_type=OptionType(data["option_type"]),
            strike=float(data["strike"]),
            maturity=float(data["maturity"]),
            barrier=float(data["barrier"]),
            barrier_type=BarrierType(data["barrier_type"]),
        )


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market state: spot, risk-free rate, volatility, and dividend yield."""

    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        check_positive(self.spot, "spot")
        check_non_negative(self.volatility, "volatility")
        if not _finite(self.rate) or not _finite(self.dividend_yield):
            raise DerivativesInputError("rate and dividend_yield must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spot": self.spot,
            "rate": self.rate,
            "volatility": self.volatility,
            "dividend_yield": self.dividend_yield,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MarketData:
        return cls(
            spot=float(data["spot"]),
            rate=float(data["rate"]),
            volatility=float(data["volatility"]),
            dividend_yield=float(data.get("dividend_yield", 0.0)),
        )


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
