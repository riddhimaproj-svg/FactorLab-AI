"""Immutable, serializable result models for the derivatives engine.

These are the public "currency" of the engine: a pricing or analytics call returns
one of these frozen dataclasses, which serialize cleanly (``to_dict`` /
``from_dict``) and render a human-readable ``summary``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Greeks", "ImpliedVolatilityResult", "MonteCarloResult", "PricingResult"]


@dataclass(frozen=True, slots=True)
class Greeks:
    """The five first/second-order option sensitivities (raw partial derivatives).

    * ``delta`` -- dV/dS (per unit of underlying)
    * ``gamma`` -- d2V/dS2
    * ``vega``  -- dV/dsigma (a 1% vol move is ``vega * 0.01``)
    * ``theta`` -- dV/dt per **year** (one calendar day is roughly ``theta / 365``)
    * ``rho``   -- dV/dr (a 1% rate move is ``rho * 0.01``)
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def to_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> Greeks:
        return cls(
            delta=float(data["delta"]),
            gamma=float(data["gamma"]),
            vega=float(data["vega"]),
            theta=float(data["theta"]),
            rho=float(data["rho"]),
        )


@dataclass(frozen=True, slots=True)
class PricingResult:
    """A priced option: value, the method used, and (optionally) its Greeks."""

    price: float
    method: str
    option_type: str
    greeks: Greeks | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "method": self.method,
            "option_type": self.option_type,
            "greeks": self.greeks.to_dict() if self.greeks is not None else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PricingResult:
        g = data.get("greeks")
        return cls(
            price=float(data["price"]),
            method=str(data["method"]),
            option_type=str(data["option_type"]),
            greeks=None if g is None else Greeks.from_dict(g),
            metadata=dict(data.get("metadata", {})),
        )

    def summary(self) -> str:
        lines = [
            f"{self.option_type.capitalize()} option ({self.method})",
            f"  Price: {self.price:.6f}",
        ]
        if self.greeks is not None:
            g = self.greeks
            lines.append(
                f"  Delta {g.delta:+.4f}  Gamma {g.gamma:+.4f}  Vega {g.vega:+.4f}  "
                f"Theta {g.theta:+.4f}  Rho {g.rho:+.4f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """A Monte Carlo price with its sampling error and variance-reduction method."""

    price: float
    standard_error: float
    n_paths: int
    method: str

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% confidence interval for the price estimate."""
        half = 1.959963984540054 * self.standard_error
        return (self.price - half, self.price + half)

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "standard_error": self.standard_error,
            "n_paths": self.n_paths,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MonteCarloResult:
        return cls(
            price=float(data["price"]),
            standard_error=float(data["standard_error"]),
            n_paths=int(data["n_paths"]),
            method=str(data["method"]),
        )

    def summary(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"Monte Carlo ({self.method}, {self.n_paths:,} paths): "
            f"{self.price:.6f} +/- {self.standard_error:.6f}  95% CI [{lo:.6f}, {hi:.6f}]"
        )


@dataclass(frozen=True, slots=True)
class ImpliedVolatilityResult:
    """The result of an implied-volatility solve."""

    implied_volatility: float
    converged: bool
    iterations: int
    method: str
    target_price: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "implied_volatility": self.implied_volatility,
            "converged": self.converged,
            "iterations": self.iterations,
            "method": self.method,
            "target_price": self.target_price,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImpliedVolatilityResult:
        return cls(
            implied_volatility=float(data["implied_volatility"]),
            converged=bool(data["converged"]),
            iterations=int(data["iterations"]),
            method=str(data["method"]),
            target_price=float(data["target_price"]),
        )
