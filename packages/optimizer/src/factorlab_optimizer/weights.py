"""PortfolioWeights: an immutable, named weight vector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from factorlab_optimizer.errors import OptimizationInputError

__all__ = ["PortfolioWeights"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PortfolioWeights:
    """Immutable portfolio weights indexed by asset name.

    Weights are stored as decimals.  They need not sum to 1 (a partially
    invested or levered portfolio is representable); helpers expose the gross,
    net, long, short, and cash exposures.
    """

    assets: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise OptimizationInputError("weights must be a 1-D array")
        if values.shape[0] != len(self.assets):
            raise OptimizationInputError("assets and weights length mismatch")
        if len(set(self.assets)) != len(self.assets):
            raise OptimizationInputError("duplicate asset names")
        if not np.all(np.isfinite(values)):
            raise OptimizationInputError("weights contain non-finite values")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return len(self.assets)

    def get(self, asset: str) -> float:
        try:
            return float(self.values[self.assets.index(asset)])
        except ValueError:
            raise KeyError(f"No weight for asset {asset!r}") from None

    def as_dict(self) -> dict[str, float]:
        return {a: float(w) for a, w in zip(self.assets, self.values, strict=True)}

    @property
    def total(self) -> float:
        """Sum of weights (1.0 for a fully-invested, unlevered portfolio)."""
        return float(np.sum(self.values))

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute weights (leverage)."""
        return float(np.sum(np.abs(self.values)))

    @property
    def net_exposure(self) -> float:
        return self.total

    @property
    def long_exposure(self) -> float:
        return float(np.sum(self.values[self.values > 0.0]))

    @property
    def short_exposure(self) -> float:
        return float(np.sum(self.values[self.values < 0.0]))

    @property
    def cash(self) -> float:
        """Residual cash weight, ``1 - sum(weights)`` (negative if levered)."""
        return float(1.0 - self.total)

    def nonzero(self, tol: float = 1e-8) -> dict[str, float]:
        """Weights whose magnitude exceeds ``tol``."""
        return {a: float(w) for a, w in self.as_dict().items() if abs(w) > tol}

    def to_dict(self) -> dict[str, Any]:
        return {"assets": list(self.assets), "values": self.values.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PortfolioWeights:
        return cls(tuple(data["assets"]), np.asarray(data["values"], dtype=np.float64))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> PortfolioWeights:
        return cls(tuple(mapping.keys()), np.array(list(mapping.values()), dtype=np.float64))

    @classmethod
    def equal_weight(cls, assets: Sequence[str]) -> PortfolioWeights:
        n = len(assets)
        return cls(tuple(assets), np.full(n, 1.0 / n))
