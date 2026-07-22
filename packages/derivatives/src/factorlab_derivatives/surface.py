r"""Volatility surface with bilinear interpolation.

A :class:`VolatilitySurface` holds implied volatilities on a rectangular grid of
strikes x maturities and interpolates (bilinearly) between grid nodes.  Queries
outside the grid are clamped to the nearest edge (flat extrapolation), which is
the conventional, arbitrage-safe default for a discrete surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from factorlab_derivatives._validation import FloatArray
from factorlab_derivatives.errors import DerivativesInputError

__all__ = ["VolatilitySurface"]


@dataclass(frozen=True, slots=True)
class VolatilitySurface:
    """An implied-volatility surface over ``(strike, maturity)`` grid nodes.

    ``vols[i, j]`` is the implied volatility at ``strikes[i]`` and ``maturities[j]``.
    Strikes and maturities must each be strictly increasing.
    """

    strikes: FloatArray
    maturities: FloatArray
    vols: FloatArray

    def __post_init__(self) -> None:
        k, t, v = self.strikes, self.maturities, self.vols
        if k.ndim != 1 or t.ndim != 1:
            raise DerivativesInputError("strikes and maturities must be 1-D")
        if k.size < 1 or t.size < 1:
            raise DerivativesInputError("surface needs >= 1 strike and >= 1 maturity")
        if v.shape != (k.size, t.size):
            raise DerivativesInputError(
                f"vols shape {v.shape} != (n_strikes, n_maturities) "
                f"({k.size}, {t.size})"
            )
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(t)):
            raise DerivativesInputError("strikes/maturities must be finite")
        if k.size > 1 and np.any(np.diff(k) <= 0.0):
            raise DerivativesInputError("strikes must be strictly increasing")
        if t.size > 1 and np.any(np.diff(t) <= 0.0):
            raise DerivativesInputError("maturities must be strictly increasing")
        if np.any(v < 0.0) or not np.all(np.isfinite(v)):
            raise DerivativesInputError("vols must be finite and non-negative")
        for arr in (k, t, v):
            arr.setflags(write=False)

    @classmethod
    def from_grid(
        cls,
        strikes: Sequence[float],
        maturities: Sequence[float],
        vols: Sequence[Sequence[float]],
    ) -> VolatilitySurface:
        """Build a surface from plain Python sequences."""
        return cls(
            strikes=np.asarray(strikes, dtype=np.float64),
            maturities=np.asarray(maturities, dtype=np.float64),
            vols=np.asarray(vols, dtype=np.float64),
        )

    def implied_vol(self, strike: float, maturity: float) -> float:
        """Bilinearly interpolate the implied vol at ``(strike, maturity)``.

        Points outside the grid are clamped to the nearest edge.
        """
        i0, i1, wi = _bracket(self.strikes, strike)
        j0, j1, wj = _bracket(self.maturities, maturity)
        v = self.vols
        top = v[i0, j0] * (1.0 - wj) + v[i0, j1] * wj
        bot = v[i1, j0] * (1.0 - wj) + v[i1, j1] * wj
        return float(top * (1.0 - wi) + bot * wi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strikes": self.strikes.tolist(),
            "maturities": self.maturities.tolist(),
            "vols": self.vols.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VolatilitySurface:
        return cls(
            strikes=np.asarray(data["strikes"], dtype=np.float64),
            maturities=np.asarray(data["maturities"], dtype=np.float64),
            vols=np.asarray(data["vols"], dtype=np.float64),
        )


def _bracket(grid: FloatArray, x: float) -> tuple[int, int, float]:
    """Return ``(lo, hi, weight)`` such that ``x ~ grid[lo]*(1-w) + grid[hi]*w``.

    Clamps to the edges when ``x`` falls outside ``[grid[0], grid[-1]]``.
    """
    n = grid.shape[0]
    if n == 1:
        return 0, 0, 0.0
    if x <= grid[0]:
        return 0, 0, 0.0
    if x >= grid[-1]:
        return n - 1, n - 1, 0.0
    hi = int(np.searchsorted(grid, x, side="right"))
    lo = hi - 1
    span = grid[hi] - grid[lo]
    w = float((x - grid[lo]) / span) if span > 0.0 else 0.0
    return lo, hi, w
