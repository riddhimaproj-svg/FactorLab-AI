"""Immutable factor panels and datasets.

A :class:`FactorPanel` is the workhorse container: a date-indexed matrix of
normalized factor returns plus its :class:`FactorMetadata`.  It is deliberately
model-agnostic -- it does not know about CAPM, FF3, or FF5 -- and can emit a
``factorlab_quant`` ``FactorSet`` on demand so any model can consume it.

A :class:`FactorDataset` groups the panels that come out of a single source file
(e.g. a Kenneth French download often contains both a monthly and an annual
panel) under one metadata umbrella.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from factorlab_data.errors import DatasetNotFoundError, FactorNotFoundError
from factorlab_data.metadata import FactorMetadata

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_quant.models import FactorSet

__all__ = ["FactorDataset", "FactorPanel"]

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]

# Risk-free column is not a priced factor; excluded from factor sets by default.
_DEFAULT_EXCLUDE = ("RF",)


@dataclass(frozen=True, slots=True)
class FactorPanel:
    """A date-indexed matrix of normalized factor returns.

    Parameters
    ----------
    dates:
        ``datetime64[D]`` array, strictly increasing, length ``n``.
    factor_names:
        Ordered column names, length ``k``.
    values:
        ``n x k`` ``float64`` matrix of returns in decimal units (NaN allowed
        for missing observations).
    metadata:
        Provenance for the panel.
    """

    dates: DateArray
    factor_names: tuple[str, ...]
    values: FloatArray
    metadata: FactorMetadata

    def __post_init__(self) -> None:
        dates = np.asarray(self.dates, dtype="datetime64[D]")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("values must be a 2-D (n_obs x n_factors) array")
        if values.shape[0] != dates.shape[0]:
            raise ValueError("dates and values disagree on the number of rows")
        if values.shape[1] != len(self.factor_names):
            raise ValueError("values columns must match factor_names length")
        dates.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "values", values)

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.dates.shape[0])

    @property
    def n_observations(self) -> int:
        return int(self.dates.shape[0])

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)

    @property
    def frequency(self) -> str:
        return self.metadata.frequency

    def has_factor(self, name: str) -> bool:
        return name in self.factor_names

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.factor_names

    def column(self, name: str) -> FloatArray:
        """Return the (read-only) column for factor ``name``."""
        try:
            idx = self.factor_names.index(name)
        except ValueError:
            raise FactorNotFoundError(name, self.factor_names) from None
        return self.values[:, idx]

    def __getitem__(self, name: str) -> FloatArray:
        return self.column(name)

    @property
    def risk_free(self) -> FloatArray | None:
        """The ``RF`` column if present (the risk-free rate), else ``None``."""
        return self.column("RF") if "RF" in self.factor_names else None

    # ------------------------------------------------------------------ #
    # Selection & slicing                                                 #
    # ------------------------------------------------------------------ #
    def select(self, names: Sequence[str]) -> FactorPanel:
        """Return a panel with exactly ``names`` (in the requested order)."""
        indices = []
        for name in names:
            if name not in self.factor_names:
                raise FactorNotFoundError(name, self.factor_names)
            indices.append(self.factor_names.index(name))
        sub_values = self.values[:, indices]
        meta = self.metadata.with_updates(factor_names=tuple(names))
        return FactorPanel(self.dates, tuple(names), sub_values, meta)

    def slice_dates(
        self, start: str | np.datetime64 | None, end: str | np.datetime64 | None
    ) -> FactorPanel:
        """Return a panel restricted to ``[start, end]`` (inclusive)."""
        mask = np.ones(self.n_observations, dtype=bool)
        if start is not None:
            mask &= self.dates >= np.asarray(start, dtype="datetime64[D]")
        if end is not None:
            mask &= self.dates <= np.asarray(end, dtype="datetime64[D]")
        return self._masked(mask)

    def dropna(self) -> FactorPanel:
        """Return a panel with rows containing any NaN removed (listwise)."""
        mask = np.all(np.isfinite(self.values), axis=1)
        return self._masked(mask)

    def _masked(self, mask: NDArray[np.bool_]) -> FactorPanel:
        meta = self.metadata.with_updates(n_observations=int(mask.sum()))
        return FactorPanel(self.dates[mask], self.factor_names, self.values[mask], meta)

    # ------------------------------------------------------------------ #
    # Bridge to the quant engine                                          #
    # ------------------------------------------------------------------ #
    def to_factor_set(
        self,
        names: Sequence[str] | None = None,
        *,
        exclude: Sequence[str] = _DEFAULT_EXCLUDE,
    ) -> FactorSet:
        """Convert to a ``factorlab_quant`` :class:`FactorSet`.

        This is the bridge that lets any quant model consume the data layer.
        ``RF`` is excluded by default since it is the risk-free rate, not a
        priced factor.  Imported lazily so the quant engine stays an *optional*
        peer dependency.
        """
        from factorlab_quant.models import Factor, FactorSet

        chosen = list(names) if names is not None else [
            n for n in self.factor_names if n not in set(exclude)
        ]
        factors = [
            Factor(
                name=name,
                values=self.column(name),
                frequency=self.frequency,
                source=self.metadata.source,
                description=f"{name} from {self.metadata.dataset_id}",
            )
            for name in chosen
        ]
        return FactorSet(factors)

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "dates": [np.datetime_as_string(d, unit="D") for d in self.dates],
            "factor_names": list(self.factor_names),
            "values": self.values.tolist(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorPanel:
        return cls(
            dates=np.array(data["dates"], dtype="datetime64[D]"),
            factor_names=tuple(data["factor_names"]),
            values=np.asarray(data["values"], dtype=np.float64),
            metadata=FactorMetadata.from_dict(data["metadata"]),
        )

    def __repr__(self) -> str:
        span = ""
        if self.n_observations:
            span = (
                f", {np.datetime_as_string(self.dates[0], unit='D')}"
                f"..{np.datetime_as_string(self.dates[-1], unit='D')}"
            )
        return (
            f"FactorPanel({self.metadata.dataset_id!r}, {self.frequency}, "
            f"n={self.n_observations}, factors={list(self.factor_names)}{span})"
        )


@dataclass(frozen=True, slots=True)
class FactorDataset:
    """A named collection of panels (by frequency) from one source.

    A single Kenneth French file typically yields more than one panel (e.g. a
    monthly panel and an annual panel); this groups them under one identity.
    """

    dataset_id: str
    panels: dict[str, FactorPanel]
    metadata: FactorMetadata

    def __post_init__(self) -> None:
        if not self.panels:
            raise ValueError("FactorDataset must contain at least one panel")

    @property
    def frequencies(self) -> tuple[str, ...]:
        return tuple(self.panels.keys())

    def panel(self, frequency: str | None = None) -> FactorPanel:
        """Return the panel for ``frequency`` (or the sole/primary panel)."""
        if frequency is None:
            # Prefer the dataset's declared frequency, else the first panel.
            frequency = (
                self.metadata.frequency
                if self.metadata.frequency in self.panels
                else next(iter(self.panels))
            )
        if frequency not in self.panels:
            raise DatasetNotFoundError(
                f"{self.dataset_id}[{frequency}]", tuple(self.panels)
            )
        return self.panels[frequency]

    def to_factor_set(
        self, frequency: str | None = None, names: Sequence[str] | None = None
    ) -> FactorSet:
        return self.panel(frequency).to_factor_set(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "panels": {freq: p.to_dict() for freq, p in self.panels.items()},
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorDataset:
        panels = {
            freq: FactorPanel.from_dict(p) for freq, p in data["panels"].items()
        }
        return cls(
            dataset_id=str(data["dataset_id"]),
            panels=panels,
            metadata=FactorMetadata.from_dict(data["metadata"]),
        )

    def __repr__(self) -> str:
        return (
            f"FactorDataset({self.dataset_id!r}, "
            f"frequencies={list(self.frequencies)})"
        )
