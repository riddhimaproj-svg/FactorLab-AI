"""Provenance metadata for factor datasets and panels.

Institutional research demands *auditable* data: every number must trace back to
a source, a vintage, a frequency, and a set of transformations.  :class:`FactorMetadata`
travels with every panel so that a downstream regression result can always be
tied to the exact data that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["VALID_FREQUENCIES", "FactorMetadata", "Frequency"]

Frequency = str
VALID_FREQUENCIES: tuple[str, ...] = ("daily", "weekly", "monthly", "annual")


@dataclass(frozen=True, slots=True)
class FactorMetadata:
    """Immutable provenance record for a factor panel.

    Attributes
    ----------
    dataset_id:
        Stable identifier of the source dataset, e.g.
        ``"F-F_Research_Data_5_Factors_2x3"``.
    name:
        Human-readable dataset name.
    source:
        Provider, e.g. ``"Kenneth French Data Library"``.
    frequency:
        One of :data:`VALID_FREQUENCIES`.
    factor_names:
        Ordered factor (column) names present in the panel.
    units:
        Units of the stored values after normalization; ``"decimal"`` once the
        raw percentages have been divided by 100.
    description, currency, provenance_url:
        Optional descriptive fields.
    start, end:
        ISO date strings for the first and last observations, if known.
    n_observations:
        Number of rows in the associated panel.
    transformations:
        Ordered log of transformations applied during parsing/normalization
        (e.g. ``("percent_to_decimal", "missing_sentinel_to_nan")``).
    extra:
        Free-form additional metadata.
    """

    dataset_id: str
    name: str
    source: str
    frequency: Frequency
    factor_names: tuple[str, ...]
    units: str = "decimal"
    description: str | None = None
    currency: str | None = None
    provenance_url: str | None = None
    start: str | None = None
    end: str | None = None
    n_observations: int = 0
    transformations: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency {self.frequency!r} not in {VALID_FREQUENCIES}"
            )

    def with_updates(self, **changes: Any) -> FactorMetadata:
        """Return a copy with ``changes`` applied (metadata is immutable)."""
        from dataclasses import replace

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "frequency": self.frequency,
            "factor_names": list(self.factor_names),
            "units": self.units,
            "description": self.description,
            "currency": self.currency,
            "provenance_url": self.provenance_url,
            "start": self.start,
            "end": self.end,
            "n_observations": self.n_observations,
            "transformations": list(self.transformations),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorMetadata:
        return cls(
            dataset_id=str(data["dataset_id"]),
            name=str(data["name"]),
            source=str(data["source"]),
            frequency=str(data["frequency"]),
            factor_names=tuple(data["factor_names"]),
            units=str(data.get("units", "decimal")),
            description=data.get("description"),
            currency=data.get("currency"),
            provenance_url=data.get("provenance_url"),
            start=data.get("start"),
            end=data.get("end"),
            n_observations=int(data.get("n_observations", 0)),
            transformations=tuple(data.get("transformations", ())),
            extra=dict(data.get("extra", {})),
        )
