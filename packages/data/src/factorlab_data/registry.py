"""A registry that routes dataset ids to the adapter that can serve them.

The registry decouples *what* data is wanted (a dataset id) from *how* it is
obtained (which provider adapter).  A model or loader asks the registry for a
dataset id; the registry returns the adapter and a descriptor.  Adding a new
provider is a registration, not a code change at any call site (Open/Closed).
"""

from __future__ import annotations

from dataclasses import dataclass

from factorlab_data.errors import DatasetNotFoundError
from factorlab_data.ports import FactorDataPort

__all__ = ["DatasetDescriptor", "FactorRegistry"]


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Routing/description record for a registered dataset."""

    dataset_id: str
    source_name: str
    description: str = ""


class FactorRegistry:
    """Routing table from dataset id -> adapter, plus source-name lookup."""

    def __init__(self) -> None:
        self._adapters: dict[str, FactorDataPort] = {}
        self._dataset_to_source: dict[str, str] = {}
        self._descriptors: dict[str, DatasetDescriptor] = {}

    # ------------------------------------------------------------------ #
    # Registration                                                       #
    # ------------------------------------------------------------------ #
    def register_adapter(
        self, adapter: FactorDataPort, *, descriptions: dict[str, str] | None = None
    ) -> None:
        """Register an adapter and index every dataset it advertises.

        ``descriptions`` optionally provides per-dataset descriptions; otherwise
        an empty description is stored.
        """
        source = adapter.source_name
        self._adapters[source] = adapter
        descriptions = descriptions or {}
        for dataset_id in adapter.available_datasets():
            self._dataset_to_source[dataset_id] = source
            self._descriptors[dataset_id] = DatasetDescriptor(
                dataset_id=dataset_id,
                source_name=source,
                description=descriptions.get(dataset_id, ""),
            )

    def register_dataset(self, descriptor: DatasetDescriptor, adapter: FactorDataPort) -> None:
        """Register a single dataset explicitly against an adapter."""
        self._adapters[adapter.source_name] = adapter
        self._dataset_to_source[descriptor.dataset_id] = adapter.source_name
        self._descriptors[descriptor.dataset_id] = descriptor

    # ------------------------------------------------------------------ #
    # Lookup                                                             #
    # ------------------------------------------------------------------ #
    def list_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self._descriptors))

    def list_sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def describe(self, dataset_id: str) -> DatasetDescriptor:
        if dataset_id not in self._descriptors:
            raise DatasetNotFoundError(dataset_id, self.list_datasets())
        return self._descriptors[dataset_id]

    def adapter_for(self, dataset_id: str) -> FactorDataPort:
        """Return the adapter registered to serve ``dataset_id``."""
        if dataset_id not in self._dataset_to_source:
            raise DatasetNotFoundError(dataset_id, self.list_datasets())
        return self._adapters[self._dataset_to_source[dataset_id]]

    def get_source(self, source_name: str) -> FactorDataPort:
        if source_name not in self._adapters:
            raise DatasetNotFoundError(source_name, self.list_sources())
        return self._adapters[source_name]
