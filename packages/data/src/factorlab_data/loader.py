"""FactorLoader: the high-level orchestration entry point.

The loader is the single object a model or application talks to.  It composes the
other pieces of the layer -- an adapter (or a registry of adapters), a cache, and
a validator -- into one workflow:

    resolve adapter -> (cache hit? return) -> parse/fetch -> validate -> cache -> return

This is where the layer's Dependency-Inversion story pays off: the loader
depends only on the :class:`FactorDataPort` abstraction and pluggable
cache/validator collaborators, so any of them can be swapped without touching
the loader.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from factorlab_data.cache import FactorCache, make_cache_key
from factorlab_data.panel import FactorDataset, FactorPanel
from factorlab_data.ports import FactorDataPort
from factorlab_data.registry import FactorRegistry
from factorlab_data.validation import FactorValidator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorlab_quant.models import FactorSet

__all__ = ["FactorLoader"]


class FactorLoader:
    """Orchestrates adapter + cache + validation to deliver factor datasets.

    Parameters
    ----------
    source:
        Either a single :class:`FactorDataPort` adapter or a
        :class:`FactorRegistry` (which routes each dataset id to its adapter).
    cache:
        Optional :class:`FactorCache`.  When present, parsed datasets are cached
        by ``(dataset_id, frequency)``.
    validator:
        Optional :class:`FactorValidator`.  When present, every loaded panel is
        validated and an invalid panel raises before being returned/cached.
    """

    def __init__(
        self,
        source: FactorDataPort | FactorRegistry,
        *,
        cache: FactorCache | None = None,
        validator: FactorValidator | None = None,
    ) -> None:
        self._source = source
        self._cache = cache
        self._validator = validator

    # ------------------------------------------------------------------ #
    # Adapter resolution                                                 #
    # ------------------------------------------------------------------ #
    def _adapter_for(self, dataset_id: str) -> FactorDataPort:
        if isinstance(self._source, FactorRegistry):
            return self._source.adapter_for(dataset_id)
        return self._source

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #
    def load(self, dataset_id: str, *, frequency: str = "monthly") -> FactorDataset:
        """Load ``dataset_id`` at ``frequency`` (fetching if necessary)."""
        key = make_cache_key(dataset_id, frequency)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        adapter = self._adapter_for(dataset_id)
        dataset = adapter.load(dataset_id, frequency=frequency)
        return self._finalize(dataset, key, frequency)

    def load_from_content(
        self,
        content: str | bytes,
        *,
        dataset_id: str,
        frequency: str = "monthly",
    ) -> FactorDataset:
        """Parse already-obtained ``content`` (offline path; no network)."""
        key = make_cache_key(dataset_id, frequency)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        adapter = self._adapter_for(dataset_id)
        dataset = adapter.parse(content, dataset_id=dataset_id, frequency=frequency)
        return self._finalize(dataset, key, frequency)

    # ------------------------------------------------------------------ #
    # Convenience: straight to a panel / factor set                      #
    # ------------------------------------------------------------------ #
    def load_panel(self, dataset_id: str, *, frequency: str = "monthly") -> FactorPanel:
        """Load a dataset and return the panel for ``frequency``."""
        return self.load(dataset_id, frequency=frequency).panel(frequency)

    def load_factor_set(
        self,
        dataset_id: str,
        *,
        frequency: str = "monthly",
        names: Sequence[str] | None = None,
    ) -> FactorSet:
        """Load a dataset and convert it to a ``factorlab_quant`` FactorSet."""
        return self.load_panel(dataset_id, frequency=frequency).to_factor_set(names)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _finalize(self, dataset: FactorDataset, key: str, frequency: str) -> FactorDataset:
        if self._validator is not None:
            for panel in dataset.panels.values():
                self._validator.assert_valid(panel)
        if self._cache is not None:
            self._cache.set(key, dataset)
        return dataset
