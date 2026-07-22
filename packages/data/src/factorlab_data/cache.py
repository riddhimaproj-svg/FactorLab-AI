"""Caching for parsed factor datasets.

Factor library files are large and change at most daily, so re-parsing (or
re-downloading) them on every call is wasteful.  :class:`FactorCache` provides a
two-tier cache -- an in-process dict backed by an optional on-disk store -- keyed
deterministically by dataset id and frequency, with an optional TTL.

The cache stores :class:`FactorDataset` objects.  Disk persistence uses pickle;
because datasets are immutable dataclasses of NumPy arrays, this round-trips
cleanly.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path

from factorlab_data.errors import FactorCacheError
from factorlab_data.panel import FactorDataset

__all__ = ["FactorCache", "make_cache_key"]


def make_cache_key(dataset_id: str, frequency: str) -> str:
    """Deterministic cache key for a (dataset, frequency) pair."""
    return f"{dataset_id}@{frequency}"


class FactorCache:
    """Two-tier (memory + optional disk) cache of parsed datasets.

    Parameters
    ----------
    cache_dir:
        Directory for on-disk persistence.  When ``None`` (default), the cache
        is memory-only and vanishes with the process.
    ttl_seconds:
        Optional time-to-live.  Entries older than this are treated as misses.
    """

    def __init__(
        self, cache_dir: str | Path | None = None, ttl_seconds: float | None = None
    ) -> None:
        self._memory: dict[str, tuple[float, FactorDataset]] = {}
        self._ttl = ttl_seconds
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def get(self, key: str) -> FactorDataset | None:
        """Return the cached dataset for ``key``, or ``None`` on a miss/expiry."""
        entry = self._memory.get(key)
        if entry is not None:
            timestamp, dataset = entry
            if self._is_fresh(timestamp):
                return dataset
            del self._memory[key]

        if self._cache_dir is not None:
            disk_dataset = self._read_disk(key)
            if disk_dataset is not None:
                self._memory[key] = (time.time(), disk_dataset)
                return disk_dataset
        return None

    def set(self, key: str, dataset: FactorDataset) -> None:
        """Store ``dataset`` under ``key`` in memory (and on disk if configured)."""
        self._memory[key] = (time.time(), dataset)
        if self._cache_dir is not None:
            self._write_disk(key, dataset)

    def clear(self) -> None:
        """Evict every entry from memory and disk."""
        self._memory.clear()
        if self._cache_dir is not None:
            for path in self._cache_dir.glob("*.pkl"):
                path.unlink(missing_ok=True)

    def keys(self) -> tuple[str, ...]:
        """In-memory keys currently held."""
        return tuple(self._memory)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _is_fresh(self, timestamp: float) -> bool:
        return self._ttl is None or (time.time() - timestamp) < self._ttl

    def _disk_path(self, key: str) -> Path:
        assert self._cache_dir is not None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self._cache_dir / f"{digest}.pkl"

    def _read_disk(self, key: str) -> FactorDataset | None:
        path = self._disk_path(key)
        if not path.exists():
            return None
        if self._ttl is not None and (time.time() - path.stat().st_mtime) >= self._ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            with path.open("rb") as handle:
                dataset = pickle.load(handle)
        except (OSError, pickle.PickleError) as exc:  # pragma: no cover - defensive
            raise FactorCacheError(f"Failed to read cache entry {path}: {exc}") from exc
        if not isinstance(dataset, FactorDataset):  # pragma: no cover - defensive
            raise FactorCacheError(f"Cache entry {path} is not a FactorDataset.")
        return dataset

    def _write_disk(self, key: str, dataset: FactorDataset) -> None:
        path = self._disk_path(key)
        try:
            with path.open("wb") as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as exc:  # pragma: no cover - defensive
            raise FactorCacheError(f"Failed to write cache entry {path}: {exc}") from exc
