"""Tests for FactorCache (memory + disk + TTL)."""

from __future__ import annotations

import time

import numpy as np

from factorlab_data.cache import FactorCache, make_cache_key
from factorlab_data.metadata import FactorMetadata
from factorlab_data.panel import FactorDataset, FactorPanel


def _dataset(dataset_id="ds") -> FactorDataset:
    dates = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[D]")
    meta = FactorMetadata(dataset_id, "n", "u", "monthly", ("A",), n_observations=2)
    panel = FactorPanel(dates, ("A",), np.array([[0.01], [0.02]]), meta)
    return FactorDataset(dataset_id, {"monthly": panel}, meta)


def test_key_is_deterministic() -> None:
    assert make_cache_key("ds", "monthly") == make_cache_key("ds", "monthly")
    assert make_cache_key("ds", "monthly") != make_cache_key("ds", "daily")


def test_memory_get_set_has() -> None:
    cache = FactorCache()
    key = make_cache_key("ds", "monthly")
    assert cache.get(key) is None
    cache.set(key, _dataset())
    assert cache.has(key)
    assert cache.get(key).dataset_id == "ds"


def test_clear() -> None:
    cache = FactorCache()
    cache.set("k", _dataset())
    cache.clear()
    assert cache.get("k") is None
    assert cache.keys() == ()


def test_ttl_expiry() -> None:
    cache = FactorCache(ttl_seconds=0.05)
    cache.set("k", _dataset())
    assert cache.get("k") is not None
    time.sleep(0.06)
    assert cache.get("k") is None


def test_disk_persistence(tmp_path) -> None:
    key = make_cache_key("ds", "monthly")
    cache1 = FactorCache(cache_dir=tmp_path)
    cache1.set(key, _dataset())

    # A fresh cache instance (cold memory) reads from disk.
    cache2 = FactorCache(cache_dir=tmp_path)
    restored = cache2.get(key)
    assert restored is not None
    np.testing.assert_allclose(restored.panel("monthly")["A"], [0.01, 0.02])


def test_disk_clear_removes_files(tmp_path) -> None:
    cache = FactorCache(cache_dir=tmp_path)
    cache.set("k", _dataset())
    assert list(tmp_path.glob("*.pkl"))
    cache.clear()
    assert not list(tmp_path.glob("*.pkl"))
