"""Tests for FactorLoader orchestration."""

from __future__ import annotations

import pytest

from factorlab_data.adapters.kenneth_french import KennethFrenchAdapter
from factorlab_data.cache import FactorCache, make_cache_key
from factorlab_data.errors import FactorValidationError
from factorlab_data.loader import FactorLoader
from factorlab_data.registry import FactorRegistry
from factorlab_data.validation import FactorValidator

_FF5 = "F-F_Research_Data_5_Factors_2x3"


def test_load_from_content(ff5_monthly_text) -> None:
    loader = FactorLoader(KennethFrenchAdapter())
    ds = loader.load_from_content(ff5_monthly_text, dataset_id=_FF5, frequency="monthly")
    assert ds.panel("monthly").n_observations == 300


def test_cache_hit_avoids_reparse(ff5_monthly_text) -> None:
    cache = FactorCache()
    loader = FactorLoader(KennethFrenchAdapter(), cache=cache)
    loader.load_from_content(ff5_monthly_text, dataset_id=_FF5, frequency="monthly")
    assert cache.has(make_cache_key(_FF5, "monthly"))
    # Second call returns the cached dataset (identity check).
    first = cache.get(make_cache_key(_FF5, "monthly"))
    second = loader.load_from_content("garbage that would fail parsing", dataset_id=_FF5)
    assert second is first


def test_validator_rejects_bad_panel() -> None:
    # A file whose single factor is entirely missing should fail validation.
    text = "\n".join(["hdr", "", ",Bad", "199001, -99.99", "199002, -99.99", "199003, -99.99"])
    loader = FactorLoader(KennethFrenchAdapter(), validator=FactorValidator())
    with pytest.raises(FactorValidationError):
        loader.load_from_content(text, dataset_id="x", frequency="monthly")


def test_load_panel_and_factor_set(ff5_monthly_text) -> None:
    loader = FactorLoader(KennethFrenchAdapter())
    loader_cache = FactorCache()
    loader = FactorLoader(KennethFrenchAdapter(), cache=loader_cache)
    loader.load_from_content(ff5_monthly_text, dataset_id=_FF5, frequency="monthly")
    # Re-load hits cache; then convert.
    panel = loader.load(_FF5, frequency="monthly").panel("monthly")
    assert panel.factor_names == ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")
    fs = panel.to_factor_set()
    assert fs.names == ("Mkt-RF", "SMB", "HML", "RMW", "CMA")


def test_loader_with_registry_routes_adapter(ff5_monthly_text) -> None:
    reg = FactorRegistry()
    reg.register_adapter(KennethFrenchAdapter())
    loader = FactorLoader(reg)
    ds = loader.load_from_content(ff5_monthly_text, dataset_id=_FF5, frequency="monthly")
    assert ds.dataset_id == _FF5


def test_load_with_fetcher_end_to_end(ff5_monthly_text) -> None:
    def fetcher(url: str) -> bytes:
        return ff5_monthly_text.encode("latin-1")

    loader = FactorLoader(KennethFrenchAdapter(fetcher=fetcher), cache=FactorCache())
    ds = loader.load(_FF5, frequency="monthly")
    assert ds.panel("monthly").n_observations == 300
