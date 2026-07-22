"""Tests for FactorRegistry."""

from __future__ import annotations

import pytest

from factorlab_data.adapters.kenneth_french import KennethFrenchAdapter
from factorlab_data.errors import DatasetNotFoundError
from factorlab_data.registry import DatasetDescriptor, FactorRegistry


def test_register_adapter_indexes_its_datasets() -> None:
    reg = FactorRegistry()
    adapter = KennethFrenchAdapter()
    reg.register_adapter(adapter)

    assert "F-F_Research_Data_5_Factors_2x3" in reg.list_datasets()
    assert reg.adapter_for("F-F_Research_Data_5_Factors_2x3") is adapter
    assert adapter.source_name in reg.list_sources()


def test_describe_returns_descriptor() -> None:
    reg = FactorRegistry()
    reg.register_adapter(
        KennethFrenchAdapter(), descriptions={"F-F_Momentum_Factor": "Momentum"}
    )
    desc = reg.describe("F-F_Momentum_Factor")
    assert isinstance(desc, DatasetDescriptor)
    assert desc.description == "Momentum"
    assert desc.source_name == "Kenneth French Data Library"


def test_unknown_dataset_lookup_raises() -> None:
    reg = FactorRegistry()
    reg.register_adapter(KennethFrenchAdapter())
    with pytest.raises(DatasetNotFoundError):
        reg.adapter_for("does-not-exist")
    with pytest.raises(DatasetNotFoundError):
        reg.describe("does-not-exist")


def test_register_single_dataset_explicitly() -> None:
    reg = FactorRegistry()
    adapter = KennethFrenchAdapter()
    reg.register_dataset(DatasetDescriptor("custom", adapter.source_name, "desc"), adapter)
    assert reg.adapter_for("custom") is adapter
    assert "custom" in reg.list_datasets()


def test_unknown_source_raises() -> None:
    reg = FactorRegistry()
    with pytest.raises(DatasetNotFoundError):
        reg.get_source("nope")
