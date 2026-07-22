"""Tests for FactorMetadata."""

from __future__ import annotations

import pytest

from factorlab_data.metadata import FactorMetadata


def _meta(**kw) -> FactorMetadata:
    base = {
        "dataset_id": "F-F_Research_Data_5_Factors_2x3",
        "name": "FF5",
        "source": "Kenneth French Data Library",
        "frequency": "monthly",
        "factor_names": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    }
    base.update(kw)
    return FactorMetadata(**base)


def test_valid_frequency_required() -> None:
    with pytest.raises(ValueError):
        _meta(frequency="hourly")


def test_with_updates_is_immutable_copy() -> None:
    m = _meta()
    m2 = m.with_updates(n_observations=100)
    assert m.n_observations == 0
    assert m2.n_observations == 100
    assert m2.dataset_id == m.dataset_id


def test_roundtrip() -> None:
    m = _meta(
        units="decimal",
        transformations=("percent_to_decimal",),
        start="1963-07-01",
        end="2020-12-01",
        n_observations=690,
        extra={"note": "test"},
    )
    restored = FactorMetadata.from_dict(m.to_dict())
    assert restored == m


def test_defaults() -> None:
    m = _meta()
    assert m.units == "decimal"
    assert m.transformations == ()
    assert m.extra == {}
