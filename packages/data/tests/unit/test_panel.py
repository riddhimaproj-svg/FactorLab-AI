"""Tests for FactorPanel and FactorDataset."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_data.errors import DatasetNotFoundError, FactorNotFoundError
from factorlab_data.metadata import FactorMetadata
from factorlab_data.panel import FactorDataset, FactorPanel


def _panel(names=("Mkt-RF", "SMB", "RF"), n=12) -> FactorPanel:
    dates = np.array(
        [np.datetime64("2000-01-01") + np.timedelta64(30 * i, "D") for i in range(n)],
        dtype="datetime64[D]",
    )
    values = np.arange(n * len(names), dtype=float).reshape(n, len(names)) / 100.0
    meta = FactorMetadata(
        dataset_id="test", name="Test", source="unit", frequency="monthly",
        factor_names=tuple(names), n_observations=n,
    )
    return FactorPanel(dates, tuple(names), values, meta)


def test_shape_and_access() -> None:
    p = _panel()
    assert len(p) == 12
    assert p.n_factors == 3
    assert "SMB" in p
    np.testing.assert_allclose(p["SMB"], p.values[:, 1])


def test_column_missing_raises() -> None:
    with pytest.raises(FactorNotFoundError):
        _panel().column("HML")


def test_values_are_read_only() -> None:
    p = _panel()
    with pytest.raises(ValueError):
        p.values[0, 0] = 1.0


def test_select_reorders() -> None:
    sub = _panel().select(["RF", "Mkt-RF"])
    assert sub.factor_names == ("RF", "Mkt-RF")
    np.testing.assert_allclose(sub["Mkt-RF"], _panel()["Mkt-RF"])


def test_risk_free_accessor() -> None:
    assert _panel().risk_free is not None
    no_rf = _panel(names=("Mkt-RF", "SMB"))
    assert no_rf.risk_free is None


def test_slice_dates_inclusive() -> None:
    p = _panel(n=12)
    sub = p.slice_dates("2000-01-01", p.dates[3])
    assert sub.n_observations == 4


def test_dropna_removes_nan_rows() -> None:
    p = _panel(n=5)
    vals = p.values.copy()
    vals[2, 0] = np.nan
    meta = p.metadata
    p2 = FactorPanel(p.dates, p.factor_names, vals, meta)
    assert p2.dropna().n_observations == 4


def test_shape_validation() -> None:
    dates = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[D]")
    meta = FactorMetadata("t", "t", "u", "monthly", ("A",))
    with pytest.raises(ValueError):
        FactorPanel(dates, ("A",), np.zeros((3, 1)), meta)  # rows mismatch


def test_to_factor_set_excludes_rf() -> None:
    fs = _panel().to_factor_set()
    assert fs.names == ("Mkt-RF", "SMB")  # RF excluded by default


def test_to_factor_set_explicit_names() -> None:
    fs = _panel().to_factor_set(names=["SMB"])
    assert fs.names == ("SMB",)


def test_panel_roundtrip() -> None:
    p = _panel()
    restored = FactorPanel.from_dict(p.to_dict())
    np.testing.assert_allclose(restored.values, p.values)
    assert restored.factor_names == p.factor_names
    assert (restored.dates == p.dates).all()


def test_dataset_panel_lookup() -> None:
    p = _panel()
    ds = FactorDataset("test", {"monthly": p}, p.metadata)
    assert ds.panel("monthly") is p
    assert ds.panel() is p  # default -> declared frequency
    with pytest.raises(DatasetNotFoundError):
        ds.panel("daily")


def test_dataset_roundtrip() -> None:
    p = _panel()
    ds = FactorDataset("test", {"monthly": p}, p.metadata)
    restored = FactorDataset.from_dict(ds.to_dict())
    assert restored.frequencies == ("monthly",)
    np.testing.assert_allclose(restored.panel("monthly").values, p.values)


def test_dataset_requires_panel() -> None:
    p = _panel()
    with pytest.raises(ValueError):
        FactorDataset("empty", {}, p.metadata)
