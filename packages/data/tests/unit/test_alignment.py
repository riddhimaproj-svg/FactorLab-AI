"""Tests for FactorAlignment."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_data.alignment import FactorAlignment
from factorlab_data.errors import FactorAlignmentError
from factorlab_data.metadata import FactorMetadata
from factorlab_data.panel import FactorPanel


def _panel(dates: np.ndarray, values: np.ndarray, names=("A", "B")) -> FactorPanel:
    meta = FactorMetadata("t", "t", "u", "monthly", tuple(names), n_observations=len(dates))
    return FactorPanel(dates, tuple(names), values, meta)


def test_common_dates_intersection() -> None:
    a = np.array(["2000-01-01", "2000-02-01", "2000-03-01"], dtype="datetime64[D]")
    b = np.array(["2000-02-01", "2000-03-01", "2000-04-01"], dtype="datetime64[D]")
    common = FactorAlignment.common_dates(a, b)
    assert list(common.astype(str)) == ["2000-02-01", "2000-03-01"]


def test_align_panels_restricts_to_common() -> None:
    d1 = np.array(["2000-01-01", "2000-02-01", "2000-03-01"], dtype="datetime64[D]")
    d2 = np.array(["2000-02-01", "2000-03-01", "2000-04-01"], dtype="datetime64[D]")
    p1 = _panel(d1, np.arange(6.0).reshape(3, 2))
    p2 = _panel(d2, np.arange(6.0).reshape(3, 2) + 10)
    a1, a2 = FactorAlignment.align_panels(p1, p2)
    assert a1.n_observations == a2.n_observations == 2
    assert (a1.dates == a2.dates).all()


def test_align_asset_to_panel_inner_join() -> None:
    pdates = np.array(["2000-01-01", "2000-02-01", "2000-03-01"], dtype="datetime64[D]")
    panel = _panel(pdates, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    asset_dates = np.array(["2000-02-01", "2000-03-01", "2000-04-01"], dtype="datetime64[D]")
    asset_values = np.array([0.02, 0.03, 0.04])

    common, aligned_asset, aligned_panel = FactorAlignment.align_asset_to_panel(
        asset_dates, asset_values, panel
    )
    assert list(common.astype(str)) == ["2000-02-01", "2000-03-01"]
    np.testing.assert_allclose(aligned_asset, [0.02, 0.03])
    np.testing.assert_allclose(aligned_panel["A"], [3.0, 5.0])


def test_align_asset_length_mismatch_raises() -> None:
    pdates = np.array(["2000-01-01"], dtype="datetime64[D]")
    panel = _panel(pdates, np.array([[1.0, 2.0]]))
    with pytest.raises(FactorAlignmentError):
        FactorAlignment.align_asset_to_panel(
            np.array(["2000-01-01"], dtype="datetime64[D]"), np.array([1.0, 2.0]), panel
        )


def test_no_common_dates_raises() -> None:
    p1 = _panel(np.array(["2000-01-01"], dtype="datetime64[D]"), np.array([[1.0, 2.0]]))
    p2 = _panel(np.array(["2001-01-01"], dtype="datetime64[D]"), np.array([[1.0, 2.0]]))
    with pytest.raises(FactorAlignmentError):
        FactorAlignment.align_panels(p1, p2)


def test_alignment_preserves_pairing_under_shuffle() -> None:
    """Even if the asset series is unsorted, values stay paired to their dates."""
    pdates = np.array(["2000-01-01", "2000-02-01", "2000-03-01"], dtype="datetime64[D]")
    panel = _panel(pdates, np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    asset_dates = np.array(["2000-03-01", "2000-01-01", "2000-02-01"], dtype="datetime64[D]")
    asset_values = np.array([0.3, 0.1, 0.2])  # paired to the dates above
    _, aligned_asset, aligned_panel = FactorAlignment.align_asset_to_panel(
        asset_dates, asset_values, panel
    )
    # common is sorted; asset values must follow their dates, not their order.
    np.testing.assert_allclose(aligned_asset, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(aligned_panel["A"], [1.0, 2.0, 3.0])
