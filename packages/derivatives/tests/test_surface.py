"""Volatility surface: exact-node recovery, bilinear interpolation, edge clamping."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_derivatives import DerivativesInputError, VolatilitySurface


@pytest.fixture
def surface() -> VolatilitySurface:
    return VolatilitySurface.from_grid(
        strikes=[90.0, 100.0, 110.0],
        maturities=[0.5, 1.0],
        vols=[[0.25, 0.24], [0.20, 0.19], [0.22, 0.21]],
    )


def test_recovers_grid_nodes_exactly(surface: VolatilitySurface) -> None:
    assert surface.implied_vol(90.0, 0.5) == pytest.approx(0.25)
    assert surface.implied_vol(100.0, 1.0) == pytest.approx(0.19)
    assert surface.implied_vol(110.0, 0.5) == pytest.approx(0.22)


def test_bilinear_midpoint(surface: VolatilitySurface) -> None:
    # centre of the [90,100] x [0.5,1.0] cell = mean of the four corners
    v = surface.implied_vol(95.0, 0.75)
    expected = np.mean([0.25, 0.24, 0.20, 0.19])
    assert v == pytest.approx(expected)


def test_interpolates_along_strike_only(surface: VolatilitySurface) -> None:
    v = surface.implied_vol(95.0, 0.5)
    assert v == pytest.approx((0.25 + 0.20) / 2.0)


def test_edge_clamping(surface: VolatilitySurface) -> None:
    assert surface.implied_vol(50.0, 0.75) == surface.implied_vol(90.0, 0.75)
    assert surface.implied_vol(200.0, 0.75) == surface.implied_vol(110.0, 0.75)
    assert surface.implied_vol(95.0, 0.1) == surface.implied_vol(95.0, 0.5)
    assert surface.implied_vol(95.0, 5.0) == surface.implied_vol(95.0, 1.0)


def test_single_maturity_column() -> None:
    s = VolatilitySurface.from_grid([90.0, 110.0], [1.0], [[0.2], [0.3]])
    assert s.implied_vol(100.0, 1.0) == pytest.approx(0.25)
    assert s.implied_vol(100.0, 99.0) == pytest.approx(0.25)


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(DerivativesInputError):
        VolatilitySurface.from_grid([90, 100], [1.0], [[0.2]])


def test_rejects_unsorted_strikes() -> None:
    with pytest.raises(DerivativesInputError):
        VolatilitySurface.from_grid([100, 90], [1.0], [[0.2], [0.3]])


def test_rejects_unsorted_maturities() -> None:
    with pytest.raises(DerivativesInputError):
        VolatilitySurface.from_grid([90.0], [1.0, 0.5], [[0.2, 0.3]])


def test_rejects_negative_vols() -> None:
    with pytest.raises(DerivativesInputError):
        VolatilitySurface.from_grid([90.0], [1.0], [[-0.2]])


def test_rejects_non_1d_axes() -> None:
    with pytest.raises(DerivativesInputError):
        VolatilitySurface(
            strikes=np.array([[90.0]]), maturities=np.array([1.0]), vols=np.array([[0.2]])
        )


def test_rejects_empty_axis() -> None:
    with pytest.raises(DerivativesInputError):
        VolatilitySurface(
            strikes=np.array([]), maturities=np.array([1.0]),
            vols=np.empty((0, 1)),
        )


def test_rejects_non_finite_axis() -> None:
    with pytest.raises(DerivativesInputError):
        VolatilitySurface.from_grid([90.0, np.inf], [1.0], [[0.2], [0.3]])


def test_serialization_round_trip(surface: VolatilitySurface) -> None:
    restored = VolatilitySurface.from_dict(surface.to_dict())
    assert np.allclose(restored.vols, surface.vols)
    assert np.allclose(restored.strikes, surface.strikes)
