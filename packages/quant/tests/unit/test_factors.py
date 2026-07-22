"""Unit tests for the Factor / FactorSet abstraction and its validation."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_quant.core.errors import (
    ConstantFactorError,
    DimensionMismatchError,
    DuplicateFactorError,
    DuplicateObservationError,
    FrequencyMismatchError,
)
from factorlab_quant.models.factors import Factor, FactorSet


@pytest.fixture
def three_factors(rng) -> FactorSet:
    n = 120
    return FactorSet(
        [
            Factor("Mkt-RF", rng.normal(0.005, 0.04, n), frequency="monthly"),
            Factor("SMB", rng.normal(0.001, 0.02, n), frequency="monthly"),
            Factor("HML", rng.normal(0.002, 0.03, n), frequency="monthly"),
        ]
    )


# -- Factor ---------------------------------------------------------------- #
def test_factor_coerces_and_freezes_values() -> None:
    f = Factor("X", [1.0, 2.0, 3.0])
    assert f.values.dtype == np.float64
    with pytest.raises(ValueError):
        f.values[0] = 9.0  # read-only


def test_factor_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Factor("  ", [1.0, 2.0])


def test_factor_label_falls_back_to_name() -> None:
    assert Factor("SMB", [1.0, 2.0]).label == "SMB"
    assert Factor("SMB", [1.0, 2.0], display_name="Small minus Big").label == "Small minus Big"


def test_factor_is_constant() -> None:
    assert Factor("C", [3.0, 3.0, 3.0]).is_constant()
    assert not Factor("V", [1.0, 2.0, 3.0]).is_constant()


# -- FactorSet construction & validation ----------------------------------- #
def test_factorset_rejects_duplicate_names() -> None:
    with pytest.raises(DuplicateFactorError):
        FactorSet([Factor("SMB", [1.0, 2.0]), Factor("SMB", [3.0, 4.0])])


def test_factorset_rejects_unequal_lengths() -> None:
    with pytest.raises(DimensionMismatchError):
        FactorSet([Factor("A", [1.0, 2.0, 3.0]), Factor("B", [1.0, 2.0])])


def test_factorset_rejects_mismatched_frequencies() -> None:
    with pytest.raises(FrequencyMismatchError):
        FactorSet(
            [
                Factor("A", [1.0, 2.0], frequency="daily"),
                Factor("B", [3.0, 4.0], frequency="monthly"),
            ]
        )


def test_factorset_requires_at_least_one_factor() -> None:
    with pytest.raises(ValueError):
        FactorSet([])


# -- Introspection --------------------------------------------------------- #
def test_names_and_shapes(three_factors) -> None:
    assert three_factors.names == ("Mkt-RF", "SMB", "HML")
    assert three_factors.n_factors == 3
    assert three_factors.n_observations == 120
    assert three_factors.frequency == "monthly"


def test_getitem_by_name_and_index(three_factors) -> None:
    assert three_factors["SMB"].name == "SMB"
    assert three_factors[0].name == "Mkt-RF"
    with pytest.raises(KeyError):
        _ = three_factors["MISSING"]


def test_contains(three_factors) -> None:
    assert "HML" in three_factors
    assert "XYZ" not in three_factors


# -- Selection & slicing --------------------------------------------------- #
def test_select_reorders_and_subsets(three_factors) -> None:
    sub = three_factors.select(["HML", "Mkt-RF"])
    assert sub.names == ("HML", "Mkt-RF")


def test_select_missing_raises(three_factors) -> None:
    with pytest.raises(KeyError):
        three_factors.select(["Mkt-RF", "NOPE"])


def test_slice_observations(three_factors) -> None:
    sliced = three_factors.slice_observations(0, 50)
    assert sliced.n_observations == 50
    assert sliced.names == three_factors.names


def test_add_returns_new_set(three_factors) -> None:
    extra = Factor("MOM", np.zeros(120) + np.arange(120), frequency="monthly")
    bigger = three_factors.add(extra)
    assert bigger.n_factors == 4
    assert three_factors.n_factors == 3  # original unchanged


# -- Design matrix --------------------------------------------------------- #
def test_to_design_matrix_with_intercept(three_factors) -> None:
    X, names = three_factors.to_design_matrix(intercept=True, intercept_name="alpha")
    assert X.shape == (120, 4)
    assert names == ("alpha", "Mkt-RF", "SMB", "HML")
    np.testing.assert_allclose(X[:, 0], 1.0)


def test_to_design_matrix_without_intercept(three_factors) -> None:
    X, names = three_factors.to_design_matrix(intercept=False)
    assert X.shape == (120, 3)
    assert names == ("Mkt-RF", "SMB", "HML")


# -- Alignment ------------------------------------------------------------- #
def test_align_drops_incomplete_rows() -> None:
    a = Factor("A", [1.0, np.nan, 3.0, 4.0])
    b = Factor("B", [1.0, 2.0, 3.0, np.nan])
    fs = FactorSet([a, b])
    response = np.array([1.0, 2.0, 3.0, 4.0])
    resp_aligned, aligned = fs.align(response)
    assert aligned.n_observations == 2
    assert resp_aligned.tolist() == [1.0, 3.0]


def test_align_length_mismatch_raises(three_factors) -> None:
    with pytest.raises(DimensionMismatchError):
        three_factors.align(np.ones(50))


# -- Regularity checks ----------------------------------------------------- #
def test_assert_regular_flags_constant_factor() -> None:
    fs = FactorSet([Factor("A", [1.0, 2.0, 3.0]), Factor("C", [5.0, 5.0, 5.0])])
    with pytest.raises(ConstantFactorError):
        fs.assert_regular()


def test_assert_regular_flags_duplicate_columns() -> None:
    fs = FactorSet([Factor("A", [1.0, 2.0, 3.0]), Factor("B", [1.0, 2.0, 3.0])])
    with pytest.raises(DuplicateFactorError):
        fs.assert_regular()


def test_duplicate_observations_detection() -> None:
    fs = FactorSet([Factor("A", [1.0, 1.0, 2.0]), Factor("B", [5.0, 5.0, 6.0])])
    assert fs.has_duplicate_observations()
    with pytest.raises(DuplicateObservationError):
        fs.assert_unique_observations()


def test_from_mapping_and_from_matrix_equivalent(rng) -> None:
    n = 30
    m = rng.normal(size=(n, 2))
    from_matrix = FactorSet.from_matrix(m, ["A", "B"])
    from_mapping = FactorSet.from_mapping({"A": m[:, 0], "B": m[:, 1]})
    np.testing.assert_allclose(from_matrix.matrix(), from_mapping.matrix())


def test_factorset_repr(three_factors) -> None:
    text = repr(three_factors)
    assert "FactorSet" in text and "n_factors=3" in text
