r"""Factor and FactorSet: the data abstraction every linear factor model consumes.

A :class:`Factor` is a single, immutable named return series (the market excess
return, SMB, HML, a momentum spread, ...), enriched with the metadata a research
platform needs to keep provenance: a display name, sampling frequency, source,
and description.

A :class:`FactorSet` is an ordered, validated collection of factors that share
an index.  It is the sole input a model needs in order to build a design matrix,
which decouples models from raw NumPy plumbing: CAPM, FF3, FF5, Carhart, the
q-factor model, and arbitrary user models all differ *only* in which factors
they place in the set.

Design notes
------------
* Both types are immutable and hold read-only arrays.
* Validation is layered: structural checks (duplicate names, length agreement,
  frequency consistency) run at construction; numerical-regularity checks
  (constant or singular factors, duplicate observations) run on demand, after
  alignment, because listwise deletion may legitimately remove offending rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from factorlab_quant.core.errors import (
    ConstantFactorError,
    DuplicateFactorError,
    DuplicateObservationError,
    FrequencyMismatchError,
)
from factorlab_quant.core.types import FloatArray
from factorlab_quant.utils.align import apply_mask, complete_case_mask
from factorlab_quant.utils.validation import as_float_vector

__all__ = ["Factor", "FactorSet"]


@dataclass(frozen=True, slots=True)
class Factor:
    """A single immutable, named factor return series.

    Parameters
    ----------
    name:
        Machine identifier, unique within a :class:`FactorSet` (e.g. ``"Mkt-RF"``).
    values:
        The per-period factor realizations.  Coerced to a read-only ``float64``
        vector.  May contain NaNs; alignment removes incomplete rows later.
    display_name:
        Human-friendly label for reports; defaults to ``name``.
    frequency:
        Sampling frequency tag, e.g. ``"daily"``, ``"monthly"``.  Optional, but
        when present it is checked for consistency across a set.
    source:
        Provenance tag, e.g. ``"Kenneth French Data Library"``.
    description:
        Free-text description for documentation and AI grounding.
    """

    name: str
    values: FloatArray
    display_name: str | None = None
    frequency: str | None = None
    source: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Factor.name must be a non-empty string")
        values = as_float_vector(self.values, name=f"factor {self.name!r}")
        values.setflags(write=False)
        # Bypass frozen to normalize the stored array.
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def label(self) -> str:
        """Display name if set, otherwise the machine name."""
        return self.display_name or self.name

    @property
    def variance(self) -> float:
        """Population variance of the (finite) values; 0.0 for a constant."""
        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            return 0.0
        return float(np.var(finite))

    def is_constant(self, tol: float = 1e-14) -> bool:
        """True when the factor has (near-)zero variance."""
        return self.variance <= tol

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible mapping."""
        return {
            "name": self.name,
            "values": self.values.tolist(),
            "display_name": self.display_name,
            "frequency": self.frequency,
            "source": self.source,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Factor:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            name=str(data["name"]),
            values=np.asarray(data["values"], dtype=np.float64),
            display_name=data.get("display_name"),
            frequency=data.get("frequency"),
            source=data.get("source"),
            description=data.get("description"),
        )

    def metadata(self) -> dict[str, object]:
        """Provenance metadata only (no values), for lightweight embedding."""
        return {
            "name": self.name,
            "display_name": self.label,
            "frequency": self.frequency,
            "source": self.source,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class FactorSet:
    """An ordered, validated collection of aligned :class:`Factor` objects.

    Structural validation (non-empty, unique names, equal lengths, consistent
    frequency) runs at construction.  Numerical-regularity checks are available
    via :meth:`assert_regular` and :meth:`assert_unique_observations` and are
    meant to be called by a model *after* alignment.
    """

    factors: tuple[Factor, ...]

    def __init__(self, factors: Iterable[Factor]) -> None:
        materialized = tuple(factors)
        object.__setattr__(self, "factors", materialized)
        self._validate_structure()

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Sequence[float] | FloatArray],
        *,
        frequency: str | None = None,
        source: str | None = None,
    ) -> FactorSet:
        """Build from an ordered ``{name: values}`` mapping.

        Insertion order of the mapping is preserved as factor order.
        """
        return cls(
            Factor(name=name, values=np.asarray(values, dtype=np.float64),
                   frequency=frequency, source=source)
            for name, values in mapping.items()
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: FloatArray,
        names: Sequence[str],
        *,
        frequency: str | None = None,
        source: str | None = None,
    ) -> FactorSet:
        """Build from an ``n x p`` matrix whose columns are factors."""
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("matrix must be 2-dimensional (n_obs x n_factors)")
        if arr.shape[1] != len(names):
            raise ValueError(
                f"matrix has {arr.shape[1]} columns but {len(names)} names given"
            )
        return cls(
            Factor(name=name, values=arr[:, j], frequency=frequency, source=source)
            for j, name in enumerate(names)
        )

    # ------------------------------------------------------------------ #
    # Structural validation                                              #
    # ------------------------------------------------------------------ #
    def _validate_structure(self) -> None:
        if not self.factors:
            raise ValueError("FactorSet must contain at least one factor")

        names = [f.name for f in self.factors]
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise DuplicateFactorError(tuple(duplicates))

        lengths = {len(f) for f in self.factors}
        if len(lengths) > 1:
            first = self.factors[0]
            for f in self.factors[1:]:
                if len(f) != len(first):
                    from factorlab_quant.core.errors import DimensionMismatchError

                    raise DimensionMismatchError(
                        expected=len(first), received=len(f), name=f"factor {f.name!r}"
                    )

        frequencies = {f.frequency for f in self.factors if f.frequency is not None}
        if len(frequencies) > 1:
            raise FrequencyMismatchError(tuple(sorted(frequencies)))

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.factors)

    def __iter__(self) -> Iterator[Factor]:
        return iter(self.factors)

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def n_observations(self) -> int:
        return len(self.factors[0])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.factors)

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(f.label for f in self.factors)

    @property
    def frequency(self) -> str | None:
        """The common frequency, or ``None`` if unspecified/heterogeneous."""
        frequencies = {f.frequency for f in self.factors if f.frequency is not None}
        return frequencies.pop() if len(frequencies) == 1 else None

    def __getitem__(self, key: str | int) -> Factor:
        """Look up a factor by machine name (str) or position (int)."""
        if isinstance(key, str):
            for f in self.factors:
                if f.name == key:
                    return f
            raise KeyError(f"No factor named {key!r}; available: {list(self.names)}")
        return self.factors[key]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names

    # ------------------------------------------------------------------ #
    # Selection & slicing                                                #
    # ------------------------------------------------------------------ #
    def select(self, names: Sequence[str]) -> FactorSet:
        """Return a new set containing exactly ``names``, in the given order.

        This is how a model enforces its factor specification and column order
        (e.g. FF3 selects ``["Mkt-RF", "SMB", "HML"]``).  Raises ``KeyError`` if
        any requested factor is missing.
        """
        return FactorSet(self[name] for name in names)

    def slice_observations(self, start: int | None, stop: int | None) -> FactorSet:
        """Return a new set with each factor's rows sliced ``[start:stop]``."""
        return FactorSet(
            Factor(
                name=f.name,
                values=f.values[start:stop],
                display_name=f.display_name,
                frequency=f.frequency,
                source=f.source,
                description=f.description,
            )
            for f in self.factors
        )

    def add(self, factor: Factor) -> FactorSet:
        """Return a new set with ``factor`` appended (re-validated)."""
        return FactorSet((*self.factors, factor))

    # ------------------------------------------------------------------ #
    # Numerical conversion                                               #
    # ------------------------------------------------------------------ #
    def matrix(self) -> FloatArray:
        """Stack the factors column-wise into an ``n x p`` matrix."""
        return np.column_stack([f.values for f in self.factors])

    def to_design_matrix(
        self, *, intercept: bool = True, intercept_name: str = "const"
    ) -> tuple[FloatArray, tuple[str, ...]]:
        """Build the regression design matrix and its ordered column names.

        Returns
        -------
        (X, param_names)
            ``X`` is ``n x (p + 1)`` with a leading column of ones when
            ``intercept`` is True, else ``n x p``.  ``param_names`` labels the
            columns, so the intercept (if present) is first.
        """
        factor_matrix = self.matrix()
        if intercept:
            n = factor_matrix.shape[0]
            design = np.column_stack([np.ones(n, dtype=np.float64), factor_matrix])
            names = (intercept_name, *self.names)
        else:
            design = factor_matrix
            names = self.names
        return np.ascontiguousarray(design), names

    # ------------------------------------------------------------------ #
    # Alignment                                                          #
    # ------------------------------------------------------------------ #
    def complete_case_mask(self, response: FloatArray | None = None) -> FloatArray:
        """Boolean mask of rows finite across every factor (and ``response``)."""
        arrays = [f.values for f in self.factors]
        if response is not None:
            arrays.append(np.asarray(response, dtype=np.float64))
        return complete_case_mask(*arrays)

    def align(self, response: FloatArray) -> tuple[FloatArray, FactorSet]:
        """Listwise-delete incomplete rows across ``response`` and all factors.

        Returns the aligned response and a new :class:`FactorSet` restricted to
        the surviving rows.
        """
        response = as_float_vector(response, name="response")
        if response.shape[0] != self.n_observations:
            from factorlab_quant.core.errors import DimensionMismatchError

            raise DimensionMismatchError(
                expected=self.n_observations, received=response.shape[0], name="response"
            )
        mask = self.complete_case_mask(response)
        (response_aligned,) = apply_mask(mask, response)
        aligned_factors = FactorSet(
            Factor(
                name=f.name,
                values=f.values[mask],
                display_name=f.display_name,
                frequency=f.frequency,
                source=f.source,
                description=f.description,
            )
            for f in self.factors
        )
        return response_aligned, aligned_factors

    # ------------------------------------------------------------------ #
    # Numerical-regularity validation                                    #
    # ------------------------------------------------------------------ #
    def assert_regular(self, *, constant_tol: float = 1e-14) -> None:
        """Raise if any factor is constant or two factors are identical.

        Intended to run *after* alignment.  General multicollinearity is caught
        downstream by the estimator's condition-number guard; this method gives
        precise, named errors for the two most common degeneracies.
        """
        for f in self.factors:
            if f.is_constant(tol=constant_tol):
                raise ConstantFactorError(f.name)

        # Exact duplicate columns -> singular design.
        matrix = self.matrix()
        n_factors = matrix.shape[1]
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                if np.allclose(matrix[:, i], matrix[:, j], rtol=0.0, atol=1e-15):
                    raise DuplicateFactorError((self.names[j],))

    def has_duplicate_observations(self, response: FloatArray | None = None) -> bool:
        """True if any full observation row is exactly repeated."""
        matrix = self.matrix()
        if response is not None:
            matrix = np.column_stack([np.asarray(response, dtype=np.float64), matrix])
        unique_rows = np.unique(matrix, axis=0)
        return bool(unique_rows.shape[0] < matrix.shape[0])

    def assert_unique_observations(self, response: FloatArray | None = None) -> None:
        """Raise :class:`DuplicateObservationError` if any row is repeated."""
        matrix = self.matrix()
        if response is not None:
            matrix = np.column_stack([np.asarray(response, dtype=np.float64), matrix])
        n_duplicates = matrix.shape[0] - np.unique(matrix, axis=0).shape[0]
        if n_duplicates > 0:
            raise DuplicateObservationError(int(n_duplicates))

    # ------------------------------------------------------------------ #
    # Serialization & metadata                                           #
    # ------------------------------------------------------------------ #
    def metadata(self) -> list[dict[str, object]]:
        """Per-factor provenance metadata (no values)."""
        return [f.metadata() for f in self.factors]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible mapping."""
        return {"factors": [f.to_dict() for f in self.factors]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorSet:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(Factor.from_dict(d) for d in data["factors"])

    def __repr__(self) -> str:
        return (
            f"FactorSet(n_factors={self.n_factors}, "
            f"n_observations={self.n_observations}, names={list(self.names)})"
        )
