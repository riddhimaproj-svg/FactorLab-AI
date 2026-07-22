"""FactorLab data layer — reusable institutional factor-data infrastructure.

A framework-independent, provider-agnostic layer for acquiring, parsing,
validating, aligning, and caching factor return data.  Its output -- immutable
:class:`FactorPanel` / :class:`FactorDataset` objects with full provenance -- can
be handed to any ``factorlab_quant`` model via :meth:`FactorPanel.to_factor_set`.

The layer is organized around Ports-and-Adapters:

* :class:`FactorDataPort` -- the interface every provider adapter implements.
* :class:`KennethFrenchAdapter` -- the first concrete adapter (FF3, FF5,
  momentum, research portfolios; daily and monthly).
* :class:`FactorPanel` / :class:`FactorDataset` / :class:`FactorMetadata` -- the
  immutable data model.
* :class:`FactorValidator`, :class:`FactorAlignment`, :class:`FactorCache`,
  :class:`FactorRegistry`, :class:`FactorLoader` -- the supporting infrastructure.

Quick start (offline, from already-downloaded content)::

    from factorlab_data import FactorLoader, KennethFrenchAdapter

    loader = FactorLoader(KennethFrenchAdapter())
    dataset = loader.load_from_content(
        csv_text, dataset_id="F-F_Research_Data_5_Factors_2x3", frequency="monthly"
    )
    factor_set = dataset.panel("monthly").to_factor_set()   # feed to a quant model
"""

from __future__ import annotations

from factorlab_data.adapters import KennethFrenchAdapter, urllib_zip_fetcher
from factorlab_data.alignment import FactorAlignment
from factorlab_data.cache import FactorCache, make_cache_key
from factorlab_data.errors import (
    DatasetNotFoundError,
    FactorAlignmentError,
    FactorCacheError,
    FactorDataError,
    FactorFetchError,
    FactorNotFoundError,
    FactorParseError,
    FactorValidationError,
)
from factorlab_data.loader import FactorLoader
from factorlab_data.metadata import VALID_FREQUENCIES, FactorMetadata
from factorlab_data.panel import FactorDataset, FactorPanel
from factorlab_data.ports import FactorDataPort
from factorlab_data.registry import DatasetDescriptor, FactorRegistry
from factorlab_data.validation import (
    FactorValidator,
    Severity,
    ValidationIssue,
    ValidationReport,
)

__version__ = "0.1.0"

__all__ = [
    "VALID_FREQUENCIES",
    "DatasetDescriptor",
    "DatasetNotFoundError",
    "FactorAlignment",
    "FactorAlignmentError",
    "FactorCache",
    "FactorCacheError",
    "FactorDataError",
    "FactorDataPort",
    "FactorDataset",
    "FactorFetchError",
    "FactorLoader",
    "FactorMetadata",
    "FactorNotFoundError",
    "FactorPanel",
    "FactorParseError",
    "FactorRegistry",
    "FactorValidationError",
    "FactorValidator",
    "KennethFrenchAdapter",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "make_cache_key",
    "urllib_zip_fetcher",
]
