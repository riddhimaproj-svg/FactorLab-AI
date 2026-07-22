"""Provider adapters implementing :class:`~factorlab_data.ports.FactorDataPort`."""

from __future__ import annotations

from factorlab_data.adapters.kenneth_french import (
    KenFrenchDatasetSpec,
    KennethFrenchAdapter,
    urllib_zip_fetcher,
)

__all__ = ["KenFrenchDatasetSpec", "KennethFrenchAdapter", "urllib_zip_fetcher"]
