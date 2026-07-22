"""The FactorDataPort interface.

Business logic and models depend on this abstraction, never on a concrete
provider.  Swapping Kenneth French for another factor source (AQR, a vendor
feed, an internal store) means writing a new adapter that satisfies this port --
no caller changes.  This is the Ports-and-Adapters (hexagonal) boundary of the
data layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from factorlab_data.panel import FactorDataset

__all__ = ["FactorDataPort"]


@runtime_checkable
class FactorDataPort(Protocol):
    """Contract every factor-data adapter must satisfy.

    Implementations separate *parsing* (pure: bytes/text -> dataset) from
    *loading* (may fetch bytes from a remote source, then parse).  Keeping
    ``parse`` pure makes adapters fully testable offline.
    """

    @property
    def source_name(self) -> str:
        """Human-readable provider name, e.g. ``"Kenneth French Data Library"``."""
        ...

    def available_datasets(self) -> tuple[str, ...]:
        """Dataset ids this adapter knows how to locate/describe."""
        ...

    def parse(
        self,
        content: str | bytes,
        *,
        dataset_id: str | None = None,
        frequency: str | None = None,
    ) -> FactorDataset:
        """Parse raw provider content into a :class:`FactorDataset` (pure)."""
        ...

    def load(self, dataset_id: str, *, frequency: str = "monthly") -> FactorDataset:
        """Locate, fetch, and parse ``dataset_id`` at ``frequency``.

        May perform I/O; implementations without a configured fetcher should
        raise :class:`~factorlab_data.errors.FactorFetchError`.
        """
        ...
