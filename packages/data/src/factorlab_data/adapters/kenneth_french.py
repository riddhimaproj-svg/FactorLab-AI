r"""Kenneth French Data Library adapter.

The Kenneth French Data Library distributes factor and portfolio returns as ZIP
archives containing a quirk-laden CSV: a free-text preamble, then one or more
*sections*, each a header row (a leading empty cell followed by column names)
and date-indexed data rows.  A single file often contains multiple sections at
different frequencies (e.g. a monthly panel followed by an annual panel).

This adapter parses that format **generically**: it discovers columns from each
section's header and infers frequency from the date-token width.  It contains no
knowledge of FF3, FF5, momentum, or any specific factor set -- the same parser
handles all of them, which is exactly what lets FF5 (and, later, Carhart, the
q-factor model, and APT) reuse it unchanged.

Values are normalized from percent to decimal, and the library's missing-value
sentinels (``-99.99``, ``-999``) are converted to NaN.

.. note::
   ``parse`` is pure (bytes/text in, dataset out) so the adapter is fully
   testable offline.  ``load`` performs network I/O only through an injected
   ``fetcher`` callable; without one it raises rather than reaching the network.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable

import numpy as np

from factorlab_data.errors import (
    DatasetNotFoundError,
    FactorFetchError,
    FactorParseError,
)
from factorlab_data.metadata import FactorMetadata
from factorlab_data.panel import FactorDataset, FactorPanel

__all__ = ["KenFrenchDatasetSpec", "KennethFrenchAdapter", "urllib_zip_fetcher"]

_SOURCE = "Kenneth French Data Library"
_BASE_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
)
# Missing-value sentinels used across the library.
_MISSING_SENTINELS = (-99.99, -999.0, -9999.0)
_SENTINEL_TOL = 5e-3
_DATE_WIDTHS = {8: "daily", 6: "monthly", 4: "annual"}

Fetcher = Callable[[str], bytes]


class KenFrenchDatasetSpec:
    """A catalog entry describing one Kenneth French dataset.

    Purely descriptive metadata (id, human name, zip filename, description).
    The parser does not use it, so unknown files can still be parsed via
    :meth:`KennethFrenchAdapter.parse`.
    """

    __slots__ = ("dataset_id", "description", "kind", "name", "zip_filename")

    def __init__(
        self,
        dataset_id: str,
        name: str,
        zip_filename: str,
        kind: str,
        description: str,
    ) -> None:
        self.dataset_id = dataset_id
        self.name = name
        self.zip_filename = zip_filename
        self.kind = kind
        self.description = description

    @property
    def url(self) -> str:
        return _BASE_URL + self.zip_filename


# A representative catalog spanning FF3, FF5, momentum, and research portfolios,
# at monthly and daily frequencies.  Extending it does not touch the parser.
_CATALOG: dict[str, KenFrenchDatasetSpec] = {
    spec.dataset_id: spec
    for spec in (
        KenFrenchDatasetSpec(
            "F-F_Research_Data_Factors", "Fama-French 3 Factors (monthly)",
            "F-F_Research_Data_Factors_CSV.zip", "factors",
            "Mkt-RF, SMB, HML, RF (monthly and annual).",
        ),
        KenFrenchDatasetSpec(
            "F-F_Research_Data_Factors_daily", "Fama-French 3 Factors (daily)",
            "F-F_Research_Data_Factors_daily_CSV.zip", "factors",
            "Mkt-RF, SMB, HML, RF (daily).",
        ),
        KenFrenchDatasetSpec(
            "F-F_Research_Data_5_Factors_2x3", "Fama-French 5 Factors (monthly)",
            "F-F_Research_Data_5_Factors_2x3_CSV.zip", "factors",
            "Mkt-RF, SMB, HML, RMW, CMA, RF (monthly and annual).",
        ),
        KenFrenchDatasetSpec(
            "F-F_Research_Data_5_Factors_2x3_daily", "Fama-French 5 Factors (daily)",
            "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "factors",
            "Mkt-RF, SMB, HML, RMW, CMA, RF (daily).",
        ),
        KenFrenchDatasetSpec(
            "F-F_Momentum_Factor", "Momentum Factor (monthly)",
            "F-F_Momentum_Factor_CSV.zip", "factors",
            "Mom (monthly and annual).",
        ),
        KenFrenchDatasetSpec(
            "F-F_Momentum_Factor_daily", "Momentum Factor (daily)",
            "F-F_Momentum_Factor_daily_CSV.zip", "factors",
            "Mom (daily).",
        ),
        KenFrenchDatasetSpec(
            "6_Portfolios_2x3", "6 Portfolios formed on Size and Book-to-Market",
            "6_Portfolios_2x3_CSV.zip", "portfolios",
            "Research portfolios, 2x3 size/BM sort.",
        ),
        KenFrenchDatasetSpec(
            "25_Portfolios_5x5", "25 Portfolios formed on Size and Book-to-Market",
            "25_Portfolios_5x5_CSV.zip", "portfolios",
            "Research portfolios, 5x5 size/BM sort.",
        ),
    )
}


def urllib_zip_fetcher(url: str) -> bytes:  # pragma: no cover - network I/O
    """Default fetcher: download ``url`` and return its bytes via ``urllib``.

    Not used in tests.  Provided so callers can opt into live downloads
    explicitly by passing ``KennethFrenchAdapter(fetcher=urllib_zip_fetcher)``.
    """
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as response:
        return bytes(response.read())


class KennethFrenchAdapter:
    """Adapter for the Kenneth French Data Library (implements ``FactorDataPort``).

    Parameters
    ----------
    fetcher:
        Optional callable mapping a URL to raw ZIP bytes.  Required only for
        :meth:`load`; :meth:`parse` never needs it.
    """

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetcher = fetcher

    @property
    def source_name(self) -> str:
        return _SOURCE

    def available_datasets(self) -> tuple[str, ...]:
        return tuple(_CATALOG)

    def describe(self, dataset_id: str) -> KenFrenchDatasetSpec:
        """Return the catalog entry for ``dataset_id``."""
        if dataset_id not in _CATALOG:
            raise DatasetNotFoundError(dataset_id, tuple(_CATALOG))
        return _CATALOG[dataset_id]

    # ------------------------------------------------------------------ #
    # Loading (I/O)                                                       #
    # ------------------------------------------------------------------ #
    def load(self, dataset_id: str, *, frequency: str = "monthly") -> FactorDataset:
        """Fetch and parse ``dataset_id``.  Requires a configured ``fetcher``."""
        spec = self.describe(dataset_id)
        if self._fetcher is None:
            raise FactorFetchError(
                f"No fetcher configured; cannot download {spec.url!r}. "
                f"Either pass fetcher=... or use parse(content=...) with local data."
            )
        raw = self._fetcher(spec.url)
        return self.parse(raw, dataset_id=dataset_id, frequency=frequency)

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                      #
    # ------------------------------------------------------------------ #
    def parse(
        self,
        content: str | bytes,
        *,
        dataset_id: str | None = None,
        frequency: str | None = None,
    ) -> FactorDataset:
        """Parse Kenneth French content into a :class:`FactorDataset`."""
        text = self._to_text(content)
        sections = _parse_sections(text)
        if not sections:
            raise FactorParseError(
                "No parseable data sections found in the provided content."
            )

        ds_id = dataset_id or "unknown"
        spec = _CATALOG.get(dataset_id) if dataset_id else None
        name = spec.name if spec else ds_id
        url = spec.url if spec else None

        panels: dict[str, FactorPanel] = {}
        for section in sections:
            panel = _section_to_panel(section, ds_id, name, url)
            # If two sections share a frequency, keep the one with more rows.
            existing = panels.get(panel.frequency)
            if existing is None or panel.n_observations > existing.n_observations:
                panels[panel.frequency] = panel

        primary_freq = frequency if frequency in panels else next(iter(panels))
        primary = panels[primary_freq]
        dataset_meta = primary.metadata.with_updates(name=name)
        return FactorDataset(dataset_id=ds_id, panels=panels, metadata=dataset_meta)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_text(content: str | bytes) -> str:
        if isinstance(content, str):
            return content
        if content[:4] == b"PK\x03\x04":  # ZIP archive
            return _read_zip_text(content)
        # Kenneth French files are Latin-1 / ASCII.
        return content.decode("latin-1")


# ---------------------------------------------------------------------------- #
# ZIP handling                                                                 #
# ---------------------------------------------------------------------------- #
def _read_zip_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [
            n for n in archive.namelist()
            if n.lower().endswith((".csv", ".txt")) and not n.startswith("__MACOSX")
        ]
        if not members:
            members = [n for n in archive.namelist() if not n.endswith("/")]
        if not members:
            raise FactorParseError("ZIP archive contains no readable data file.")
        return archive.read(members[0]).decode("latin-1")


# ---------------------------------------------------------------------------- #
# Section parsing                                                              #
# ---------------------------------------------------------------------------- #
class _Section:
    __slots__ = ("dates", "frequency", "names", "raw_values")

    def __init__(
        self,
        names: list[str],
        dates: list[np.datetime64],
        raw_values: list[list[float]],
        frequency: str,
    ) -> None:
        self.names = names
        self.dates = dates
        self.raw_values = raw_values
        self.frequency = frequency


def _is_date_token(token: str) -> bool:
    token = token.strip()
    return token.isdigit() and len(token) in _DATE_WIDTHS


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_header(line: str) -> bool:
    parts = line.split(",")
    if len(parts) < 2:
        return False
    if _is_date_token(parts[0]):
        return False
    rest = [p.strip() for p in parts[1:]]
    return any(p and not _is_number(p) for p in rest)


def _parse_date(token: str) -> tuple[np.datetime64, str]:
    token = token.strip()
    freq = _DATE_WIDTHS[len(token)]
    if freq == "daily":
        iso = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    elif freq == "monthly":
        iso = f"{token[:4]}-{token[4:6]}-01"
    else:  # annual
        iso = f"{token}-01-01"
    return np.datetime64(iso, "D"), freq


def _parse_value(token: str) -> float:
    token = token.strip()
    if token == "":
        return float("nan")
    try:
        value = float(token)
    except ValueError:
        return float("nan")
    for sentinel in _MISSING_SENTINELS:
        if abs(value - sentinel) < _SENTINEL_TOL:
            return float("nan")
    return value


def _parse_sections(text: str) -> list[_Section]:
    lines = text.splitlines()
    sections: list[_Section] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not _looks_like_header(line):
            i += 1
            continue

        names = [p.strip() for p in line.split(",")]
        # Drop the leading (empty) date-column label.
        if names and names[0] == "":
            names = names[1:]
        names = [nm for nm in names if nm != ""]
        if not names:
            i += 1
            continue

        dates: list[np.datetime64] = []
        rows: list[list[float]] = []
        section_freq: str | None = None
        j = i + 1
        while j < n:
            row = lines[j].strip()
            if not row:
                break
            tokens = [t.strip() for t in row.split(",")]
            if not _is_date_token(tokens[0]):
                break
            date, freq = _parse_date(tokens[0])
            if section_freq is None:
                section_freq = freq
            values = [_parse_value(t) for t in tokens[1 : 1 + len(names)]]
            while len(values) < len(names):
                values.append(float("nan"))
            dates.append(date)
            rows.append(values)
            j += 1

        if dates and section_freq is not None:
            sections.append(_Section(names, dates, rows, section_freq))
        i = max(j, i + 1)

    return sections


def _section_to_panel(
    section: _Section, dataset_id: str, name: str, url: str | None
) -> FactorPanel:
    dates = np.array(section.dates, dtype="datetime64[D]")
    if dates.size and np.any(np.diff(dates) <= np.timedelta64(0, "D")):
        raise FactorParseError(
            f"Dates in section {section.frequency!r} are not strictly increasing."
        )
    raw = np.array(section.raw_values, dtype=np.float64)
    # Normalize: percent -> decimal (missing already NaN).
    values = raw / 100.0

    metadata = FactorMetadata(
        dataset_id=dataset_id,
        name=name,
        source=_SOURCE,
        frequency=section.frequency,
        factor_names=tuple(section.names),
        units="decimal",
        provenance_url=url,
        start=np.datetime_as_string(dates[0], unit="D") if dates.size else None,
        end=np.datetime_as_string(dates[-1], unit="D") if dates.size else None,
        n_observations=int(dates.shape[0]),
        transformations=("percent_to_decimal", "missing_sentinel_to_nan"),
    )
    return FactorPanel(dates, tuple(section.names), values, metadata)
