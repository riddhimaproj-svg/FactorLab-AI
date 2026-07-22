"""Tests for the KennethFrenchAdapter — generic parsing across factor sets."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from factorlab_data.adapters.kenneth_french import KennethFrenchAdapter
from factorlab_data.errors import DatasetNotFoundError, FactorFetchError, FactorParseError
from factorlab_data.ports import FactorDataPort
from tests.conftest import make_kf_monthly_text


@pytest.fixture
def adapter() -> KennethFrenchAdapter:
    return KennethFrenchAdapter()


def test_satisfies_port(adapter) -> None:
    assert isinstance(adapter, FactorDataPort)
    assert adapter.source_name == "Kenneth French Data Library"


def test_available_datasets_span_all_kinds(adapter) -> None:
    ids = adapter.available_datasets()
    assert "F-F_Research_Data_Factors" in ids          # FF3
    assert "F-F_Research_Data_5_Factors_2x3" in ids     # FF5
    assert "F-F_Momentum_Factor" in ids                 # Momentum
    assert "6_Portfolios_2x3" in ids                    # Portfolios
    assert "F-F_Research_Data_5_Factors_2x3_daily" in ids  # daily


def test_parse_ff5_monthly(adapter, ff5_monthly_text) -> None:
    ds = adapter.parse(
        ff5_monthly_text, dataset_id="F-F_Research_Data_5_Factors_2x3", frequency="monthly"
    )
    panel = ds.panel("monthly")
    assert panel.factor_names == ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")
    assert panel.n_observations == 300
    assert panel.frequency == "monthly"


def test_parser_is_generic_ff3(adapter, ff3_monthly_text) -> None:
    """Same parser, different factor set — no hardcoded columns."""
    ds = adapter.parse(ff3_monthly_text, dataset_id="F-F_Research_Data_Factors")
    assert ds.panel("monthly").factor_names == ("Mkt-RF", "SMB", "HML", "RF")


def test_parser_is_generic_momentum(adapter, momentum_monthly_text) -> None:
    ds = adapter.parse(momentum_monthly_text, dataset_id="F-F_Momentum_Factor")
    assert ds.panel("monthly").factor_names == ("Mom",)


def test_parser_is_generic_portfolios(adapter, portfolios_monthly_text) -> None:
    ds = adapter.parse(portfolios_monthly_text, dataset_id="6_Portfolios_2x3")
    assert ds.panel("monthly").factor_names == ("SMALL LoBM", "ME1 BM2", "BIG HiBM")


def test_parses_daily_frequency(adapter, ff3_daily_text) -> None:
    ds = adapter.parse(ff3_daily_text, dataset_id="F-F_Research_Data_Factors_daily")
    panel = ds.panel("daily")
    assert panel.frequency == "daily"
    assert panel.n_observations == 90
    # consecutive daily dates one day apart
    assert (panel.dates[1] - panel.dates[0]) == np.timedelta64(1, "D")


def test_percent_to_decimal_normalization(adapter) -> None:
    text, decimals = make_kf_monthly_text(["Mkt-RF", "RF"], n_months=24, seed=3, with_annual=False)
    panel = adapter.parse(text, dataset_id="x").panel("monthly")
    np.testing.assert_allclose(panel["Mkt-RF"], decimals[:, 0], atol=1e-6)
    assert panel.metadata.units == "decimal"
    assert "percent_to_decimal" in panel.metadata.transformations


def test_missing_sentinels_become_nan(adapter) -> None:
    text, _ = make_kf_monthly_text(
        ["Mkt-RF", "RF"], n_months=24, seed=3, with_annual=False, missing_last=True
    )
    panel = adapter.parse(text, dataset_id="x").panel("monthly")
    assert np.isnan(panel["Mkt-RF"][-1])
    assert np.isfinite(panel["Mkt-RF"][:-1]).all()


def test_monthly_and_annual_sections_both_parsed(adapter, ff5_monthly_text) -> None:
    ds = adapter.parse(ff5_monthly_text, dataset_id="F-F_Research_Data_5_Factors_2x3")
    assert set(ds.frequencies) == {"monthly", "annual"}
    assert ds.panel("annual").frequency == "annual"


def test_parse_from_zip_bytes(adapter, ff3_monthly_text) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors.CSV", ff3_monthly_text)
    ds = adapter.parse(buffer.getvalue(), dataset_id="F-F_Research_Data_Factors")
    assert ds.panel("monthly").factor_names == ("Mkt-RF", "SMB", "HML", "RF")


def test_parse_empty_raises(adapter) -> None:
    with pytest.raises(FactorParseError):
        adapter.parse("just some prose with no data\nand no header")


def test_describe_unknown_dataset_raises(adapter) -> None:
    with pytest.raises(DatasetNotFoundError):
        adapter.describe("NOPE")


def test_load_without_fetcher_raises(adapter) -> None:
    with pytest.raises(FactorFetchError):
        adapter.load("F-F_Research_Data_5_Factors_2x3")


def test_load_with_injected_fetcher(ff5_monthly_text) -> None:
    calls = []

    def fake_fetcher(url: str) -> bytes:
        calls.append(url)
        return ff5_monthly_text.encode("latin-1")

    adapter = KennethFrenchAdapter(fetcher=fake_fetcher)
    ds = adapter.load("F-F_Research_Data_5_Factors_2x3", frequency="monthly")
    assert ds.panel("monthly").n_observations == 300
    assert calls and calls[0].endswith(".zip")


def test_non_strictly_increasing_dates_raise(adapter) -> None:
    text = "\n".join(
        [
            "hdr", "",
            ",Mkt-RF",
            "199001,   1.00",
            "199001,   2.00",  # duplicate date
        ]
    )
    with pytest.raises(FactorParseError):
        adapter.parse(text, dataset_id="x")
