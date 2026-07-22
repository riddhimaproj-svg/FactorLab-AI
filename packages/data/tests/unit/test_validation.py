"""Tests for FactorValidator."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_data.errors import FactorValidationError
from factorlab_data.metadata import FactorMetadata
from factorlab_data.panel import FactorPanel
from factorlab_data.validation import FactorValidator, Severity


def _panel(values: np.ndarray, names=("A", "B"), frequency="monthly", dates=None) -> FactorPanel:
    n = values.shape[0]
    if dates is None:
        dates = np.array(
            [np.datetime64("2000-01-01") + np.timedelta64(30 * i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
    meta = FactorMetadata("t", "t", "u", frequency, tuple(names), n_observations=n)
    return FactorPanel(dates, tuple(names), values, meta)


def test_clean_panel_is_valid() -> None:
    rng = np.random.default_rng(0)
    panel = _panel(rng.normal(0, 0.03, size=(60, 2)))
    report = FactorValidator().validate(panel)
    assert report.is_valid
    assert report.errors == ()


def test_missing_flagged_when_disallowed() -> None:
    values = np.zeros((10, 2))
    values[3, 0] = np.nan
    panel = _panel(values)
    report = FactorValidator(allow_missing=False, flag_constant=False).validate(panel)
    assert not report.is_valid
    assert any(i.code == "missing" for i in report.errors)


def test_missing_allowed_by_default() -> None:
    values = np.zeros((10, 2))
    values[3, 0] = np.nan
    values[:, 1] = np.arange(10)  # non-constant second column
    values[:, 0] = np.arange(10) * 0.01
    values[3, 0] = np.nan
    panel = _panel(values)
    report = FactorValidator().validate(panel)
    assert report.is_valid  # NaNs allowed


def test_all_missing_column_is_error() -> None:
    values = np.full((10, 2), np.nan)
    panel = _panel(values)
    report = FactorValidator().validate(panel)
    assert any(i.code == "all_missing" for i in report.errors)


def test_constant_factor_warns() -> None:
    values = np.column_stack([np.arange(10) * 0.01, np.ones(10)])
    panel = _panel(values)
    report = FactorValidator().validate(panel)
    assert any(i.code == "constant" and i.severity == Severity.WARNING for i in report.warnings)


def test_extreme_value_warns_percent_not_normalized() -> None:
    # values look like percentages (e.g. 5.0 = 500%): flags a units mistake.
    values = np.column_stack([np.full(10, 5.0), np.arange(10) * 0.01])
    panel = _panel(values)
    report = FactorValidator().validate(panel)
    assert any(i.code == "extreme_value" for i in report.warnings)


def test_frequency_mismatch_warns() -> None:
    # Declared monthly, but dates are 1 day apart (daily spacing).
    dates = np.array(
        [np.datetime64("2000-01-01") + np.timedelta64(i, "D") for i in range(10)],
        dtype="datetime64[D]",
    )
    values = np.arange(20, dtype=float).reshape(10, 2) * 0.001
    panel = _panel(values, dates=dates)
    report = FactorValidator().validate(panel)
    assert any(i.code == "frequency_mismatch" for i in report.warnings)


def test_assert_valid_raises_on_error() -> None:
    panel = _panel(np.full((5, 2), np.nan))
    with pytest.raises(FactorValidationError):
        FactorValidator().assert_valid(panel)


def test_report_summary_text() -> None:
    panel = _panel(np.full((5, 1), np.nan), names=("A",))
    report = FactorValidator().validate(panel)
    assert "all_missing" in report.summary()
