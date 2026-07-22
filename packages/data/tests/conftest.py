"""Shared fixtures: synthetic Kenneth-French-format content generators.

Tests never touch the network.  These helpers build byte-for-byte realistic
Kenneth French files (preamble, header rows, monthly/daily/annual sections,
percent units, missing-value sentinels) so the adapter is exercised against the
real format offline.
"""

from __future__ import annotations

import numpy as np
import pytest


def _fmt_row(date: str, values: list[float]) -> str:
    return ",".join([date, *[f"{v:9.2f}" for v in values]])


def make_kf_monthly_text(
    names: list[str],
    *,
    start_year: int = 1990,
    n_months: int = 240,
    seed: int = 0,
    with_annual: bool = True,
    missing_last: bool = False,
) -> tuple[str, np.ndarray]:
    """Return (content, decimal_values) for a monthly KF-format file.

    ``decimal_values`` is the ground-truth ``n x k`` matrix in *decimal* units
    (the file itself stores percentages), so tests can check normalization.
    """
    rng = np.random.default_rng(seed)
    k = len(names)
    # Store at 2-decimal percent precision (as real Kenneth French files do);
    # return the *stored* decimals so tests can compare like-for-like.
    decimals = np.round(rng.normal(0.0, 0.03, size=(n_months, k)) * 100.0, 2) / 100.0

    lines = ["This file was created for testing purposes only.", ""]
    lines.append("," + ",".join(names))
    y, m = start_year, 1
    for i in range(n_months):
        pct = [v * 100.0 for v in decimals[i]]
        if missing_last and i == n_months - 1:
            pct[0] = -99.99  # missing sentinel
        lines.append(_fmt_row(f"{y:04d}{m:02d}", pct))
        m += 1
        if m > 12:
            m = 1
            y += 1

    if with_annual:
        lines.extend(["", " Annual Factors: January-December", "," + ",".join(names)])
        lines.append(_fmt_row(f"{start_year:04d}", [5.0] * k))
        lines.append(_fmt_row(f"{start_year + 1:04d}", [4.0] * k))

    lines.extend(["", "Copyright 2024 synthetic."])
    return "\n".join(lines), decimals


def make_kf_daily_text(
    names: list[str], *, n_days: int = 60, seed: int = 1
) -> tuple[str, np.ndarray]:
    """Return (content, decimal_values) for a daily KF-format file (YYYYMMDD)."""
    rng = np.random.default_rng(seed)
    k = len(names)
    decimals = np.round(rng.normal(0.0, 0.01, size=(n_days, k)) * 100.0, 2) / 100.0
    dates = np.datetime64("2020-01-01") + np.arange(n_days)
    lines = ["Daily synthetic file.", "", "," + ",".join(names)]
    for i in range(n_days):
        token = str(dates[i]).replace("-", "")
        lines.append(_fmt_row(token, [v * 100.0 for v in decimals[i]]))
    lines.extend(["", "Copyright."])
    return "\n".join(lines), decimals


@pytest.fixture
def ff5_monthly_text() -> str:
    text, _ = make_kf_monthly_text(
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], n_months=300, seed=5
    )
    return text


@pytest.fixture
def ff3_monthly_text() -> str:
    text, _ = make_kf_monthly_text(["Mkt-RF", "SMB", "HML", "RF"], n_months=180, seed=6)
    return text


@pytest.fixture
def ff3_daily_text() -> str:
    text, _ = make_kf_daily_text(["Mkt-RF", "SMB", "HML", "RF"], n_days=90, seed=7)
    return text


@pytest.fixture
def momentum_monthly_text() -> str:
    text, _ = make_kf_monthly_text(["Mom"], n_months=120, seed=8)
    return text


@pytest.fixture
def portfolios_monthly_text() -> str:
    text, _ = make_kf_monthly_text(
        ["SMALL LoBM", "ME1 BM2", "BIG HiBM"], n_months=120, seed=9
    )
    return text
