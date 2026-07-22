"""Estimation layer: OLS with classical, White, and Newey--West covariance."""

from __future__ import annotations

from factorlab_quant.estimation.hac import (
    bartlett_weights,
    default_hac_lags,
    newey_west_covariance,
    white_covariance,
)
from factorlab_quant.estimation.ols import OLS

__all__ = [
    "OLS",
    "bartlett_weights",
    "default_hac_lags",
    "newey_west_covariance",
    "white_covariance",
]
