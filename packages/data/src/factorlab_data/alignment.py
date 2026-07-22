"""Date alignment across factor panels and asset series.

Factor models require every series -- the asset and every factor -- to share a
single, gap-free date index.  In practice the asset return series and the factor
panel come from different sources with different calendars, so they must be
inner-joined on their common dates before estimation.

:class:`FactorAlignment` provides the pure, index-based join primitives.  It
never imputes: alignment is exact set intersection on dates, which is the only
defensible default for return data (a fabricated return is worse than a dropped
one).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factorlab_data.errors import FactorAlignmentError
from factorlab_data.panel import FactorPanel

__all__ = ["FactorAlignment"]

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


class FactorAlignment:
    """Stateless date-alignment utilities (all methods are static)."""

    @staticmethod
    def common_dates(*date_arrays: DateArray) -> DateArray:
        """Return the sorted intersection of several date arrays."""
        if not date_arrays:
            raise FactorAlignmentError("common_dates requires at least one array")
        common = date_arrays[0].astype("datetime64[D]")
        for arr in date_arrays[1:]:
            common = np.intersect1d(common, arr.astype("datetime64[D]"))
        return common

    @staticmethod
    def align_panels(*panels: FactorPanel) -> tuple[FactorPanel, ...]:
        """Restrict every panel to the dates common to all of them."""
        if not panels:
            raise FactorAlignmentError("align_panels requires at least one panel")
        common = FactorAlignment.common_dates(*[p.dates for p in panels])
        if common.size == 0:
            raise FactorAlignmentError("Panels share no common dates.")
        return tuple(FactorAlignment._restrict_panel(p, common) for p in panels)

    @staticmethod
    def align_asset_to_panel(
        asset_dates: DateArray,
        asset_values: FloatArray,
        panel: FactorPanel,
    ) -> tuple[DateArray, FloatArray, FactorPanel]:
        """Inner-join an asset return series with a factor panel on dates.

        Returns the common dates, the asset values on those dates, and the panel
        restricted to those dates -- ready to hand to a model.
        """
        asset_dates = np.asarray(asset_dates, dtype="datetime64[D]")
        asset_values = np.asarray(asset_values, dtype=np.float64)
        if asset_dates.shape[0] != asset_values.shape[0]:
            raise FactorAlignmentError(
                "asset_dates and asset_values disagree on length "
                f"({asset_dates.shape[0]} vs {asset_values.shape[0]})."
            )
        common = FactorAlignment.common_dates(asset_dates, panel.dates)
        if common.size == 0:
            raise FactorAlignmentError("Asset and panel share no common dates.")

        asset_idx = FactorAlignment._index_of(asset_dates, common)
        aligned_panel = FactorAlignment._restrict_panel(panel, common)
        return common, asset_values[asset_idx], aligned_panel

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _index_of(dates: DateArray, targets: DateArray) -> NDArray[np.intp]:
        order = np.argsort(dates)
        positions = np.searchsorted(dates[order], targets)
        return order[positions]

    @staticmethod
    def _restrict_panel(panel: FactorPanel, dates: DateArray) -> FactorPanel:
        idx = FactorAlignment._index_of(panel.dates, dates)
        meta = panel.metadata.with_updates(
            n_observations=int(dates.shape[0]),
            start=np.datetime_as_string(dates[0], unit="D") if dates.size else None,
            end=np.datetime_as_string(dates[-1], unit="D") if dates.size else None,
        )
        return FactorPanel(dates, panel.factor_names, panel.values[idx], meta)
