"""Integration: Kenneth French FF3 + Momentum -> Factor Layer -> Carhart.

Proves the momentum factor is sourced through the data layer and combined with
the FF3 factors via the layer's alignment, then consumed by the Carhart model.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_data import FactorAlignment, FactorLoader, KennethFrenchAdapter
from tests.conftest import make_kf_monthly_text

pytestmark = pytest.mark.integration


def _load_factors():
    """Load FF3 and Momentum panels from the layer and align them on dates."""
    ff3_text, _ = make_kf_monthly_text(["Mkt-RF", "SMB", "HML", "RF"], n_months=300, seed=31)
    mom_text, _ = make_kf_monthly_text(["Mom"], n_months=300, seed=32)

    loader = FactorLoader(KennethFrenchAdapter())
    ff3 = loader.load_from_content(
        ff3_text, dataset_id="F-F_Research_Data_Factors", frequency="monthly"
    ).panel("monthly")
    mom = loader.load_from_content(
        mom_text, dataset_id="F-F_Momentum_Factor", frequency="monthly"
    ).panel("monthly")

    ff3_aligned, mom_aligned = FactorAlignment.align_panels(ff3, mom)
    return ff3_aligned, mom_aligned


def _combined_factor_set(ff3_panel, mom_panel):
    # FF3 factors from the layer, then append the momentum factor (named "Mom").
    fs = ff3_panel.to_factor_set()  # Mkt-RF, SMB, HML (RF excluded)
    return fs.add(mom_panel.to_factor_set()["Mom"])


def test_momentum_sourced_from_layer_feeds_carhart() -> None:
    from factorlab_quant.models.carhart import CarhartModel, CarhartResult

    ff3, mom = _load_factors()
    factor_set = _combined_factor_set(ff3, mom)
    assert "Mom" in factor_set.names  # library name before the model normalizes it

    rng = np.random.default_rng(77)
    betas = {"Mkt-RF": 1.0, "SMB": -0.2, "HML": 0.3, "Mom": 0.5}
    alpha = 0.001
    asset_excess = alpha + rng.normal(0, 0.006, ff3.n_observations)
    for name, b in betas.items():
        col = ff3[name] if name in ff3.factor_names else mom[name]
        asset_excess = asset_excess + b * col

    result = CarhartModel().fit(asset_excess, factor_set, returns_are_excess=True)
    assert isinstance(result, CarhartResult)
    assert result.factor_names == ("Mkt-RF", "SMB", "HML", "MOM")  # normalized
    assert result.alpha.estimate == pytest.approx(alpha, abs=0.002)
    assert result.momentum_loading.estimate == pytest.approx(0.5, abs=0.03)


def test_alignment_of_mismatched_spans() -> None:
    """Momentum with a shorter span still aligns to the FF3 panel intersection."""
    from factorlab_quant.models.carhart import CarhartModel

    ff3_text, _ = make_kf_monthly_text(["Mkt-RF", "SMB", "HML", "RF"], n_months=300, seed=41)
    mom_text, _ = make_kf_monthly_text(["Mom"], n_months=240, seed=42)  # shorter
    loader = FactorLoader(KennethFrenchAdapter())
    ff3 = loader.load_from_content(
        ff3_text, dataset_id="F-F_Research_Data_Factors"
    ).panel("monthly")
    mom = loader.load_from_content(mom_text, dataset_id="F-F_Momentum_Factor").panel("monthly")

    ff3_a, mom_a = FactorAlignment.align_panels(ff3, mom)
    assert ff3_a.n_observations == mom_a.n_observations == 240

    factor_set = ff3_a.to_factor_set().add(mom_a.to_factor_set()["Mom"])
    rng = np.random.default_rng(5)
    asset = rng.normal(0, 0.01, 240) + 1.0 * ff3_a["Mkt-RF"] + 0.4 * mom_a["Mom"]
    res = CarhartModel().fit(asset, factor_set, returns_are_excess=True)
    assert res.n_observations == 240
