"""Integration: Kenneth French text -> Factor Layer -> FF5 model.

This proves the factor infrastructure delivers data a real model can consume,
end to end.  It lives in the data package's suite because that environment has
both ``factorlab_data`` and ``factorlab_quant`` installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_data import FactorCache, FactorLoader, FactorValidator, KennethFrenchAdapter
from tests.conftest import make_kf_monthly_text

pytestmark = pytest.mark.integration

FF5 = "F-F_Research_Data_5_Factors_2x3"


def _load_ff5_panel(seed: int = 21):
    text, _ = make_kf_monthly_text(
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], n_months=360, seed=seed
    )
    loader = FactorLoader(
        KennethFrenchAdapter(), cache=FactorCache(), validator=FactorValidator()
    )
    return loader.load_from_content(text, dataset_id=FF5, frequency="monthly").panel("monthly")


def test_full_pipeline_recovers_ff5_parameters() -> None:
    from factorlab_quant.models.fama_french_5 import FamaFrench5Model, FamaFrench5Result

    panel = _load_ff5_panel()
    rng = np.random.default_rng(99)
    betas = {"Mkt-RF": 1.05, "SMB": -0.20, "HML": 0.30, "RMW": 0.40, "CMA": 0.25}
    alpha = 0.001
    asset_excess = alpha + rng.normal(0, 0.006, panel.n_observations)
    for name, b in betas.items():
        asset_excess = asset_excess + b * panel[name]

    result = FamaFrench5Model().fit(asset_excess, panel, returns_are_excess=True)
    assert isinstance(result, FamaFrench5Result)
    assert result.alpha.estimate == pytest.approx(alpha, abs=0.002)
    for name, b in betas.items():
        assert result.factor_loading(name).estimate == pytest.approx(b, abs=0.03)


def test_pipeline_via_factor_set_matches_via_panel() -> None:
    """Feeding FF5 a FactorSet or the panel directly must be identical."""
    from factorlab_quant.models.fama_french_5 import FamaFrench5Model

    panel = _load_ff5_panel(seed=22)
    rng = np.random.default_rng(7)
    asset = 0.001 + 1.1 * panel["Mkt-RF"] + 0.3 * panel["RMW"] + rng.normal(0, 0.01, len(panel))

    via_panel = FamaFrench5Model().fit(asset, panel, returns_are_excess=True)
    via_set = FamaFrench5Model().fit(asset, panel.to_factor_set(), returns_are_excess=True)
    np.testing.assert_allclose(via_panel.params, via_set.params, rtol=1e-12)


def test_pipeline_excess_conversion_with_panel_rf() -> None:
    """Use the panel's RF column to convert a raw asset to excess."""
    from factorlab_quant.models.fama_french_5 import FamaFrench5Model

    panel = _load_ff5_panel(seed=23)
    rf = panel.risk_free
    assert rf is not None
    rng = np.random.default_rng(3)
    asset_excess = 0.0008 + 1.0 * panel["Mkt-RF"] + rng.normal(0, 0.008, len(panel))
    asset_raw = asset_excess + rf

    via_raw = FamaFrench5Model().fit(asset_raw, panel, risk_free=rf)
    via_excess = FamaFrench5Model().fit(asset_excess, panel, returns_are_excess=True)
    np.testing.assert_allclose(via_raw.params, via_excess.params, rtol=1e-9)
