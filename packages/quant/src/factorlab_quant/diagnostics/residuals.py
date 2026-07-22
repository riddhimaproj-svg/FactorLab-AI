"""Residual moment and autocorrelation statistics.

Small, well-tested numerical primitives that the specification tests and the
regression assembler build upon.  Kept separate from :mod:`.tests` so they can
be reused (e.g. by risk metrics) without importing the hypothesis-test layer.
"""

from __future__ import annotations

import numpy as np

from factorlab_quant.core.types import FloatArray

__all__ = ["durbin_watson", "sample_excess_kurtosis", "sample_skewness"]


def sample_skewness(x: FloatArray) -> float:
    r"""Biased (population) sample skewness :math:`m_3 / m_2^{3/2}`.

    Uses the maximum-likelihood (biased) moment definition, matching the
    convention of the Jarque--Bera statistic it feeds.
    """
    x = np.asarray(x, dtype=np.float64)
    deviations = x - x.mean()
    m2 = np.mean(deviations**2)
    m3 = np.mean(deviations**3)
    if m2 == 0.0:
        return 0.0
    return float(m3 / m2**1.5)


def sample_excess_kurtosis(x: FloatArray) -> float:
    r"""Biased sample *excess* kurtosis :math:`m_4 / m_2^{2} - 3`.

    Excess kurtosis is zero for a normal distribution; positive values indicate
    fat tails, a hallmark of financial return residuals.
    """
    x = np.asarray(x, dtype=np.float64)
    deviations = x - x.mean()
    m2 = np.mean(deviations**2)
    m4 = np.mean(deviations**4)
    if m2 == 0.0:
        return 0.0
    return float(m4 / m2**2 - 3.0)


def durbin_watson(residuals: FloatArray) -> float:
    r"""Durbin--Watson statistic for first-order residual autocorrelation.

    .. math::

        d = \frac{\sum_{t=2}^{n} (\hat e_t - \hat e_{t-1})^2}
                 {\sum_{t=1}^{n} \hat e_t^2}.

    :math:`d \approx 2` indicates no first-order autocorrelation, :math:`d \to 0`
    strong positive autocorrelation, and :math:`d \to 4` strong negative
    autocorrelation.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    denom = float(np.sum(residuals**2))
    if denom == 0.0:
        return float("nan")
    diff = np.diff(residuals)
    return float(np.sum(diff**2) / denom)
