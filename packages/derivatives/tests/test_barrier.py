"""Barrier options: in/out parity, breach handling, and Monte Carlo sanity."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab_derivatives import (
    BarrierType,
    DerivativesInputError,
    OptionType,
    barrier_price,
    black_scholes_price,
)

COMMON = {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "volatility": 0.2}


@pytest.mark.parametrize(
    ("in_type", "out_type", "barrier"),
    [
        (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT, 90.0),
        (BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT, 120.0),
    ],
)
@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_in_out_parity(
    opt: OptionType, in_type: BarrierType, out_type: BarrierType, barrier: float
) -> None:
    ki = barrier_price(opt, barrier=barrier, barrier_type=in_type, **COMMON)
    ko = barrier_price(opt, barrier=barrier, barrier_type=out_type, **COMMON)
    vanilla = black_scholes_price(opt, **COMMON)
    assert ki + ko == pytest.approx(vanilla, abs=1e-9)


def test_barrier_prices_are_non_negative() -> None:
    for bt in BarrierType:
        barrier = 90.0 if bt.is_down else 115.0
        price = barrier_price(OptionType.CALL, barrier=barrier, barrier_type=bt, **COMMON)
        assert price >= -1e-12


def test_already_breached_down_in_equals_vanilla() -> None:
    # spot below a down barrier: the knock-in is already alive -> vanilla.
    ki = barrier_price(
        OptionType.CALL, spot=85.0, strike=100.0, maturity=1.0, rate=0.05,
        volatility=0.2, barrier=90.0, barrier_type=BarrierType.DOWN_AND_IN,
    )
    vanilla = black_scholes_price(OptionType.CALL, 85.0, 100.0, 1.0, 0.05, 0.2)
    assert ki == pytest.approx(vanilla, abs=1e-12)


def test_already_breached_down_out_is_worthless() -> None:
    ko = barrier_price(
        OptionType.CALL, spot=85.0, strike=100.0, maturity=1.0, rate=0.05,
        volatility=0.2, barrier=90.0, barrier_type=BarrierType.DOWN_AND_OUT,
    )
    assert ko == 0.0


def test_monte_carlo_agreement_down_and_out_call() -> None:
    # Independent path-simulation check (continuous monitoring approximated finely).
    s0, k, t, r, sigma, h = 100.0, 100.0, 1.0, 0.05, 0.2, 90.0
    rng = np.random.default_rng(7)
    n_paths, n_steps = 60_000, 250
    dt = t / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    log_s = np.full(n_paths, np.log(s0))
    alive = np.ones(n_paths, dtype=bool)
    for _ in range(n_steps):
        log_s += drift + diff * rng.standard_normal(n_paths)
        alive &= np.exp(log_s) > h
    payoff = np.where(alive, np.maximum(np.exp(log_s) - k, 0.0), 0.0)
    mc = np.exp(-r * t) * payoff.mean()
    analytic = barrier_price(
        OptionType.CALL, spot=s0, strike=k, maturity=t, rate=r, volatility=sigma,
        barrier=h, barrier_type=BarrierType.DOWN_AND_OUT,
    )
    # Discrete monitoring slightly over-prices the knock-out; loose tolerance.
    assert mc == pytest.approx(analytic, abs=0.5)


def test_requires_positive_maturity_and_vol() -> None:
    with pytest.raises(DerivativesInputError):
        barrier_price(
            OptionType.CALL, spot=100, strike=100, maturity=0.0, rate=0.05,
            volatility=0.2, barrier=90, barrier_type=BarrierType.DOWN_AND_IN,
        )
    with pytest.raises(DerivativesInputError):
        barrier_price(
            OptionType.CALL, spot=100, strike=100, maturity=1.0, rate=0.05,
            volatility=0.0, barrier=90, barrier_type=BarrierType.DOWN_AND_IN,
        )
