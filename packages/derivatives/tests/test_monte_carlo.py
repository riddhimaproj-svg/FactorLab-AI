"""Monte Carlo: convergence to Black-Scholes and variance-reduction effectiveness."""

from __future__ import annotations

import pytest

from factorlab_derivatives import (
    DerivativesInputError,
    MonteCarloResult,
    OptionType,
    black_scholes_price,
    monte_carlo_european,
)


@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_converges_to_black_scholes_within_standard_error(opt: OptionType) -> None:
    s, k, t, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    bs = black_scholes_price(opt, s, k, t, r, sigma)
    mc = monte_carlo_european(opt, s, k, t, r, sigma, n_paths=400_000, seed=0)
    # Estimate must sit within ~4 standard errors of the analytical price.
    assert abs(mc.price - bs) < 4.0 * mc.standard_error + 1e-9
    lo, hi = mc.confidence_interval
    assert lo < mc.price < hi


def test_control_variate_reduces_standard_error() -> None:
    args = (OptionType.CALL, 100, 100, 1.0, 0.05, 0.2)
    plain = monte_carlo_european(*args, n_paths=100_000, antithetic=False,
                                 control_variate=False, seed=1)
    reduced = monte_carlo_european(*args, n_paths=100_000, antithetic=True,
                                   control_variate=True, seed=1)
    assert reduced.standard_error < plain.standard_error


def test_antithetic_only_path() -> None:
    mc = monte_carlo_european(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2,
                              n_paths=50_000, antithetic=True, control_variate=False, seed=2)
    assert "antithetic" in mc.method
    assert mc.standard_error > 0.0


def test_plain_method_name() -> None:
    mc = monte_carlo_european(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2,
                              n_paths=10_000, antithetic=False, control_variate=False, seed=3)
    assert mc.method == "monte_carlo_plain"


def test_deterministic_with_seed() -> None:
    a = monte_carlo_european(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, n_paths=20_000, seed=42)
    b = monte_carlo_european(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, n_paths=20_000, seed=42)
    assert a.price == b.price


def test_zero_maturity_is_intrinsic_with_no_error() -> None:
    mc = monte_carlo_european(OptionType.CALL, 120, 100, 0.0, 0.05, 0.2, n_paths=10_000)
    assert mc.price == pytest.approx(20.0)
    assert mc.standard_error == 0.0


def test_zero_vol_is_deterministic() -> None:
    mc = monte_carlo_european(OptionType.CALL, 100, 90, 1.0, 0.05, 0.0, n_paths=10_000)
    assert mc.standard_error == 0.0
    assert mc.price > 0.0


def test_rejects_too_few_paths() -> None:
    with pytest.raises(DerivativesInputError):
        monte_carlo_european(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, n_paths=1)


def test_result_serializes() -> None:
    mc = monte_carlo_european(OptionType.CALL, 100, 100, 1.0, 0.05, 0.2, n_paths=10_000, seed=0)
    restored = MonteCarloResult.from_dict(mc.to_dict())
    assert restored == mc
