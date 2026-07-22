r"""Monte Carlo pricing of European options under geometric Brownian motion.

The terminal price is simulated exactly (GBM has a closed-form transition):

.. math::

    S_T = S_0 \exp\!\big((r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}\, Z\big),
    \qquad Z \sim \mathcal N(0, 1).

Two optional variance-reduction techniques are supported:

* **Antithetic variates** — pair each draw ``Z`` with ``-Z`` to cancel sampling
  noise in the drift.
* **Control variates** — use the discounted terminal price (whose expectation is
  the known forward ``S_0 e^{-qT}``) as a control, removing variance analytically.

The returned :class:`MonteCarloResult` carries the estimate's standard error and a
95% confidence interval.
"""

from __future__ import annotations

import numpy as np

from factorlab_derivatives._validation import check_non_negative, check_positive
from factorlab_derivatives.errors import DerivativesInputError
from factorlab_derivatives.instruments import OptionType
from factorlab_derivatives.reports import MonteCarloResult

__all__ = ["monte_carlo_european"]


def monte_carlo_european(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    *,
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European option by simulating terminal GBM prices."""
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_non_negative(maturity, "maturity")
    check_non_negative(volatility, "volatility")
    if n_paths < 2:
        raise DerivativesInputError("n_paths must be >= 2")

    phi = option_type.sign
    disc = float(np.exp(-rate * maturity))

    # Degenerate: no randomness left, price is the discounted deterministic payoff.
    if maturity <= 0.0 or volatility <= 0.0:
        fwd = spot * np.exp((rate - dividend) * maturity)
        payoff = max(phi * (fwd - strike), 0.0)
        method = _method_name(antithetic, control_variate)
        return MonteCarloResult(
            price=disc * payoff, standard_error=0.0, n_paths=n_paths, method=method
        )

    rng = np.random.default_rng(seed)
    drift = (rate - dividend - 0.5 * volatility**2) * maturity
    diffusion = volatility * np.sqrt(maturity)

    if antithetic:
        half = (n_paths + 1) // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)

    terminal = spot * np.exp(drift + diffusion * z)
    discounted_payoff = disc * np.maximum(phi * (terminal - strike), 0.0)

    if control_variate:
        control = disc * terminal
        expected_control = spot * np.exp(-dividend * maturity)
        cov = np.cov(discounted_payoff, control, ddof=1)
        var_control = cov[1, 1]
        beta = cov[0, 1] / var_control if var_control > 0.0 else 0.0
        samples = discounted_payoff - beta * (control - expected_control)
    else:
        samples = discounted_payoff

    price = float(np.mean(samples))
    std_err = float(np.std(samples, ddof=1) / np.sqrt(samples.shape[0]))
    return MonteCarloResult(
        price=price,
        standard_error=std_err,
        n_paths=n_paths,
        method=_method_name(antithetic, control_variate),
    )


def _method_name(antithetic: bool, control_variate: bool) -> str:
    parts = ["monte_carlo"]
    if antithetic:
        parts.append("antithetic")
    if control_variate:
        parts.append("control_variate")
    return "+".join(parts) if len(parts) > 1 else "monte_carlo_plain"
