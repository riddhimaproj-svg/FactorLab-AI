# factorlab-quant

**Institutional-grade multi-factor asset-pricing engine.**
A generic *linear factor modeling framework* — the numerical core of the FactorLab AI platform.

Pure, typed, side-effect-free. It knows nothing about HTTP, databases, or data providers:
it takes return arrays in and returns immutable, fully-documented, serializable result
objects. That isolation makes it independently installable and testable, and reusable from
a notebook, a CLI, a batch job, or the FastAPI service.

---

## What this package is

A single engine (`LinearFactorModel`) performs *all* regression work — alignment, design
construction, estimation, robust inference, diagnostics, prediction, and serialization.
Concrete asset-pricing models are **thin subclasses that specify only their factors**.
Adding a model is additive; it writes no regression code.

| Model | Factors | Status |
|-------|---------|--------|
| **CAPM** (Sharpe–Lintner, Jensen's α) | Mkt-RF | ✅ Production |
| **Fama–French 3** (1993) | Mkt-RF, SMB, HML | ✅ Production |
| Carhart 4 | Mkt-RF, SMB, HML, MOM | 🔜 trivial subclass |
| Fama–French 5 | Mkt-RF, SMB, HML, RMW, CMA | 🔜 trivial subclass |
| Hou–Xue–Zhang q | Mkt-RF, ME, I/A, ROE | 🔜 trivial subclass |
| APT / user-defined | arbitrary priced factors | ✅ use `LinearFactorModel` directly |

---

## The theory

A linear factor model explains an asset's excess return as a linear combination of
systematic factor returns plus an intercept and idiosyncratic noise:

```
R_i,t − R_f,t  =  α_i  +  Σ_k β_i,k · F_k,t  +  ε_i,t
```

- **β_i,k** — the asset's loading (exposure) on factor *k*.
- **α_i** — the intercept (Jensen's alpha): average return not explained by the factors.
  The central test is `H0: α = 0`.
- Standard errors default to **Newey–West (HAC)** with a Bartlett kernel, because return
  residuals are heteroskedastic and serially correlated.

**CAPM is the one-factor special case** (K = 1, F₁ = Mkt-RF). FF3/FF5/Carhart/q differ
*only* in which factors enter the sum — which is exactly why the framework needs to be
written once. Each future model is roughly:

```python
@register_model("FF3")
class FamaFrench3(LinearFactorModel):
    def __init__(self, estimator=None):
        super().__init__("Fama-French 3-Factor",
                         factor_names=("Mkt-RF", "SMB", "HML"),
                         estimator=estimator)
```

---

## Architecture

```
                         ┌──────────────────────────────┐
                         │        AbstractFactorModel     │  thin ABC:
                         │  estimator + name + factors    │  identity & DI
                         └───────────────┬────────────────┘
                                         │ subclass
                         ┌───────────────▼────────────────┐
   Factor  ─────────────▶│        LinearFactorModel        │──────▶ FactorModelResult
   FactorSet  (design) ─▶│  align · design · fit · infer   │       (coeffs, diagnostics,
                         │  diagnostics · predict · meta   │        prediction, to_dict)
                         └───────────────┬────────────────┘
                                         │ subclass (factors only)
                         ┌───────────────▼────────────────┐
                         │   CAPM   FF3   FF5   Carhart …  │──────▶ CAPMResult (+α, β test,
                         └────────────────────────────────┘        Treynor, risk split)
        registry:  register_model / get_model / list_models  ◀── models self-register

        estimation/  OLS + White(HC0/HC1) + Newey–West(HAC)     ← injected (Dependency Inversion)
        diagnostics/ Jarque–Bera · Breusch–Pagan · Durbin–Watson · F-test
        core/        RegressionResult · CoefficientEstimate · errors  (framework-free)
```

Dependency direction points inward: `models → estimation → diagnostics → core`. `core`
depends only on NumPy/SciPy. The estimator is **injected** into every model, so inference
behavior is swappable without touching model code.

### Package layout

```
src/factorlab_quant/
├── core/            # framework-free primitives: typed results, protocols, errors
├── estimation/      # OLS + White (HC0/HC1) + Newey–West (HAC) covariance
├── diagnostics/     # residual moments + specification tests
├── models/
│   ├── base.py                 # AbstractFactorModel (thin ABC)
│   ├── factors.py              # Factor, FactorSet (immutable data abstraction)
│   ├── linear_factor_model.py  # LinearFactorModel + FactorModelResult (all shared logic)
│   ├── capm.py                 # CAPM + CAPMResult (thin subclass)
│   └── registry.py             # register_model / get_model / list_models
└── utils/           # validation guards + observation alignment
```

---

## Installation

```bash
pip install -e ".[dev]"     # includes pandas, statsmodels, pytest, hypothesis
# or runtime only:
pip install -e .
```

Requires Python ≥ 3.11.

---

## Quick start — CAPM

```python
import numpy as np
from factorlab_quant import CAPM

rng = np.random.default_rng(42)
market_excess = rng.normal(0.006, 0.043, size=180)
asset_excess = 0.002 + 1.15 * market_excess + rng.normal(0, 0.018, size=180)

result = CAPM().fit(asset_excess, market_excess, returns_are_excess=True)
print(result.summary())

result.alpha.estimate            # Jensen's alpha (per period)
result.beta.estimate             # market beta
result.beta_p_vs_one             # H0: beta = 1
result.annualized_alpha          # geometrically annualized alpha
result.systematic_variance_ratio # R² — priced (market) risk
result.treynor_ratio
```

## Quick start — arbitrary factor model

```python
import numpy as np
from factorlab_quant import LinearFactorModel, Factor, FactorSet

rng = np.random.default_rng(0)
n = 240
mkt = rng.normal(0.005, 0.04, n)
smb = rng.normal(0.001, 0.02, n)
hml = rng.normal(0.002, 0.03, n)
asset = 0.001 + 1.1*mkt - 0.3*smb + 0.5*hml + rng.normal(0, 0.01, n)

factors = FactorSet([
    Factor("Mkt-RF", mkt, frequency="monthly", source="Kenneth French Data Library"),
    Factor("SMB", smb, frequency="monthly"),
    Factor("HML", hml, frequency="monthly"),
])

model = LinearFactorModel("My 3-Factor", factor_names=("Mkt-RF", "SMB", "HML"))
result = model.fit(asset, factors, covariance_type="HAC")

print(result.summary())
result.factor_loading("HML").estimate
```

## Prediction API

Available on every `FactorModelResult`:

```python
result.predict({"Mkt-RF": 0.02, "SMB": -0.01, "HML": 0.03})   # model-implied excess return
result.predict_excess_return(scenario)                         # alias, asset-pricing framing
result.expected_return(scenario, risk_free=0.002)              # excess + risk-free
result.confidence_interval(scenario, level=0.95)               # (lower, upper) for the mean
result.prediction_interval(scenario, level=0.95)               # (lower, upper) for a new obs
```

The confidence interval reflects the sampling uncertainty of the fitted mean
(`x₀' Σ_β x₀`); the prediction interval additionally includes idiosyncratic residual
variance and is therefore always wider.

## Registry

```python
from factorlab_quant import get_model, list_models, create_model
list_models()            # ['CAPM']  (grows as models are registered)
model = create_model("CAPM")
```

## Serialization

```python
d = result.to_dict()                       # JSON-compatible dict (schema_version tagged)
s = result.to_json()
restored = FactorModelResult.from_json(s)   # dispatches back to the correct subclass
import pickle; pickle.loads(pickle.dumps(result))   # pickle-compatible
```

---

## Developer guide — adding a new model

1. **Subclass `LinearFactorModel`.** In `__init__`, call `super().__init__` with the model
   name and ordered `factor_names`. Decorate with `@register_model("NAME")`.
2. **(Optional) override `fit`** if the model needs bespoke input handling (e.g. CAPM
   builds the market-excess factor from raw returns). Otherwise the inherited `fit`
   accepts a `FactorSet` directly.
3. **(Optional) return a richer result** by overriding `_make_result` to construct a
   subclass of `FactorModelResult` (as `CAPMResult` does). Only do this for
   model-specific statistics; the generic result already covers coefficients, diagnostics,
   prediction, and serialization.
4. **Add tests**: parameter recovery on a known DGP, a `statsmodels` cross-validation, and
   any model-specific invariants.

Because the specification is the *only* thing that changes, FF3/FF5/Carhart/q are each a
few lines plus tests.

### Design rules enforced

- **Immutability**: results and factors are frozen dataclasses with read-only arrays.
- **Dependency Inversion**: the OLS estimator is injected into models.
- **Fail loud, fail early**: malformed input raises precise typed errors before any math
  (`DuplicateFactorError`, `ConstantFactorError`, `FrequencyMismatchError`,
  `InsufficientDataError`, `CollinearityError`, `NonFiniteError`, …).
- **Validation coverage**: duplicate factor names, constant/singular factors, misaligned
  frequencies, non-finite data, and (opt-in) duplicate observations are all rejected.

---

## Testing

```bash
pytest                      # full suite (unit + property + statsmodels cross-validation)
pytest -m validation        # only cross-validation vs statsmodels
pytest -m property          # only Hypothesis property tests
pytest --cov=factorlab_quant
```

The suite validates the engine three ways: against a **known DGP** (parameter recovery),
against **statsmodels** (standard errors, p-values, information criteria, diagnostics to
machine tolerance), and via **property-based invariants** (R² ∈ [0,1], PSD covariance,
prediction-interval ⊇ confidence-interval, serialization fidelity, CAPM ≡ generic
one-factor model).

---

## References

- Sharpe (1964); Lintner (1965); Mossin (1966) — CAPM.
- Jensen (1968) — the time-series alpha test.
- Fama & French (1993, 2015); Carhart (1997); Hou, Xue & Zhang (2015) — multi-factor models.
- White (1980); Newey & West (1987, 1994) — robust covariance estimation.
- Breusch & Pagan (1979); Koenker (1981); Jarque & Bera (1980) — specification tests.
```
