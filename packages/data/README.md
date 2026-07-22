# factorlab-data

**Reusable institutional factor-data infrastructure.**
Acquire, parse, validate, normalize, align, and cache factor return data — then hand it to
any `factorlab_quant` model.

This package is the shared data layer for the FactorLab platform. It is provider-agnostic
and model-agnostic: the same infrastructure serves CAPM, FF3, FF5, and (in future) Carhart,
the q-factor model, and APT **unchanged**.

---

## Architecture — Ports & Adapters

```
                          ┌──────────────────────────────┐
   dataset id  ─────────▶ │        FactorRegistry         │  routes id -> adapter
                          └───────────────┬───────────────┘
                                          │
                          ┌───────────────▼───────────────┐
                          │       FactorDataPort (port)    │  the interface
                          └───────────────┬───────────────┘
                                          │ implements
          Kenneth French .zip/.csv ─────▶ │  KennethFrenchAdapter          (parse: pure)
                                          │  - generic section parser
                                          │  - percent -> decimal, sentinels -> NaN
                                          └───────────────┬───────────────┘
                                                          │ produces
   FactorLoader ──(cache? validate?)──────────────▶ FactorDataset ─▶ FactorPanel(+FactorMetadata)
        │  FactorCache (memory + disk, TTL)                                    │ .to_factor_set()
        │  FactorValidator (severity-tagged report)                           ▼
        │  FactorAlignment (inner-join on dates)                     factorlab_quant.FactorSet
        └───────────────────────────────────────────────────────────────────▶ (any quant model)
```

**Dependency direction.** The layer depends only on NumPy. It produces `factorlab_quant`
FactorSets through a *lazy* import (`FactorPanel.to_factor_set`), so the quant engine is an
**optional peer dependency** — the two packages stay decoupled and install in any order.

---

## Components

| Component | Responsibility |
|-----------|----------------|
| `FactorDataPort` | The interface every provider adapter implements (`parse`, `load`, `available_datasets`). |
| `KennethFrenchAdapter` | Concrete adapter for the Kenneth French Data Library. Generic section parser — **no FF5-specific logic**. |
| `FactorPanel` | Immutable date-indexed matrix of normalized factor returns + metadata. |
| `FactorDataset` | Named group of panels (e.g. monthly + annual) from one source file. |
| `FactorMetadata` | Provenance: source, frequency, units, transformations, date span. |
| `FactorValidator` | Severity-tagged validation (date order, missing/all-missing, constant, extreme values, frequency spacing). |
| `FactorAlignment` | Exact date-intersection joins across panels and asset series (never imputes). |
| `FactorCache` | Two-tier (memory + optional disk) cache, keyed by `(dataset, frequency)`, optional TTL. |
| `FactorRegistry` | Routing table from dataset id → adapter. |
| `FactorLoader` | Orchestration: resolve adapter → cache → parse/fetch → validate → cache → return. |

### The adapter supports
FF3, FF5, Momentum, and research portfolios, at **daily and monthly** frequencies — because
the parser discovers columns from each section header and infers frequency from the date-token
width (8 = daily, 6 = monthly, 4 = annual). Nothing in the parser names a specific factor.

---

## Design rationale

- **Parsing is pure; loading is the only I/O.** `parse(content)` takes bytes/text and returns a
  dataset with zero side effects, so adapters are fully testable offline. `load()` performs
  network I/O only through an **injected `fetcher`** — without one it raises rather than reaching
  the network. Tests never touch the network.
- **Normalization is explicit and logged.** Raw percentages become decimals; the library's
  missing sentinels (`-99.99`, `-999`) become NaN. Both transformations are recorded in
  `FactorMetadata.transformations` for auditability.
- **Validation is separate from parsing.** A panel can be validated regardless of provenance,
  and the caller chooses strictness (`validate` returns a report; `assert_valid` raises).
- **Alignment never fabricates data.** Joins are exact set-intersection on dates. A dropped
  observation is always preferable to an imputed return.
- **Immutability + provenance everywhere.** Every panel is a frozen dataclass carrying the
  metadata needed to trace a regression result back to its exact data vintage.

---

## Quick start (offline)

```python
from factorlab_data import FactorLoader, KennethFrenchAdapter, FactorValidator, FactorCache

loader = FactorLoader(
    KennethFrenchAdapter(),
    cache=FactorCache(),
    validator=FactorValidator(),
)

# `csv_text` is the content of a Kenneth French CSV (already downloaded/unzipped).
dataset = loader.load_from_content(
    csv_text, dataset_id="F-F_Research_Data_5_Factors_2x3", frequency="monthly"
)

panel = dataset.panel("monthly")          # FactorPanel: Mkt-RF, SMB, HML, RMW, CMA, RF
factor_set = panel.to_factor_set()        # -> factorlab_quant FactorSet (RF excluded)
rf = panel.risk_free                      # the RF column, for excess-return conversion
```

### Feeding a model (data → FF5)

```python
from factorlab_quant.models.fama_french_5 import FamaFrench5Model

# `asset` aligned to the panel dates (see FactorAlignment.align_asset_to_panel)
result = FamaFrench5Model().fit(asset, panel, risk_free=rf)
print(result.summary())
```

### Live download (opt-in, requires network)

```python
from factorlab_data import KennethFrenchAdapter, urllib_zip_fetcher

adapter = KennethFrenchAdapter(fetcher=urllib_zip_fetcher)   # you opt into I/O explicitly
dataset = adapter.load("F-F_Research_Data_5_Factors_2x3", frequency="monthly")
```

---

## Testing

```bash
pip install -e ".[dev]"        # plus: pip install -e ../quant  (peer, for integration tests)
pytest
ruff check src tests
mypy src
```

Parsing is validated against synthetic Kenneth-French-format fixtures; an integration test
drives the full path **Kenneth French text → Factor Layer → FF5 model**, proving the design.

---

## References

- Kenneth R. French, Data Library.
- Fama, E. F., & French, K. R. (1993, 2015). Common risk factors; A five-factor asset pricing model.
