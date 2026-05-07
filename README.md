# Blueprint

**Blueprint** is a pure-Python library for generating realistic synthetic datasets. Define features, population segments, and causal relationships between columns — then emit a reproducible `pandas.DataFrame` in one call.

```python
from blueprint import Blueprint, Feature, Class, Influence

df = (
    Blueprint(n=1000, seed=42)
    .add_feature(
        Feature("sqft",  dtype=int,   base=1800, std=400,  clip=(500, 5000)),
        Feature("price", dtype=float, base=0,    std=0,    derived=True),
        Feature("tax",   dtype=float, base=0,    std=0,    derived=True),
    )
    .add_class(Class("luxury", when=("sqft", ">=", 2500)))
    .add_influence(
        Influence("sqft").on("price", effect="+155 per unit"),
        Influence("price").on("tax",  effect="+0.012 per unit"),
    )
    .emit()
)
```

## Why Blueprint?

Most synthetic data tools generate columns independently. Blueprint lets you specify **why** one column affects another — and preserves those relationships in the output:

- **Features** — numeric, boolean, categorical, datetime, text/template, computed, and derived columns
- **Classes** — named population segments that override feature parameters for a subset of rows
- **Influences** — causal edges (`source → target`) with rich effect types, optional row-level noise, class-conditional behavior, and gating conditions
- **DAG** — dependencies are topologically sorted so multi-hop chains always evaluate in the right order
- **Reproducibility** — every run with the same `seed` produces identical data; influence noise has its own deterministic sub-seed per edge

## Installation

```bash
pip install blueprint-synth
```

Requires Python 3.10+ and only depends on `numpy` and `pandas`.

```python
import blueprint
```

## Feature overview

### Features

```python
Feature("age",       dtype=int,        base=35,   std=10,  clip=(18, 80))
Feature("active",    dtype=bool,        p=0.7)
Feature("tier",      dtype="category",  values=["bronze", "silver", "gold"], weights=[5, 3, 1])
Feature("joined",    dtype="datetime",  start="2020-01-01", end="2024-12-31")
Feature("user_id",   dtype="id",        style="uuid")
Feature("score",     dtype="computed",  formula=lambda df: df["a"] * 2 + df["b"])
Feature("revenue",   dtype=float,       base=0, std=0, derived=True)  # accumulates from influences only
```

Modifiers chain onto any numeric feature: `.trend()`, `.seasonality()`, `.noise()`, `.clip()`, `.round()`.

### Classes (population segments)

```python
Class("high_value", when=("income", ">=", 100000))
Class("churned",    when=("days_inactive", ">", 90))
Class("sampled",    when=("__random__", 0.2))          # random 20% of rows
Class("custom",     when=lambda df: df["x"] > df["y"])
```

Override any feature parameter for rows that match a class:

```python
Class("vip", when=("tier", "==", "gold")).override("spend", base=5000, std=800)
```

### Influences (causal relationships)

```python
Influence("sqft").on("price", effect="+155 per unit")   # per-unit additive
Influence("has_pool").on("price", effect="+8%")          # percentage
Influence("is_member").on("fee", effect="-20")           # flat additive (boolean source)
Influence("region").on("price", by_class={              # class-conditional
    "urban": "+15%", "suburban": "+5%"
}, effect="+0%")
Influence("distance").on("price",                        # custom function
    fn=lambda src, tgt, df: tgt - src * 250)
```

Add row-level noise to any numeric effect for more realistic variation:

```python
Influence("sqft").on("price", effect="+155 per unit", noise_std=0.1)
# effective rate ~ N(155, 15.5) per row, fully reproducible
```

### Output formats

```python
df = bp.emit()                         # pandas DataFrame
df = bp.emit(describe=True)            # prints blueprint summary first
df = bp.emit(manifest="meta.json")     # writes a JSON config sidecar
bp.to_csv("data.csv")
bp.to_json("data.json", manifest="meta.json")
```

## Notebook guide

The `docs/notebooks/` directory contains a step-by-step notebook series covering every aspect of the library:

| Notebook | Topic |
|---|---|
| [01 — Getting Started](docs/notebooks/01_getting_started.ipynb) | Installation, minimal example, reproducibility |
| [02 — Features Deep Dive](docs/notebooks/02_features_deep_dive.ipynb) | All dtype options, modifiers, computed & derived columns |
| [03 — Classes](docs/notebooks/03_classes.ipynb) | Population segments, condition types, presets |
| [04 — Influences](docs/notebooks/04_influences.ipynb) | Effect strings, by_class, when=, fn=, noise_std, presets |
| [05 — The Dependency DAG](docs/notebooks/05_dependency_dag.ipynb) | Topological sort, cycle detection, visualization |
| [06 — Assembly & Emission](docs/notebooks/06_assembly_and_emission.ipynb) | Blueprint construction, validate, describe, emit, output formats |

## Preset library

```python
from blueprint.presets.classes import RandomClass, HighValueClass, LowValueClass, OutlierClass
from blueprint.presets.influences import ScalesWith, CorrelatedWith, Caps
```

```python
bp.add_class(HighValueClass("rich", feature="income", top_pct=0.2))
bp.add_influence(CorrelatedWith("income", "spend", correlation=0.75))
bp.add_influence(Caps("experience", "salary", threshold=10, decay=0.05))
```

## License

MIT — see [LICENSE](LICENSE).
