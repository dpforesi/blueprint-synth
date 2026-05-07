# Blueprint — Synthetic Dataset Generation Library
## Design & Development Specification

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Design Principles](#2-design-principles)
3. [Core Concepts](#3-core-concepts)
4. [Architecture](#4-architecture)
5. [API Reference](#5-api-reference)
   - 5.1 [Feature](#51-feature)
   - 5.2 [Class](#52-class)
   - 5.3 [Influence](#53-influence)
   - 5.4 [Blueprint](#54-blueprint)
6. [Column Generators](#6-column-generators)
7. [Effect Syntax](#7-effect-syntax)
8. [Dependency DAG](#8-dependency-dag)
9. [Evaluation Order](#9-evaluation-order)
10. [Emitter](#10-emitter)
11. [Presets & Recipes](#11-presets--recipes)
12. [Full Example — Real Estate Dataset](#12-full-example--real-estate-dataset)
13. [File Structure](#13-file-structure)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Scope & Complexity Estimates](#15-scope--complexity-estimates)

---

## 1. Project Overview

**Blueprint** is a pure Python library for generating realistic, structured synthetic datasets. It allows developers, data scientists, and ML practitioners to declaratively define the shape, distribution, and causal relationships of data — then emit that data as a pandas DataFrame, CSV, or JSON.

The core differentiator from other synthetic data tools is that Blueprint treats **feature interactions and contextual effects as first-class citizens**. The same feature can behave differently depending on which "class" of row it belongs to, and features can conditionally influence one another through a resolved dependency graph.

### Primary Use Cases

- Generating ML training datasets with known causal structure
- Building realistic test fixtures for applications and dashboards
- Prototyping data pipelines before real data is available
- Benchmarking analytics tools against data with controlled properties
- Teaching and demonstrating statistical concepts with reproducible datasets

---

## 2. Design Principles

### P1 — Pure Python, minimal dependencies

The only permitted external dependencies are:

- **`numpy`** — vectorized signal generation, random sampling, linear algebra (e.g. Cholesky decomposition for correlated columns)
- **`pandas`** — DataFrame assembly, datetime index generation, CSV/JSON emission

No scikit-learn, scipy, faker, or any other library. The library must be installable with `pip install numpy pandas` and nothing else.

### P2 — Native Python API, no DSL parsing

Blueprint is a Python library, not a configuration language. Users write Python. This means:

- Full access to loops, conditionals, and functions when building blueprints
- No tokenizer or parser to maintain
- Blueprints can be composed from other blueprints programmatically
- Works natively in Jupyter notebooks, scripts, and application code

### P3 — Declarative but not magic

The user declares *what* the data should look like. The library figures out *how* to produce it. But the evaluation order, dependency resolution, and column construction are all inspectable and debuggable — no hidden black boxes.

### P4 — Context-sensitive by design

The same feature can mean different things in different populations. A `has_pool` column should be able to positively influence `price` in luxury neighborhoods and negatively influence it in entry-level ones. This contextual awareness is not an afterthought — it is the central design goal.

### P5 — Composable and extensible

Every component (Feature, Class, Influence) is a standalone Python object. Users can subclass any of them to add custom behavior. Pre-built presets and recipes ship with the library and are themselves just Python.

### P6 — Reproducible by default

Every `Blueprint` accepts a `seed` parameter. With the same seed, the same blueprint always produces identical output.

---

## 3. Core Concepts

There are four first-class concepts in Blueprint:

| Concept | Role | Analogy |
|---|---|---|
| **Feature** | Defines a column and how its raw values are generated | A column schema + distribution |
| **Class** | Defines a named subset of rows with shared characteristics | A population segment or cohort |
| **Influence** | Defines how one feature's value affects another's | A causal edge in a DAG |
| **Blueprint** | Orchestrates all of the above and emits the dataset | The dataset factory |

### How they interact

```
Classes assign rows to populations
    ↓
Features generate raw column values (shaped by Class overrides)
    ↓
Influences modify column values based on other columns and Class membership
    ↓
Blueprint assembles everything into a DataFrame
```

The **target variable** (e.g. `price`, `churn`, `rating`) is just a regular Feature that happens to receive many Influences. Nothing is special-cased.

---

## 4. Architecture

```
blueprint/
│
├── core/
│   ├── feature.py        # Feature definition + generator dispatch
│   ├── klass.py          # Class definition + mask resolution
│   ├── influence.py      # Influence definition + effect application
│   ├── dag.py            # Dependency graph + topological sort
│   └── blueprint.py      # Orchestrator — owns all components, runs emit()
│
├── generators/
│   ├── numeric.py        # Continuous and integer signal generation
│   ├── categorical.py    # Weighted category sampling
│   ├── boolean.py        # Probabilistic and rule-based boolean generation
│   ├── temporal.py       # Datetime column generation (ranges, offsets, timezones)
│   ├── identity.py       # Unique IDs, sequential integers, row numbers
│   └── text.py           # Template-based string columns
│
├── effects/
│   ├── parser.py         # Effect string parsing: "+12%", "+500 per unit", etc.
│   └── applicators.py    # Effect functions: multiply, add, per_unit, fn
│
├── emitter/
│   └── formats.py        # DataFrame, CSV, JSON emission
│
└── presets/
    ├── classes.py         # Built-in Class presets
    ├── influences.py      # Built-in Influence presets
    └── recipes.py         # Full dataset recipes: real_estate(), ecommerce(), etc.
```

---

## 5. API Reference

### 5.1 `Feature`

A `Feature` defines a single column. It specifies the column name, data type, and the generator parameters that control how raw values are sampled. It does not know about other features — that is the job of `Influence`.

#### Constructor

```python
Feature(
    name: str,
    dtype: str | type,           # float, int, bool, "category", "datetime", "id", "str"
    base: float = 0,             # center / mean of the distribution
    std: float = 0,              # standard deviation (numeric types)
    clip: tuple = (None, None),  # hard min/max after generation
    p: float = 0.5,              # probability of True (bool dtype only)
    values: list = None,         # pool of values (category dtype only)
    weights: list = None,        # sampling weights for values (optional)
    derived: bool = False,       # if True, no base sampling — value comes entirely from Influences
    nullable: float = 0.0,       # fraction of rows to set as NaN after generation
    seed: int = None,            # per-feature seed override
)
```

#### Modifier Methods (chainable)

Modifiers are applied to numeric features in the order they are declared. They transform the raw sampled values.

```python
feature.noise(std=50, distribution="gaussian")
# Add random noise to each value.
# distribution options: "gaussian" (default), "uniform", "poisson"

feature.trend(rate=0.005, style="linear")
# Apply a drift that grows over row index.
# rate: fractional change per row (e.g. 0.005 = +0.5% per step)
# style: "linear" (default) | "exponential"

feature.seasonality(period=7, amplitude=200, phase=0)
# Add a sinusoidal oscillation.
# period: in units of rows
# amplitude: peak deviation from base
# phase: offset in radians

feature.spike(at, magnitude, duration=1, shape="flat")
# Inject a transient event at a specific row index or label.
# at: int (row index) or string label (if index is labeled)
# magnitude: multiplier (e.g. 3.0 = 3x) or absolute delta ("+500")
# duration: number of rows the spike persists
# shape: "flat" | "triangle" (ramps up and down)

feature.dropout(rate=0.01, fill=None)
# Randomly set a fraction of values to NaN or a fill value.

feature.clip(min=None, max=None)
# Hard-clamp values after all modifiers are applied.

feature.round(decimals=0)
# Round to specified decimal places after all other transforms.
```

#### Examples

```python
# Continuous float with trend, seasonality, and noise
revenue = Feature("revenue", dtype=float, base=10_000, std=500)
    .trend(rate=0.003)
    .seasonality(period=30, amplitude=2000)
    .noise(std=300)
    .clip(min=0)

# Integer drawn from a distribution
bedrooms = Feature("bedrooms", dtype=int, base=3, std=1, clip=(1, 8))

# Boolean with 30% probability of True
has_pool = Feature("has_pool", dtype=bool, p=0.3)

# Categorical with weighted sampling
tier = Feature("neighborhood_tier", dtype="category",
               values=["luxury", "mid", "entry"],
               weights=[0.2, 0.5, 0.3])

# Derived — value is computed entirely from Influences, no base sampling
price_per_sqft = Feature("price_per_sqft", dtype=float, derived=True)
```

---

### 5.2 `Class`

A `Class` is a named population segment. It defines which rows belong to it (via a mask condition) and optionally overrides the generation parameters for one or more features within that population.

Classes are the mechanism for expressing context-dependent behavior. They do not change how features are defined — they change what parameters are used to generate values *for a specific subset of rows*.

#### Constructor

```python
Class(
    name: str,
    when: tuple | callable,   # condition that selects which rows belong to this class
)
```

The `when` argument can be:

| Form | Example | Meaning |
|---|---|---|
| Equality tuple | `("tier", "==", "luxury")` | Rows where column equals value |
| Comparison tuple | `("income", ">", 80_000)` | Rows where column meets condition |
| Range tuple | `("age", "between", (25, 45))` | Rows where column is within range |
| In-set tuple | `("region", "in", ["North", "East"])` | Rows where column is in a set |
| Callable | `lambda df: df["age"] > df["retirement_age"]` | Arbitrary logic over the DataFrame |
| Probabilistic | `("__random__", p=0.2)` | Randomly assign 20% of rows |

#### Override Methods (chainable)

```python
klass.override(
    feature_name: str,
    **kwargs                  # same kwargs as Feature constructor
)
```

Overrides replace the base Feature's generator parameters for rows in this class. Any parameter not overridden falls back to the Feature's own definition.

```python
luxury = (
    Class("luxury", when=("neighborhood_tier", "==", "luxury"))
    .override("price",      base=900_000, std=200_000)
    .override("sqft",       base=4000, std=1200)
    .override("has_pool",   p=0.75)
    .override("crime_rate", base=4, std=2)
)
```

#### Notes on Class Resolution

- Classes are not mutually exclusive by default. A row can belong to multiple classes if the `when` conditions overlap.
- When multiple classes override the same feature for the same row, the last registered class wins. This is a deliberate design choice — define more specific classes after more general ones.
- A row that belongs to no class uses the base Feature parameters.

---

### 5.3 `Influence`

An `Influence` expresses a causal relationship between two features: *"when feature A has a certain value, feature B's value changes in a certain way."*

Influences are applied after all raw feature values are generated. They are resolved in topological order so that chains of dependencies always evaluate correctly.

#### Constructor & Chaining

```python
Influence(source: str)          # the feature that causes the effect
    .on(
        target: str,            # the feature that is affected
        effect: str | None,     # effect string (see Effect Syntax section)
        by_class: dict | None,  # class-conditional effects {class_name: effect_string}
        fn: callable | None,    # arbitrary function override
        when: tuple | None,     # condition that gates the influence
    )
```

At least one of `effect`, `by_class`, or `fn` must be provided.

#### Effect Forms

```python
# Unconditional scalar effect
Influence("sqft").on("price", effect="+110 per unit")

# Class-conditional effect — the pool example
Influence("has_pool").on("price",
    by_class={
        "luxury": "+12%",
        "mid":    "+4%",
        "entry":  "-3%",
    }
)

# Gated influence — only applies when a condition is met
Influence("crime_rate").on("price",
    effect="-0.6% per unit",
    when=("crime_rate", ">", 10)
)

# Functional influence — full control
Influence("distance_to_school").on("price",
    fn=lambda source_col, target_col, df: target_col - (source_col * 300)
)

# Mixed: class-conditional with a fallback default
Influence("has_garage").on("price",
    by_class={"luxury": "+5%", "mid": "+6%", "entry": "+2%"},
    effect="+3%"              # default for rows not in any named class
)
```

#### Multiple `.on()` calls

A single `Influence` source can affect multiple targets:

```python
Influence("is_weekend")
    .on("foot_traffic", effect="+40%")
    .on("revenue",      effect="+25%")
    .on("staff_needed", effect="+2 per unit")
```

---

### 5.4 `Blueprint`

The `Blueprint` is the top-level orchestrator. It owns all Features, Classes, and Influences, resolves the dependency graph, and runs the generation pipeline.

#### Constructor

```python
Blueprint(
    n: int,          # number of rows to generate
    seed: int = 42,  # global random seed for reproducibility
)
```

#### Registration Methods

```python
bp.add_feature(*features: Feature)         # register one or more Features
bp.add_class(*classes: Class)              # register one or more Classes
bp.add_influence(*influences: Influence)   # register one or more Influences
```

All registration methods return `self` for chaining.

#### Inspection Methods

```python
bp.describe()
# Prints a human-readable summary:
# - Number of rows and columns
# - Each feature: dtype, base, modifiers
# - Each class: condition, row coverage estimate, overrides
# - Each influence: source → target, effect type
# - Dependency graph edges and resolved evaluation order

bp.validate()
# Checks for:
# - Circular dependencies in the Influence DAG
# - References to undefined feature names in Influences or Classes
# - Type conflicts (e.g. Influence on a derived feature with no base)
# - Overlapping class override conflicts
# - Missing required parameters for derived features
# Raises descriptive errors with fix suggestions.
```

#### Emission Methods

```python
# Returns a pandas DataFrame
df = bp.emit()

# Writes CSV to disk, returns the DataFrame
df = bp.to_csv("output.csv", index=False)

# Writes JSON to disk, returns the DataFrame
df = bp.to_json("output.json", orient="records", indent=2)

# Convenience: emit and immediately display summary stats
df = bp.emit(describe=True)
```

---

## 6. Column Generators

Beyond the core numeric/categorical/boolean types, Blueprint ships with a set of specialized column generators for common real-world column patterns. These are specified via the `dtype` argument on `Feature`.

### 6.1 Numeric Generators

| dtype | Description |
|---|---|
| `float` | Continuous values sampled from a normal distribution centered on `base` with `std` |
| `int` | Same as float, rounded to nearest integer |
| `positive_float` | Float clipped at 0 — equivalent to `float` with `clip=(0, None)` |
| `percentage` | Float clipped to [0, 1] |

### 6.2 Categorical Generator

```python
Feature("region", dtype="category",
        values=["North", "South", "East", "West"],
        weights=[0.4, 0.2, 0.3, 0.1])
```

- `values`: list of strings, ints, or any hashable type
- `weights`: optional list of relative sampling weights (need not sum to 1)
- If `weights` is omitted, uniform sampling is used
- Output column uses `pandas.Categorical` dtype for memory efficiency

### 6.3 Boolean Generator

```python
Feature("has_pool", dtype=bool, p=0.3)
```

- `p`: probability of `True` for each row
- Can be overridden per-class
- Class overrides that specify `"+15%"` or `"-10%"` shift `p` by that amount relative to the base

### 6.4 Datetime Generator

Generates a column of datetime values. Does not control the row index — it creates a standalone datetime column.

```python
Feature("signup_date", dtype="datetime",
        start="2020-01-01",
        end="2024-12-31",
        distribution="uniform",    # uniform | recent_biased | early_biased
        tz="UTC",
        freq="D")                  # pandas freq string for rounding output
```

Useful for: signup dates, event timestamps, last-modified columns.

### 6.5 Timestamp Offset Generator

Generates a datetime that is derived from another datetime column plus a random offset. Useful for modeling time-between-events.

```python
Feature("purchase_date", dtype="datetime_offset",
        source="signup_date",
        offset_min="1D",
        offset_max="180D",
        distribution="exponential")  # exponential | uniform | normal
```

### 6.6 Unique ID Generator

```python
Feature("user_id", dtype="id",
        style="uuid4")     # uuid4 | uuid1 | sequential | prefixed

# Sequential integers starting from a base
Feature("order_id", dtype="id",
        style="sequential",
        start=10_000,
        step=1)

# Prefixed IDs like "ORD-00001"
Feature("order_id", dtype="id",
        style="prefixed",
        prefix="ORD-",
        padding=5)
```

### 6.7 Row Number Generator

```python
Feature("row_num", dtype="row_number")
# Simple 0-indexed integer column — equivalent to DataFrame.reset_index()
# Useful as an explicit integer key or for downstream operations that need an ordinal
```

### 6.8 Template String Generator

Generates strings from a template where `{placeholders}` are drawn from other columns or random pools.

```python
Feature("address", dtype="str",
        template="{number} {street_name} {suffix}",
        pools={
            "number":      Feature("__num", dtype=int, base=100, std=800, clip=(1, 9999)),
            "street_name": Feature("__st",  dtype="category", values=["Oak", "Maple", "Main", "Park"]),
            "suffix":      Feature("__sfx", dtype="category", values=["St", "Ave", "Blvd", "Dr"]),
        }
)
```

### 6.9 Computed / Formula Column

Defines a column as an explicit formula over other already-resolved columns. Evaluated after all Influences are applied.

```python
Feature("price_per_sqft", dtype="computed",
        formula=lambda df: df["price"] / df["sqft"])

Feature("profit_margin", dtype="computed",
        formula=lambda df: (df["revenue"] - df["cost"]) / df["revenue"])
```

The difference between `derived=True` and `dtype="computed"`:
- `derived=True` means the column has no base distribution and accumulates Influence effects
- `dtype="computed"` means the column is a deterministic function of other columns with no randomness of its own

---

## 7. Effect Syntax

Influences use a simple string mini-language for expressing common effect types. This avoids the need for lambda functions in most cases while remaining readable.

### Effect String Reference

| String | Meaning | Example |
|---|---|---|
| `"+12%"` | Multiply target by 1.12 | Pool adds 12% to price |
| `"-3%"` | Multiply target by 0.97 | Pool subtracts 3% from price |
| `"+500"` | Add flat value to target | Garage adds $500 |
| `"-200"` | Subtract flat value from target | Crime subtracts $200 |
| `"+110 per unit"` | Add `110 × source_value` | $110 per sqft |
| `"-0.6% per unit"` | Multiply target by `(1 - 0.006 × source_value)` | -0.6% per crime index point |
| `"= 0"` | Set target to a fixed value | Force to zero |
| `"= source"` | Set target equal to source column's value | Copy |
| `"* 2.5"` | Multiply by a constant (non-percentage) | Exact multiplier |

### Boolean Source Features

When the source feature is boolean (`has_pool`, `is_weekend`, etc.), the effect is applied only to rows where the source is `True`. For rows where source is `False`, the target is unchanged.

```python
Influence("is_weekend").on("revenue", effect="+30%")
# Revenue is multiplied by 1.30 on weekend rows.
# Revenue is unchanged on non-weekend rows.
```

### Class Effects on Boolean Features

When an Influence targets a boolean feature, percentage effects shift its `p` parameter:

```python
Influence("neighborhood_tier").on("has_pool",
    by_class={"luxury": "+45%", "entry": "-20%"}
)
# luxury rows: base p=0.3 → p=0.3+0.45=0.75
# entry rows:  base p=0.3 → p=0.3-0.20=0.10
```

---

## 8. Dependency DAG

The Dependency DAG is the internal mechanism that ensures Influences are applied in the correct order — even when chains of dependencies exist.

### How it works

1. Each `Influence` registers a directed edge: `source → target`
2. `Feature`s with `derived=True` or `dtype="computed"` are also registered as nodes
3. The DAG resolver performs a **topological sort** (Kahn's algorithm) to determine evaluation order
4. If a cycle is detected, a `BlueprintCycleError` is raised with the cycle path included in the error message

### Example DAG

```
sqft ──────────────────────────────────────→ price
bedrooms ──────────────────────────────────→ price
has_pool ──────────────────────────────────→ price
crime_rate ────────────────────────────────→ price
neighborhood_tier ─→ has_pool (p override)
price ─────────────────────────────────────→ price_per_sqft (computed)
sqft ──────────────────────────────────────→ price_per_sqft (computed)
```

Evaluation order resolved from this DAG:

```
1. neighborhood_tier   (no dependencies)
2. sqft                (no dependencies)
3. bedrooms            (no dependencies)
4. crime_rate          (no dependencies)
5. has_pool            (depends on neighborhood_tier override)
6. price               (depends on sqft, bedrooms, has_pool, crime_rate)
7. price_per_sqft      (depends on price and sqft)
```

### Rules

- A Feature cannot depend on itself (immediate cycle)
- Influence chains of any length are supported, as long as no cycle exists
- Independent features are generated in registration order (deterministic given a seed)

---

## 9. Evaluation Order

When `emit()` is called, the following steps are executed in strict order:

```
Step 1:  Validate the Blueprint (unless validate=False is passed)
Step 2:  Seed global numpy RNG
Step 3:  Resolve class masks
         - Evaluate each Class's `when` condition against a skeleton DataFrame
         - Store boolean mask arrays indexed by class name
Step 4:  Topological sort of Influence DAG → resolved column order
Step 5:  For each column in resolved order:
    5a.  Identify which Class (if any) each row belongs to
    5b.  Apply Class overrides to generator parameters for relevant rows
    5c.  Generate raw values using the appropriate Generator
    5d.  Apply column-level modifiers (trend, seasonality, noise, spike, dropout)
    5e.  Apply any Influences targeting this column (in DAG order)
    5f.  Apply clip and round
Step 6:  Apply computed columns (dtype="computed", formula=...)
Step 7:  Cast dtypes
Step 8:  Assemble DataFrame (columns in registration order)
Step 9:  Emit to requested format
```

---

## 10. Emitter

The emitter handles serialization. All emit methods return the generated DataFrame in addition to writing files.

```python
# DataFrame (default — no file written)
df = bp.emit()

# CSV
df = bp.to_csv(
    path: str,
    index: bool = False,
    encoding: str = "utf-8",
)

# JSON
df = bp.to_json(
    path: str,
    orient: str = "records",   # pandas orient: records | split | index | columns
    indent: int = 2,
    date_format: str = "iso",
)
```

### Manifest

Every emit can optionally produce a `_manifest.json` alongside the dataset. This file documents the exact generation parameters, class assignments, and influence graph used — making results fully reproducible and auditable.

```python
df = bp.emit(manifest="real_estate_manifest.json")
```

Manifest contents:

```json
{
  "generated_at": "2024-11-01T14:32:00Z",
  "seed": 42,
  "n_rows": 2000,
  "n_columns": 8,
  "features": { ... },
  "classes": {
    "luxury": { "n_rows": 412, "coverage": 0.206, "condition": "neighborhood_tier == luxury" },
    "entry":  { "n_rows": 589, "coverage": 0.295, "condition": "neighborhood_tier == entry" }
  },
  "influence_graph": {
    "edges": [
      { "source": "sqft", "target": "price", "effect": "+110 per unit" },
      ...
    ],
    "evaluation_order": ["neighborhood_tier", "sqft", "bedrooms", "crime_rate", "has_pool", "price", "price_per_sqft"]
  }
}
```

---

## 11. Presets & Recipes

### Built-in Class Presets

Common class patterns that ship ready to use:

```python
from blueprint.presets import HighValueClass, LowValueClass, OutlierClass, RandomClass

# Select top N% of rows by a numeric feature
high_earners = HighValueClass("high_earners", feature="income", top_pct=0.1)

# Randomly assign rows to a class
test_group = RandomClass("test_group", p=0.5)

# Inject a small population of statistical outliers
outliers = OutlierClass("outliers", p=0.02, features=["revenue", "units_sold"], magnitude=5.0)
```

### Built-in Influence Presets

Common influence patterns:

```python
from blueprint.presets import CorrelatedWith, ScalesWith, Caps

# Make two features correlated via a shared influence
CorrelatedWith("revenue", "units_sold", correlation=0.85)

# Soft cap: values above threshold get diminishing returns
Caps("experience_years", target="salary", threshold=20, decay=0.3)
```

### Full Dataset Recipes

Pre-built blueprints for common dataset types. Each is a function that returns a configured `Blueprint` ready to emit:

```python
from blueprint.presets.recipes import real_estate, ecommerce, employee_survey, web_events

df = real_estate(n=5000, seed=42).emit()
df = ecommerce(n=10_000, include_returns=True).emit()
df = employee_survey(n=500, departments=["Engineering", "Sales", "HR"]).emit()
```

Recipes are also good reference implementations showing how to compose the full API.

---

## 12. Full Example — Real Estate Dataset

This is the canonical end-to-end example showing all four primitives working together.

```python
from blueprint import Blueprint, Feature, Class, Influence

# ── Features ────────────────────────────────────────────────────────────────

listing_id     = Feature("listing_id",        dtype="id",       style="prefixed", prefix="LST-", padding=6)
tier           = Feature("neighborhood_tier", dtype="category", values=["luxury", "mid", "entry"], weights=[0.2, 0.5, 0.3])
sqft           = Feature("sqft",              dtype=int,        base=1800, std=600,    clip=(400, 8000))
bedrooms       = Feature("bedrooms",          dtype=int,        base=3,    std=1,      clip=(1, 8))
bathrooms      = Feature("bathrooms",         dtype=float,      base=2.0,  std=0.5,   clip=(1.0, 6.0)).round(1)
has_pool       = Feature("has_pool",          dtype=bool,       p=0.25)
has_garage     = Feature("has_garage",        dtype=bool,       p=0.60)
crime_rate     = Feature("crime_rate",        dtype=float,      base=15,   std=10,    clip=(0, 100))
school_dist_mi = Feature("school_dist_mi",    dtype=float,      base=1.5,  std=1.0,   clip=(0.1, 10.0)).round(2)
price          = Feature("price",             dtype=float,      base=300_000, std=60_000, clip=(50_000, None)).round(-3)
list_date      = Feature("list_date",         dtype="datetime", start="2022-01-01", end="2024-12-31", distribution="recent_biased")
price_per_sqft = Feature("price_per_sqft",    dtype="computed", formula=lambda df: (df["price"] / df["sqft"]).round(2))

# ── Classes ──────────────────────────────────────────────────────────────────

luxury = (
    Class("luxury", when=("neighborhood_tier", "==", "luxury"))
    .override("price",          base=900_000, std=200_000, clip=(400_000, None))
    .override("sqft",           base=4000,    std=1200)
    .override("has_pool",       p=0.75)
    .override("has_garage",     p=0.95)
    .override("crime_rate",     base=4,  std=2)
    .override("school_dist_mi", base=0.8, std=0.4)
)

entry = (
    Class("entry", when=("neighborhood_tier", "==", "entry"))
    .override("price",          base=160_000, std=30_000, clip=(50_000, 250_000))
    .override("sqft",           base=1100,    std=300)
    .override("has_pool",       p=0.08)
    .override("has_garage",     p=0.35)
    .override("crime_rate",     base=28, std=12)
    .override("school_dist_mi", base=2.5, std=1.5)
)

# mid class needs no overrides — base Feature parameters apply

# ── Influences ───────────────────────────────────────────────────────────────

influences = [
    Influence("sqft").on("price",          effect="+110 per unit"),
    Influence("bedrooms").on("price",      effect="+8000 per unit"),
    Influence("bathrooms").on("price",     effect="+6000 per unit"),

    Influence("has_pool").on("price",
        by_class={"luxury": "+12%", "mid": "+4%", "entry": "-3%"}),

    Influence("has_garage").on("price",
        by_class={"luxury": "+5%", "mid": "+6%", "entry": "+2%"}),

    Influence("crime_rate").on("price",
        effect="-0.6% per unit",
        when=("crime_rate", ">", 5)),

    Influence("school_dist_mi").on("price",
        effect="-0.8% per unit"),
]

# ── Assembly ─────────────────────────────────────────────────────────────────

bp = Blueprint(n=2000, seed=7)

bp.add_feature(listing_id, tier, sqft, bedrooms, bathrooms,
               has_pool, has_garage, crime_rate, school_dist_mi,
               price, list_date, price_per_sqft)

bp.add_class(luxury, entry)

bp.add_influence(*influences)

# ── Emit ─────────────────────────────────────────────────────────────────────

bp.validate()
bp.describe()

df = bp.emit()
bp.to_csv("real_estate.csv")
bp.to_json("real_estate.json", manifest="real_estate_manifest.json")
```

---

## 13. File Structure

```
blueprint/
│
├── __init__.py                  # Public API exports
│
├── core/
│   ├── __init__.py
│   ├── feature.py               # Feature class
│   ├── klass.py                 # Class class (named klass to avoid Python keyword)
│   ├── influence.py             # Influence class
│   ├── dag.py                   # DAG node, edge, topological sort (Kahn's algorithm)
│   └── blueprint.py             # Blueprint orchestrator
│
├── generators/
│   ├── __init__.py
│   ├── numeric.py               # float, int, positive_float, percentage
│   ├── categorical.py           # category (weighted sampling)
│   ├── boolean.py               # bool (probabilistic + rule-based)
│   ├── temporal.py              # datetime, datetime_offset
│   ├── identity.py              # uuid4, uuid1, sequential, prefixed, row_number
│   └── text.py                  # template strings
│
├── effects/
│   ├── __init__.py
│   ├── parser.py                # effect string → (type, params) tuple
│   └── applicators.py           # apply_pct, apply_flat, apply_per_unit, apply_fn
│
├── emitter/
│   ├── __init__.py
│   └── formats.py               # to_dataframe, to_csv, to_json, to_manifest
│
├── presets/
│   ├── __init__.py
│   ├── classes.py               # HighValueClass, LowValueClass, OutlierClass, RandomClass
│   ├── influences.py            # CorrelatedWith, ScalesWith, Caps
│   └── recipes.py               # real_estate(), ecommerce(), employee_survey(), web_events()
│
└── tests/
    ├── test_feature.py
    ├── test_class.py
    ├── test_influence.py
    ├── test_dag.py
    ├── test_blueprint.py
    ├── test_generators.py
    ├── test_effects.py
    └── test_recipes.py
```

---

## 14. Implementation Roadmap

### Phase 0 — Scaffolding
- Create package structure and `__init__.py` files
- Define public API exports in top-level `__init__.py`
- Set up test scaffolding

### Phase 1 — Core: Feature + Generators
- Implement `Feature` class with dtype dispatch
- Implement `numeric.py` (float, int)
- Implement `categorical.py`
- Implement `boolean.py`
- Implement `identity.py` (sequential, uuid, prefixed)
- Basic `Blueprint.emit()` with no Influences or Classes (flat table generation)

**Milestone:** Can generate a flat DataFrame with mixed column types.

### Phase 2 — Classes
- Implement `Class` with `when` condition parsing and mask resolution
- Implement `override()` modifier — merge override params with base Feature params
- Integrate class mask resolution into `Blueprint.emit()` pipeline
- Add class coverage reporting to `Blueprint.describe()`

**Milestone:** Can generate datasets where different populations have different distributions.

### Phase 3 — Effects & Influences
- Implement `effects/parser.py` — parse effect strings into typed tuples
- Implement `effects/applicators.py` — apply_pct, apply_flat, apply_per_unit
- Implement `Influence` class with `.on()` chaining
- Wire Influences into Blueprint (apply after raw generation, before clip/round)

**Milestone:** Can express causal relationships between columns.

### Phase 4 — DAG & Dependency Resolution
- Implement `dag.py` — node/edge registration, Kahn's topological sort
- Cycle detection with descriptive error output
- Wire DAG resolver into `Blueprint.emit()` so column generation order is correct
- Implement `Blueprint.validate()`

**Milestone:** Chains of Influences resolve correctly. Cycles are caught at validation time.

### Phase 5 — Extended Generators
- Implement `temporal.py` (datetime, datetime_offset)
- Implement `text.py` (template strings)
- Add `computed` dtype (formula columns)
- Add `derived` feature support (pure Influence accumulation, no base sampling)

**Milestone:** Full column type coverage for realistic dataset schemas.

### Phase 6 — Emitter & Manifest
- Implement `to_csv()`, `to_json()` with full options
- Implement manifest generation
- Implement `Blueprint.describe()` human-readable output

**Milestone:** Full emit pipeline. Datasets are usable.

### Phase 7 — Presets & Recipes
- Implement `presets/classes.py` convenience classes
- Implement `presets/influences.py` convenience influences
- Implement `presets/recipes.py` — at minimum: `real_estate()`, `ecommerce()`, `employee_survey()`
- Recipes serve as both utilities and reference implementations

**Milestone:** Library is immediately useful without writing a Blueprint from scratch.

### Phase 8 — Polish & Testing
- Full test suite for all components
- Edge case handling: zero-row blueprints, all-null columns, single-row blueprints
- Performance profiling — ensure 100k rows generates in < 5 seconds on a laptop
- README and docstrings

---

## 15. Scope & Complexity Estimates

| Module | Complexity | Est. Lines |
|---|---|---|
| `core/feature.py` | Medium | ~180 |
| `core/klass.py` | Medium | ~150 |
| `core/influence.py` | Medium | ~160 |
| `core/dag.py` | High | ~140 |
| `core/blueprint.py` | High | ~200 |
| `generators/numeric.py` | Low | ~80 |
| `generators/categorical.py` | Low | ~60 |
| `generators/boolean.py` | Low | ~60 |
| `generators/temporal.py` | Medium | ~120 |
| `generators/identity.py` | Low | ~80 |
| `generators/text.py` | Medium | ~90 |
| `effects/parser.py` | Medium | ~100 |
| `effects/applicators.py` | Medium | ~100 |
| `emitter/formats.py` | Low | ~80 |
| `presets/` (all) | Low-Medium | ~200 |
| `tests/` | — | ~400 |
| **Total** | | **~2,200** |

A focused developer should be able to complete Phases 1–6 in a few days of part-time work, resulting in a fully functional ~1,500 line core library. Phases 7–8 add polish and make it shareable.
