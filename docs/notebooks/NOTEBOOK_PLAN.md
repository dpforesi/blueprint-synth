# Blueprint Notebook Guide — Plan & Checklist

Comprehensive, example-driven Jupyter notebook series covering every aspect of the Blueprint synthetic data library.

---

## Notebook 1 — Getting Started

- [x] What is Blueprint? Purpose, use cases, design philosophy
- [x] Installation and imports
- [x] Minimal working example: one Feature, one Blueprint, `emit()`
- [x] Inspecting the output DataFrame
- [x] Reproducibility with `seed`

## Notebook 2 — Features Deep Dive

- [x] Feature anatomy: `name`, `dtype`, `base`, `std`, `clip`
- [x] Numeric features: `int` and `float` (normal distribution, clipping)
- [x] Boolean features: `dtype=bool`, `p` parameter
- [x] Categorical features: `dtype="category"`, `values`, `weights`
- [x] Identity features: `dtype="id"` — uuid4, sequential, prefixed styles
- [x] Datetime features: `dtype="datetime"`, `start`/`end` range
- [x] Text/template features: `dtype="str"`, `template`, `pools`
- [x] Computed features: `dtype="computed"`, `formula` lambdas
- [x] Derived features: `derived=True` — accumulates from influences only
- [x] Nullable features: `nullable` parameter for injecting NaNs
- [x] Feature modifiers: `.trend()`, `.seasonality()`, `.noise()`, `.clip()`, `.round()`
- [x] Modifier chaining and order of application

## Notebook 3 — Classes (Population Segments)

- [x] What classes represent and why they matter
- [x] Condition types:
  - [x] Equality: `("col", "==", value)`
  - [x] Comparison: `>`, `>=`, `<`, `<=`, `!=`
  - [x] Between: `("col", "between", (lo, hi))`
  - [x] Membership: `("col", "in", [values])`
  - [x] Random: `("__random__", p)`
  - [x] Callable: `lambda df: ...`
- [x] Overriding feature parameters with `.override()`
- [x] Class overlap behavior: registration order as tiebreaker
- [x] Preset classes: `RandomClass`, `HighValueClass`, `LowValueClass`, `OutlierClass`

## Notebook 4 — Influences (Causal Relationships)

- [x] What influences represent: source -> target causal edges
- [x] `Influence("source").on("target", effect=...)` pattern
- [x] Effect types:
  - [x] Flat additive: `"+5000"`
  - [x] Percentage: `"+12%"`
  - [x] Per-unit additive: `"+110 per unit"`
  - [x] Per-unit percentage: `"-0.6% per unit"`
  - [x] Set: `"=value"` (overwrite)
  - [x] Multiply: `"*1.5"`
- [x] Class-conditional effects with `by_class={...}`
- [x] Gated influences with `when=` conditions
- [x] Custom influence functions with `fn=`
- [x] Boolean sources: effect triggers only on True rows
- [x] Preset influences: `ScalesWith`, `CorrelatedWith`, `Caps`
- [x] Influence variability: `noise_std` — deterministic vs. stochastic effects, reproducibility, `fn=` RNG passthrough

## Notebook 5 — The Dependency DAG

- [x] Why evaluation order matters
- [x] How Blueprint builds the dependency graph from influences
- [x] Topological sort and the emit pipeline
- [x] Cycle detection: `BlueprintCycleError`
- [x] Inspecting the DAG with `bp.validate()`
- [x] Visualizing the dependency graph (networkx/matplotlib example)

## Notebook 6 — Blueprint Assembly & Emission

- [x] Creating a Blueprint: `Blueprint(n=, seed=)`
- [x] Registering components: `add_feature()`, `add_class()`, `add_influence()`
- [x] Method chaining pattern
- [x] `validate()` — what it checks and when to call it
- [x] `describe()` — human-readable blueprint summary
- [x] `emit()` — full pipeline walkthrough
- [x] `emit(describe=True)` and `emit(manifest=)`
- [x] Output formats: `to_csv()`, `to_json()`, `to_manifest()`
- [x] Manifest files: what they contain and how to use them

## Notebook 7 — Presets & Recipes

- [ ] Philosophy: presets are just Python convenience wrappers
- [ ] Class presets walkthrough with examples:
  - [ ] `RandomClass`
  - [ ] `HighValueClass`
  - [ ] `LowValueClass`
  - [ ] `OutlierClass`
- [ ] Influence presets walkthrough with examples:
  - [ ] `ScalesWith`
  - [ ] `CorrelatedWith`
  - [ ] `Caps`
- [ ] Combining presets with custom components
- [ ] When to use presets vs. raw API

## Notebook 8 — Realistic Worked Example

- [ ] End-to-end walkthrough: real estate dataset (mirrors `demo_real_estate.py`)
- [ ] Step-by-step construction with explanations at each stage
- [ ] Analyzing the output: distribution checks, correlation validation
- [ ] Verifying that class overrides and influences produced expected patterns
- [ ] Exporting to CSV/JSON with manifest

## Notebook 9 — Advanced Patterns & Tips

- [ ] Building blueprints programmatically with loops and functions
- [ ] Composing multiple blueprints (e.g., train/test splits with different seeds)
- [ ] Using computed features for complex derived columns
- [ ] Custom influence functions for non-standard transformations
- [ ] Time-series datasets: combining `.trend()`, `.seasonality()`, and computed dates
- [ ] Controlling correlation structure across multiple features
- [ ] Debugging: inspecting intermediate state, reading the DAG, tracing influences
- [ ] Performance considerations for large datasets
