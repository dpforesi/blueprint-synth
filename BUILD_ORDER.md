# Blueprint Build Order

Each phase unblocks the next. Complete items in order.

---

## Phase 1 — Feature + Basic Generators ✅
*Unblocks: Classes, flat emit milestone*

- [x] `core/feature.py` — Feature class: constructor, modifier chain (`modifiers` list), generator dispatch
- [x] `generators/numeric.py` — generate_float, generate_int, generate_positive_float, generate_percentage
- [x] `generators/categorical.py` — generate_categorical (weighted sampling, pandas.Categorical)
- [x] `generators/boolean.py` — generate_boolean (Bernoulli sampling)
- [x] `generators/identity.py` — generate_id (uuid4, sequential, prefixed), generate_row_number
- [x] `core/blueprint.py` — add_feature(), flat emit() (no classes/influences)

**Milestone:** `bp.emit()` returns a DataFrame with mixed column types.

---

## Phase 2 — Classes ✅
*Unblocks: class-conditional Influences*

- [x] `core/klass.py` — Class: `when`-condition parsing, mask resolution against a DataFrame, `override()`
- [x] `core/blueprint.py` — `add_class()`, per-row class mask resolution integrated into `emit()`

**Milestone:** Different row populations have different feature distributions.

---

## Phase 3 — Effects & Influences ✅
*Unblocks: DAG (needs Influence edges to sort)*

- [x] `effects/parser.py` — `parse_effect()`: effect string → `(type, params)` tuple
- [x] `effects/applicators.py` — apply_pct, apply_flat, apply_per_unit, apply_per_unit_pct, apply_fn, apply_set, apply_multiply
- [x] `core/influence.py` — Influence: source, `.on()` chaining (target, effect, by_class, fn, when)
- [x] `core/blueprint.py` — `add_influence()`, apply Influences after raw generation, before clip/round

**Milestone:** Causal relationships between columns produce realistic co-movement.

---

## Phase 4 — DAG & Dependency Resolution ✅
*Unblocks: correct multi-level influence chains; `validate()`*

- [x] `core/dag.py` — DAG: add_node, add_edge, topological sort (Kahn's algorithm), cycle detection, BlueprintCycleError
- [x] `core/blueprint.py` — wire DAG into `emit()` column ordering; implement `Blueprint.validate()`

**Milestone:** Influence chains resolve in correct order. Cycles caught at `validate()` time.

---

## Phase 5 — Extended Generators ✅
*Unblocks: full real-estate example; any schema using dates or template strings*

- [x] `generators/temporal.py` — generate_datetime (range + distribution bias), generate_datetime_offset
- [x] `generators/text.py` — generate_text (template strings with pool sampling)
- [x] `core/feature.py` — `dtype="computed"` (formula columns evaluated after influences)

**Milestone:** Full column type coverage — any realistic dataset schema is expressible.

---

## Phase 6 — Emitter & Manifest ✅
*Unblocks: Recipes (need working emit-to-disk to build on)*

- [x] `emitter/formats.py` — to_dataframe, to_csv, to_json, to_manifest
- [x] `core/blueprint.py` — `to_csv()`, `to_json()`, manifest generation
- [x] `core/blueprint.py` — `describe()` human-readable summary; `emit(describe=True)` and `emit(manifest=...)`

**Milestone:** Datasets can be written to disk with full metadata. `describe()` is usable.

---

## Phase 7 — Presets & Recipes ✅
*Unblocks: out-of-the-box usability; reference implementations for tests*

- [x] `presets/classes.py` — HighValueClass, LowValueClass, OutlierClass, RandomClass
- [x] `presets/influences.py` — CorrelatedWith, ScalesWith, Caps
- [x] `presets/recipes.py` — real_estate(), ecommerce(), employee_survey(), web_events()

**Milestone:** Library is immediately useful without writing a Blueprint from scratch.

---

## Phase 8 — Polish & Testing
*Terminal phase — no downstream blockers*

- [x] `tests/test_feature.py` — Feature construction, modifier chain, all dtypes
- [x] `tests/test_class.py` — Class mask resolution (all operators), override merging, edge cases (zero-rows, p=0/1)
- [x] `tests/test_influence.py` — Influence `.on()` chaining, by_class, fn, when
- [x] `tests/test_dag.py` — Topological sort, cycle detection, edge cases (empty, diamond, duplicate edges)
- [x] `tests/test_blueprint.py` — Full emit pipeline, `validate()`, `describe()`, computed columns, presets, manifests
- [x] `tests/test_generators.py` — All generator functions in isolation
- [x] `tests/test_effects.py` — Effect parser and all applicators
- [x] `tests/test_recipes.py` — End-to-end recipe execution and output validation
- [x] Edge cases: zero-row blueprints, all-null columns, single-row blueprints
- [x] Performance: 100k rows generates in < 5s on a laptop (~0.05s measured)

**Milestone:** Library is fully tested, edge-safe, and ready to share.
