"""
Blueprint — Real Estate Dataset Demo
=====================================
Demonstrates the full Blueprint API:
  - Feature: all dtypes, modifiers (trend, seasonality, noise, clip, round)
  - Class: condition types, overrides
  - Influence: flat, pct, per_unit, by_class, gated (when=), fn
  - Computed columns and derived features
  - Presets: RandomClass, HighValueClass, ScalesWith, CorrelatedWith, Caps
  - emit(), validate(), describe(), to_csv(), to_json() with manifest

Run from the DataSmythe root:
    python demo_real_estate.py
"""

import pandas as pd
import numpy as np

from blueprint import Blueprint, Feature, Class, Influence
from blueprint.presets.classes import RandomClass, HighValueClass, LowValueClass
from blueprint.presets.influences import ScalesWith, CorrelatedWith, Caps

pd.set_option("display.float_format", "${:,.0f}".format)
pd.set_option("display.max_columns", 14)
pd.set_option("display.width", 140)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FEATURES
# Each Feature defines one column: its name, data type, and generation params.
# Features are independent — they don't know about each other. Interactions
# are expressed via Influences (see Section 3).
# ──────────────────────────────────────────────────────────────────────────────

# dtype="id"  →  unique identifiers; style options: "uuid4", "sequential", "prefixed"
listing_id = Feature(
    "listing_id",
    dtype="id",
    style="prefixed",
    prefix="LST-",
    padding=6,          # zero-padded: LST-000001, LST-000002, …
)

# dtype="category"  →  weighted categorical sampling (returns pd.Categorical)
neighborhood_tier = Feature(
    "neighborhood_tier",
    dtype="category",
    values=["luxury", "mid", "entry"],
    weights=[0.20, 0.50, 0.30],    # 20% luxury, 50% mid, 30% entry
)

# dtype=int  →  normal distribution rounded to integer; clip=(min, max) hard-clamps
sqft = Feature("sqft", dtype=int, base=1_800, std=600, clip=(400, 8_000))

bedrooms  = Feature("bedrooms",  dtype=int,   base=3,   std=1,   clip=(1, 8))
bathrooms = Feature(
    "bathrooms",
    dtype=float,
    base=2.0,
    std=0.5,
    clip=(1.0, 6.0),
)

# Modifier: .round(decimals) — applied after clip
bathrooms.round(1)

# dtype=bool  →  Bernoulli sampling; p = probability of True
has_pool   = Feature("has_pool",   dtype=bool, p=0.25)
has_garage = Feature("has_garage", dtype=bool, p=0.60)

crime_rate     = Feature("crime_rate",     dtype=float, base=15,  std=10,  clip=(0, 100))
school_dist_mi = Feature("school_dist_mi", dtype=float, base=1.5, std=1.0, clip=(0.1, 10.0))

# Modifier: .round() chains directly after Feature construction
school_dist_mi.round(2)

# Price with TIME-SERIES MODIFIERS:
#   .trend(rate)       →  adds base × rate × row_index to each value
#                         rate=0.0003 gives ~12% appreciation over 400 rows
#   .seasonality(...)  →  sinusoidal oscillation over row index
#                         period=52 mimics annual cycles if rows ≈ weekly listings
#   .noise(std)        →  per-row Gaussian jitter on top of trend + seasonality
#   .round(-3)         →  round to nearest $1,000
#
# KEY POINT: trend and seasonality operate on ROW INDEX, not calendar date.
# To make this meaningful as a time-series, either:
#   a) use dtype="computed" for list_date so row order == time order (shown below), or
#   b) sort the emitted DataFrame by list_date after emit().
price = (
    Feature("price", dtype=float, base=300_000, std=60_000, clip=(50_000, None))
    .trend(rate=0.0003, style="linear")          # linear market appreciation
    .seasonality(period=52, amplitude=12_000)    # annual demand cycle
    .noise(std=5_000, distribution="gaussian")   # per-listing idiosyncratic noise
    .round(-3)
)

# dtype="datetime"  →  uniform random timestamps between start and end
list_date_random = Feature(
    "list_date_random",
    dtype="datetime",
    start="2022-01-01",
    end="2024-12-31",
)

# dtype="computed"  →  formula runs AFTER all influences are applied
# Evenly-spaced dates: row 0 = 2022-01-01, row 399 = ~2024-12-31
# This makes row order == time order, so .trend() on price is meaningful.
list_date = Feature(
    "list_date",
    dtype="computed",
    formula=lambda df: pd.Series([
        pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(i * (3 * 365 / 400)))
        for i in range(len(df))
    ]),
)

# dtype="computed"  →  derived formula column evaluated after all influences
# price_per_sqft depends on price (which has been shaped by all Influences first)
price_per_sqft = Feature(
    "price_per_sqft",
    dtype="computed",
    formula=lambda df: (df["price"] / df["sqft"]).round(2),
)

# dtype="str"  →  template string; {placeholders} filled from pool Features each row
address = Feature(
    "address",
    dtype="str",
    template="{number} {street_name} {suffix}",
    pools={
        "number":      Feature("__num", dtype=int, base=500, std=400, clip=(1, 9999)),
        "street_name": Feature("__st",  dtype="category",
                               values=["Oak", "Maple", "Main", "Park", "Cedar", "Elm"]),
        "suffix":      Feature("__sfx", dtype="category",
                               values=["St", "Ave", "Blvd", "Dr", "Ln"]),
    },
)

# derived=True  →  column starts at zero and accumulates Influence effects only
# (no base distribution of its own — its value is entirely determined by Influences)
price_change_pct = Feature("price_change_pct", dtype=float, derived=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CLASSES
# A Class defines a named population segment and optionally overrides Feature
# parameters for rows in that segment. Classes are NOT mutually exclusive.
# When multiple classes overlap on the same row, the last-registered class wins.
# ──────────────────────────────────────────────────────────────────────────────

# Condition type: equality tuple ("col", "==", value)
# .override() merges new params into the base Feature; un-overridden params stay.
luxury = (
    Class("luxury", when=("neighborhood_tier", "==", "luxury"))
    .override("price",          base=900_000, std=200_000, clip=(400_000, None))
    .override("sqft",           base=4_000,   std=1_200)
    .override("has_pool",       p=0.75)   # 75% of luxury homes have a pool
    .override("has_garage",     p=0.95)
    .override("crime_rate",     base=4,   std=2)
    .override("school_dist_mi", base=0.8, std=0.4)
)

entry = (
    Class("entry", when=("neighborhood_tier", "==", "entry"))
    .override("price",          base=160_000, std=30_000, clip=(50_000, 250_000))
    .override("sqft",           base=1_100,   std=300)
    .override("has_pool",       p=0.08)
    .override("has_garage",     p=0.35)
    .override("crime_rate",     base=28,  std=12)
    .override("school_dist_mi", base=2.5, std=1.5)
)

# mid class uses base Feature params — no override needed, but MUST be registered
# so that by_class={"mid": "+4%"} in Influences can find its mask.
mid = Class("mid", when=("neighborhood_tier", "==", "mid"))

# Condition type: comparison tuple ("col", ">", value)
# Rows with crime_rate > 40 are tagged as high-crime regardless of tier
high_crime = (
    Class("high_crime", when=("crime_rate", ">", 40))
    .override("price", base=200_000, std=50_000, clip=(50_000, 350_000))
)

# Condition type: callable  →  arbitrary pandas logic over the full skeleton DataFrame
large_home = Class(
    "large_home",
    when=lambda df: (df["sqft"] > 3_000) & (df["bedrooms"] >= 4),
)
large_home.override("price", base=600_000, std=100_000, clip=(300_000, None))

# Condition type: ("__random__", p)  →  randomly assign p fraction of rows
# Using the RandomClass preset (equivalent to Class("name", when=("__random__", p)))
investors = RandomClass("investor_purchase", p=0.15)

# HighValueClass preset: selects top N% of rows by a numeric feature
top_sqft = HighValueClass("large_sqft_tier", feature="sqft", top_pct=0.10)

# Condition type: ("col", "between", (lo, hi))
mid_sqft_class = Class("mid_sqft", when=("sqft", "between", (1_400, 2_200)))

# Condition type: ("col", "in", [values])
coastal_tiers = Class(
    "coastal_tier",
    when=("neighborhood_tier", "in", ["luxury", "mid"]),
)
coastal_tiers.override("school_dist_mi", base=0.9, std=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — INFLUENCES
# An Influence expresses a causal relationship: "when source has a value,
# target's value changes." Influences are resolved in topological order.
# ──────────────────────────────────────────────────────────────────────────────

# Effect type: "+N per unit"  →  target += N × source
# Each extra square foot adds $110 to price
sqft_influence = Influence("sqft").on("price", effect="+110 per unit")

# Effect type: "+N" (flat)  →  target += N
bedrooms_influence  = Influence("bedrooms").on("price",  effect="+8000 per unit")
bathrooms_influence = Influence("bathrooms").on("price", effect="+6000 per unit")

# Effect type: by_class dict  →  different pct effects per population
# Boolean source: effect only applies to rows where has_pool == True
pool_influence = Influence("has_pool").on(
    "price",
    by_class={
        "luxury": "+12%",   # pool adds 12% in luxury neighborhoods
        "mid":    "+4%",    # adds 4% in mid
        "entry":  "-3%",    # actually hurts value in entry (maintenance cost signal)
    },
)

garage_influence = Influence("has_garage").on(
    "price",
    by_class={
        "luxury": "+5%",
        "mid":    "+6%",
        "entry":  "+2%",
    },
    effect="+3%",   # default fallback for rows not in any of the named classes
)

# Effect type: "N% per unit"  →  target *= (1 + N/100 × source)
# when= gate: only fires for rows satisfying the condition
crime_influence = Influence("crime_rate").on(
    "price",
    effect="-0.6% per unit",
    when=("crime_rate", ">", 5),   # only penalize if crime index > 5
)

school_influence = Influence("school_dist_mi").on(
    "price",
    effect="-0.8% per unit",
)

# Effect type: fn= callable  →  full control over the transformation
# fn receives (source_col, target_col, df) and returns new target_col
def garage_size_bonus(source_col, target_col, df):
    """Luxury homes with garage get an extra $20k premium (size adjustment)."""
    luxury_mask = df["neighborhood_tier"] == "luxury"
    bonus = np.where(source_col & luxury_mask, 20_000, 0)
    return target_col + bonus

luxury_garage_influence = Influence("has_garage").on(
    "price",
    fn=garage_size_bonus,
)

# ScalesWith preset: shorthand for "+rate per unit"
sqft_to_price_change = ScalesWith("sqft", "price_change_pct", rate=0.00005)

# CorrelatedWith preset: imposes approximate Pearson correlation via normalized fn
# crime_rate and school_dist_mi are both negatively correlated with desirability
crime_school_correlation = CorrelatedWith("crime_rate", "school_dist_mi", correlation=0.4)

# Caps preset: soft cap — values of source above threshold cause diminishing returns
# Experience (here school proximity) beyond 5 miles has less marginal price impact
dist_cap = Caps("school_dist_mi", "price", threshold=5.0, decay=0.02)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BLUEPRINT ASSEMBLY
# Blueprint is the orchestrator. It owns all Features, Classes, and Influences,
# resolves the dependency graph, and runs the emit pipeline.
# ──────────────────────────────────────────────────────────────────────────────

bp = Blueprint(n=400, seed=7)

# Registration order = column order in the emitted DataFrame
bp.add_feature(
    listing_id,
    neighborhood_tier,
    sqft,
    bedrooms,
    bathrooms,
    has_pool,
    has_garage,
    crime_rate,
    school_dist_mi,
    price,
    list_date,          # computed: evenly-spaced dates (row 0 = earliest)
    price_per_sqft,     # computed: price / sqft (evaluated after influences)
    address,            # template string: "123 Oak St"
    price_change_pct,   # derived: starts at 0, accumulates influence effects only
)

# Classes are evaluated in registration order.
# If rows match multiple classes, the last-registered class wins per feature.
bp.add_class(
    luxury,
    mid,            # no overrides, but needed for by_class influence lookup
    entry,
    high_crime,
    large_home,
    coastal_tiers,
)

# Influences are applied in topological order (dependency-resolved).
# Blueprint.validate() will raise BlueprintCycleError if a cycle is detected.
bp.add_influence(
    sqft_influence,
    bedrooms_influence,
    bathrooms_influence,
    pool_influence,
    garage_influence,
    crime_influence,
    school_influence,
    luxury_garage_influence,
    sqft_to_price_change,
    crime_school_correlation,
    dist_cap,
)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — VALIDATE & DESCRIBE
# validate() checks for cycles, undefined references, and missing formulas.
# describe() prints a human-readable summary of the blueprint.
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("BLUEPRINT DESCRIPTION")
print("=" * 70)
bp.validate()   # raises BlueprintCycleError or ValueError on problems
bp.describe()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EMIT
# emit() runs the full pipeline:
#   1. Generate base values per feature
#   2. Resolve class masks and apply overrides
#   3. Apply influences in topological order
#   4. Evaluate computed columns (formula=)
#   5. Return a pandas DataFrame
# ──────────────────────────────────────────────────────────────────────────────

df = bp.emit()

print()
print("=" * 70)
print(f"EMITTED: {len(df)} rows × {len(df.columns)} columns")
print("=" * 70)
print(df[["listing_id", "neighborhood_tier", "sqft", "bedrooms",
          "price", "price_per_sqft", "list_date", "address"]].head(8).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — OUTPUT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("PRICE BY NEIGHBORHOOD TIER")
print("=" * 70)
for tier, grp in df.groupby("neighborhood_tier", observed=True):
    print(
        f"  {tier:<8}  n={len(grp):>4}  "
        f"mean=${grp['price'].mean():>10,.0f}  "
        f"min=${grp['price'].min():>8,.0f}  "
        f"max=${grp['price'].max():>9,.0f}  "
        f"ppsf=${grp['price_per_sqft'].mean():>6.2f}"
    )

print()
print("=" * 70)
print("POOL PREMIUM BY TIER  (shows by_class influence working)")
print("=" * 70)
for tier, grp in df.groupby("neighborhood_tier", observed=True):
    with_pool    = grp[grp["has_pool"] == True]["price"].mean()
    without_pool = grp[grp["has_pool"] == False]["price"].mean()
    if not (pd.isna(with_pool) or pd.isna(without_pool)):
        delta_pct = (with_pool / without_pool - 1) * 100
        sign = "+" if delta_pct > 0 else ""
        print(f"  {tier:<8}  pool=${with_pool:>10,.0f}  no_pool=${without_pool:>10,.0f}  "
              f"delta={sign}{delta_pct:.1f}%")

print()
print("=" * 70)
print("PRICE TREND OVER TIME  (row order == time order via computed list_date)")
print("Shows .trend() + .seasonality() modifiers at work")
print("=" * 70)
chunk = len(df) // 4
labels = ["2022 H1", "2022 H2–2023 H1", "2023 H2–2024 H1", "2024 H2"]
for i, label in enumerate(labels):
    grp     = df.iloc[i * chunk : (i + 1) * chunk]
    mid_grp = grp[grp["neighborhood_tier"] == "mid"]
    date_lo = grp["list_date"].min()
    date_hi = grp["list_date"].max()
    mid_mean = mid_grp["price"].mean() if len(mid_grp) else float("nan")
    print(
        f"  {label:<18}  "
        f"dates {date_lo}–{date_hi}  "
        f"all=${grp['price'].mean():>10,.0f}  "
        f"mid-only=${mid_mean:>10,.0f}"
    )

print()
print("=" * 70)
print("CRIME ↔ SCHOOL DISTANCE CORRELATION")
print("(CorrelatedWith preset: crime_rate → school_dist_mi, r≈0.4)")
print("=" * 70)
r = np.corrcoef(df["crime_rate"], df["school_dist_mi"])[0, 1]
print(f"  Pearson r(crime_rate, school_dist_mi) = {r:.3f}")

print()
print("=" * 70)
print("PRICE_CHANGE_PCT  (derived=True — accumulates Influence effects only)")
print("=" * 70)
print(f"  mean = {df['price_change_pct'].mean():.4f}  "
      f"std = {df['price_change_pct'].std():.4f}  "
      f"range = [{df['price_change_pct'].min():.4f}, {df['price_change_pct'].max():.4f}]")

print()
print("=" * 70)
print("ADDRESS COLUMN  (dtype='str' template generator)")
print("=" * 70)
print("  " + "\n  ".join(df["address"].head(8).tolist()))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EMIT TO DISK
# to_csv() and to_json() both emit the DataFrame and write a file.
# Both return the DataFrame so you can chain further operations.
# manifest= writes a JSON sidecar with full generation metadata.
# ──────────────────────────────────────────────────────────────────────────────

df_csv = bp.to_csv("real_estate.csv")
print()
print("=" * 70)
print("WRITTEN: real_estate.csv")
print("=" * 70)
print(f"  Rows: {len(df_csv)}  Columns: {list(df_csv.columns)}")

df_json = bp.to_json("real_estate.json", manifest="real_estate_manifest.json")
print()
print("=" * 70)
print("WRITTEN: real_estate.json  +  real_estate_manifest.json")
print("=" * 70)

import json, os
with open("real_estate_manifest.json") as f:
    manifest = json.load(f)
print(f"  Manifest keys: {list(manifest.keys())}")
print(f"  seed={manifest['seed']}  n_rows={manifest['n_rows']}")
print(f"  Features: {list(manifest['features'].keys())}")

# emit() also accepts manifest= directly
df2 = bp.emit(manifest="real_estate_manifest2.json")
print()
print("  Also wrote real_estate_manifest2.json via emit(manifest=...)")

# emit(describe=True) prints the describe() summary before returning
print()
print("  emit(describe=True) would reprint the blueprint description above.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PRESET CLASSES DEMO  (separate small blueprint)
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("PRESET CLASSES DEMO")
print("=" * 70)

bp_preset = Blueprint(n=1_000, seed=99)
bp_preset.add_feature(
    Feature("income",       dtype=float, base=55_000, std=20_000, clip=(15_000, None)),
    Feature("credit_score", dtype=int,   base=680,    std=80,     clip=(300, 850)),
    Feature("loan_amount",  dtype=float, base=200_000, std=60_000, clip=(10_000, None)),
)

# HighValueClass: selects top 10% by income; override loan eligibility
bp_preset.add_class(
    HighValueClass("high_earners", feature="income", top_pct=0.10)
    .override("loan_amount", base=500_000, std=100_000),
)

# LowValueClass: selects bottom 20% by credit_score; restrict loan amount
bp_preset.add_class(
    LowValueClass("poor_credit", feature="credit_score", bottom_pct=0.20)
    .override("loan_amount", base=80_000, std=20_000, clip=(10_000, 150_000)),
)

# RandomClass: randomly flag 5% of applicants for manual review
bp_preset.add_class(RandomClass("manual_review", p=0.05))

df_preset = bp_preset.emit()
print(f"  high_earners  mean_loan=${df_preset[df_preset['income'] >= df_preset['income'].quantile(0.90)]['loan_amount'].mean():>10,.0f}")
print(f"  poor_credit   mean_loan=${df_preset[df_preset['credit_score'] <= df_preset['credit_score'].quantile(0.20)]['loan_amount'].mean():>10,.0f}")
print(f"  all           mean_loan=${df_preset['loan_amount'].mean():>10,.0f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — REPRODUCIBILITY
# Same seed always produces identical output.
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("REPRODUCIBILITY CHECK")
print("=" * 70)
df_a = Blueprint(n=50, seed=42).add_feature(
    Feature("x", dtype=float, base=100, std=20)
).emit()
df_b = Blueprint(n=50, seed=42).add_feature(
    Feature("x", dtype=float, base=100, std=20)
).emit()
match = df_a["x"].equals(df_b["x"])
print(f"  Same seed produces identical output: {match}")

df_c = Blueprint(n=50, seed=99).add_feature(
    Feature("x", dtype=float, base=100, std=20)
).emit()
differ = not df_a["x"].equals(df_c["x"])
print(f"  Different seed produces different output: {differ}")

# Clean up output files
for fname in ["real_estate.csv", "real_estate.json",
              "real_estate_manifest.json", "real_estate_manifest2.json"]:
    if os.path.exists(fname):
        os.remove(fname)
print()
print("Demo complete.")
