import pytest
import numpy as np
import pandas as pd
from blueprint.generators.numeric import generate_float, generate_int, generate_positive_float, generate_percentage
from blueprint.generators.categorical import generate_categorical
from blueprint.generators.boolean import generate_boolean
from blueprint.generators.temporal import generate_datetime, generate_datetime_offset
from blueprint.generators.identity import generate_id, generate_row_number
from blueprint.generators.text import generate_text


RNG = np.random.default_rng(42)


class TestNumericGenerators:
    def test_float_output_shape(self):
        out = generate_float(100, base=0, std=1, rng=RNG)
        assert len(out) == 100

    def test_float_output_type(self):
        out = generate_float(100, base=0, std=1, rng=RNG)
        assert out.dtype == float or np.issubdtype(out.dtype, np.floating)

    def test_float_mean_approx(self):
        out = generate_float(10_000, base=500, std=10, rng=np.random.default_rng(0))
        assert abs(out.mean() - 500) < 5

    def test_float_std_approx(self):
        out = generate_float(10_000, base=0, std=50, rng=np.random.default_rng(0))
        assert abs(out.std() - 50) < 5

    def test_int_output_is_integer(self):
        out = generate_int(100, base=3, std=1, rng=RNG)
        assert np.issubdtype(out.dtype, np.integer)

    def test_positive_float_no_negatives(self):
        out = generate_positive_float(1000, base=5, std=10, rng=np.random.default_rng(0))
        assert (out >= 0).all()

    def test_percentage_clipped_to_unit_interval(self):
        out = generate_percentage(1000, base=0.5, std=1.0, rng=np.random.default_rng(0))
        assert (out >= 0).all() and (out <= 1).all()


class TestCategoricalGenerator:
    def test_output_length(self):
        out = generate_categorical(100, ["a", "b", "c"], [0.5, 0.3, 0.2], RNG)
        assert len(out) == 100

    def test_only_valid_values(self):
        out = generate_categorical(100, ["a", "b", "c"], None, RNG)
        assert set(out).issubset({"a", "b", "c"})

    def test_uniform_weights_when_none(self):
        out = generate_categorical(10_000, ["a", "b"], None, np.random.default_rng(0))
        counts = pd.Series(out).value_counts(normalize=True)
        assert abs(counts["a"] - 0.5) < 0.05

    def test_weighted_sampling(self):
        out = generate_categorical(10_000, ["a", "b"], [0.9, 0.1], np.random.default_rng(0))
        counts = pd.Series(out).value_counts(normalize=True)
        assert counts["a"] > 0.8

    def test_returns_pandas_categorical(self):
        out = generate_categorical(50, ["x", "y"], None, RNG)
        assert isinstance(out, pd.Categorical)


class TestBooleanGenerator:
    def test_output_length(self):
        out = generate_boolean(100, p=0.5, rng=RNG)
        assert len(out) == 100

    def test_output_dtype_bool(self):
        out = generate_boolean(100, p=0.5, rng=RNG)
        assert out.dtype == bool

    def test_p_zero_all_false(self):
        out = generate_boolean(100, p=0.0, rng=RNG)
        assert not out.any()

    def test_p_one_all_true(self):
        out = generate_boolean(100, p=1.0, rng=RNG)
        assert out.all()

    def test_p_approx_respected(self):
        out = generate_boolean(10_000, p=0.3, rng=np.random.default_rng(0))
        assert abs(out.mean() - 0.3) < 0.02


class TestTemporalGenerator:
    def test_datetime_output_length(self):
        out = generate_datetime(100, "2020-01-01", "2024-12-31", rng=np.random.default_rng(0))
        assert len(out) == 100

    def test_datetime_within_range(self):
        out = generate_datetime(100, "2020-01-01", "2024-12-31", rng=np.random.default_rng(0))
        assert (out >= pd.Timestamp("2020-01-01")).all()
        assert (out <= pd.Timestamp("2024-12-31")).all()

    def test_datetime_returns_series(self):
        out = generate_datetime(10, "2020-01-01", "2021-01-01", rng=np.random.default_rng(0))
        assert isinstance(out, pd.Series)

    def test_datetime_offset_after_source(self):
        source = pd.Series(pd.date_range("2020-01-01", periods=100, freq="D"))
        out = generate_datetime_offset(
            100, source, "1D", "30D", distribution="uniform", rng=np.random.default_rng(0)
        )
        assert (out >= source).all()

    def test_datetime_offset_length(self):
        source = pd.Series(pd.date_range("2020-01-01", periods=50, freq="D"))
        out = generate_datetime_offset(50, source, "1D", "10D", rng=np.random.default_rng(0))
        assert len(out) == 50


class TestIdentityGenerator:
    def test_uuid4_length(self):
        out = generate_id(100, style="uuid4", rng=np.random.default_rng(0))
        assert len(out) == 100

    def test_uuid4_unique(self):
        out = generate_id(100, style="uuid4", rng=np.random.default_rng(0))
        assert len(set(out)) == 100

    def test_sequential_starts_at_zero(self):
        out = generate_id(10, style="sequential", start=0, step=1)
        assert list(out) == list(range(10))

    def test_sequential_custom_start(self):
        out = generate_id(5, style="sequential", start=10_000, step=1)
        assert list(out) == [10_000, 10_001, 10_002, 10_003, 10_004]

    def test_prefixed_format(self):
        out = generate_id(5, style="prefixed", prefix="ORD-", padding=5)
        assert list(out) == ["ORD-00001", "ORD-00002", "ORD-00003", "ORD-00004", "ORD-00005"]

    def test_row_number_is_zero_indexed(self):
        out = generate_row_number(5)
        assert list(out) == [0, 1, 2, 3, 4]

    def test_row_number_length(self):
        out = generate_row_number(100)
        assert len(out) == 100


class TestTextGenerator:
    def test_output_length(self):
        from blueprint.core.feature import Feature
        pools = {
            "num": Feature("__num", dtype=int, base=100, std=50, clip=(1, 999)),
            "street": Feature("__st", dtype="category", values=["Oak", "Main"]),
        }
        out = generate_text(10, "{num} {street}", pools, np.random.default_rng(0))
        assert len(out) == 10

    def test_output_contains_pool_values(self):
        from blueprint.core.feature import Feature
        pools = {
            "street": Feature("__st", dtype="category", values=["Oak", "Main"]),
        }
        out = generate_text(20, "{street} St", pools, np.random.default_rng(0))
        assert all("Oak" in v or "Main" in v for v in out)
