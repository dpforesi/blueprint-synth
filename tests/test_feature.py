import pytest
from blueprint import Feature


class TestFeatureConstruction:
    def test_basic_float_feature(self):
        f = Feature("price", dtype=float, base=300_000, std=60_000)
        assert f.name == "price"
        assert f.dtype == float
        assert f.base == 300_000
        assert f.std == 60_000

    def test_int_feature(self):
        f = Feature("bedrooms", dtype=int, base=3, std=1, clip=(1, 8))
        assert f.name == "bedrooms"
        assert f.dtype == int
        assert f.clip == (1, 8)

    def test_bool_feature(self):
        f = Feature("has_pool", dtype=bool, p=0.3)
        assert f.name == "has_pool"
        assert f.p == 0.3

    def test_category_feature(self):
        f = Feature(
            "tier",
            dtype="category",
            values=["luxury", "mid", "entry"],
            weights=[0.2, 0.5, 0.3],
        )
        assert f.name == "tier"
        assert f.values == ["luxury", "mid", "entry"]
        assert f.weights == [0.2, 0.5, 0.3]

    def test_derived_feature(self):
        f = Feature("price_per_sqft", dtype=float, derived=True)
        assert f.derived is True

    def test_nullable_fraction(self):
        f = Feature("income", dtype=float, base=50_000, std=10_000, nullable=0.1)
        assert f.nullable == 0.1

    def test_seed_override(self):
        f = Feature("col", dtype=float, base=0, std=1, seed=99)
        assert f.seed == 99

    def test_defaults(self):
        f = Feature("x", dtype=float)
        assert f.base == 0
        assert f.std == 0
        assert f.clip == (None, None)
        assert f.nullable == 0.0
        assert f.derived is False
        assert f.seed is None


class TestFeatureModifiers:
    def test_noise_returns_self(self):
        f = Feature("x", dtype=float, base=0, std=1)
        result = f.noise(std=50)
        assert result is f

    def test_trend_returns_self(self):
        f = Feature("x", dtype=float, base=0, std=1)
        result = f.trend(rate=0.005)
        assert result is f

    def test_seasonality_returns_self(self):
        f = Feature("x", dtype=float, base=0, std=1)
        result = f.seasonality(period=7, amplitude=200)
        assert result is f

    def test_spike_returns_self(self):
        f = Feature("x", dtype=float, base=0, std=1)
        result = f.spike(at=100, magnitude=3.0)
        assert result is f

    def test_dropout_returns_self(self):
        f = Feature("x", dtype=float, base=0, std=1)
        result = f.dropout(rate=0.05)
        assert result is f

    def test_clip_returns_self(self):
        f = Feature("x", dtype=float, base=0, std=1)
        result = f.clip(min=0, max=1000)
        assert result is f

    def test_round_returns_self(self):
        f = Feature("x", dtype=float, base=0, std=1)
        result = f.round(decimals=2)
        assert result is f

    def test_chain(self):
        f = (
            Feature("revenue", dtype=float, base=10_000, std=500)
            .trend(rate=0.003)
            .seasonality(period=30, amplitude=2000)
            .noise(std=300)
            .clip(min=0)
        )
        assert f.name == "revenue"

    def test_modifiers_recorded_in_order(self):
        f = Feature("x", dtype=float, base=0, std=1)
        f.trend(rate=0.01).noise(std=5).clip(min=0)
        names = [m["type"] for m in f.modifiers]
        assert names == ["trend", "noise", "clip"]

    def test_noise_distribution_options(self):
        for dist in ("gaussian", "uniform", "poisson"):
            f = Feature("x", dtype=float, base=0, std=1).noise(std=10, distribution=dist)
            modifier = f.modifiers[-1]
            assert modifier["distribution"] == dist

    def test_trend_style_options(self):
        for style in ("linear", "exponential"):
            f = Feature("x", dtype=float, base=0, std=1).trend(rate=0.01, style=style)
            modifier = f.modifiers[-1]
            assert modifier["style"] == style

    def test_spike_shape_options(self):
        for shape in ("flat", "triangle"):
            f = Feature("x", dtype=float, base=0, std=1).spike(at=50, magnitude=2.0, shape=shape)
            modifier = f.modifiers[-1]
            assert modifier["shape"] == shape
