import numpy as np
import pandas as pd
import pytest
from blueprint import Class


class TestClassConstruction:
    def test_equality_condition(self):
        c = Class("luxury", when=("neighborhood_tier", "==", "luxury"))
        assert c.name == "luxury"
        assert c.when == ("neighborhood_tier", "==", "luxury")

    def test_comparison_condition(self):
        c = Class("high_income", when=("income", ">", 80_000))
        assert c.when == ("income", ">", 80_000)

    def test_range_condition(self):
        c = Class("mid_age", when=("age", "between", (25, 45)))
        assert c.when == ("age", "between", (25, 45))

    def test_in_set_condition(self):
        c = Class("northern", when=("region", "in", ["North", "East"]))
        assert c.when == ("region", "in", ["North", "East"])

    def test_callable_condition(self):
        fn = lambda df: df["age"] > df["retirement_age"]
        c = Class("retired", when=fn)
        assert callable(c.when)

    def test_probabilistic_condition(self):
        c = Class("test_group", when=("__random__", 0.2))
        assert c.when == ("__random__", 0.2)


class TestClassOverrides:
    def test_override_returns_self(self):
        c = Class("luxury", when=("tier", "==", "luxury"))
        result = c.override("price", base=900_000, std=200_000)
        assert result is c

    def test_single_override_stored(self):
        c = Class("luxury", when=("tier", "==", "luxury"))
        c.override("price", base=900_000, std=200_000)
        assert "price" in c.overrides
        assert c.overrides["price"]["base"] == 900_000
        assert c.overrides["price"]["std"] == 200_000

    def test_multiple_overrides(self):
        c = (
            Class("luxury", when=("tier", "==", "luxury"))
            .override("price", base=900_000, std=200_000)
            .override("sqft", base=4000, std=1200)
            .override("has_pool", p=0.75)
        )
        assert len(c.overrides) == 3
        assert c.overrides["has_pool"]["p"] == 0.75

    def test_override_is_chainable(self):
        c = Class("luxury", when=("tier", "==", "luxury"))
        result = c.override("price", base=900_000).override("sqft", base=4000)
        assert result is c

    def test_empty_overrides_by_default(self):
        c = Class("vanilla", when=("region", "==", "North"))
        assert c.overrides == {}


class TestClassResolveMask:
    RNG = np.random.default_rng(0)

    def _df(self, data):
        return pd.DataFrame(data)

    def test_equality_mask(self):
        c = Class("lux", when=("tier", "==", "luxury"))
        df = self._df({"tier": ["luxury", "mid", "luxury", "entry"]})
        mask = c.resolve_mask(df, 4, self.RNG)
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_greater_than_mask(self):
        c = Class("hi", when=("score", ">", 50))
        df = self._df({"score": [10, 60, 50, 70]})
        mask = c.resolve_mask(df, 4, self.RNG)
        np.testing.assert_array_equal(mask, [False, True, False, True])

    def test_between_mask(self):
        c = Class("mid", when=("age", "between", (25, 45)))
        df = self._df({"age": [20, 30, 45, 50]})
        mask = c.resolve_mask(df, 4, self.RNG)
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_in_mask(self):
        c = Class("north", when=("region", "in", ["North", "East"]))
        df = self._df({"region": ["North", "South", "East", "West"]})
        mask = c.resolve_mask(df, 4, self.RNG)
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_callable_mask(self):
        c = Class("old", when=lambda df: df["age"] >= df["cutoff"])
        df = self._df({"age": [20, 50, 65], "cutoff": [65, 65, 65]})
        mask = c.resolve_mask(df, 3, self.RNG)
        np.testing.assert_array_equal(mask, [False, False, True])

    def test_random_mask_approximate_fraction(self):
        c = Class("half", when=("__random__", 0.5))
        df = self._df({"x": list(range(10000))})
        mask = c.resolve_mask(df, 10000, np.random.default_rng(1))
        assert 4500 < mask.sum() < 5500

    def test_random_p0_all_false(self):
        c = Class("none", when=("__random__", 0.0))
        df = self._df({"x": list(range(100))})
        mask = c.resolve_mask(df, 100, np.random.default_rng(0))
        assert not mask.any()

    def test_random_p1_all_true(self):
        c = Class("all", when=("__random__", 1.0))
        df = self._df({"x": list(range(100))})
        mask = c.resolve_mask(df, 100, np.random.default_rng(0))
        assert mask.all()

    def test_zero_rows(self):
        c = Class("lux", when=("tier", "==", "luxury"))
        df = pd.DataFrame({"tier": pd.Series([], dtype=str)})
        mask = c.resolve_mask(df, 0, np.random.default_rng(0))
        assert len(mask) == 0
