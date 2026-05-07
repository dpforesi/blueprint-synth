import pytest
from blueprint import Influence


class TestInfluenceConstruction:
    def test_source_stored(self):
        inf = Influence("sqft")
        assert inf.source == "sqft"

    def test_no_edges_by_default(self):
        inf = Influence("sqft")
        assert inf.edges == []


class TestInfluenceOn:
    def test_on_returns_self(self):
        inf = Influence("sqft")
        result = inf.on("price", effect="+110 per unit")
        assert result is inf

    def test_single_edge_registered(self):
        inf = Influence("sqft").on("price", effect="+110 per unit")
        assert len(inf.edges) == 1
        edge = inf.edges[0]
        assert edge["target"] == "price"
        assert edge["effect"] == "+110 per unit"

    def test_by_class_stored(self):
        inf = Influence("has_pool").on(
            "price",
            by_class={"luxury": "+12%", "mid": "+4%", "entry": "-3%"},
        )
        edge = inf.edges[0]
        assert edge["by_class"] == {"luxury": "+12%", "mid": "+4%", "entry": "-3%"}

    def test_fn_stored(self):
        fn = lambda src, tgt, df: tgt - src * 300
        inf = Influence("distance").on("price", fn=fn)
        assert inf.edges[0]["fn"] is fn

    def test_when_condition_stored(self):
        inf = Influence("crime_rate").on(
            "price",
            effect="-0.6% per unit",
            when=("crime_rate", ">", 10),
        )
        assert inf.edges[0]["when"] == ("crime_rate", ">", 10)

    def test_multiple_targets_from_single_source(self):
        inf = (
            Influence("is_weekend")
            .on("foot_traffic", effect="+40%")
            .on("revenue", effect="+25%")
            .on("staff_needed", effect="+2 per unit")
        )
        assert len(inf.edges) == 3
        targets = [e["target"] for e in inf.edges]
        assert targets == ["foot_traffic", "revenue", "staff_needed"]

    def test_effect_and_by_class_and_default(self):
        inf = Influence("has_garage").on(
            "price",
            by_class={"luxury": "+5%", "mid": "+6%", "entry": "+2%"},
            effect="+3%",
        )
        edge = inf.edges[0]
        assert edge["effect"] == "+3%"
        assert edge["by_class"]["luxury"] == "+5%"

    def test_requires_at_least_one_of_effect_by_class_fn(self):
        with pytest.raises((ValueError, TypeError)):
            Influence("sqft").on("price")

    def test_noise_std_stored(self):
        inf = Influence("x").on("y", effect="+10%", noise_std=0.15)
        assert inf.edges[0]["noise_std"] == 0.15

    def test_noise_std_default_none(self):
        inf = Influence("x").on("y", effect="+10%")
        assert inf.edges[0]["noise_std"] is None

    def test_noise_std_zero_stored(self):
        inf = Influence("x").on("y", effect="+5", noise_std=0.0)
        assert inf.edges[0]["noise_std"] == 0.0
