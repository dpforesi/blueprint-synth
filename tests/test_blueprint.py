import json
import os

import numpy as np
import pytest
import pandas as pd
from blueprint import Blueprint, Feature, Class, Influence
from blueprint.presets.classes import RandomClass, HighValueClass, LowValueClass
from blueprint.presets.influences import ScalesWith, CorrelatedWith, Caps


@pytest.fixture
def simple_bp():
    bp = Blueprint(n=100, seed=42)
    bp.add_feature(
        Feature("x", dtype=float, base=0, std=1),
        Feature("y", dtype=float, base=5, std=2),
    )
    return bp


class TestBlueprintConstruction:
    def test_n_stored(self):
        bp = Blueprint(n=500, seed=1)
        assert bp.n == 500

    def test_seed_stored(self):
        bp = Blueprint(n=100, seed=7)
        assert bp.seed == 7

    def test_default_seed(self):
        bp = Blueprint(n=100)
        assert bp.seed == 42


class TestBlueprintRegistration:
    def test_add_feature_returns_self(self):
        bp = Blueprint(n=10)
        f = Feature("x", dtype=float, base=0, std=1)
        result = bp.add_feature(f)
        assert result is bp

    def test_add_class_returns_self(self):
        bp = Blueprint(n=10)
        c = Class("grp", when=("x", ">", 0))
        result = bp.add_class(c)
        assert result is bp

    def test_add_influence_returns_self(self):
        bp = Blueprint(n=10)
        inf = Influence("x").on("y", effect="+10%")
        result = bp.add_influence(inf)
        assert result is bp

    def test_add_multiple_features_at_once(self):
        bp = Blueprint(n=10)
        f1 = Feature("a", dtype=float, base=0, std=1)
        f2 = Feature("b", dtype=float, base=0, std=1)
        bp.add_feature(f1, f2)
        assert len(bp.features) == 2

    def test_chaining(self):
        f1 = Feature("a", dtype=float, base=0, std=1)
        f2 = Feature("b", dtype=float, base=0, std=1)
        c = Class("grp", when=("a", ">", 0))
        inf = Influence("a").on("b", effect="+10%")
        bp = Blueprint(n=10).add_feature(f1, f2).add_class(c).add_influence(inf)
        assert isinstance(bp, Blueprint)


class TestBlueprintEmit:
    def test_emit_returns_dataframe(self, simple_bp):
        df = simple_bp.emit()
        assert isinstance(df, pd.DataFrame)

    def test_emit_correct_row_count(self, simple_bp):
        df = simple_bp.emit()
        assert len(df) == 100

    def test_emit_correct_columns(self, simple_bp):
        df = simple_bp.emit()
        assert "x" in df.columns
        assert "y" in df.columns

    def test_columns_in_registration_order(self):
        bp = Blueprint(n=10, seed=1)
        bp.add_feature(
            Feature("z", dtype=float, base=0, std=1),
            Feature("a", dtype=float, base=0, std=1),
            Feature("m", dtype=float, base=0, std=1),
        )
        df = bp.emit()
        assert list(df.columns) == ["z", "a", "m"]

    def test_reproducible_with_same_seed(self):
        bp1 = Blueprint(n=50, seed=42)
        bp1.add_feature(Feature("v", dtype=float, base=10, std=2))
        bp2 = Blueprint(n=50, seed=42)
        bp2.add_feature(Feature("v", dtype=float, base=10, std=2))
        df1 = bp1.emit()
        df2 = bp2.emit()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        bp1 = Blueprint(n=100, seed=1)
        bp1.add_feature(Feature("v", dtype=float, base=10, std=2))
        bp2 = Blueprint(n=100, seed=2)
        bp2.add_feature(Feature("v", dtype=float, base=10, std=2))
        df1 = bp1.emit()
        df2 = bp2.emit()
        assert not df1["v"].equals(df2["v"])


class TestBlueprintValidation:
    def test_validate_passes_clean_blueprint(self, simple_bp):
        simple_bp.validate()

    def test_validate_raises_on_undefined_feature_in_influence(self):
        bp = Blueprint(n=10)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1))
        bp.add_influence(Influence("x").on("nonexistent", effect="+10%"))
        with pytest.raises(Exception):
            bp.validate()

    def test_validate_raises_on_cycle(self):
        bp = Blueprint(n=10)
        bp.add_feature(
            Feature("a", dtype=float, base=0, std=1),
            Feature("b", dtype=float, base=0, std=1),
        )
        bp.add_influence(Influence("a").on("b", effect="+10%"))
        bp.add_influence(Influence("b").on("a", effect="+10%"))
        with pytest.raises(Exception):
            bp.validate()

    def test_validate_raises_on_undefined_class_feature_reference(self):
        bp = Blueprint(n=10)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1))
        bp.add_class(Class("grp", when=("nonexistent_col", "==", "val")))
        with pytest.raises(Exception):
            bp.validate()


class TestBlueprintEdgeCases:
    def test_zero_row_emit(self):
        bp = Blueprint(n=0, seed=42)
        bp.add_feature(
            Feature("x", dtype=float, base=0, std=1),
            Feature("cat", dtype="category", values=["a", "b"]),
            Feature("flag", dtype=bool, p=0.5),
        )
        df = bp.emit()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["x", "cat", "flag"]

    def test_zero_row_with_class(self):
        bp = Blueprint(n=0, seed=42)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1))
        bp.add_class(Class("grp", when=("x", ">", 0)).override("x", base=100))
        df = bp.emit()
        assert len(df) == 0

    def test_zero_row_with_influence(self):
        bp = Blueprint(n=0, seed=42)
        bp.add_feature(
            Feature("x", dtype=float, base=0, std=1),
            Feature("y", dtype=float, base=0, std=1),
        )
        bp.add_influence(Influence("x").on("y", effect="+10%"))
        df = bp.emit()
        assert len(df) == 0

    def test_single_row_emit(self):
        bp = Blueprint(n=1, seed=42)
        bp.add_feature(Feature("x", dtype=float, base=10, std=2))
        df = bp.emit()
        assert len(df) == 1
        assert "x" in df.columns

    def test_nullable_all_null(self):
        bp = Blueprint(n=100, seed=42)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1, nullable=1.0))
        df = bp.emit()
        assert df["x"].isna().all()

    def test_nullable_partial(self):
        bp = Blueprint(n=10000, seed=42)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1, nullable=0.3))
        df = bp.emit()
        null_frac = df["x"].isna().mean()
        assert 0.25 < null_frac < 0.35


class TestBlueprintComputedColumns:
    def test_computed_column_formula(self):
        bp = Blueprint(n=100, seed=42)
        bp.add_feature(
            Feature("price", dtype=float, base=100, std=0),
            Feature("sqft", dtype=float, base=50, std=0),
            Feature("pps", dtype="computed", formula=lambda df: df["price"] / df["sqft"]),
        )
        df = bp.emit()
        np.testing.assert_allclose(df["pps"].values, 2.0)

    def test_computed_column_present_in_output(self):
        bp = Blueprint(n=50, seed=1)
        bp.add_feature(
            Feature("a", dtype=float, base=3, std=0),
            Feature("b", dtype="computed", formula=lambda df: df["a"] * 2),
        )
        df = bp.emit()
        assert "b" in df.columns
        np.testing.assert_allclose(df["b"].values, 6.0)

    def test_computed_column_after_influences(self):
        bp = Blueprint(n=50, seed=1)
        bp.add_feature(
            Feature("x", dtype=float, base=100, std=0),
            Feature("y", dtype=float, base=50, std=0),
            Feature("total", dtype="computed", formula=lambda df: df["x"] + df["y"]),
        )
        bp.add_influence(Influence("x").on("y", effect="+50"))
        df = bp.emit()
        # x=100, y=50+50(influence)=100, total=200
        np.testing.assert_allclose(df["total"].values, 200.0)

    def test_computed_column_missing_formula_raises(self):
        bp = Blueprint(n=10, seed=1)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1))
        f = Feature("bad", dtype="computed")
        bp.add_feature(f)
        with pytest.raises(ValueError, match="formula"):
            bp.emit()


class TestBlueprintDescribe:
    def test_describe_runs(self, simple_bp, capsys):
        simple_bp.describe()
        out = capsys.readouterr().out
        assert "Blueprint" in out
        assert "x" in out
        assert "y" in out

    def test_emit_describe_true(self, simple_bp, capsys):
        df = simple_bp.emit(describe=True)
        out = capsys.readouterr().out
        assert "Blueprint" in out
        assert isinstance(df, pd.DataFrame)

    def test_describe_shows_classes(self):
        bp = Blueprint(n=10, seed=1)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1))
        bp.add_class(Class("hi", when=("x", ">", 0)))
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        bp.describe()
        sys.stdout = sys.__stdout__
        assert "hi" in buf.getvalue()

    def test_describe_shows_influences(self):
        bp = Blueprint(n=10, seed=1)
        bp.add_feature(
            Feature("a", dtype=float, base=0, std=1),
            Feature("b", dtype=float, base=0, std=1),
        )
        bp.add_influence(Influence("a").on("b", effect="+10%"))
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        bp.describe()
        sys.stdout = sys.__stdout__
        assert "a" in buf.getvalue()
        assert "b" in buf.getvalue()


class TestBlueprintEmitManifest:
    def test_emit_with_manifest(self, tmp_path, simple_bp):
        path = str(tmp_path / "manifest.json")
        simple_bp.emit(manifest=path)
        assert os.path.exists(path)
        with open(path) as f:
            m = json.load(f)
        assert m["seed"] == 42
        assert m["n_rows"] == 100


class TestBlueprintPresets:
    def test_random_class_produces_subset(self):
        bp = Blueprint(n=1000, seed=42)
        bp.add_feature(Feature("x", dtype=float, base=0, std=1))
        bp.add_class(RandomClass("half", p=0.5).override("x", base=100, std=0))
        df = bp.emit()
        assert isinstance(df, pd.DataFrame)
        high_count = (df["x"] > 50).sum()
        assert 400 < high_count < 600

    def test_high_value_class_selects_top(self):
        bp = Blueprint(n=1000, seed=42)
        bp.add_feature(Feature("income", dtype=float, base=50000, std=10000))
        bp.add_class(HighValueClass("rich", feature="income", top_pct=0.2))
        df = bp.emit()
        assert isinstance(df, pd.DataFrame)

    def test_low_value_class_selects_bottom(self):
        bp = Blueprint(n=1000, seed=42)
        bp.add_feature(Feature("score", dtype=float, base=50, std=15))
        bp.add_class(LowValueClass("low", feature="score", bottom_pct=0.1))
        df = bp.emit()
        assert isinstance(df, pd.DataFrame)

    def test_scales_with(self):
        bp = Blueprint(n=100, seed=42)
        bp.add_feature(
            Feature("sqft", dtype=float, base=100, std=0),
            Feature("price", dtype=float, base=0, std=0),
        )
        bp.add_influence(ScalesWith("sqft", "price", rate=2))
        df = bp.emit()
        np.testing.assert_allclose(df["price"].values, 200.0)

    def test_correlated_with_produces_correlation(self):
        bp = Blueprint(n=2000, seed=42)
        bp.add_feature(
            Feature("x", dtype=float, base=0, std=10),
            Feature("y", dtype=float, base=0, std=10),
        )
        bp.add_influence(CorrelatedWith("x", "y", correlation=0.8))
        df = bp.emit()
        r = np.corrcoef(df["x"], df["y"])[0, 1]
        assert r > 0.5

    def test_caps_reduces_high_values(self):
        bp = Blueprint(n=500, seed=42)
        bp.add_feature(
            Feature("experience", dtype=float, base=15, std=5, clip=(0, 40)),
            Feature("salary", dtype=float, base=100000, std=0),
        )
        bp.add_influence(Caps("experience", "salary", threshold=10, decay=0.05))
        df = bp.emit()
        assert (df["salary"] <= 100000).all()


class TestBlueprintEmitCSV:
    def test_to_csv_writes_file(self, tmp_path, simple_bp):
        path = str(tmp_path / "out.csv")
        df = simple_bp.to_csv(path)
        import os
        assert os.path.exists(path)
        assert isinstance(df, pd.DataFrame)

    def test_to_csv_roundtrip(self, tmp_path, simple_bp):
        path = str(tmp_path / "out.csv")
        original = simple_bp.to_csv(path)
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(original.columns)
        assert len(loaded) == len(original)


class TestBlueprintEmitJSON:
    def test_to_json_writes_file(self, tmp_path, simple_bp):
        path = str(tmp_path / "out.json")
        df = simple_bp.to_json(path)
        import os
        assert os.path.exists(path)
        assert isinstance(df, pd.DataFrame)

    def test_to_json_with_manifest(self, tmp_path, simple_bp):
        data_path = str(tmp_path / "out.json")
        manifest_path = str(tmp_path / "manifest.json")
        simple_bp.to_json(data_path, manifest=manifest_path)
        import os, json
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "seed" in manifest
        assert "n_rows" in manifest


class TestInfluenceVariability:
    def _make_per_unit_bp(self, noise_std=None, seed=42, n=500):
        bp = Blueprint(n=n, seed=seed)
        bp.add_feature(
            Feature("x", dtype=float, base=10, std=0),
            Feature("y", dtype=float, base=0,  std=0, derived=True),
        )
        kw = {} if noise_std is None else {"noise_std": noise_std}
        bp.add_influence(Influence("x").on("y", effect="+2 per unit", **kw))
        return bp

    def test_zero_noise_matches_no_noise(self):
        df_det   = self._make_per_unit_bp(noise_std=None).emit()
        df_zero  = self._make_per_unit_bp(noise_std=0.0).emit()
        pd.testing.assert_frame_equal(df_det, df_zero)

    def test_nonzero_noise_adds_row_level_variation(self):
        df = self._make_per_unit_bp(noise_std=0.2).emit()
        assert df["y"].std() > 0

    def test_nonzero_noise_centered_on_nominal(self):
        df = self._make_per_unit_bp(noise_std=0.1, n=1000).emit()
        # x=10, rate=2, E[noise]=1.0 → E[y] ≈ 20
        assert abs(df["y"].mean() - 20.0) < 1.5

    def test_reproducible_same_seed(self):
        df1 = self._make_per_unit_bp(noise_std=0.2, seed=42).emit()
        df2 = self._make_per_unit_bp(noise_std=0.2, seed=42).emit()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_blueprint_seeds_differ(self):
        df1 = self._make_per_unit_bp(noise_std=0.2, seed=1).emit()
        df2 = self._make_per_unit_bp(noise_std=0.2, seed=2).emit()
        assert not df1["y"].equals(df2["y"])

    def test_flat_effect_with_noise(self):
        bp = Blueprint(n=500, seed=42)
        bp.add_feature(
            Feature("src", dtype=float, base=1, std=0),
            Feature("tgt", dtype=float, base=100, std=0),
        )
        bp.add_influence(Influence("src").on("tgt", effect="+10", noise_std=0.2))
        df = bp.emit()
        assert df["tgt"].std() > 0
        assert abs(df["tgt"].mean() - 110.0) < 3.0

    def test_pct_effect_with_noise(self):
        bp = Blueprint(n=500, seed=42)
        bp.add_feature(
            Feature("src", dtype=float, base=1, std=0),
            Feature("tgt", dtype=float, base=100, std=0),
        )
        bp.add_influence(Influence("src").on("tgt", effect="+10%", noise_std=0.2))
        df = bp.emit()
        assert df["tgt"].std() > 0
        assert abs(df["tgt"].mean() - 110.0) < 3.0

    def test_per_unit_pct_effect_with_noise(self):
        bp = Blueprint(n=500, seed=42)
        bp.add_feature(
            Feature("units", dtype=float, base=5, std=0),
            Feature("price", dtype=float, base=100, std=0),
        )
        bp.add_influence(Influence("units").on("price", effect="+10% per unit", noise_std=0.2))
        df = bp.emit()
        assert df["price"].std() > 0

    def test_multiply_effect_with_noise(self):
        bp = Blueprint(n=500, seed=42)
        bp.add_feature(
            Feature("flag", dtype=bool, p=1.0),
            Feature("val",  dtype=float, base=100, std=0),
        )
        bp.add_influence(Influence("flag").on("val", effect="*1.5", noise_std=0.2))
        df = bp.emit()
        assert df["val"].std() > 0
        assert abs(df["val"].mean() - 150.0) < 10.0

    def test_noise_with_by_class(self):
        bp = Blueprint(n=500, seed=42)
        bp.add_feature(
            Feature("x",   dtype=float,      base=10, std=0),
            Feature("cat", dtype="category", values=["a", "b"]),
            Feature("y",   dtype=float,      base=0,  std=0, derived=True),
        )
        bp.add_class(Class("group_a", when=("cat", "==", "a")))
        bp.add_influence(
            Influence("x").on(
                "y",
                by_class={"group_a": "+3 per unit"},
                effect="+2 per unit",
                noise_std=0.2,
            )
        )
        df = bp.emit()
        assert df["y"].std() > 0

    def test_noise_with_when_condition(self):
        bp = Blueprint(n=500, seed=42)
        bp.add_feature(
            Feature("x",     dtype=float, base=10, std=0),
            Feature("gated", dtype=float, base=0,  std=1),
            Feature("y",     dtype=float, base=0,  std=0, derived=True),
        )
        bp.add_influence(
            Influence("x").on("y", effect="+2 per unit", when=("gated", ">", 0), noise_std=0.15)
        )
        df = bp.emit()
        above = df[df["gated"] > 0]["y"]
        below = df[df["gated"] <= 0]["y"]
        assert above.std() > 0
        assert (below == 0).all()

    def test_validate_rejects_negative_noise_std(self):
        bp = Blueprint(n=10, seed=1)
        bp.add_feature(
            Feature("x", dtype=float, base=0, std=1),
            Feature("y", dtype=float, base=0, std=1),
        )
        bp.add_influence(Influence("x").on("y", effect="+1", noise_std=-0.1))
        with pytest.raises(ValueError, match="noise_std"):
            bp.validate()

    def test_describe_shows_noise_std(self, capsys):
        bp = Blueprint(n=10, seed=1)
        bp.add_feature(
            Feature("x", dtype=float, base=0, std=1),
            Feature("y", dtype=float, base=0, std=1),
        )
        bp.add_influence(Influence("x").on("y", effect="+10%", noise_std=0.1))
        bp.describe()
        out = capsys.readouterr().out
        assert "noise_std" in out
        assert "0.1" in out

    def test_manifest_includes_influence_noise(self, tmp_path):
        bp = Blueprint(n=10, seed=42)
        bp.add_feature(
            Feature("x", dtype=float, base=0, std=1),
            Feature("y", dtype=float, base=0, std=1),
        )
        bp.add_influence(Influence("x").on("y", effect="+10%", noise_std=0.2))
        path = str(tmp_path / "manifest.json")
        bp.emit(manifest=path)
        with open(path) as f:
            m = json.load(f)
        assert "influence_noise" in m
        assert m["influence_noise"]["x→y"] == 0.2

    def test_manifest_no_noise_key_when_none(self, tmp_path):
        bp = Blueprint(n=10, seed=42)
        bp.add_feature(
            Feature("x", dtype=float, base=0, std=1),
            Feature("y", dtype=float, base=0, std=1),
        )
        bp.add_influence(Influence("x").on("y", effect="+10%"))
        path = str(tmp_path / "manifest.json")
        bp.emit(manifest=path)
        with open(path) as f:
            m = json.load(f)
        assert "influence_noise" not in m

    def test_fn_edge_rng_passed_when_accepted(self):
        received = {}

        def custom_fn(src, tgt, df, rng=None):
            received["rng"] = rng
            return tgt + src

        bp = Blueprint(n=10, seed=42)
        bp.add_feature(
            Feature("x", dtype=float, base=1, std=0),
            Feature("y", dtype=float, base=0, std=0),
        )
        bp.add_influence(Influence("x").on("y", fn=custom_fn, noise_std=0.1))
        bp.emit()
        assert received.get("rng") is not None

    def test_fn_edge_rng_not_passed_to_old_3arg_fn(self):
        results = []

        def old_fn(src, tgt, df):
            results.append(1)
            return tgt + src

        bp = Blueprint(n=10, seed=42)
        bp.add_feature(
            Feature("x", dtype=float, base=1, std=0),
            Feature("y", dtype=float, base=0, std=0),
        )
        bp.add_influence(Influence("x").on("y", fn=old_fn, noise_std=0.1))
        bp.emit()
        assert len(results) > 0  # fn was called without error
