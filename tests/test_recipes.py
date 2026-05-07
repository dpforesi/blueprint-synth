import pytest
import pandas as pd
from blueprint import Blueprint
from blueprint.presets.recipes import real_estate, ecommerce, employee_survey, web_events


class TestRealEstateRecipe:
    def test_returns_blueprint(self):
        bp = real_estate()
        assert isinstance(bp, Blueprint)

    def test_emit_returns_dataframe(self):
        df = real_estate(n=100, seed=42).emit()
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = real_estate(n=50, seed=1).emit()
        assert len(df) == 50

    def test_expected_columns_present(self):
        df = real_estate(n=50, seed=1).emit()
        for col in ["listing_id", "neighborhood_tier", "sqft", "bedrooms", "price"]:
            assert col in df.columns

    def test_price_positive(self):
        df = real_estate(n=200, seed=1).emit()
        assert (df["price"] > 0).all()

    def test_reproducible(self):
        df1 = real_estate(n=100, seed=42).emit()
        df2 = real_estate(n=100, seed=42).emit()
        pd.testing.assert_frame_equal(df1, df2)


class TestEcommerceRecipe:
    def test_returns_blueprint(self):
        bp = ecommerce()
        assert isinstance(bp, Blueprint)

    def test_emit_returns_dataframe(self):
        df = ecommerce(n=50, seed=42).emit()
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = ecommerce(n=75, seed=1).emit()
        assert len(df) == 75


class TestEmployeeSurveyRecipe:
    def test_returns_blueprint(self):
        bp = employee_survey()
        assert isinstance(bp, Blueprint)

    def test_emit_returns_dataframe(self):
        df = employee_survey(n=50, seed=42).emit()
        assert isinstance(df, pd.DataFrame)

    def test_custom_departments(self):
        df = employee_survey(n=100, seed=1, departments=["Engineering", "Sales"]).emit()
        assert "department" in df.columns
        assert set(df["department"].unique()).issubset({"Engineering", "Sales"})


class TestWebEventsRecipe:
    def test_returns_blueprint(self):
        bp = web_events()
        assert isinstance(bp, Blueprint)

    def test_emit_returns_dataframe(self):
        df = web_events(n=50, seed=42).emit()
        assert isinstance(df, pd.DataFrame)
