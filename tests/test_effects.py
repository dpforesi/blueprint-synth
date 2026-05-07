import pytest
import numpy as np
from blueprint.effects.parser import parse_effect
from blueprint.effects.applicators import (
    apply_pct,
    apply_flat,
    apply_per_unit,
    apply_per_unit_pct,
    apply_fn,
    apply_set,
    apply_multiply,
)


class TestEffectParser:
    def test_positive_pct(self):
        kind, params = parse_effect("+12%")
        assert kind == "pct"
        assert abs(params - 0.12) < 1e-9

    def test_negative_pct(self):
        kind, params = parse_effect("-3%")
        assert kind == "pct"
        assert abs(params - (-0.03)) < 1e-9

    def test_positive_flat(self):
        kind, params = parse_effect("+500")
        assert kind == "flat"
        assert params == 500

    def test_negative_flat(self):
        kind, params = parse_effect("-200")
        assert kind == "flat"
        assert params == -200

    def test_per_unit(self):
        kind, params = parse_effect("+110 per unit")
        assert kind == "per_unit"
        assert params == 110

    def test_negative_per_unit(self):
        kind, params = parse_effect("-50 per unit")
        assert kind == "per_unit"
        assert params == -50

    def test_per_unit_pct(self):
        kind, params = parse_effect("-0.6% per unit")
        assert kind == "per_unit_pct"
        assert abs(params - (-0.006)) < 1e-9

    def test_set_numeric(self):
        kind, params = parse_effect("= 0")
        assert kind == "set"
        assert params == 0

    def test_set_source(self):
        kind, params = parse_effect("= source")
        assert kind == "set_source"

    def test_multiply(self):
        kind, params = parse_effect("* 2.5")
        assert kind == "multiply"
        assert abs(params - 2.5) < 1e-9

    def test_invalid_string_raises(self):
        with pytest.raises((ValueError, KeyError)):
            parse_effect("gibberish")


class TestApplicators:
    BASE = np.array([1000.0, 2000.0, 3000.0])
    SOURCE = np.array([1.0, 2.0, 3.0])

    def test_apply_pct_no_mask(self):
        result = apply_pct(self.BASE.copy(), pct=0.1)
        expected = np.array([1100.0, 2200.0, 3300.0])
        np.testing.assert_allclose(result, expected)

    def test_apply_pct_negative(self):
        result = apply_pct(self.BASE.copy(), pct=-0.1)
        np.testing.assert_allclose(result, [900.0, 1800.0, 2700.0])

    def test_apply_pct_with_mask(self):
        mask = np.array([True, False, True])
        result = apply_pct(self.BASE.copy(), pct=0.1, mask=mask)
        np.testing.assert_allclose(result, [1100.0, 2000.0, 3300.0])

    def test_apply_flat_no_mask(self):
        result = apply_flat(self.BASE.copy(), delta=500)
        np.testing.assert_allclose(result, [1500.0, 2500.0, 3500.0])

    def test_apply_flat_with_mask(self):
        mask = np.array([True, False, True])
        result = apply_flat(self.BASE.copy(), delta=100, mask=mask)
        np.testing.assert_allclose(result, [1100.0, 2000.0, 3100.0])

    def test_apply_per_unit_no_mask(self):
        result = apply_per_unit(self.BASE.copy(), rate=100, source=self.SOURCE)
        np.testing.assert_allclose(result, [1100.0, 2200.0, 3300.0])

    def test_apply_per_unit_with_mask(self):
        mask = np.array([True, False, True])
        result = apply_per_unit(self.BASE.copy(), rate=100, source=self.SOURCE, mask=mask)
        np.testing.assert_allclose(result, [1100.0, 2000.0, 3300.0])

    def test_apply_per_unit_pct(self):
        result = apply_per_unit_pct(self.BASE.copy(), rate=-0.006, source=self.SOURCE)
        expected = self.BASE * (1 + (-0.006) * self.SOURCE)
        np.testing.assert_allclose(result, expected)

    def test_apply_fn(self):
        fn = lambda src, tgt, df: tgt - src * 300
        result = apply_fn(self.BASE.copy(), self.SOURCE, fn=fn, df=None)
        expected = self.BASE - self.SOURCE * 300
        np.testing.assert_allclose(result, expected)

    def test_apply_set_scalar(self):
        result = apply_set(self.BASE.copy(), value=0)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_apply_set_with_mask(self):
        mask = np.array([True, False, True])
        result = apply_set(self.BASE.copy(), value=0, mask=mask)
        np.testing.assert_allclose(result, [0.0, 2000.0, 0.0])

    def test_apply_multiply(self):
        result = apply_multiply(self.BASE.copy(), factor=2.0)
        np.testing.assert_allclose(result, [2000.0, 4000.0, 6000.0])

    def test_apply_multiply_with_mask(self):
        mask = np.array([True, False, True])
        result = apply_multiply(self.BASE.copy(), factor=2.0, mask=mask)
        np.testing.assert_allclose(result, [2000.0, 2000.0, 6000.0])
