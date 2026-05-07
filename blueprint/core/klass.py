import numpy as np
import pandas as pd


class Class:
    def __init__(self, name: str, when):
        self.name = name
        self.when = when
        self.overrides: dict = {}

    def override(self, feature_name: str, **kwargs) -> "Class":
        self.overrides[feature_name] = kwargs
        return self

    def resolve_mask(self, df: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
        when = self.when
        if callable(when):
            return np.asarray(when(df), dtype=bool)
        if isinstance(when, tuple):
            if len(when) == 2 and when[0] == "__random__":
                return rng.random(n) < when[1]
            col, op, val = when
            s = df[col]
            if op == "==":
                return np.asarray(s == val, dtype=bool)
            if op == "!=":
                return np.asarray(s != val, dtype=bool)
            if op == ">":
                return np.asarray(s > val, dtype=bool)
            if op == ">=":
                return np.asarray(s >= val, dtype=bool)
            if op == "<":
                return np.asarray(s < val, dtype=bool)
            if op == "<=":
                return np.asarray(s <= val, dtype=bool)
            if op == "between":
                lo, hi = val
                return np.asarray((s >= lo) & (s <= hi), dtype=bool)
            if op == "in":
                return np.asarray(s.isin(val), dtype=bool)
            raise ValueError(f"Unknown condition operator: {op!r}")
        raise TypeError(f"Invalid 'when' type: {type(when)!r}")
