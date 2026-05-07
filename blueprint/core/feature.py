import numpy as np


class _ClipProxy(tuple):
    """Tuple subclass that makes f.clip both a readable attribute and a chainable method call."""

    def __new__(cls, feature: "Feature", lo, hi):
        obj = super().__new__(cls, (lo, hi))
        obj._feature = feature
        return obj

    def __call__(self, min=None, max=None) -> "Feature":
        self._feature.modifiers.append({"type": "clip", "min": min, "max": max})
        return self._feature


class Feature:
    def __init__(
        self,
        name: str,
        dtype,
        base: float = 0,
        std: float = 0,
        clip: tuple = (None, None),
        p: float = 0.5,
        values: list = None,
        weights: list = None,
        derived: bool = False,
        nullable: float = 0.0,
        seed: int = None,
        **kwargs,
    ):
        self.name = name
        self.dtype = dtype
        self.base = base
        self.std = std
        self._constructor_clip = tuple(clip)
        self.p = p
        self.values = values
        self.weights = weights
        self.derived = derived
        self.nullable = nullable
        self.seed = seed
        self._kwargs = kwargs
        self.modifiers: list = []

    @property
    def clip(self) -> "_ClipProxy":
        return _ClipProxy(self, *self._constructor_clip)

    def noise(self, std: float = 0, distribution: str = "gaussian") -> "Feature":
        self.modifiers.append({"type": "noise", "std": std, "distribution": distribution})
        return self

    def trend(self, rate: float = 0.005, style: str = "linear") -> "Feature":
        self.modifiers.append({"type": "trend", "rate": rate, "style": style})
        return self

    def seasonality(self, period: int = 7, amplitude: float = 200, phase: float = 0) -> "Feature":
        self.modifiers.append({"type": "seasonality", "period": period, "amplitude": amplitude, "phase": phase})
        return self

    def spike(self, at, magnitude, duration: int = 1, shape: str = "flat") -> "Feature":
        self.modifiers.append({"type": "spike", "at": at, "magnitude": magnitude, "duration": duration, "shape": shape})
        return self

    def dropout(self, rate: float = 0.01, fill=None) -> "Feature":
        self.modifiers.append({"type": "dropout", "rate": rate, "fill": fill})
        return self

    def round(self, decimals: int = 0) -> "Feature":
        self.modifiers.append({"type": "round", "decimals": decimals})
        return self

    def _is_numeric(self) -> bool:
        return self.dtype in (float, "float", int, "int", "positive_float", "percentage") or self.derived

    def generate(self, n: int, rng: np.random.Generator):
        from blueprint.generators.boolean import generate_boolean
        from blueprint.generators.categorical import generate_categorical
        from blueprint.generators.identity import generate_id, generate_row_number
        from blueprint.generators.numeric import (
            generate_float,
            generate_int,
            generate_percentage,
            generate_positive_float,
        )

        if self.derived:
            return np.zeros(n, dtype=float)

        dt = self.dtype
        if dt in (float, "float"):
            return generate_float(n, self.base, self.std, rng)
        if dt in (int, "int"):
            return generate_int(n, self.base, self.std, rng)
        if dt == "positive_float":
            return generate_positive_float(n, self.base, self.std, rng)
        if dt == "percentage":
            return generate_percentage(n, self.base, self.std, rng)
        if dt == "category":
            return generate_categorical(n, self.values, self.weights, rng)
        if dt in (bool, "bool"):
            return generate_boolean(n, self.p, rng)
        if dt == "id":
            return generate_id(
                n,
                style=self._kwargs.get("style", "uuid4"),
                start=self._kwargs.get("start", 1),
                step=self._kwargs.get("step", 1),
                prefix=self._kwargs.get("prefix", ""),
                padding=self._kwargs.get("padding", 0),
                rng=rng,
            )
        if dt == "row_number":
            return generate_row_number(n)
        if dt == "computed":
            return np.zeros(n, dtype=float)
        if dt == "datetime":
            from blueprint.generators.temporal import generate_datetime
            return generate_datetime(
                n,
                start=self._kwargs["start"],
                end=self._kwargs["end"],
                distribution=self._kwargs.get("distribution", "uniform"),
                tz=self._kwargs.get("tz"),
                freq=self._kwargs.get("freq"),
                rng=rng,
            )
        if dt == "str":
            from blueprint.generators.text import generate_text
            return generate_text(n, self._kwargs["template"], self._kwargs["pools"], rng)
        raise NotImplementedError(f"dtype={dt!r} not yet implemented")

    def apply_modifiers(self, values, n: int, rng: np.random.Generator):
        if not self._is_numeric():
            return self._apply_nullable(values, n, rng)

        values = np.asarray(values, dtype=float)
        idx = np.arange(n)

        for mod in self.modifiers:
            t = mod["type"]
            if t == "noise":
                values = self._mod_noise(values, mod, rng)
            elif t == "trend":
                values = self._mod_trend(values, idx, mod)
            elif t == "seasonality":
                values = self._mod_seasonality(values, idx, mod)
            elif t == "spike":
                values = self._mod_spike(values, mod)
            elif t == "dropout":
                values = self._mod_dropout(values, mod, rng)
            elif t == "clip":
                lo, hi = mod.get("min"), mod.get("max")
                if lo is not None:
                    values = np.maximum(values, lo)
                if hi is not None:
                    values = np.minimum(values, hi)
            elif t == "round":
                values = np.round(values, mod["decimals"])

        lo, hi = self._constructor_clip
        if lo is not None:
            values = np.maximum(values, lo)
        if hi is not None:
            values = np.minimum(values, hi)

        if self.dtype in (int, "int"):
            if not any(m["type"] == "round" for m in self.modifiers):
                values = np.round(values, 0)
            values = values.astype(np.int64)

        return self._apply_nullable(values, n, rng)

    def _apply_nullable(self, values, n: int, rng: np.random.Generator):
        if self.nullable <= 0:
            return values
        mask = rng.random(n) < self.nullable
        if self._is_numeric():
            return np.where(mask, np.nan, np.asarray(values, dtype=float))
        result = np.asarray(values, dtype=object)
        result[mask] = None
        return result

    def _mod_noise(self, values: np.ndarray, mod: dict, rng: np.random.Generator) -> np.ndarray:
        std, dist = mod["std"], mod["distribution"]
        if dist == "gaussian":
            return values + rng.normal(0, std, len(values))
        if dist == "uniform":
            return values + rng.uniform(-std, std, len(values))
        if dist == "poisson":
            lam = max(std, 0)
            return values + rng.poisson(lam, len(values)) - lam
        raise ValueError(f"Unknown noise distribution: {dist!r}")

    def _mod_trend(self, values: np.ndarray, idx: np.ndarray, mod: dict) -> np.ndarray:
        rate, style = mod["rate"], mod["style"]
        if style == "linear":
            return values + self.base * rate * idx
        if style == "exponential":
            return values * (1 + rate) ** idx
        raise ValueError(f"Unknown trend style: {style!r}")

    def _mod_seasonality(self, values: np.ndarray, idx: np.ndarray, mod: dict) -> np.ndarray:
        return values + mod["amplitude"] * np.sin(2 * np.pi * idx / mod["period"] + mod["phase"])

    def _mod_spike(self, values: np.ndarray, mod: dict) -> np.ndarray:
        at, magnitude, duration, shape = mod["at"], mod["magnitude"], mod["duration"], mod["shape"]
        values = values.copy()
        n = len(values)

        if isinstance(magnitude, str) and (magnitude.startswith("+") or magnitude.startswith("-")):
            is_delta, mag_val = True, float(magnitude)
        else:
            is_delta, mag_val = False, float(magnitude)

        half = max(1, (duration + 1) // 2)
        for d in range(duration):
            i = at + d
            if 0 <= i < n:
                t = min(d + 1, duration - d) / half if shape == "triangle" else 1.0
                if is_delta:
                    values[i] += mag_val * t
                else:
                    values[i] *= 1 + (mag_val - 1) * t

        return values

    def _mod_dropout(self, values: np.ndarray, mod: dict, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(len(values)) < mod["rate"]
        fill = mod["fill"]
        values = values.astype(float)
        values[mask] = fill if fill is not None else np.nan
        return values
