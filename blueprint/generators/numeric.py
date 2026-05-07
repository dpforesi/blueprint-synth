import numpy as np


def generate_float(n: int, base: float, std: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(base, std, n) if std > 0 else np.full(n, float(base))


def generate_int(n: int, base: float, std: float, rng: np.random.Generator) -> np.ndarray:
    return np.round(generate_float(n, base, std, rng)).astype(np.int64)


def generate_positive_float(n: int, base: float, std: float, rng: np.random.Generator) -> np.ndarray:
    return np.maximum(0.0, generate_float(n, base, std, rng))


def generate_percentage(n: int, base: float, std: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(generate_float(n, base, std, rng), 0.0, 1.0)
