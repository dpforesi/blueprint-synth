import numpy as np


def generate_boolean(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n) < p
