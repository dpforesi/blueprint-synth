import numpy as np
import pandas as pd


def generate_categorical(
    n: int,
    values: list,
    weights: list,
    rng: np.random.Generator,
) -> pd.Categorical:
    if values is None:
        raise ValueError("Feature with dtype='category' must specify values")
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    indices = rng.choice(len(values), size=n, p=w)
    return pd.Categorical([values[i] for i in indices], categories=values)
