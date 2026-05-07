import uuid as _uuid

import numpy as np
import pandas as pd


def generate_id(
    n: int,
    style: str = "uuid4",
    start: int = 1,
    step: int = 1,
    prefix: str = "",
    padding: int = 0,
    rng: np.random.Generator = None,
) -> pd.Series:
    if rng is None:
        rng = np.random.default_rng()

    if style == "uuid4":
        all_bytes = rng.integers(0, 256, (n, 16), dtype=np.uint8)
        all_bytes[:, 6] = (all_bytes[:, 6] & 0x0F) | 0x40
        all_bytes[:, 8] = (all_bytes[:, 8] & 0x3F) | 0x80
        return pd.Series([str(_uuid.UUID(bytes=b.tobytes())) for b in all_bytes])

    if style == "uuid1":
        return pd.Series([str(_uuid.uuid1()) for _ in range(n)])

    if style == "sequential":
        return pd.Series(range(start, start + n * step, step), dtype=np.int64)

    if style == "prefixed":
        seq = range(start, start + n)
        if padding:
            return pd.Series([f"{prefix}{str(i).zfill(padding)}" for i in seq])
        return pd.Series([f"{prefix}{i}" for i in seq])

    raise ValueError(f"Unknown id style: {style!r}")


def generate_row_number(n: int) -> np.ndarray:
    return np.arange(n)
