import numpy as np
import pandas as pd


def generate_datetime(
    n: int,
    start: str,
    end: str,
    distribution: str = "uniform",
    tz: str = None,
    freq: str = None,
    rng: np.random.Generator = None,
) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    delta_ns = end_ts.value - start_ts.value
    offsets_ns = rng.integers(0, delta_ns + 1, size=n)
    result = pd.Series(pd.to_datetime(start_ts.value + offsets_ns))
    if tz:
        result = result.dt.tz_localize(tz)
    return result


def generate_datetime_offset(
    n: int,
    source: pd.Series,
    offset_min: str,
    offset_max: str,
    distribution: str = "uniform",
    rng: np.random.Generator = None,
) -> pd.Series:
    min_s = pd.Timedelta(offset_min).total_seconds()
    max_s = pd.Timedelta(offset_max).total_seconds()
    offsets_s = rng.uniform(min_s, max_s, size=n)
    result = pd.Series([
        source.iloc[i] + pd.Timedelta(seconds=offsets_s[i])
        for i in range(n)
    ])
    return result
