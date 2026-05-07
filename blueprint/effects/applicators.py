import numpy as np


def apply_pct(target: np.ndarray, pct: float, mask: np.ndarray = None) -> np.ndarray:
    out = np.array(target, dtype=float)
    if mask is None:
        out *= (1 + pct)
    else:
        out[mask] *= (1 + pct)
    return out


def apply_flat(target: np.ndarray, delta: float, mask: np.ndarray = None) -> np.ndarray:
    out = np.array(target, dtype=float)
    if mask is None:
        out += delta
    else:
        out[mask] += delta
    return out


def apply_per_unit(target: np.ndarray, rate: float, source: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    out = np.array(target, dtype=float)
    src = np.asarray(source, dtype=float)
    if mask is None:
        out += rate * src
    else:
        out[mask] += rate * src[mask]
    return out


def apply_per_unit_pct(target: np.ndarray, rate: float, source: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    out = np.array(target, dtype=float)
    src = np.asarray(source, dtype=float)
    if mask is None:
        out *= (1 + rate * src)
    else:
        out[mask] *= (1 + rate * src[mask])
    return out


def apply_fn(target: np.ndarray, source: np.ndarray, fn, df, mask: np.ndarray = None, rng=None) -> np.ndarray:
    import inspect
    out = np.array(target, dtype=float)
    src = np.asarray(source, dtype=float)
    if rng is not None:
        try:
            accepts_rng = "rng" in inspect.signature(fn).parameters
        except (ValueError, TypeError):
            accepts_rng = False
        result = np.asarray(fn(src, out, df, rng=rng) if accepts_rng else fn(src, out, df), dtype=float)
    else:
        result = np.asarray(fn(src, out, df), dtype=float)
    if mask is not None:
        out[mask] = result[mask]
        return out
    return result


def apply_set(target: np.ndarray, value, mask: np.ndarray = None) -> np.ndarray:
    out = np.array(target, dtype=float)
    if mask is None:
        out[:] = value
    else:
        out[mask] = value
    return out


def apply_multiply(target: np.ndarray, factor: float, mask: np.ndarray = None) -> np.ndarray:
    out = np.array(target, dtype=float)
    if mask is None:
        out *= factor
    else:
        out[mask] *= factor
    return out
