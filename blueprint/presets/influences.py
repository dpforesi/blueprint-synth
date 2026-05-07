import numpy as np

from blueprint.core.influence import Influence


def ScalesWith(source: str, target: str, rate: float) -> Influence:
    return Influence(source).on(target, effect=f"+{rate} per unit")


def CorrelatedWith(source: str, target: str, correlation: float) -> Influence:
    def _fn(source_col, target_col, df):
        src_std = source_col.std()
        if src_std < 1e-10:
            return target_col
        src_norm = (source_col - source_col.mean()) / src_std
        tgt_std = target_col.std()
        return target_col + correlation * src_norm * tgt_std

    return Influence(source).on(target, fn=_fn)


def Caps(source: str, target: str, threshold: float, decay: float) -> Influence:
    def _fn(source_col, target_col, df):
        excess = np.maximum(source_col - threshold, 0)
        factor = 1.0 / (1.0 + decay * excess)
        return target_col * factor

    return Influence(source).on(target, fn=_fn)
