import re

import numpy as np
import pandas as pd


def generate_text(
    n: int,
    template: str,
    pools: dict,
    rng: np.random.Generator,
) -> pd.Series:
    keys = re.findall(r'\{(\w+)\}', template)

    generated = {}
    for key in set(keys):
        feature = pools[key]
        raw = feature.generate(n, rng)
        generated[key] = feature.apply_modifiers(raw, n, rng)

    results = []
    for i in range(n):
        row = {k: generated[k][i] for k in generated}
        results.append(template.format(**row))

    return pd.Series(results)
