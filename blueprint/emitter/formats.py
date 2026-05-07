import json

import pandas as pd


def to_dataframe(columns: dict, column_order: list) -> pd.DataFrame:
    return pd.DataFrame({k: columns[k] for k in column_order})


def to_csv(df: pd.DataFrame, path: str, index: bool = False, encoding: str = "utf-8") -> pd.DataFrame:
    df.to_csv(path, index=index, encoding=encoding)
    return df


def to_json(
    df: pd.DataFrame,
    path: str,
    orient: str = "records",
    indent: int = 2,
    date_format: str = "iso",
) -> pd.DataFrame:
    df.to_json(path, orient=orient, indent=indent, date_format=date_format)
    return df


def to_manifest(
    path: str,
    seed: int,
    n_rows: int,
    features: dict,
    classes: dict,
    influence_graph: dict,
    influence_noise: dict = None,
) -> None:
    manifest = {
        "seed": seed,
        "n_rows": n_rows,
        "features": features,
        "classes": classes,
        "influence_graph": influence_graph,
    }
    if influence_noise:
        manifest["influence_noise"] = influence_noise
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
