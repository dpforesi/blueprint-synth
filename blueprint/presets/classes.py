from blueprint.core.klass import Class


def RandomClass(name: str, p: float) -> Class:
    return Class(name, when=("__random__", p))


def HighValueClass(name: str, feature: str, top_pct: float) -> Class:
    return Class(name, when=lambda df: df[feature] >= df[feature].quantile(1.0 - top_pct))


def LowValueClass(name: str, feature: str, bottom_pct: float) -> Class:
    return Class(name, when=lambda df: df[feature] <= df[feature].quantile(bottom_pct))


def OutlierClass(name: str, p: float, features: list, magnitude: float) -> Class:
    c = Class(name, when=("__random__", p))
    c._outlier_features = features
    c._outlier_magnitude = magnitude
    return c
