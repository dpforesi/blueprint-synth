class Influence:
    def __init__(self, source: str):
        self.source = source
        self.edges: list = []

    def on(
        self,
        target: str,
        effect: str = None,
        by_class: dict = None,
        fn=None,
        when=None,
        noise_std: float = None,
    ) -> "Influence":
        if effect is None and by_class is None and fn is None:
            raise ValueError(
                f"Influence.on('{target}'): at least one of 'effect', 'by_class', or 'fn' is required"
            )
        self.edges.append({
            "target": target,
            "effect": effect,
            "by_class": by_class,
            "fn": fn,
            "when": when,
            "noise_std": noise_std,
        })
        return self
