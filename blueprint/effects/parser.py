def parse_effect(effect_str: str) -> tuple:
    s = effect_str.strip()

    if s.startswith("="):
        val = s[1:].strip()
        if val == "source":
            return ("set_source", None)
        try:
            num = float(val)
            return ("set", int(num) if num == int(num) else num)
        except ValueError:
            raise ValueError(f"Cannot parse effect string: {effect_str!r}")

    if s.startswith("*"):
        return ("multiply", float(s[1:].strip()))

    if "per unit" in s:
        base = s.replace("per unit", "").strip()
        if base.endswith("%"):
            return ("per_unit_pct", float(base[:-1]) / 100)
        return ("per_unit", float(base))

    if s.endswith("%"):
        return ("pct", float(s[:-1]) / 100)

    try:
        return ("flat", float(s))
    except ValueError:
        raise ValueError(f"Cannot parse effect string: {effect_str!r}")
