import numpy as np
import pandas as pd

from blueprint.core.feature import Feature
from blueprint.core.dag import DAG, BlueprintCycleError
from blueprint.effects.parser import parse_effect
from blueprint.effects.applicators import (
    apply_pct, apply_flat, apply_per_unit, apply_per_unit_pct,
    apply_fn, apply_set, apply_multiply,
)


class Blueprint:
    def __init__(self, n: int, seed: int = 42):
        self.n = n
        self.seed = seed
        self.features: list = []
        self._classes: list = []
        self._influences: list = []
        self._feature_map: dict = {}

    def add_feature(self, *features) -> "Blueprint":
        for f in features:
            self.features.append(f)
            self._feature_map[f.name] = f
        return self

    def add_class(self, *classes) -> "Blueprint":
        for c in classes:
            self._classes.append(c)
        return self

    def add_influence(self, *influences) -> "Blueprint":
        for inf in influences:
            self._influences.append(inf)
        return self

    def validate(self) -> None:
        feature_names = set(self._feature_map.keys())

        for klass in self._classes:
            when = klass.when
            if isinstance(when, tuple) and len(when) >= 1 and when[0] != "__random__":
                col_name = when[0]
                if col_name not in feature_names:
                    raise ValueError(
                        f"Class '{klass.name}': 'when' references undefined feature '{col_name}'"
                    )
            for feat_name in klass.overrides:
                if feat_name not in feature_names:
                    raise ValueError(
                        f"Class '{klass.name}': override references undefined feature '{feat_name}'"
                    )

        adj = {name: set() for name in feature_names}
        for inf in self._influences:
            if inf.source not in feature_names:
                raise ValueError(f"Influence: source '{inf.source}' is not a defined feature")
            for edge in inf.edges:
                tgt = edge["target"]
                if tgt not in feature_names:
                    raise ValueError(f"Influence: target '{tgt}' is not a defined feature")
                noise_std = edge.get("noise_std")
                if noise_std is not None:
                    if not isinstance(noise_std, (int, float)) or noise_std < 0:
                        raise ValueError(
                            f"Influence '{inf.source}' → '{tgt}': "
                            f"noise_std must be a non-negative number, got {noise_std!r}"
                        )
                adj[inf.source].add(tgt)

        if self._detect_cycle(adj):
            raise BlueprintCycleError("Cycle detected in influence graph")

    def describe(self) -> None:
        lines = [f"Blueprint(n={self.n}, seed={self.seed})"]

        lines.append(f"\nFeatures ({len(self.features)}):")
        for f in self.features:
            if f.dtype == "computed":
                lines.append(f"  {f.name:<20} computed")
            elif f.dtype in (float, "float", int, "int", "positive_float", "percentage") or f.derived:
                clip_str = f", clip={f._constructor_clip}" if any(v is not None for v in f._constructor_clip) else ""
                lines.append(f"  {f.name:<20} {str(f.dtype):<12} base={f.base}, std={f.std}{clip_str}")
            else:
                lines.append(f"  {f.name:<20} {str(f.dtype)}")

        if self._classes:
            lines.append(f"\nClasses ({len(self._classes)}):")
            for c in self._classes:
                when = c.when
                if callable(when):
                    cond = "<callable>"
                elif isinstance(when, tuple) and when[0] == "__random__":
                    cond = f"random p={when[1]}"
                else:
                    col, op, val = when
                    cond = f"{col} {op} {val}"
                overrides_str = f", overrides: {list(c.overrides.keys())}" if c.overrides else ""
                lines.append(f"  {c.name:<20} when: {cond}{overrides_str}")

        if self._influences:
            lines.append(f"\nInfluences ({len(self._influences)}):")
            for inf in self._influences:
                for edge in inf.edges:
                    effect = edge["by_class"] or edge["fn"] or edge["effect"] or ""
                    noise_str = f"  [noise_std={edge['noise_std']}]" if edge.get("noise_std") else ""
                    lines.append(f"  {inf.source} → {edge['target']}  {effect}{noise_str}")

        if self._influences or any(f.dtype == "computed" for f in self.features):
            order = self._topological_order()
            lines.append(f"\nEvaluation order: {order}")

        print("\n".join(lines))

    def emit(self, describe: bool = False, manifest: str = None) -> pd.DataFrame:
        if describe:
            self.describe()

        rng = np.random.default_rng(self.seed)
        columns = {}

        for feature in self.features:
            if feature.dtype == "computed":
                columns[feature.name] = np.zeros(self.n, dtype=float)
                continue
            feature_rng = np.random.default_rng(feature.seed) if feature.seed is not None else rng
            raw = feature.generate(self.n, feature_rng)
            columns[feature.name] = feature.apply_modifiers(raw, self.n, feature_rng)

        class_masks = {}
        if self._classes:
            skeleton_df = pd.DataFrame(columns)
            mask_rng = np.random.default_rng([self.seed, 99])
            for klass in self._classes:
                class_masks[klass.name] = klass.resolve_mask(skeleton_df, self.n, mask_rng)

        for klass in self._classes:
            for feat_name, override_params in klass.overrides.items():
                feature = self._feature_map[feat_name]
                if feature.dtype == "computed":
                    continue
                mask = class_masks[klass.name]
                count = int(mask.sum())
                if count == 0:
                    continue
                override_feature = self._build_override_feature(feature, override_params)
                class_rng = np.random.default_rng(self._class_feature_seed(klass.name, feat_name))
                raw_override = override_feature.generate(count, class_rng)
                vals_override = override_feature.apply_modifiers(raw_override, count, class_rng)
                columns[feat_name] = self._splice_values(columns[feat_name], mask, vals_override)

        if self._influences:
            self._apply_influences(columns, class_masks)

        for feature in self.features:
            if feature.dtype == "computed":
                formula = feature._kwargs.get("formula")
                if formula is None:
                    raise ValueError(f"Feature '{feature.name}': dtype='computed' requires a formula= kwarg")
                df_current = pd.DataFrame(columns)
                result = formula(df_current)
                columns[feature.name] = result.values if isinstance(result, pd.Series) else np.asarray(result)

        df = pd.DataFrame(columns)

        if manifest is not None:
            from blueprint.emitter.formats import to_manifest
            influence_noise = {
                f"{inf.source}→{edge['target']}": edge["noise_std"]
                for inf in self._influences
                for edge in inf.edges
                if edge.get("noise_std") is not None
            }
            to_manifest(
                manifest,
                seed=self.seed,
                n_rows=self.n,
                features={f.name: str(f.dtype) for f in self.features},
                classes={c.name: str(c.when) for c in self._classes},
                influence_graph={
                    inf.source: [e["target"] for e in inf.edges]
                    for inf in self._influences
                },
                influence_noise=influence_noise or None,
            )

        return df

    def _apply_influences(self, columns: dict, class_masks: dict) -> None:
        topo_order = self._topological_order()

        influence_by_target: dict = {name: [] for name in [f.name for f in self.features]}
        for inf in self._influences:
            for edge in inf.edges:
                influence_by_target[edge["target"]].append((inf, edge))

        df = pd.DataFrame({name: np.asarray(v) for name, v in columns.items()})

        for feat_name in topo_order:
            edges = influence_by_target.get(feat_name, [])
            if not edges:
                continue

            for inf, edge in edges:
                source_raw = df[inf.source].values
                target_col = df[feat_name].values.astype(float)
                source_col = source_raw.astype(float) if source_raw.dtype != object else source_raw

                base_mask = np.ones(self.n, dtype=bool)

                if edge["when"] is not None:
                    base_mask &= self._resolve_condition(edge["when"], df)

                if source_raw.dtype == bool or (
                    hasattr(source_raw, 'dtype') and np.issubdtype(source_raw.dtype, np.bool_)
                ):
                    base_mask &= source_raw

                noise_std = edge.get("noise_std")
                edge_rng = None
                if noise_std is not None and noise_std > 0:
                    edge_rng = np.random.default_rng(
                        self._influence_edge_seed(inf.source, edge["target"])
                    )

                if edge["fn"] is not None:
                    new_col = apply_fn(
                        target_col, source_col, fn=edge["fn"], df=df, mask=base_mask, rng=edge_rng
                    )
                else:
                    noise_multipliers = (
                        edge_rng.normal(1.0, noise_std, self.n) if edge_rng is not None else None
                    )
                    if edge["by_class"] is not None:
                        new_col = target_col.copy()
                        class_handled = np.zeros(self.n, dtype=bool)

                        for class_name, class_effect_str in edge["by_class"].items():
                            if class_name in class_masks:
                                class_mask = class_masks[class_name] & base_mask
                                if class_mask.any():
                                    kind, params = parse_effect(class_effect_str)
                                    new_col = self._apply_effect(
                                        new_col, source_col, kind, params, class_mask, df, noise_multipliers
                                    )
                                    class_handled |= class_masks[class_name]

                        default_effect = edge["effect"]
                        if default_effect is not None:
                            remaining = base_mask & ~class_handled
                            if remaining.any():
                                kind, params = parse_effect(default_effect)
                                new_col = self._apply_effect(
                                    new_col, source_col, kind, params, remaining, df, noise_multipliers
                                )
                    elif edge["effect"] is not None:
                        kind, params = parse_effect(edge["effect"])
                        new_col = self._apply_effect(
                            target_col, source_col, kind, params, base_mask, df, noise_multipliers
                        )
                    else:
                        continue

                df[feat_name] = new_col

            columns[feat_name] = df[feat_name].values

    def _apply_effect(self, target, source, kind, params, mask, df, noise=None):
        if noise is not None:
            return self._apply_effect_noisy(target, source, kind, params, mask, noise)
        if kind == "pct":
            return apply_pct(target, params, mask)
        if kind == "flat":
            return apply_flat(target, params, mask)
        if kind == "per_unit":
            return apply_per_unit(target, params, source, mask)
        if kind == "per_unit_pct":
            return apply_per_unit_pct(target, params, source, mask)
        if kind == "set":
            return apply_set(target, params, mask)
        if kind == "set_source":
            out = target.copy()
            out[mask] = source[mask]
            return out
        if kind == "multiply":
            return apply_multiply(target, params, mask)
        raise ValueError(f"Unknown effect kind: {kind}")

    def _apply_effect_noisy(self, target, source, kind, params, mask, noise):
        out = np.array(target, dtype=float)
        src = np.asarray(source, dtype=float)
        n = noise[mask]
        if kind == "flat":
            out[mask] += params * n
        elif kind == "pct":
            out[mask] *= 1.0 + params * n
        elif kind == "per_unit":
            out[mask] += params * n * src[mask]
        elif kind == "per_unit_pct":
            out[mask] *= 1.0 + params * n * src[mask]
        elif kind == "multiply":
            out[mask] *= params * n
        else:
            # set / set_source: noise is not meaningful — apply deterministically
            if kind == "set":
                out[mask] = params
            elif kind == "set_source":
                out[mask] = src[mask]
        return out

    def _resolve_condition(self, when, df: pd.DataFrame) -> np.ndarray:
        if callable(when):
            result = when(df)
            return result.values if isinstance(result, pd.Series) else np.asarray(result, dtype=bool)
        col, op, val = when
        series = df[col]
        if op == "==":
            return (series == val).values
        if op == "!=":
            return (series != val).values
        if op == ">":
            return (series > val).values
        if op == ">=":
            return (series >= val).values
        if op == "<":
            return (series < val).values
        if op == "<=":
            return (series <= val).values
        if op == "between":
            low, high = val
            return ((series >= low) & (series <= high)).values
        if op == "in":
            return series.isin(val).values
        raise ValueError(f"Unknown condition operator: {op!r}")

    def _topological_order(self) -> list:
        dag = DAG()
        for f in self.features:
            dag.add_node(f.name)
        for inf in self._influences:
            for edge in inf.edges:
                dag.add_edge(inf.source, edge["target"])
        return dag.topological_sort()

    def _detect_cycle(self, adj: dict) -> bool:
        dag = DAG()
        for name in adj:
            dag.add_node(name)
        for src, targets in adj.items():
            for tgt in targets:
                dag.add_edge(src, tgt)
        return dag.has_cycle()

    def _class_feature_seed(self, class_name: str, feature_name: str) -> int:
        h = self.seed
        for ch in f"{class_name}\x00{feature_name}":
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _influence_edge_seed(self, source: str, target: str) -> int:
        h = self.seed
        for ch in f"__edge_noise__{source}\x00{target}":
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _build_override_feature(self, base: Feature, override_params: dict) -> Feature:
        params = dict(
            name=base.name,
            dtype=base.dtype,
            base=base.base,
            std=base.std,
            clip=base._constructor_clip,
            p=base.p,
            values=base.values,
            weights=base.weights,
            derived=base.derived,
            nullable=base.nullable,
            seed=base.seed,
            **base._kwargs,
        )
        params.update(override_params)
        f = Feature(**params)
        f.modifiers = list(base.modifiers)
        return f

    def _splice_values(self, base, mask: np.ndarray, override_vals):
        if isinstance(base, pd.Categorical):
            result = list(base)
            ov_list = list(override_vals)
            for i, val in zip(np.where(mask)[0], ov_list):
                result[i] = val
            return pd.Categorical(result)
        base_arr = np.asarray(base)
        result = base_arr.copy()
        result[mask] = np.asarray(override_vals, dtype=base_arr.dtype)
        return result

    def to_csv(self, path: str, index: bool = False, encoding: str = "utf-8") -> pd.DataFrame:
        from blueprint.emitter.formats import to_csv as _to_csv
        df = self.emit()
        return _to_csv(df, path, index=index, encoding=encoding)

    def to_json(
        self,
        path: str,
        orient: str = "records",
        indent: int = 2,
        date_format: str = "iso",
        manifest: str = None,
    ) -> pd.DataFrame:
        from blueprint.emitter.formats import to_json as _to_json, to_manifest
        df = self.emit()
        _to_json(df, path, orient=orient, indent=indent, date_format=date_format)
        if manifest is not None:
            influence_noise = {
                f"{inf.source}→{edge['target']}": edge["noise_std"]
                for inf in self._influences
                for edge in inf.edges
                if edge.get("noise_std") is not None
            }
            to_manifest(
                manifest,
                seed=self.seed,
                n_rows=self.n,
                features={f.name: str(f.dtype) for f in self.features},
                classes={c.name: str(c.when) for c in self._classes},
                influence_graph={
                    inf.source: [e["target"] for e in inf.edges]
                    for inf in self._influences
                },
                influence_noise=influence_noise or None,
            )
        return df
