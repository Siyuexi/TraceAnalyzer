"""Bonus-map case taxonomy and experiment-scope filtering."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from p2a.core import BonusMapStore


DIRECT_CASE = "direct"
LATENT_CASE = "latent"
EXPOSED_CASE = "exposed"
LEGACY_STANDARD_CASE = "standard"
OTHER_CASE = "others"

PRIMARY_CASE_TYPES = (DIRECT_CASE, LATENT_CASE, EXPOSED_CASE)
PATH_CASE_TYPES = frozenset((*PRIMARY_CASE_TYPES, LEGACY_STANDARD_CASE))
CASE_FILTER_BUCKETS = (*PRIMARY_CASE_TYPES, OTHER_CASE)

LEGACY_STANDARD_SUBTYPE_ALIASES = {
    "latent": LATENT_CASE,
    "non_obvious_root": LATENT_CASE,
    "non-obvious-root": LATENT_CASE,
    "nonobvious": LATENT_CASE,
    "pattern": LATENT_CASE,
    "pattern_computable": LATENT_CASE,
    "exposed": EXPOSED_CASE,
    "obvious_root": EXPOSED_CASE,
    "obvious-root": EXPOSED_CASE,
    "collapsed": EXPOSED_CASE,
}


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, Iterable):
        return {str(item) for item in value if isinstance(item, str) and item}
    return set()


def _projection_roots_anchors(projection: dict[str, Any]) -> tuple[set[str], set[str]]:
    return _string_set(projection.get("roots")), _string_set(projection.get("anchors"))


def root_anchor_overlap(roots: Iterable[str], anchors: Iterable[str]) -> str:
    root_set = set(roots)
    if not root_set:
        return "unknown"
    anchor_set = set(anchors)
    overlap = root_set & anchor_set
    if not overlap:
        return "none"
    if root_set <= anchor_set:
        return "all"
    return "partial"


def bonus_map_root_anchor_overlap(bonus_map: dict[str, Any] | None) -> str:
    if not isinstance(bonus_map, dict):
        return "unknown"
    return root_anchor_overlap(
        _string_set(bonus_map.get("root_cause_nodes")),
        _string_set(bonus_map.get("selected_issue_anchor_nodes")),
    )


def _edge_endpoints(edge: Any) -> tuple[str | None, str | None]:
    if isinstance(edge, dict):
        return edge.get("caller") or edge.get("source"), edge.get("callee") or edge.get("target")
    if isinstance(edge, list | tuple) and len(edge) >= 2:
        return edge[0], edge[1]
    return None, None


def _path_edges(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _standard_case_from_projection(roots: set[str], anchors: set[str], path_edges: list[Any]) -> str:
    if not roots or not anchors or roots & anchors or not path_edges:
        return EXPOSED_CASE
    return LATENT_CASE


def canonical_case_type_from_parts(
    case_type: Any,
    *,
    roots: Iterable[str] = (),
    anchors: Iterable[str] = (),
    path_edges: Iterable[Any] = (),
    legacy_standard_fallback: str = EXPOSED_CASE,
) -> str:
    value = str(case_type or "").strip()
    root_set = set(roots)
    anchor_set = set(anchors)
    edge_list = list(path_edges)
    if value == LATENT_CASE and (root_set or anchor_set or edge_list):
        return _standard_case_from_projection(root_set, anchor_set, edge_list)
    if value in PRIMARY_CASE_TYPES:
        return value
    if value == LEGACY_STANDARD_CASE:
        if root_set and anchor_set and edge_list:
            return _standard_case_from_projection(root_set, anchor_set, edge_list)
        return legacy_standard_fallback
    return value


def _bonus_map_pattern_structure(bonus_map: dict[str, Any] | None) -> bool:
    if not isinstance(bonus_map, dict):
        return False
    anchors = _string_set(bonus_map.get("selected_issue_anchor_nodes"))
    roots = _string_set(bonus_map.get("root_cause_nodes"))
    path_edges = _path_edges(bonus_map.get("reward_path_edges"))
    if _standard_case_from_projection(roots, anchors, path_edges) != LATENT_CASE:
        return False
    nodes = bonus_map.get("call_graph_nodes") if isinstance(bonus_map.get("call_graph_nodes"), dict) else {}
    path_node_keys = set(anchors) | set(roots)
    for edge in path_edges:
        caller, callee = _edge_endpoints(edge)
        if isinstance(caller, str):
            path_node_keys.add(caller)
        if isinstance(callee, str):
            path_node_keys.add(callee)
    distances = {
        float(nodes[key]["normalized_distance"])
        for key in path_node_keys
        if key in nodes and isinstance(nodes[key], dict) and nodes[key].get("normalized_distance") is not None
    }
    return len(distances) >= 2


def canonical_bonus_case_type(bonus_map: dict[str, Any] | None) -> str:
    if not isinstance(bonus_map, dict):
        return ""
    raw = str(bonus_map.get("case_type") or "").strip()
    if raw in {LEGACY_STANDARD_CASE, LATENT_CASE}:
        return LATENT_CASE if _bonus_map_pattern_structure(bonus_map) else EXPOSED_CASE
    return canonical_case_type_from_parts(
        raw,
        roots=_string_set(bonus_map.get("root_cause_nodes")),
        anchors=_string_set(bonus_map.get("selected_issue_anchor_nodes")),
        path_edges=_path_edges(bonus_map.get("reward_path_edges")),
    )


def canonical_detail_case_type(detail: dict[str, Any]) -> str:
    raw = str(detail.get("bonus_case_type") or detail.get("path_case_kind") or detail.get("chain_case_kind") or "").strip()
    if raw in PRIMARY_CASE_TYPES:
        return raw
    projection = detail.get("path_projection") or detail.get("chain_projection") or {}
    roots, anchors = _projection_roots_anchors(projection if isinstance(projection, dict) else {})
    edges = []
    if isinstance(projection, dict):
        edges = _path_edges(projection.get("path_edges") or projection.get("chain_edges"))
    return canonical_case_type_from_parts(raw, roots=roots, anchors=anchors, path_edges=edges)


def traceable_case_family(case_type: str) -> str:
    if case_type in {LATENT_CASE, EXPOSED_CASE, LEGACY_STANDARD_CASE}:
        return LEGACY_STANDARD_CASE
    return case_type


def bonus_map_pattern_computable(bonus_map: dict[str, Any] | None) -> bool:
    if not isinstance(bonus_map, dict):
        return False
    if str(bonus_map.get("case_type") or "").strip() == LATENT_CASE:
        return _bonus_map_pattern_structure(bonus_map)
    return canonical_bonus_case_type(bonus_map) == LATENT_CASE


def enrich_bonus_map_case_metadata(bonus_map: dict[str, Any]) -> dict[str, Any]:
    raw_case_type = str(bonus_map.get("case_type") or "")
    case_type = canonical_bonus_case_type(bonus_map)
    if raw_case_type == LEGACY_STANDARD_CASE and case_type in {LATENT_CASE, EXPOSED_CASE}:
        bonus_map["legacy_case_type"] = LEGACY_STANDARD_CASE
        bonus_map["case_type"] = case_type
    bonus_map["case_type_version"] = "latent_exposed_v1"
    bonus_map["traceable_case_family"] = traceable_case_family(case_type)
    bonus_map["root_anchor_overlap"] = bonus_map_root_anchor_overlap(bonus_map)
    bonus_map["pattern_computable"] = bonus_map_pattern_computable(bonus_map)
    return bonus_map


@dataclass(frozen=True)
class BonusMapInstanceFilter:
    case_types: tuple[str, ...] = ()
    pattern_computable: bool | None = None
    require_bonus_map: bool = True

    @property
    def active(self) -> bool:
        return bool(self.case_types) or self.pattern_computable is not None

    def matches(self, bonus_map: dict[str, Any] | None) -> bool:
        if not isinstance(bonus_map, dict):
            return not self.require_bonus_map and not self.active
        case_type = canonical_bonus_case_type(bonus_map)
        if self.case_types and case_type not in self.case_types:
            return False
        if self.pattern_computable is not None and bonus_map_pattern_computable(bonus_map) is not self.pattern_computable:
            return False
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "case_types": list(self.case_types),
            "pattern_computable": self.pattern_computable,
            "require_bonus_map": self.require_bonus_map,
            "case_type_version": "latent_exposed_v1",
        }


def _normalize_case_type_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Iterable):
        raw_values = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw_values = [str(value).strip()]

    out: list[str] = []
    for raw in raw_values:
        mapped = LEGACY_STANDARD_SUBTYPE_ALIASES.get(raw, raw)
        if mapped == LEGACY_STANDARD_CASE:
            mapped_values = (LATENT_CASE, EXPOSED_CASE)
        else:
            mapped_values = (mapped,)
        for item in mapped_values:
            if item not in (*PRIMARY_CASE_TYPES,):
                raise ValueError(f"unknown bonus-map case type: {raw!r}")
            if item not in out:
                out.append(item)
    return tuple(out)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"expected boolean value, got {value!r}")


def parse_bonus_map_instance_filter(value: Any) -> BonusMapInstanceFilter:
    if value in (None, "", False):
        return BonusMapInstanceFilter()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = {"case_type": value}
        value = parsed
    if not isinstance(value, dict):
        raise ValueError("bonus_map_instance_filter must be a mapping")

    case_value = value.get("case_types", value.get("case_type"))
    standard_subtype = value.get("standard_subtype")
    if standard_subtype is not None and case_value in (None, LEGACY_STANDARD_CASE):
        case_value = standard_subtype
    case_types = _normalize_case_type_value(case_value)
    pattern_computable = value.get("pattern_computable", value.get("pattern_eligible"))
    if pattern_computable is not None:
        pattern_computable = _coerce_bool(pattern_computable)
    require_bonus_map = _coerce_bool(value.get("require_bonus_map", True))
    return BonusMapInstanceFilter(
        case_types=case_types,
        pattern_computable=pattern_computable,
        require_bonus_map=require_bonus_map,
    )


@dataclass(frozen=True)
class ScopedRows:
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def select_rows_by_bonus_map_scope(
    rows: list[dict[str, Any]],
    *,
    bonus_map_dir: Path | None,
    instance_id: Callable[[dict[str, Any]], str | None],
    scope_filter: BonusMapInstanceFilter,
    limit: int | None = None,
    offset: int = 0,
) -> ScopedRows:
    source_rows = list(rows)
    store = BonusMapStore(str(bonus_map_dir)) if bonus_map_dir is not None else None
    selected: list[dict[str, Any]] = []
    matched_case_counts: dict[str, int] = {}
    missing_bonus_maps = 0
    for row in source_rows:
        row_instance_id = instance_id(row)
        bonus_map = store.get(row_instance_id) if store is not None and row_instance_id else None
        if store is not None and bonus_map is None:
            missing_bonus_maps += 1
        if scope_filter.active and not scope_filter.matches(bonus_map):
            continue
        if bonus_map is not None:
            case_type = canonical_bonus_case_type(bonus_map)
            matched_case_counts[case_type] = matched_case_counts.get(case_type, 0) + 1
        selected.append(row)

    filtered_before_window = len(selected)
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]

    metadata = {
        "source_size": len(source_rows),
        "selected_size_before_window": filtered_before_window,
        "selected_size": len(selected),
        "limit": limit,
        "offset": offset,
        "bonus_map_dir": str(bonus_map_dir) if bonus_map_dir is not None else None,
        "filter": scope_filter.metadata(),
        "matched_case_types": dict(sorted(matched_case_counts.items())),
        "missing_bonus_maps": missing_bonus_maps,
    }
    return ScopedRows(rows=selected, metadata=metadata)
