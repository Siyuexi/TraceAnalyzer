#!/usr/bin/env python3
"""Relabel bonus-map node roles after role-priority fixes."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROLE_LIST_KEYS = {
    "symptom": "symptom_nodes",
    "test_adapter": "test_adapter_nodes",
    "root_cause": "root_cause_nodes",
    "fix_adapter": "fix_adapter_nodes",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _node_role(node: dict[str, Any]) -> str:
    role = node.get("node_role")
    return "test_adapter" if role == "pre_symptom" else str(role or "")


def _sync_role_lists(data: dict[str, Any]) -> None:
    nodes = data.get("call_graph_nodes")
    if not isinstance(nodes, dict):
        return
    grouped = {role: [] for role in ROLE_LIST_KEYS}
    for node_key, node in nodes.items():
        if isinstance(node, dict):
            role = _node_role(node)
            if role in grouped:
                grouped[role].append(str(node_key))
    for role, key in ROLE_LIST_KEYS.items():
        data[key] = sorted(grouped[role])


def _sync_edge_roles(data: dict[str, Any]) -> None:
    nodes = data.get("call_graph_nodes")
    metadata = data.get("call_graph_edge_metadata")
    if not isinstance(nodes, dict) or not isinstance(metadata, list):
        return
    direct_edges: list[list[str]] = []
    for item in metadata:
        if not isinstance(item, dict):
            continue
        caller = item.get("caller")
        callee = item.get("callee")
        caller_role = _node_role(nodes.get(caller, {})) if caller in nodes else str(item.get("caller_role") or "")
        callee_role = _node_role(nodes.get(callee, {})) if callee in nodes else str(item.get("callee_role") or "")
        item["caller_role"] = caller_role
        item["callee_role"] = callee_role
        item["role_transition"] = f"{caller_role}->{callee_role}"
        direct = caller_role == "symptom" and callee_role == "root_cause"
        item["direct_symptom_to_root_cause"] = direct
        if direct and isinstance(caller, str) and isinstance(callee, str):
            direct_edges.append([caller, callee])
    data["direct_symptom_to_root_cause_edges"] = direct_edges


def relabel_bonus_map(data: dict[str, Any]) -> list[dict[str, str]]:
    nodes = data.get("call_graph_nodes")
    if not isinstance(nodes, dict):
        return []
    anchors = {str(key) for key in data.get("selected_issue_anchor_nodes") or []}
    changes: list[dict[str, str]] = []
    for node_key in sorted(anchors):
        node = nodes.get(node_key)
        if not isinstance(node, dict):
            continue
        old_role = _node_role(node)
        if old_role != "fix_adapter":
            continue
        node["node_role"] = "symptom"
        changes.append({"node": node_key, "old_role": old_role, "new_role": "symptom"})
    if changes:
        data["issue_anchor_source"] = "issue_anchor"
        _sync_role_lists(data)
        _sync_edge_roles(data)
    return changes


def _backup_path(path: Path, backup_dir: Path) -> Path:
    relative = path.name
    return backup_dir / relative


def relabel_path(path: Path, *, dry_run: bool, backup_dir: Path | None) -> list[dict[str, str]]:
    data = _load_json(path)
    changes = relabel_bonus_map(data)
    if not changes or dry_run:
        return changes
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, _backup_path(path, backup_dir))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return changes


def _iter_bonus_maps(bonus_map_dir: Path, instance_ids: set[str] | None) -> list[Path]:
    if instance_ids:
        return [bonus_map_dir / f"{instance_id}.json" for instance_id in sorted(instance_ids)]
    return sorted(bonus_map_dir.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bonus_map_dir", type=Path)
    parser.add_argument("--instance-id", action="append", dest="instance_ids", help="Restrict relabeling to one instance id. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing JSON files.")
    parser.add_argument("--backup-dir", type=Path, help="Directory for original JSON copies before writing.")
    args = parser.parse_args()

    bonus_map_dir = args.bonus_map_dir
    if not bonus_map_dir.is_dir():
        raise SystemExit(f"bonus map directory not found: {bonus_map_dir}")
    instance_ids = set(args.instance_ids or []) or None
    backup_dir = args.backup_dir
    if backup_dir is None and not args.dry_run:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("/data/tmp") / f"bonus_map_role_relabel_backup_{stamp}"

    files = _iter_bonus_maps(bonus_map_dir, instance_ids)
    missing = [path for path in files if not path.exists()]
    if missing:
        raise SystemExit("missing bonus maps: " + ", ".join(str(path) for path in missing))

    changed: dict[str, list[dict[str, str]]] = {}
    for path in files:
        changes = relabel_path(path, dry_run=args.dry_run, backup_dir=backup_dir)
        if changes:
            changed[path.name] = changes

    total_nodes = sum(len(items) for items in changed.values())
    mode = "would relabel" if args.dry_run else "relabeled"
    print(f"{mode} {total_nodes} node(s) in {len(changed)} bonus map(s)")
    if backup_dir is not None and changed and not args.dry_run:
        print(f"backup_dir={backup_dir}")
    for name, changes in changed.items():
        print(name)
        for change in changes:
            print(f"  {change['node']}: {change['old_role']} -> {change['new_role']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
