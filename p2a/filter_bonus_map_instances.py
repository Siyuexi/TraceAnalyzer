"""Filter eval rows by bonus-map case taxonomy without splitting datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from p2a.bonus_map_scope import parse_bonus_map_instance_filter, select_rows_by_bonus_map_scope


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _extra_info(row: dict[str, Any]) -> dict[str, Any]:
    value = _maybe_json(row.get("extra_info"))
    return value if isinstance(value, dict) else {}


def _instance_id(row: dict[str, Any]) -> str | None:
    extra = _extra_info(row)
    for value in (
        row.get("instance_id"),
        extra.get("instance_id"),
        ((extra.get("tools_kwargs") or {}).get("reward") or {}).get("metadata", {}).get("instance_id"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("records", [])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    if suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
        return
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _filter_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.filter_json:
        payload = json.loads(args.filter_json)
        if not isinstance(payload, dict):
            raise ValueError("--filter-json must decode to a JSON object")
    else:
        payload = {}
    if args.case_type:
        payload["case_types"] = args.case_type
    if args.pattern_computable is not None:
        payload["pattern_computable"] = args.pattern_computable
    return payload


def _bool_arg(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=Path)
    parser.add_argument("--bonus-map-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-type", action="append", default=[], help="Bonus-map case type to keep: direct, latent, exposed. May repeat.")
    parser.add_argument("--pattern-computable", type=_bool_arg, default=None)
    parser.add_argument("--filter-json", default=None, help="JSON object matching bonus_map_instance_filter config.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scope_filter = parse_bonus_map_instance_filter(_filter_config(args))
    rows = _load_rows(args.data_file)
    scoped = select_rows_by_bonus_map_scope(
        rows,
        bonus_map_dir=args.bonus_map_dir,
        instance_id=_instance_id,
        scope_filter=scope_filter,
        limit=args.limit,
        offset=args.offset,
    )
    if scope_filter.active and not scoped.rows:
        raise SystemExit("bonus-map instance filter selected zero rows")
    _write_rows(args.out, scoped.rows)
    metadata = {
        **scoped.metadata,
        "source_data_file": str(args.data_file),
        "selected_data_file": str(args.out),
    }
    metadata_out = args.metadata_out or args.out.with_suffix(args.out.suffix + ".scope.json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "metadata_out": str(metadata_out), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
