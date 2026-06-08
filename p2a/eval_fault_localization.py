"""Score eval rollouts against P2A bonus maps.

This is an offline diagnostic.  The eval-set bonus maps are not used to reshape
training advantages; they provide a reference graph for measuring whether a
model's eval rollout reads files/functions near the eventual fault.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from p2a.core import (
    BonusMapStore,
    compute_p2a_multiplier,
    match_reads_to_callgraph,
    parse_read_actions,
    parse_read_actions_from_tool_calls,
)

SUMMARY_SCHEMA_VERSION = "p2a_eval_fault_localization_v1"
TEXT_FIELDS = (
    "response_text",
    "assistant_response",
    "completion",
    "response",
    "output",
    "text",
)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _records_from_json_payload(payload: Any) -> Iterable[dict]:
    payload = _maybe_json(payload)
    if isinstance(payload, list):
        for item in payload:
            item = _maybe_json(item)
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("records", "rollouts", "data", "items", "samples"):
            nested = _maybe_json(payload.get(key))
            if isinstance(nested, list):
                yield from _records_from_json_payload(nested)
                return
        yield payload


def iter_records(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        for record in df.to_dict(orient="records"):
            yield record
        return

    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL record: {exc}") from exc
                yield from _records_from_json_payload(payload)
        return

    with path.open(encoding="utf-8") as f:
        yield from _records_from_json_payload(json.load(f))


def _get_nested(mapping: dict, *path: str) -> Any:
    value: Any = mapping
    for key in path:
        value = _maybe_json(value)
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _maybe_json(value)


def _as_str(value: Any) -> str | None:
    value = _maybe_json(value)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def extract_instance_id(record: dict) -> str | None:
    direct_keys = (
        "instance_id",
        "uid",
        "task_id",
        "sample_id",
        "id",
    )
    for key in direct_keys:
        value = _as_str(record.get(key))
        if value and ("__" in value or key == "instance_id"):
            return value

    nested_paths = (
        ("metadata", "instance_id"),
        ("extra_fields", "instance_id"),
        ("extra_info", "instance_id"),
        ("extra_info", "tools_kwargs", "reward", "metadata", "instance_id"),
        ("tools_kwargs", "reward", "metadata", "instance_id"),
        ("reward", "metadata", "instance_id"),
    )
    for path in nested_paths:
        value = _as_str(_get_nested(record, *path))
        if value:
            return value
    return None


def _normalize_tool_call(tool_call: Any) -> dict | None:
    tool_call = _maybe_json(tool_call)
    if not isinstance(tool_call, dict):
        return None
    if isinstance(tool_call.get("function"), dict):
        return tool_call
    if "name" in tool_call and ("arguments" in tool_call or "args" in tool_call):
        return {
            "function": {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments", tool_call.get("args", {})),
            }
        }
    if "tool_name" in tool_call:
        return {
            "function": {
                "name": tool_call.get("tool_name"),
                "arguments": tool_call.get("arguments", tool_call.get("args", {})),
            }
        }
    return None


def _normalize_tool_calls(value: Any) -> list[dict]:
    value = _maybe_json(value)
    if value is None:
        return []
    if isinstance(value, dict):
        nested = value.get("tool_calls")
        if nested is not None:
            return _normalize_tool_calls(nested)
        normalized = _normalize_tool_call(value)
        return [normalized] if normalized else []
    if isinstance(value, list):
        calls = []
        for item in value:
            normalized = _normalize_tool_call(item)
            if normalized:
                calls.append(normalized)
        return calls
    return []


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    value = _maybe_json(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            item = _maybe_json(item)
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _reads_from_tool_calls(tool_calls: Any) -> list[dict]:
    normalized = _normalize_tool_calls(tool_calls)
    if not normalized:
        return []
    return parse_read_actions_from_tool_calls(normalized)


def _dedupe_reads(reads: Iterable[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for read in reads:
        if not isinstance(read, dict):
            continue
        file_path = read.get("file_path")
        if not file_path:
            continue
        start = int(read.get("start_line", 1))
        end = int(read.get("end_line", 999999))
        key = (file_path, start, end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"file_path": file_path, "start_line": start, "end_line": end})
    return deduped


def _extract_reads_from_text(value: Any, tracking_mode: str) -> list[dict]:
    text = _content_to_text(value)
    if not text:
        return []
    return parse_read_actions(text, tracking_mode=tracking_mode)


def _messages_reads(messages: Any, tracking_mode: str) -> list[dict]:
    messages = _maybe_json(messages)
    if not isinstance(messages, list):
        return []

    reads = []
    for message in messages:
        message = _maybe_json(message)
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        is_assistant = role in {"assistant", "agent", "model"} or "tool_calls" in message
        if not is_assistant:
            continue
        reads.extend(_reads_from_tool_calls(message.get("tool_calls")))
        reads.extend(_extract_reads_from_text(message.get("content"), tracking_mode))
    return reads


def _candidate_containers(record: dict) -> list[dict]:
    containers = [record]
    for key in ("extra_fields", "extra_info", "metadata"):
        value = _maybe_json(record.get(key))
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _step_trace_items(record: dict) -> list[Any]:
    for container in _candidate_containers(record):
        value = _maybe_json(container.get("p2a_step_traces"))
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("p2a_step_traces", "steps", "traces"):
                nested = _maybe_json(value.get(key))
                if isinstance(nested, list):
                    return nested
    return []


def _reads_from_step(trace: Any, tracking_mode: str) -> list[dict]:
    trace = _maybe_json(trace)
    reads = []
    if isinstance(trace, dict):
        reads.extend(_reads_from_tool_calls(trace.get("tool_calls")))
        for field in TEXT_FIELDS:
            reads.extend(_extract_reads_from_text(trace.get(field), tracking_mode))
        reads.extend(_messages_reads(trace.get("messages"), tracking_mode))
    elif isinstance(trace, list):
        reads.extend(_reads_from_tool_calls(trace))
        for item in trace:
            reads.extend(_extract_reads_from_text(item, tracking_mode))
    elif isinstance(trace, str):
        reads.extend(_extract_reads_from_text(trace, tracking_mode))
    return _dedupe_reads(reads)


def extract_record_reads(record: dict, tracking_mode: str) -> tuple[list[list[dict]], list[dict]]:
    step_reads = [_reads_from_step(trace, tracking_mode) for trace in _step_trace_items(record)]
    step_reads = [reads for reads in step_reads if reads]

    reads = []
    for container in _candidate_containers(record):
        reads.extend(_reads_from_tool_calls(container.get("tool_calls")))
        reads.extend(_messages_reads(container.get("messages"), tracking_mode))
        for field in TEXT_FIELDS:
            reads.extend(_extract_reads_from_text(container.get(field), tracking_mode))

    all_reads = _dedupe_reads([read for reads_for_step in step_reads for read in reads_for_step] + reads)
    return step_reads, all_reads


def _first_matching_step(step_reads: list[list[dict]], bonus_map: dict, *, require_gt: bool) -> int | None:
    for idx, reads in enumerate(step_reads):
        distance = match_reads_to_callgraph(reads, bonus_map)
        if distance < 0:
            continue
        if require_gt and distance != 0.0:
            continue
        return idx
    return None


def score_record(
    record: dict,
    *,
    index: int,
    bonus_maps: BonusMapStore,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
) -> dict:
    instance_id = extract_instance_id(record)
    bonus_map = bonus_maps.get(instance_id) if instance_id else None
    step_reads, reads = extract_record_reads(record, tracking_mode)

    result = {
        "record_index": index,
        "instance_id": instance_id,
        "has_bonus_map": bonus_map is not None,
        "has_step_traces": bool(step_reads),
        "n_steps_with_reads": len(step_reads),
        "n_reads": len(reads),
        "read_files": sorted({read["file_path"] for read in reads}),
        "bonus_case_type": None,
        "bonus_traceable": None,
        "has_call_graph": False,
        "hit_call_graph": False,
        "hit_ground_truth": False,
        "hit_near": False,
        "min_distance": None,
        "best_positive_multiplier": 1.0,
        "first_hit_step": None,
        "first_ground_truth_step": None,
    }
    if not bonus_map:
        return result

    result["bonus_case_type"] = bonus_map.get("case_type")
    result["bonus_traceable"] = bool(bonus_map.get("traceable"))
    result["has_call_graph"] = bool(bonus_map.get("call_graph_nodes"))

    distance = match_reads_to_callgraph(reads, bonus_map)
    if distance < 0:
        return result

    result["hit_call_graph"] = True
    result["hit_ground_truth"] = distance == 0.0
    result["hit_near"] = distance <= near_threshold
    result["min_distance"] = distance
    result["best_positive_multiplier"] = compute_p2a_multiplier(distance, m_max, 1)
    if step_reads:
        result["first_hit_step"] = _first_matching_step(step_reads, bonus_map, require_gt=False)
        result["first_ground_truth_step"] = _first_matching_step(step_reads, bonus_map, require_gt=True)
    else:
        result["first_hit_step"] = 0
        if distance == 0.0:
            result["first_ground_truth_step"] = 0
    return result


def _rate(num: int | float, denom: int | float) -> float | None:
    if not denom:
        return None
    return num / denom


def summarize(details: list[dict], *, source: Path, bonus_map_dir: Path, tracking_mode: str, near_threshold: float, m_max: float) -> dict:
    counts = Counter()
    by_case: dict[str, Counter] = defaultdict(Counter)
    distances = []
    multipliers = []

    for item in details:
        counts["n_records"] += 1
        if item["instance_id"]:
            counts["n_with_instance_id"] += 1
        if item["has_bonus_map"]:
            counts["n_with_bonus_map"] += 1
        if item["has_call_graph"]:
            counts["n_with_call_graph"] += 1
        if item["bonus_traceable"]:
            counts["n_traceable_bonus"] += 1
        if item["n_reads"]:
            counts["n_with_reads"] += 1
        if item["hit_call_graph"]:
            counts["n_hit_call_graph"] += 1
            distances.append(item["min_distance"])
            multipliers.append(item["best_positive_multiplier"])
        if item["hit_ground_truth"]:
            counts["n_hit_ground_truth"] += 1
        if item["hit_near"]:
            counts["n_hit_near"] += 1

        case_type = item["bonus_case_type"] or "missing_bonus_map"
        bucket = by_case[case_type]
        bucket["n"] += 1
        if item["has_call_graph"]:
            bucket["n_with_call_graph"] += 1
        if item["n_reads"]:
            bucket["n_with_reads"] += 1
        if item["hit_call_graph"]:
            bucket["n_hit_call_graph"] += 1
        if item["hit_ground_truth"]:
            bucket["n_hit_ground_truth"] += 1
        if item["hit_near"]:
            bucket["n_hit_near"] += 1

    n_records = counts["n_records"]
    n_with_bonus = counts["n_with_bonus_map"]
    n_with_graph = counts["n_with_call_graph"]
    summary_by_case = {}
    for case_type, bucket in sorted(by_case.items()):
        denom = bucket["n_with_call_graph"] or bucket["n"]
        summary_by_case[case_type] = {
            **dict(bucket),
            "read_rate": _rate(bucket["n_with_reads"], bucket["n"]),
            "graph_hit_rate": _rate(bucket["n_hit_call_graph"], denom),
            "ground_truth_hit_rate": _rate(bucket["n_hit_ground_truth"], denom),
            "near_hit_rate": _rate(bucket["n_hit_near"], denom),
        }

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source": str(source),
        "bonus_map_dir": str(bonus_map_dir),
        "tracking_mode": tracking_mode,
        "near_threshold": near_threshold,
        "m_max": m_max,
        "step_index_origin": "zero_based",
        "counts": dict(counts),
        "rates": {
            "instance_id_rate": _rate(counts["n_with_instance_id"], n_records),
            "bonus_map_coverage": _rate(n_with_bonus, n_records),
            "call_graph_coverage": _rate(n_with_graph, n_records),
            "read_rate": _rate(counts["n_with_reads"], n_records),
            "graph_hit_rate_over_bonus_maps": _rate(counts["n_hit_call_graph"], n_with_bonus),
            "graph_hit_rate_over_call_graphs": _rate(counts["n_hit_call_graph"], n_with_graph),
            "ground_truth_hit_rate_over_call_graphs": _rate(counts["n_hit_ground_truth"], n_with_graph),
            "near_hit_rate_over_call_graphs": _rate(counts["n_hit_near"], n_with_graph),
        },
        "averages": {
            "avg_min_distance_on_hits": sum(distances) / len(distances) if distances else None,
            "avg_best_positive_multiplier_on_hits": sum(multipliers) / len(multipliers) if multipliers else None,
        },
        "by_case_type": summary_by_case,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score eval rollout fault-localization reads against P2A bonus maps.")
    parser.add_argument("rollouts", type=Path, help="Rollout dump (.jsonl, .json, or .parquet)")
    parser.add_argument("--bonus-map-dir", type=Path, required=True, help="Directory containing <instance_id>.json bonus maps")
    parser.add_argument("--tracking-mode", choices=["view_only", "view_and_bash"], default="view_and_bash")
    parser.add_argument("--near-threshold", type=float, default=0.5, help="Distance threshold for near-fault hits")
    parser.add_argument("--m-max", type=float, default=3.0, help="Multiplier used only for diagnostic best_positive_multiplier")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional path for summary JSON")
    parser.add_argument("--details-out", type=Path, default=None, help="Optional path for per-record JSONL details")
    args = parser.parse_args()

    if not args.rollouts.exists():
        raise FileNotFoundError(args.rollouts)
    if not args.bonus_map_dir.is_dir():
        raise NotADirectoryError(args.bonus_map_dir)

    bonus_maps = BonusMapStore(str(args.bonus_map_dir))
    details = [
        score_record(
            record,
            index=index,
            bonus_maps=bonus_maps,
            tracking_mode=args.tracking_mode,
            near_threshold=args.near_threshold,
            m_max=args.m_max,
        )
        for index, record in enumerate(iter_records(args.rollouts))
    ]
    summary = summarize(
        details,
        source=args.rollouts,
        bonus_map_dir=args.bonus_map_dir,
        tracking_mode=args.tracking_mode,
        near_threshold=args.near_threshold,
        m_max=args.m_max,
    )

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if args.details_out:
        write_jsonl(args.details_out, details)

    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
