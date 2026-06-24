"""Normalized data adapter for the unified P2A HTML dashboard."""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from p2a.core import BonusMapStore, normalize_action, reads_from_step_trace
from p2a.eval_cache import aggregate_model_metrics, ensure_db, json_loads
from p2a.eval_fault_localization import (
    _json_default,
    iter_records,
    score_record,
    summarize,
    summarize_trends,
)


DASHBOARD_SCHEMA_VERSION = "p2a_unified_dashboard_v1"
THIRD_PARTY_PROVIDER_SOURCES = {
    "internal_api",
    "openai_compatible",
    "third_party_api",
    "third_party",
    "api",
}
CHAIN_BAD_PATTERN_KEYS = (
    "missed_anchor",
    "missed_root_after_anchor",
    "root_before_anchor",
    "chain_stall",
    "chain_read_loop",
    "off_chain_read_spree",
    "error_spiral_on_chain",
)
VERIFY_MARKERS = (
    "reward_spec",
    "Eval report:",
    "Beginning environment shutdown",
    "Environment shutdown completed",
    "num_turns:",
)
RUNNING_MARKERS = (
    "Beginning environment startup",
    "Runtime initialized",
    "STEP ",
    "MODEL INPUT",
    "ACTION:",
)


@dataclass(frozen=True)
class DashboardRequest:
    rollouts: tuple[Path, ...] = ()
    details: tuple[Path, ...] = ()
    db_path: Path | None = None
    log_dir: Path | None = None
    bonus_map_dir: Path | None = None
    experiment_id: str | None = None
    provider_source: str | None = None
    dataset: str | None = None
    tracking_mode: str = "view_and_bash"
    near_threshold: float = 0.5
    m_max: float = 3.0
    detail_limit: int = 500


def _safe_json_loads(value: str | None, default: Any) -> Any:
    loaded = json_loads(value, default)
    return loaded if loaded is not None else default


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, ensure_ascii=False))


def _as_mapping(value: Any) -> dict[str, Any]:
    value = _safe_json_loads(value, value) if isinstance(value, str) else value
    return value if isinstance(value, dict) else {}


def _as_sequence(value: Any) -> list[Any]:
    value = _safe_json_loads(value, value) if isinstance(value, str) else value
    return value if isinstance(value, list) else []


def _tool_name(tool_call: Any) -> str:
    call = _as_mapping(tool_call)
    function = _as_mapping(call.get("function"))
    if function.get("name"):
        return str(function.get("name"))
    return str(call.get("name") or call.get("tool_name") or "unknown")


def _tool_args(tool_call: Any) -> Any:
    call = _as_mapping(tool_call)
    function = _as_mapping(call.get("function"))
    args = function.get("arguments") if function else call.get("arguments", call.get("args", {}))
    return _safe_json_loads(args, args) if isinstance(args, str) else args


def _source_kind(*, provider_source: str | None, schema_version: str | None, run_step: Any, log_dir: bool = False) -> str:
    source = (provider_source or "").lower()
    schema = (schema_version or "").lower()
    if source in THIRD_PARTY_PROVIDER_SOURCES or "third_party" in schema or "api" in source:
        return "third_party_api"
    if log_dir or "uni_agent" in schema:
        return "local_inference"
    if run_step is not None:
        return "local_training"
    return "offline_artifact"


def _experiment_key(parts: dict[str, Any]) -> str:
    fields = (
        parts.get("source_kind") or "unknown",
        parts.get("experiment_id") or "adhoc",
        parts.get("provider_source") or "unknown-provider",
        parts.get("dataset") or parts.get("data_source") or "unknown-dataset",
        parts.get("model_api_name") or parts.get("model_label") or "unknown-model",
        parts.get("model_label") or parts.get("model_api_name") or "unknown-label",
    )
    return "::".join(str(field) for field in fields)


def _dataset_name(item: dict[str, Any]) -> str:
    return str(item.get("dataset") or item.get("data_source") or "unknown-dataset")


def _eval_cell_key(parts: dict[str, Any]) -> str:
    return str(parts.get("eval_cell_key") or parts.get("experiment_key") or _experiment_key(parts))


def _record_metadata(record: dict[str, Any], request: DashboardRequest, *, log_dir: bool = False) -> dict[str, Any]:
    extra = _as_mapping(record.get("extra_fields")) or _as_mapping(record.get("extra_info")) or _as_mapping(record.get("metadata"))
    provider_source = (
        request.provider_source
        or record.get("provider_source")
        or extra.get("provider_source")
        or ("local" if log_dir else None)
    )
    dataset = request.dataset or record.get("dataset") or record.get("data_source") or extra.get("data_source")
    run_step = record.get("run_step") or record.get("global_step") or record.get("trainer_step") or extra.get("run_step")
    schema_version = str(record.get("schema_version") or "")
    kind = _source_kind(
        provider_source=str(provider_source) if provider_source else None,
        schema_version=schema_version,
        run_step=run_step,
        log_dir=log_dir,
    )
    run_id = record.get("run_id") or extra.get("run_id")
    model_label = record.get("model_label") or record.get("model") or extra.get("model_label") or extra.get("model")
    model_api_name = record.get("model_api_name") or extra.get("model_api_name") or model_label
    experiment_id = (
        request.experiment_id
        or record.get("experiment_id")
        or extra.get("experiment_id")
        or (run_id if kind == "local_inference" and run_id else None)
        or dataset
        or "adhoc"
    )
    metadata = {
        "experiment_id": str(experiment_id) if experiment_id is not None else None,
        "source_kind": kind,
        "provider_source": str(provider_source) if provider_source is not None else None,
        "dataset": str(dataset) if dataset is not None else None,
        "model_api_name": str(model_api_name) if model_api_name is not None else None,
        "model_label": str(model_label) if model_label is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "run_step": run_step,
        "artifact_root": str(record.get("artifact_root") or record.get("artifact_rollouts") or ""),
        "schema_version": schema_version or None,
    }
    metadata["experiment_key"] = _experiment_key(metadata)
    metadata["eval_cell_key"] = metadata["experiment_key"]
    return metadata


def _step_inspection(record: dict[str, Any], step_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = _as_sequence(record.get("p2a_step_traces"))
    by_trace_index = {
        int(detail.get("trace_index", index)): detail
        for index, detail in enumerate(step_details or [])
        if isinstance(detail, dict)
    }
    out = []
    for index, trace_value in enumerate(traces):
        trace = _as_mapping(trace_value)
        tool_calls = _as_sequence(trace.get("tool_calls"))
        tool_names = [_tool_name(call) for call in tool_calls] or ["no-tool"]
        tool_args = [_tool_args(call) for call in tool_calls]
        primary_args = _as_mapping(tool_args[0]) if tool_args else {}
        tool_results = _as_sequence(trace.get("tool_results"))
        observation = _tool_observation(tool_results)
        scored = by_trace_index.get(index, {})
        action = normalize_action(trace, tracking_mode="view_and_bash")
        recovered_reads = scored.get("reads") or reads_from_step_trace(trace, tracking_mode="view_and_bash")
        out.append(
            {
                "trace_index": index,
                "step_index": trace.get("step_idx", scored.get("step_index", index)),
                "tool_names": tool_names,
                "tool_name": tool_names[0] if tool_names else "no-tool",
                "tool_args": _jsonable(tool_args),
                "action_family": action.get("family"),
                "target_path": action.get("target_path"),
                "command": primary_args.get("command"),
                "path": primary_args.get("path") or primary_args.get("file"),
                "view_range": primary_args.get("view_range"),
                "old_str": primary_args.get("old_str"),
                "new_str": primary_args.get("new_str"),
                "thought": trace.get("thought") or "",
                "think": trace.get("thought") or "",
                "response_text": trace.get("response_text") or trace.get("response") or trace.get("assistant_response") or "",
                "tool_calls": _jsonable(tool_calls),
                "tool_results": _jsonable(tool_results),
                "observation": observation,
                "raw_action": _jsonable(tool_calls[0]) if tool_calls else "",
                "status": "error" if _looks_like_error(observation) else "ok",
                "recovered_reads": _jsonable(recovered_reads),
                "exit_reason": trace.get("exit_reason"),
                "parse_error": trace.get("parse_error"),
                "scored": scored,
            }
        )
    return out


def _tool_observation(tool_results: list[Any]) -> str:
    chunks = []
    for result_value in tool_results:
        result = _as_mapping(result_value)
        for key in ("observation", "content", "result", "output", "stderr", "stdout", "error"):
            value = result.get(key)
            if value not in (None, ""):
                chunks.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=_json_default))
                break
        if not result and isinstance(result_value, str):
            chunks.append(result_value)
    return "\n\n".join(chunks)


def _looks_like_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("traceback", "error:", "exception", "command failed", "no such file"))


def _enrich_detail_from_record(
    detail: dict[str, Any],
    record: dict[str, Any],
    request: DashboardRequest,
    *,
    log_dir: bool = False,
) -> dict[str, Any]:
    metadata = _record_metadata(record, request, log_dir=log_dir)
    enriched = {**metadata, **detail}
    if metadata.get("dataset") and enriched.get("data_source") in {None, "unknown"}:
        enriched["data_source"] = metadata["dataset"]
    enriched["experiment_key"] = metadata["experiment_key"]
    enriched["eval_cell_key"] = metadata["eval_cell_key"]
    enriched["model_label"] = metadata.get("model_label") or enriched.get("model_label")
    enriched["model_api_name"] = metadata.get("model_api_name") or enriched.get("model_api_name")
    enriched["run_id"] = metadata.get("run_id") or enriched.get("run_id")
    enriched["raw_available"] = bool(record.get("p2a_step_traces") or record.get("messages") or record.get("trajectory"))
    enriched["messages"] = _jsonable(record.get("messages") or [])
    enriched["trajectory"] = _jsonable(record.get("trajectory") or [])
    enriched["raw_response_text"] = record.get("response_text") or ""
    enriched["reward"] = record.get("reward")
    enriched["resolved"] = record.get("resolved")
    enriched["termination_reason"] = record.get("termination_reason")
    enriched["error"] = record.get("error")
    enriched["system_error"] = record.get("system_error")
    enriched["step_inspection"] = _step_inspection(record, enriched.get("step_details") or [])
    return enriched


def _is_scored_detail(record: dict[str, Any]) -> bool:
    return "record_index" in record and (
        "chain_evaluable" in record or "hit_call_graph" in record or "step_details" in record
    )


def _normalize_detail(detail: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    normalized = {
        "record_index": index if index is not None else detail.get("record_index", 0),
        "instance_id": None,
        "data_source": "unknown",
        "run_step": None,
        "has_bonus_map": False,
        "has_step_traces": False,
        "n_steps_with_reads": 0,
        "n_reads": 0,
        "read_files": [],
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
        "hit_precision": None,
        "hit_recall": None,
        "hit_f1": None,
        "n_hit_nodes": 0,
        "n_call_graph_nodes": 0,
        "order_score": None,
        "order_defined": False,
        "miracle_step": None,
        "miracle_severity": None,
        "block_order_score": None,
        "block_order_defined": False,
        "block_miracle_step": None,
        "block_miracle_severity": None,
        "graph_topology": None,
        "chain_evaluable": False,
        "not_chain_evaluable_reason": "missing_bonus_map",
        "chain_case_kind": None,
        "chain_graph_covered": False,
        "chain_projection": None,
        "chain_hit": False,
        "anchor_hit": False,
        "root_hit": False,
        "chain_node_recall": None,
        "chain_read_precision": None,
        "first_anchor_step": None,
        "first_root_step": None,
        "steps_anchor_to_root": None,
        "anchor_before_root": None,
        "n_chain_nodes": 0,
        "n_context_nodes": 0,
        "n_hit_chain_nodes": 0,
        "chain_bad_patterns": {},
        "step_details": [],
        "purpose_blocks": [],
        "bad_patterns": {},
        "n_blocks": 0,
        "n_scored_read_blocks": 0,
        "n_achieving_blocks": 0,
        "n_wasted_blocks": 0,
        "n_loop_blocks": 0,
        "n_block_steps": 0,
        "n_scored_read_block_steps": 0,
        "n_achieving_block_steps": 0,
        "n_wasted_block_steps": 0,
        "n_loop_block_steps": 0,
        "block_efficiency": None,
    }
    normalized.update(detail)
    return normalized


def _normalize_details(details: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, detail in enumerate(details):
        item = _normalize_detail(detail, index=index)
        item.setdefault("source_kind", _source_kind(
            provider_source=item.get("provider_source"),
            schema_version=item.get("schema_version"),
            run_step=item.get("run_step"),
        ))
        item.setdefault("experiment_key", _experiment_key(item))
        item.setdefault("eval_cell_key", item.get("experiment_key"))
        normalized.append(item)
    return normalized


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    counts = summary.setdefault("counts", {})
    for key in (
        "n_records",
        "n_with_instance_id",
        "n_with_bonus_map",
        "n_with_call_graph",
        "n_with_reads",
        "n_chain_evaluable",
        "n_not_chain_evaluable",
    ):
        counts.setdefault(key, 0)
    summary.setdefault("rates", {})
    summary.setdefault("averages", {})
    summary.setdefault("distributions", {})
    summary.setdefault("by_case_type", {})
    return summary


def _empty_summary(request: DashboardRequest) -> dict[str, Any]:
    bonus_map_dir = request.bonus_map_dir or Path(".")
    return _finalize_summary(summarize(
        [],
        source=Path("empty"),
        bonus_map_dir=bonus_map_dir,
        tracking_mode=request.tracking_mode,
        near_threshold=request.near_threshold,
        m_max=request.m_max,
    ))


def _score_records(
    records: Iterable[dict[str, Any]],
    *,
    request: DashboardRequest,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    if request.bonus_map_dir is None:
        return []
    bonus_maps = BonusMapStore(str(request.bonus_map_dir))
    scored = []
    for index, record in enumerate(records):
        detail = score_record(
            record,
            index=start_index + index,
            bonus_maps=bonus_maps,
            tracking_mode=request.tracking_mode,
            near_threshold=request.near_threshold,
            m_max=request.m_max,
        )
        scored.append(_enrich_detail_from_record(detail, record, request))
    return scored


def _load_record_path(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for record in iter_records(path):
        if _is_scored_detail(record):
            details.append(record)
        else:
            raw_records.append(record)
    return raw_records, details


def _load_rollout_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        raw, _details = _load_record_path(path)
        records.extend(raw)
    return records


def _load_detail_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for path in paths:
        raw, scored = _load_record_path(path)
        details.extend(scored)
        details.extend(record for record in raw if _is_scored_detail(record))
    return _normalize_details(details)


def _read_pickle_dict(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)  # noqa: S301 - local trusted Uni-Agent artifact.
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError):
        return {}
    return value if isinstance(value, dict) else {}


def _record_from_uni_agent_run(run_dir: Path) -> dict[str, Any] | None:
    result_path = run_dir / "interaction_result.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict):
        return None

    rollout_cache = _read_pickle_dict(run_dir / "rollout_cache.pkl")
    extra_fields = rollout_cache.get("extra_fields") if isinstance(rollout_cache.get("extra_fields"), dict) else {}
    record = {
        "schema_version": "p2a_uni_agent_run_v1",
        "run_id": run_dir.name,
        "instance_id": extra_fields.get("instance_id") or result.get("instance_id") or run_dir.name,
        "data_source": extra_fields.get("data_source") or result.get("data_source") or "local-uni-agent",
        "messages": result.get("messages") or [],
        "trajectory": result.get("trajectory") or [],
        "p2a_step_traces": extra_fields.get("p2a_step_traces") or result.get("p2a_step_traces") or [],
        "response_text": extra_fields.get("response_text") or result.get("response_text") or "",
        "reward": result.get("reward_score"),
        "resolved": result.get("resolved"),
        "metrics": result.get("metrics") or rollout_cache.get("metrics") or {},
        "token_usage": rollout_cache.get("token_usage") or {},
        "execution_time": result.get("execution_time"),
        "extra_fields": dict(extra_fields),
    }
    return record


def _load_uni_agent_records(log_dir: Path | None) -> list[dict[str, Any]]:
    if log_dir is None or not log_dir.exists():
        return []
    records = []
    for child in sorted(log_dir.iterdir()):
        if child.is_dir():
            record = _record_from_uni_agent_run(child)
            if record is not None:
                records.append(record)
    return records


def _db_where(
    *,
    experiment_id: str | None,
    provider_source: str | None,
    dataset: str | None,
) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    if experiment_id:
        where.append("c.experiment_id = ?")
        params.append(experiment_id)
    if provider_source:
        where.append("c.provider_source = ?")
        params.append(provider_source)
    if dataset:
        where.append("c.dataset = ?")
        params.append(dataset)
    return ("WHERE " + " AND ".join(where) if where else ""), params


def _load_db_records(conn: sqlite3.Connection, request: DashboardRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    where_sql, params = _db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
    )
    rows = conn.execute(
        f"""
        SELECT
          c.experiment_id,
          c.provider_source,
          c.dataset,
          c.model_api_name,
          c.model_label,
          c.status,
          c.error,
          c.artifact_rollouts,
          c.artifact_details,
          c.run_id,
          r.rollout_json,
          q.metrics_json
        FROM run_cells c
        LEFT JOIN raw_rollouts r ON r.cell_id = c.id
        LEFT JOIN quantitative_metrics q ON q.cell_id = c.id
        {where_sql}
        ORDER BY c.model_label, c.instance_id
        """,
        params,
    ).fetchall()

    records: list[dict[str, Any]] = []
    stored_details: list[dict[str, Any]] = []
    for row in rows:
        record = _safe_json_loads(row["rollout_json"], {})
        if isinstance(record, dict) and record:
            record.setdefault("experiment_id", row["experiment_id"])
            record.setdefault("provider_source", row["provider_source"])
            record.setdefault("data_source", row["dataset"])
            record.setdefault("model", row["model_label"])
            record.setdefault("model_label", row["model_label"])
            record.setdefault("model_api_name", row["model_api_name"])
            record.setdefault("run_id", row["run_id"])
            record.setdefault("artifact_rollouts", row["artifact_rollouts"])
            records.append(record)

        metrics = _safe_json_loads(row["metrics_json"], {})
        detail = metrics.get("detail") if isinstance(metrics, dict) else None
        if isinstance(detail, dict) and detail:
            if isinstance(record, dict) and record:
                stored_details.append(_enrich_detail_from_record(detail, record, request))
            else:
                fallback_record = {
                    "experiment_id": row["experiment_id"],
                    "provider_source": row["provider_source"],
                    "dataset": row["dataset"],
                    "data_source": row["dataset"],
                    "model_label": row["model_label"],
                    "model_api_name": row["model_api_name"],
                    "run_id": row["run_id"],
                    "artifact_rollouts": row["artifact_rollouts"],
                }
                stored_details.append(_enrich_detail_from_record(detail, fallback_record, request))
    return records, stored_details


def _tail(path: Path, max_chars: int = 60_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_chars))
            payload = handle.read()
    except OSError:
        return ""
    return payload.decode("utf-8", errors="replace")


def _infer_status(file_names: set[str], log_excerpt: str) -> str:
    if "interaction_result.json" in file_names:
        return "completed"
    if "run.log" not in file_names:
        return "queued"
    if any(marker in log_excerpt for marker in VERIFY_MARKERS):
        return "verify"
    if any(marker in log_excerpt for marker in RUNNING_MARKERS):
        return "running"
    return "queued"


def _run_snapshot(run_dir: Path) -> dict[str, Any]:
    file_names = []
    latest = run_dir.stat().st_mtime
    for child in sorted(run_dir.rglob("*")):
        if child.is_file():
            file_names.append(child.relative_to(run_dir).as_posix())
            try:
                latest = max(latest, child.stat().st_mtime)
            except OSError:
                pass
    run_log = run_dir / "run.log"
    log_excerpt = _tail(run_log) if run_log.exists() else ""
    file_set = set(file_names)
    return {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "status": _infer_status(file_set, log_excerpt),
        "last_update": latest,
        "files": file_names,
        "log_sources": [
            {"key": rel, "label": "Run log" if rel == "run.log" else Path(rel).name}
            for rel in file_names
            if rel == "run.log" or rel.endswith((".log", ".txt"))
        ],
        "log_excerpt": log_excerpt,
    }


def _scan_log_dir(log_dir: Path | None) -> list[dict[str, Any]]:
    if log_dir is None or not log_dir.exists():
        return []
    return [_run_snapshot(child) for child in sorted(log_dir.iterdir()) if child.is_dir()]


def _scan_db_artifact_runs(conn: sqlite3.Connection, request: DashboardRequest) -> list[dict[str, Any]]:
    where_sql, params = _db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
    )
    rows = conn.execute(
        f"""
        SELECT DISTINCT
          experiment_id,
          provider_source,
          dataset,
          model_api_name,
          model_label,
          artifact_rollouts
        FROM run_cells c
        {where_sql}
        """,
        params,
    ).fetchall()
    run_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["artifact_rollouts"]:
            continue
        path = Path(str(row["artifact_rollouts"])).expanduser()
        if path.name == "rollouts.jsonl":
            path = path.parent
        metadata = {
            "source_kind": _source_kind(
                provider_source=row["provider_source"],
                schema_version=None,
                run_step=None,
            ),
            "experiment_id": row["experiment_id"],
            "provider_source": row["provider_source"],
            "dataset": row["dataset"],
            "model_api_name": row["model_api_name"],
            "model_label": row["model_label"],
        }
        metadata["experiment_key"] = _experiment_key(metadata)
        metadata["eval_cell_key"] = metadata["experiment_key"]
        item = run_meta.setdefault(
            str(path),
            {
                "path": path,
                "eval_cell_keys": set(),
                "datasets": set(),
                "model_labels": set(),
                "experiment_ids": set(),
            },
        )
        item["eval_cell_keys"].add(metadata["eval_cell_key"])
        item["datasets"].add(str(row["dataset"]))
        item["model_labels"].add(str(row["model_label"]))
        item["experiment_ids"].add(str(row["experiment_id"]))
    runs = []
    for item in sorted(run_meta.values(), key=lambda value: str(value["path"])):
        path = item["path"]
        if not path.exists():
            continue
        run = _run_snapshot(path)
        run["eval_cell_keys"] = sorted(item["eval_cell_keys"])
        run["datasets"] = sorted(item["datasets"])
        run["model_labels"] = sorted(item["model_labels"])
        run["experiment_ids"] = sorted(item["experiment_ids"])
        runs.append(run)
    return runs


def _read_log_from_runs(runs: Iterable[dict[str, Any]], run_id: str, source: str) -> dict[str, Any]:
    for run in runs:
        if run.get("run_id") != run_id:
            continue
        path = Path(str(run.get("path") or "")) / source
        text = _tail(path, max_chars=120_000)
        return {
            "run_id": run_id,
            "source": source,
            "text": text,
            "file_size": path.stat().st_size if path.exists() else 0,
        }
    raise FileNotFoundError(f"Run {run_id!r} log source {source!r} not found")


def _summary_source(request: DashboardRequest) -> Path:
    if request.db_path:
        return request.db_path
    if request.rollouts:
        return request.rollouts[0] if len(request.rollouts) == 1 else Path("multiple_rollout_sources")
    if request.details:
        return request.details[0] if len(request.details) == 1 else Path("multiple_detail_sources")
    if request.log_dir:
        return request.log_dir
    return Path("empty")


def _source_list(request: DashboardRequest) -> list[dict[str, str]]:
    sources = []
    for path in request.rollouts:
        sources.append({"kind": "rollouts", "path": str(path)})
    for path in request.details:
        sources.append({"kind": "details", "path": str(path)})
    if request.db_path:
        sources.append({"kind": "db", "path": str(request.db_path)})
    if request.log_dir:
        sources.append({"kind": "log_dir", "path": str(request.log_dir)})
    if request.bonus_map_dir:
        sources.append({"kind": "bonus_map_dir", "path": str(request.bonus_map_dir)})
    return sources


def _avg(values: Iterable[Any]) -> float | None:
    real = [float(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    return sum(real) / len(real) if real else None


def _rate(values: Iterable[Any]) -> float | None:
    real = [1 if bool(value) else 0 for value in values if value is not None]
    return sum(real) / len(real) if real else None


def _sum_int(items: Iterable[dict[str, Any]], key: str) -> int:
    total = 0
    for item in items:
        value = item.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            total += int(value)
    return total


def _distribution(items: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        value = item.get(key)
        if value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _instance_key(detail: dict[str, Any]) -> str:
    return str(detail.get("instance_id") or f"record-{detail.get('record_index', 0)}")


def _unique_dataset_details(details: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for detail in details:
        dataset = _dataset_name(detail)
        instance = _instance_key(detail)
        current = by_dataset[dataset].get(instance)
        if current is None:
            by_dataset[dataset][instance] = detail
            continue
        current_score = int(bool(current.get("has_bonus_map"))) + int(bool(current.get("chain_evaluable")))
        new_score = int(bool(detail.get("has_bonus_map"))) + int(bool(detail.get("chain_evaluable")))
        if new_score > current_score:
            by_dataset[dataset][instance] = detail
    return {dataset: list(items.values()) for dataset, items in sorted(by_dataset.items())}


def _dataset_distributions(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset, items in _unique_dataset_details(details).items():
        case_types: dict[str, int] = defaultdict(int)
        not_chain: dict[str, int] = defaultdict(int)
        availability = {
            "with_bonus_map": 0,
            "with_call_graph": 0,
            "chain_evaluable": 0,
            "not_chain_evaluable": 0,
        }
        for item in items:
            case_types[str(item.get("bonus_case_type") or "missing_bonus_map")] += 1
            if item.get("has_bonus_map"):
                availability["with_bonus_map"] += 1
            if item.get("has_call_graph"):
                availability["with_call_graph"] += 1
            if item.get("chain_evaluable"):
                availability["chain_evaluable"] += 1
            else:
                availability["not_chain_evaluable"] += 1
                not_chain[str(item.get("not_chain_evaluable_reason") or "unknown")] += 1
        out[dataset] = {
            "dataset": dataset,
            "n_instances": len(items),
            "distributions": {
                "case_types": dict(sorted(case_types.items())),
                "not_chain_evaluable_reasons": dict(sorted(not_chain.items())),
                "availability": availability,
            },
        }
    return out


def _chain_bad_distribution(details: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for detail in details:
        patterns = detail.get("chain_bad_patterns")
        if not isinstance(patterns, dict):
            continue
        for key in CHAIN_BAD_PATTERN_KEYS:
            if patterns.get(key):
                counts[key] += 1
    return dict(sorted(counts.items()))


def _detail_model_metrics(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        key = (
            str(detail.get("experiment_key") or _experiment_key(detail)),
            str(detail.get("source_kind") or "offline_artifact"),
            str(detail.get("experiment_id") or "adhoc"),
            str(detail.get("provider_source") or "unknown-provider"),
            str(detail.get("dataset") or detail.get("data_source") or "unknown-dataset"),
            str(detail.get("model_label") or detail.get("model_api_name") or "unknown-model"),
        )
        groups[key].append(detail)

    rows = []
    for (experiment_key, source_kind, experiment_id, provider_source, dataset, model_label), items in sorted(groups.items()):
        chain_items = [item for item in items if item.get("chain_evaluable") is True]
        order_items = [item for item in items if item.get("order_defined") is True]
        block_order_items = [item for item in items if item.get("block_order_defined") is True]
        scored_blocks = _sum_int(items, "n_scored_read_blocks")
        total_blocks = _sum_int(items, "n_blocks")
        scored_block_steps = _sum_int(items, "n_scored_read_block_steps")
        block_steps = _sum_int(items, "n_block_steps")
        row = {
            "experiment_key": experiment_key,
            "eval_cell_key": experiment_key,
            "source_kind": source_kind,
            "experiment_id": experiment_id,
            "provider_source": provider_source,
            "dataset": dataset,
            "model_api_name": str(items[0].get("model_api_name") or model_label),
            "model_label": model_label,
            "target": len(items),
            "done": len(items),
            "errors": sum(1 for item in items if item.get("error") or item.get("system_error")),
            "pending": 0,
            "resolved_rate": _rate(item.get("resolved") for item in items),
            "reward_rate": _avg(item.get("reward") for item in items),
            "p2a_read_rate": _rate((item.get("n_reads") or 0) > 0 for item in items),
            "call_graph_hit_rate": _rate(item.get("hit_call_graph") for item in items),
            "ground_truth_hit_rate": _rate(item.get("hit_ground_truth") for item in items),
            "near_hit_rate": _rate(item.get("hit_near") for item in items),
            "avg_min_distance": _avg(item.get("min_distance") for item in items),
            "avg_read_precision": _avg(item.get("hit_precision") for item in items),
            "avg_node_recall": _avg(item.get("hit_recall") for item in items),
            "avg_hit_f1": _avg(item.get("hit_f1") for item in items),
            "chain_graph_coverage": _rate(item.get("chain_graph_covered") for item in items),
            "chain_hit_rate": _rate(item.get("chain_hit") for item in chain_items),
            "anchor_hit_rate": _rate(item.get("anchor_hit") for item in chain_items),
            "root_hit_rate": _rate(item.get("root_hit") for item in chain_items),
            "avg_chain_node_recall": _avg(item.get("chain_node_recall") for item in chain_items),
            "avg_chain_read_precision": _avg(item.get("chain_read_precision") for item in chain_items),
            "avg_first_anchor_step": _avg(item.get("first_anchor_step") for item in chain_items),
            "avg_first_root_step": _avg(item.get("first_root_step") for item in chain_items),
            "avg_steps_anchor_to_root": _avg(item.get("steps_anchor_to_root") for item in chain_items),
            "anchor_before_root_rate": _rate(item.get("anchor_before_root") for item in chain_items),
            "avg_order_score": _avg(item.get("order_score") for item in order_items),
            "reverse_order_rate": _rate(
                (item.get("order_score") < 0)
                for item in order_items
                if isinstance(item.get("order_score"), int | float)
            ),
            "miracle_rate": _rate(
                item.get("miracle_step")
                for item in items
                if item.get("hit_ground_truth") and item.get("miracle_step") is not None
            ),
            "avg_miracle_severity": _avg(item.get("miracle_severity") for item in items),
            "avg_block_order_score": _avg(item.get("block_order_score") for item in block_order_items),
            "block_reverse_order_rate": _rate(
                (item.get("block_order_score") < 0)
                for item in block_order_items
                if isinstance(item.get("block_order_score"), int | float)
            ),
            "block_miracle_rate": _rate(
                item.get("block_miracle_step")
                for item in items
                if item.get("hit_ground_truth") and item.get("block_miracle_step") is not None
            ),
            "avg_block_efficiency": _avg(item.get("block_efficiency") for item in items),
            "avg_blocks_per_trace": (total_blocks / len(items)) if items else None,
            "block_achieve_rate": (_sum_int(items, "n_achieving_blocks") / scored_blocks) if scored_blocks else None,
            "block_waste_rate": (_sum_int(items, "n_wasted_blocks") / scored_blocks) if scored_blocks else None,
            "block_loop_rate": (_sum_int(items, "n_loop_blocks") / total_blocks) if total_blocks else None,
            "achieving_block_step_share": (_sum_int(items, "n_achieving_block_steps") / scored_block_steps)
            if scored_block_steps
            else None,
            "wasted_block_step_share": (_sum_int(items, "n_wasted_block_steps") / scored_block_steps)
            if scored_block_steps
            else None,
            "loop_block_step_share": (_sum_int(items, "n_loop_block_steps") / block_steps) if block_steps else None,
            "loop_trace_rate": _rate((item.get("bad_patterns") or {}).get("has_loop") for item in items),
            "error_spiral_rate": _rate((item.get("bad_patterns") or {}).get("error_spiral") for item in items),
            "not_chain_evaluable_reasons": _distribution(
                [item for item in items if not item.get("chain_evaluable")],
                "not_chain_evaluable_reason",
            ),
            "chain_bad_patterns": _chain_bad_distribution(items),
        }
        rows.append(row)
    return rows


def _normalize_model_row(row: dict[str, Any]) -> dict[str, Any]:
    source_kind = _source_kind(
        provider_source=row.get("provider_source"),
        schema_version=None,
        run_step=row.get("run_step"),
    )
    normalized = {**row, "source_kind": row.get("source_kind") or source_kind}
    normalized["experiment_key"] = row.get("experiment_key") or _experiment_key(normalized)
    normalized["eval_cell_key"] = row.get("eval_cell_key") or normalized["experiment_key"]
    return normalized


def _merge_model_metrics(base_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {_normalize_model_row(row)["eval_cell_key"]: _normalize_model_row(row) for row in base_rows}
    for row in detail_rows:
        current = merged.get(row["eval_cell_key"], {})
        merged[row["eval_cell_key"]] = {**row, **current, **{key: value for key, value in row.items() if value is not None}}
        for key in ("target", "done", "errors", "pending"):
            if current.get(key) is not None:
                merged[row["eval_cell_key"]][key] = current[key]
    return sorted(merged.values(), key=lambda item: (str(item.get("experiment_id")), str(item.get("model_label"))))


def _eval_cell_registry(model_metrics: list[dict[str, Any]], details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail_counts: dict[str, int] = defaultdict(int)
    trace_counts: dict[str, int] = defaultdict(int)
    for detail in details:
        key = _eval_cell_key(detail)
        detail_counts[key] += 1
        if detail.get("raw_available") or detail.get("step_details"):
            trace_counts[key] += 1

    eval_cells = []
    for row in model_metrics:
        key = _eval_cell_key(row)
        eval_cells.append(
            {
                "eval_cell_key": key,
                "experiment_key": key,
                "source_kind": row.get("source_kind"),
                "experiment_id": row.get("experiment_id"),
                "provider_source": row.get("provider_source"),
                "dataset": row.get("dataset"),
                "model_api_name": row.get("model_api_name"),
                "model_label": row.get("model_label"),
                "target": row.get("target"),
                "done": row.get("done"),
                "errors": row.get("errors"),
                "pending": row.get("pending"),
                "detail_count": detail_counts.get(key, 0),
                "trajectory_count": trace_counts.get(key, 0),
                "resolved_rate": row.get("resolved_rate"),
                "root_hit_rate": row.get("root_hit_rate") or row.get("ground_truth_hit_rate"),
                "chain_node_recall": row.get("avg_chain_node_recall") or row.get("avg_node_recall"),
                "read_precision": row.get("avg_chain_read_precision") or row.get("avg_read_precision"),
            }
        )
    return eval_cells


def _dataset_registry(
    details: list[dict[str, Any]],
    eval_cells: list[dict[str, Any]],
    distributions_by_dataset: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    trajectories: dict[str, int] = defaultdict(int)
    cells: dict[str, set[str]] = defaultdict(set)
    models: dict[str, set[str]] = defaultdict(set)
    source_kinds: dict[str, set[str]] = defaultdict(set)
    for detail in details:
        dataset = _dataset_name(detail)
        trajectories[dataset] += 1
        cells[dataset].add(_eval_cell_key(detail))
        if detail.get("model_label"):
            models[dataset].add(str(detail.get("model_label")))
        if detail.get("source_kind"):
            source_kinds[dataset].add(str(detail.get("source_kind")))
    for cell in eval_cells:
        dataset = _dataset_name(cell)
        cells[dataset].add(_eval_cell_key(cell))
        if cell.get("model_label"):
            models[dataset].add(str(cell.get("model_label")))
        if cell.get("source_kind"):
            source_kinds[dataset].add(str(cell.get("source_kind")))
    dataset_names = sorted(set(distributions_by_dataset) | set(cells) | set(trajectories))
    return [
        {
            "dataset": dataset,
            "n_instances": (distributions_by_dataset.get(dataset) or {}).get("n_instances", 0),
            "n_eval_cells": len(cells.get(dataset, set())),
            "n_trajectories": trajectories.get(dataset, 0),
            "models": sorted(models.get(dataset, set())),
            "source_kinds": sorted(source_kinds.get(dataset, set())),
        }
        for dataset in dataset_names
    ]


def build_dashboard_snapshot(request: DashboardRequest) -> dict[str, Any]:
    raw_records = _load_rollout_paths(request.rollouts)
    details = _load_detail_paths(request.details)
    runs = _scan_log_dir(request.log_dir)
    raw_records.extend(_load_uni_agent_records(request.log_dir))

    base_model_metrics: list[dict[str, Any]] = []
    if request.db_path:
        with ensure_db(request.db_path) as conn:
            base_model_metrics = aggregate_model_metrics(
                conn,
                experiment_id=request.experiment_id,
                provider_source=request.provider_source,
                dataset=request.dataset,
            )
            db_records, db_details = _load_db_records(conn, request)
            raw_records.extend(db_records)
            if request.bonus_map_dir is None:
                details.extend(db_details)
            runs.extend(_scan_db_artifact_runs(conn, request))

    if request.bonus_map_dir is not None and raw_records:
        details.extend(_score_records(raw_records, request=request, start_index=len(details)))

    if details:
        details = _normalize_details(details)
        summary = _finalize_summary(summarize(
            details,
            source=_summary_source(request),
            bonus_map_dir=request.bonus_map_dir or Path("."),
            tracking_mode=request.tracking_mode,
            near_threshold=request.near_threshold,
            m_max=request.m_max,
        ))
        summary["trends"] = summarize_trends(
            details,
            tracking_mode=request.tracking_mode,
            near_threshold=request.near_threshold,
            m_max=request.m_max,
        )
    else:
        summary = _empty_summary(request)
        summary["trends"] = []

    detail_model_metrics = _detail_model_metrics(details)
    model_metrics = _merge_model_metrics(base_model_metrics, detail_model_metrics)
    eval_cells = _eval_cell_registry(model_metrics, details)
    distributions_by_dataset = _dataset_distributions(details)
    datasets = _dataset_registry(details, eval_cells, distributions_by_dataset)
    summary["by_dataset"] = {
        dataset: {
            "counts": {"n_instances": payload["n_instances"]},
            "distributions": payload["distributions"],
        }
        for dataset, payload in distributions_by_dataset.items()
    }
    summary["distributions_by_dataset"] = distributions_by_dataset

    deduped_runs = {str(run.get("path") or run.get("run_id")): run for run in runs}
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at": time.time(),
        "filters": {
            "experiment_id": request.experiment_id,
            "provider_source": request.provider_source,
            "dataset": request.dataset,
        },
        "sources": _source_list(request),
        "datasets": datasets,
        "eval_cells": eval_cells,
        "experiments": eval_cells,
        "summary": summary,
        "model_metrics": model_metrics,
        "runs": sorted(deduped_runs.values(), key=lambda run: (str(run.get("status")), str(run.get("run_id")))),
        "details": details[: request.detail_limit],
        "detail_count": len(details),
        "raw_record_count": len(raw_records),
    }


def read_dashboard_log(request: DashboardRequest, run_id: str, source: str = "run.log") -> dict[str, Any]:
    runs = _scan_log_dir(request.log_dir)
    if request.db_path:
        with ensure_db(request.db_path) as conn:
            runs.extend(_scan_db_artifact_runs(conn, request))
    return _read_log_from_runs(runs, run_id=run_id, source=source)


def snapshot_to_json(snapshot: dict[str, Any], *, indent: int | None = None) -> str:
    return json.dumps(snapshot, ensure_ascii=False, default=_json_default, indent=indent)
