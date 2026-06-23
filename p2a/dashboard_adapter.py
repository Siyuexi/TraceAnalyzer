"""Normalized data adapter for the unified P2A HTML dashboard."""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from p2a.core import BonusMapStore
from p2a.eval_cache import aggregate_model_metrics, ensure_db, json_loads
from p2a.eval_fault_localization import (
    _json_default,
    iter_records,
    score_record,
    summarize,
    summarize_trends,
)


DASHBOARD_SCHEMA_VERSION = "p2a_unified_dashboard_v1"
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
    return [_normalize_detail(detail, index=index) for index, detail in enumerate(details)]


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
    return [
        score_record(
            record,
            index=start_index + index,
            bonus_maps=bonus_maps,
            tracking_mode=request.tracking_mode,
            near_threshold=request.near_threshold,
            m_max=request.m_max,
        )
        for index, record in enumerate(records)
    ]


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
            records.append(record)

        metrics = _safe_json_loads(row["metrics_json"], {})
        detail = metrics.get("detail") if isinstance(metrics, dict) else None
        if isinstance(detail, dict) and detail:
            detail.setdefault("experiment_id", row["experiment_id"])
            detail.setdefault("provider_source", row["provider_source"])
            detail.setdefault("model_label", row["model_label"])
            detail.setdefault("data_source", row["dataset"])
            stored_details.append(detail)
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
        SELECT DISTINCT artifact_rollouts
        FROM run_cells c
        {where_sql}
        """,
        params,
    ).fetchall()
    run_dirs: dict[str, Path] = {}
    for row in rows:
        if not row["artifact_rollouts"]:
            continue
        path = Path(str(row["artifact_rollouts"])).expanduser()
        if path.name == "rollouts.jsonl":
            run_dirs[str(path.parent)] = path.parent
    return [_run_snapshot(path) for path in sorted(run_dirs.values()) if path.exists()]


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


def build_dashboard_snapshot(request: DashboardRequest) -> dict[str, Any]:
    raw_records = _load_rollout_paths(request.rollouts)
    details = _load_detail_paths(request.details)
    runs = _scan_log_dir(request.log_dir)
    raw_records.extend(_load_uni_agent_records(request.log_dir))

    model_metrics: list[dict[str, Any]] = []
    if request.db_path:
        with ensure_db(request.db_path) as conn:
            model_metrics = aggregate_model_metrics(
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
