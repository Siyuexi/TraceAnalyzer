"""SQLite cache for API/local evaluation rollouts and trace-quality metrics."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from p2a.eval_fault_localization import _json_default, iter_records


SCHEMA_VERSION = 1
DONE_STATUS = "done"
ERROR_STATUS = "error"
PENDING_STATUS = "pending"
RUNNING_STATUS = "running"
CHAIN_BAD_PATTERN_KEYS = (
    "missed_anchor",
    "missed_root_after_anchor",
    "root_before_anchor",
    "chain_stall",
    "chain_read_loop",
    "off_chain_read_spree",
    "error_spiral_on_chain",
)
DYNAMIC_TRACEABLE_CASE_TYPES = {"direct", "standard"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _nested_mappings(record: dict[str, Any]) -> list[dict[str, Any]]:
    extra = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    tools = extra.get("tools_kwargs") if isinstance(extra.get("tools_kwargs"), dict) else {}
    reward = tools.get("reward") if isinstance(tools.get("reward"), dict) else {}
    metadata = reward.get("metadata") if isinstance(reward.get("metadata"), dict) else {}
    return [record, extra, tools, reward, metadata]


def _first_text_field(record: dict[str, Any], fields: Iterable[str]) -> str | None:
    for mapping in _nested_mappings(record):
        for field in fields:
            value = mapping.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _issue_description(record: dict[str, Any]) -> str | None:
    return _first_text_field(record, ("problem_statement", "issue_description", "issue_text", "issue", "description", "problem", "title"))


def _golden_patch(record: dict[str, Any]) -> str | None:
    return _first_text_field(record, ("golden_patch", "patch", "base_patch", "fix_patch"))


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def connect_readonly(db_path: Path | str, *, timeout: float = 2.0) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)
    uri_path = quote(str(path.resolve()), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA query_only = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS experiments (
          experiment_id TEXT NOT NULL,
          provider_source TEXT NOT NULL,
          dataset TEXT NOT NULL,
          config_snapshot TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (experiment_id, provider_source, dataset)
        );

        CREATE TABLE IF NOT EXISTS run_cells (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          experiment_id TEXT NOT NULL,
          provider_source TEXT NOT NULL,
          model_api_name TEXT NOT NULL,
          model_label TEXT NOT NULL,
          dataset TEXT NOT NULL,
          instance_id TEXT NOT NULL,
          status TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          run_id TEXT,
          artifact_rollouts TEXT,
          artifact_details TEXT,
          started_at TEXT,
          ended_at TEXT,
          error TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE (experiment_id, provider_source, model_api_name, dataset, instance_id)
        );

        CREATE TABLE IF NOT EXISTS raw_rollouts (
          run_id TEXT PRIMARY KEY,
          cell_id INTEGER NOT NULL UNIQUE REFERENCES run_cells(id) ON DELETE CASCADE,
          messages_json TEXT,
          trajectory_json TEXT,
          p2a_step_traces_json TEXT,
          final_response TEXT,
          reward_json TEXT,
          resolved INTEGER,
          token_usage_json TEXT,
          cache_metrics_json TEXT,
          issue_description TEXT,
          golden_patch TEXT,
          rollout_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quantitative_metrics (
          cell_id INTEGER PRIMARY KEY REFERENCES run_cells(id) ON DELETE CASCADE,
          reward REAL,
          resolved INTEGER,
          p2a_read INTEGER,
          call_graph_hit INTEGER,
          ground_truth_hit INTEGER,
          near_hit INTEGER,
          min_distance REAL,
          turns INTEGER,
          tool_calls INTEGER,
          wall_time REAL,
          input_tokens REAL,
          output_tokens REAL,
          reasoning_tokens REAL,
          cache_hit_tokens REAL,
          cache_write_tokens REAL,
          cost REAL,
          metrics_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_run_cells_exp_model
          ON run_cells (experiment_id, provider_source, dataset, model_label);
        CREATE INDEX IF NOT EXISTS idx_run_cells_status
          ON run_cells (experiment_id, provider_source, dataset, status);
        """
    )
    _ensure_column(conn, "raw_rollouts", "issue_description", "TEXT")
    _ensure_column(conn, "raw_rollouts", "golden_patch", "TEXT")
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, utc_now()),
    )
    conn.commit()


def ensure_db(db_path: Path | str) -> sqlite3.Connection:
    conn = connect(db_path)
    init_db(conn)
    return conn


def upsert_experiment(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    dataset: str,
    config_snapshot: dict[str, Any] | str,
) -> None:
    now = utc_now()
    snapshot = config_snapshot if isinstance(config_snapshot, str) else json_dumps(config_snapshot)
    conn.execute(
        """
        INSERT INTO experiments(experiment_id, provider_source, dataset, config_snapshot, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, provider_source, dataset) DO UPDATE SET
          config_snapshot = excluded.config_snapshot,
          updated_at = excluded.updated_at
        """,
        (experiment_id, provider_source, dataset, snapshot, now, now),
    )


def upsert_planned_cells(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    instance_ids: Iterable[str],
) -> None:
    now = utc_now()
    rows = [
        (
            experiment_id,
            provider_source,
            model_api_name,
            model_label,
            dataset,
            instance_id,
            PENDING_STATUS,
            now,
            now,
        )
        for instance_id in instance_ids
    ]
    conn.executemany(
        """
        INSERT INTO run_cells(
          experiment_id, provider_source, model_api_name, model_label, dataset,
          instance_id, status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, provider_source, model_api_name, dataset, instance_id) DO UPDATE SET
          model_label = excluded.model_label,
          updated_at = excluded.updated_at
        """,
        rows,
    )


def mark_cells_running(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    dataset: str,
    instance_ids: Iterable[str],
) -> None:
    now = utc_now()
    conn.executemany(
        """
        UPDATE run_cells
        SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
        WHERE experiment_id = ?
          AND provider_source = ?
          AND model_api_name = ?
          AND dataset = ?
          AND instance_id = ?
          AND status != ?
        """,
        [
            (
                RUNNING_STATUS,
                now,
                now,
                experiment_id,
                provider_source,
                model_api_name,
                dataset,
                instance_id,
                DONE_STATUS,
            )
            for instance_id in instance_ids
        ],
    )


def completed_instance_ids(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    dataset: str,
) -> set[str]:
    rows = conn.execute(
        """
        SELECT instance_id
        FROM run_cells
        WHERE experiment_id = ?
          AND provider_source = ?
          AND model_api_name = ?
          AND dataset = ?
          AND status = ?
        """,
        (experiment_id, provider_source, model_api_name, dataset, DONE_STATUS),
    ).fetchall()
    return {str(row["instance_id"]) for row in rows}


def _bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _tool_call_count(record: dict[str, Any]) -> int:
    total = 0
    for trace in record.get("p2a_step_traces") or []:
        if isinstance(trace, dict):
            calls = trace.get("tool_calls") or []
            if isinstance(calls, list):
                total += len(calls)
    return total


def _reward_number(record: dict[str, Any]) -> float | None:
    value = record.get("reward")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _number(value)


def _cell_id(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    instance_id: str,
) -> int:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO run_cells(
          experiment_id, provider_source, model_api_name, model_label, dataset,
          instance_id, status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, provider_source, model_api_name, dataset, instance_id) DO UPDATE SET
          model_label = excluded.model_label,
          updated_at = excluded.updated_at
        """,
        (
            experiment_id,
            provider_source,
            model_api_name,
            model_label,
            dataset,
            instance_id,
            PENDING_STATUS,
            now,
            now,
        ),
    )
    row = conn.execute(
        """
        SELECT id FROM run_cells
        WHERE experiment_id = ?
          AND provider_source = ?
          AND model_api_name = ?
          AND dataset = ?
          AND instance_id = ?
        """,
        (experiment_id, provider_source, model_api_name, dataset, instance_id),
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to create or load run cell")
    return int(row["id"])


def _unique_raw_run_id(conn: sqlite3.Connection, requested_run_id: str, cell_id: int) -> str:
    candidate = requested_run_id
    suffix = 1
    while True:
        row = conn.execute(
            "SELECT cell_id FROM raw_rollouts WHERE run_id = ? AND cell_id != ?",
            (candidate, cell_id),
        ).fetchone()
        if row is None:
            return candidate
        suffix += 1
        candidate = f"{requested_run_id}:{cell_id}:{suffix}"


def upsert_rollout_record(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    record: dict[str, Any],
    detail: dict[str, Any] | None = None,
    artifact_rollouts: Path | str | None = None,
    artifact_details: Path | str | None = None,
) -> None:
    instance_id = str(record.get("instance_id") or (detail or {}).get("instance_id") or "")
    if not instance_id:
        raise ValueError("rollout record has no instance_id; cannot key DB cell")

    now = utc_now()
    requested_run_id = str(record.get("run_id") or f"{experiment_id}:{provider_source}:{model_api_name}:{dataset}:{instance_id}")
    cell_id = _cell_id(
        conn,
        experiment_id=experiment_id,
        provider_source=provider_source,
        model_api_name=model_api_name,
        model_label=model_label,
        dataset=dataset,
        instance_id=instance_id,
    )
    run_id = _unique_raw_run_id(conn, requested_run_id, cell_id)
    status = ERROR_STATUS if record.get("error") else DONE_STATUS
    conn.execute(
        """
        UPDATE run_cells
        SET status = ?,
            attempts = attempts + 1,
            run_id = ?,
            artifact_rollouts = ?,
            artifact_details = ?,
            ended_at = ?,
            error = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            status,
            run_id,
            str(artifact_rollouts) if artifact_rollouts else None,
            str(artifact_details) if artifact_details else None,
            now,
            str(record.get("error")) if record.get("error") else None,
            now,
            cell_id,
        ),
    )

    token_usage = record.get("token_usage") if isinstance(record.get("token_usage"), dict) else {}
    cache_metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    issue_description = _issue_description(record)
    golden_patch = _golden_patch(record)
    conn.execute("DELETE FROM raw_rollouts WHERE cell_id = ?", (cell_id,))
    conn.execute(
        """
        INSERT INTO raw_rollouts(
          run_id, cell_id, messages_json, trajectory_json, p2a_step_traces_json,
          final_response, reward_json, resolved, token_usage_json, cache_metrics_json,
          issue_description, golden_patch, rollout_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            cell_id,
            json_dumps(record.get("messages") or []),
            json_dumps(record.get("trajectory") or []),
            json_dumps(record.get("p2a_step_traces") or []),
            record.get("response_text") or "",
            json_dumps(record.get("reward")),
            _bool_int(record.get("resolved")),
            json_dumps(token_usage),
            json_dumps(cache_metrics),
            issue_description,
            golden_patch,
            json_dumps(record),
            now,
        ),
    )

    turns = len(record.get("p2a_step_traces") or record.get("trajectory") or [])
    metrics = {
        "token_usage": token_usage,
        "cache_metrics": cache_metrics,
    }
    conn.execute(
        """
        INSERT INTO quantitative_metrics(
          cell_id, reward, resolved, p2a_read, call_graph_hit, ground_truth_hit,
          near_hit, min_distance, turns, tool_calls, wall_time, input_tokens,
          output_tokens, reasoning_tokens, cache_hit_tokens, cache_write_tokens,
          cost, metrics_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cell_id) DO UPDATE SET
          reward = excluded.reward,
          resolved = excluded.resolved,
          p2a_read = excluded.p2a_read,
          call_graph_hit = excluded.call_graph_hit,
          ground_truth_hit = excluded.ground_truth_hit,
          near_hit = excluded.near_hit,
          min_distance = excluded.min_distance,
          turns = excluded.turns,
          tool_calls = excluded.tool_calls,
          wall_time = excluded.wall_time,
          input_tokens = excluded.input_tokens,
          output_tokens = excluded.output_tokens,
          reasoning_tokens = excluded.reasoning_tokens,
          cache_hit_tokens = excluded.cache_hit_tokens,
          cache_write_tokens = excluded.cache_write_tokens,
          cost = excluded.cost,
          metrics_json = excluded.metrics_json,
          updated_at = excluded.updated_at
        """,
        (
            cell_id,
            _reward_number(record),
            _bool_int(record.get("resolved")),
            None,
            None,
            None,
            None,
            None,
            turns,
            _tool_call_count(record),
            _number(record.get("wall_time") or record.get("execution_time")),
            _number(token_usage.get("input_tokens")),
            _number(token_usage.get("output_tokens")),
            _number(token_usage.get("reasoning_tokens")),
            _number(token_usage.get("cache_hit_tokens")),
            _number(token_usage.get("cache_write_tokens")),
            _number(token_usage.get("cost")),
            json_dumps(metrics),
            now,
        ),
    )


def ingest_artifacts(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    rollouts_path: Path,
    details_path: Path | None = None,
) -> int:
    n_records = 0
    for record in iter_records(rollouts_path):
        upsert_rollout_record(
            conn,
            experiment_id=experiment_id,
            provider_source=provider_source,
            model_api_name=model_api_name,
            model_label=model_label,
            dataset=dataset,
            record=record,
            artifact_rollouts=rollouts_path,
            artifact_details=details_path,
        )
        n_records += 1
    return n_records


def _avg(values: list[float | None]) -> float | None:
    real = [value for value in values if value is not None]
    return sum(real) / len(real) if real else None


def _rate(values: list[int | None]) -> float | None:
    real = [value for value in values if value is not None]
    return sum(real) / len(real) if real else None


def _detail_from_metric_row(row: sqlite3.Row) -> dict[str, Any]:
    metrics = json_loads(row["metrics_json"], {})
    detail = metrics.get("detail") if isinstance(metrics, dict) else None
    return detail if isinstance(detail, dict) else {}


def _detail_number(detail: dict[str, Any], key: str) -> float | None:
    return _number(detail.get(key))


def _detail_bool(detail: dict[str, Any], key: str) -> int | None:
    return _bool_int(detail.get(key))


def _avg_detail(details: list[dict[str, Any]], key: str) -> float | None:
    return _avg([_detail_number(detail, key) for detail in details])


def _rate_detail(details: list[dict[str, Any]], key: str) -> float | None:
    return _rate([_detail_bool(detail, key) for detail in details])


def _sum_detail(details: list[dict[str, Any]], key: str) -> int:
    total = 0
    for detail in details:
        value = detail.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            total += int(value)
    return total


def _chain_node_precision(detail: dict[str, Any]) -> float | None:
    projection = detail.get("chain_projection") or {}
    chain_nodes = projection.get("chain_nodes") or []
    context_nodes = projection.get("context_nodes") or []
    if not isinstance(chain_nodes, list) or not isinstance(context_nodes, list):
        return None
    hit_chain = sum(1 for node in chain_nodes if isinstance(node, dict) and node.get("hit"))
    hit_context = sum(1 for node in context_nodes if isinstance(node, dict) and node.get("hit"))
    denom = hit_chain + hit_context
    return (hit_chain / denom) if denom else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


def _chain_node_f1(detail: dict[str, Any]) -> float | None:
    return _f1(_chain_node_precision(detail), _detail_number(detail, "chain_node_recall"))


def _detail_distribution(details: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for detail in details:
        value = detail.get(key)
        if value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _detail_case_type(detail: dict[str, Any]) -> str:
    return str(detail.get("bonus_case_type") or detail.get("chain_case_kind") or "")


def _is_dynamic_traceable_detail(detail: dict[str, Any]) -> bool:
    return detail.get("chain_evaluable") is True and _detail_case_type(detail) in DYNAMIC_TRACEABLE_CASE_TYPES


def _has_dual_symptom_root(detail: dict[str, Any]) -> bool:
    projection = detail.get("chain_projection") or {}
    anchors = set(projection.get("anchors") or [])
    roots = set(projection.get("roots") or [])
    return bool(anchors & roots)


def _has_reward_path_edges(detail: dict[str, Any]) -> bool:
    projection = detail.get("chain_projection") or {}
    return bool(projection.get("chain_edges") or [])


def _is_order_metric_detail(detail: dict[str, Any]) -> bool:
    return _is_dynamic_traceable_detail(detail) and _has_reward_path_edges(detail) and not _has_dual_symptom_root(detail)


def _chain_bad_distribution(details: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for detail in details:
        patterns = detail.get("chain_bad_patterns")
        if not isinstance(patterns, dict):
            continue
        for key in CHAIN_BAD_PATTERN_KEYS:
            if patterns.get(key):
                counts[key] += 1
    return dict(sorted(counts.items()))


def aggregate_model_metrics(
    conn: sqlite3.Connection,
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
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
    where_sql = "WHERE " + " AND ".join(where) if where else ""

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
          q.reward,
          q.resolved,
          q.p2a_read,
          q.call_graph_hit,
          q.ground_truth_hit,
          q.near_hit,
          q.min_distance,
          q.turns,
          q.tool_calls,
          q.wall_time,
          q.input_tokens,
          q.output_tokens,
          q.reasoning_tokens,
          q.cache_hit_tokens,
          q.cache_write_tokens,
          q.cost,
          q.metrics_json
        FROM run_cells c
        LEFT JOIN quantitative_metrics q ON q.cell_id = c.id
        {where_sql}
        ORDER BY c.model_label, c.instance_id
        """,
        params,
    ).fetchall()

    groups: dict[tuple[str, str, str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment_id"]),
            str(row["provider_source"]),
            str(row["dataset"]),
            str(row["model_api_name"]),
            str(row["model_label"]),
        )
        groups[key].append(row)

    out = []
    for (exp_id, source, ds, api_name, label), group in sorted(groups.items(), key=lambda item: item[0][-1]):
        done = [row for row in group if row["status"] == DONE_STATUS]
        errors = [row for row in group if row["status"] == ERROR_STATUS]
        metric_rows = [row for row in done if row["turns"] is not None]
        distances = [row["min_distance"] for row in metric_rows if row["min_distance"] is not None]
        cache_hit = sum(float(row["cache_hit_tokens"] or 0) for row in metric_rows)
        cache_write = sum(float(row["cache_write_tokens"] or 0) for row in metric_rows)
        input_tokens = sum(float(row["input_tokens"] or 0) for row in metric_rows)
        details = [_detail_from_metric_row(row) for row in metric_rows]
        details = [detail for detail in details if detail]
        bonus_details = [detail for detail in details if _is_dynamic_traceable_detail(detail)]
        order_metric_details = [detail for detail in details if _is_order_metric_detail(detail)]
        order_details = [detail for detail in order_metric_details if detail.get("order_defined") is True]
        block_order_details = [detail for detail in order_metric_details if detail.get("block_order_defined") is True]
        scored_read_blocks = _sum_detail(bonus_details, "n_scored_read_blocks")
        total_blocks = _sum_detail(bonus_details, "n_blocks")
        block_steps = _sum_detail(bonus_details, "n_block_steps")
        scored_block_steps = _sum_detail(bonus_details, "n_scored_read_block_steps")
        out.append(
            {
                "experiment_id": exp_id,
                "provider_source": source,
                "dataset": ds,
                "model_api_name": api_name,
                "model_label": label,
                "target": len(group),
                "done": len(done),
                "errors": len(errors),
                "pending": sum(1 for row in group if row["status"] in {PENDING_STATUS, RUNNING_STATUS}),
                "resolved_rate": _rate([row["resolved"] for row in metric_rows]),
                "reward_rate": _avg([row["reward"] for row in metric_rows]),
                "p2a_read_rate": _rate([row["p2a_read"] for row in metric_rows]),
                "call_graph_hit_rate": _rate([row["call_graph_hit"] for row in metric_rows]),
                "ground_truth_hit_rate": _rate([row["ground_truth_hit"] for row in metric_rows]),
                "near_hit_rate": _rate([row["near_hit"] for row in metric_rows]),
                "avg_min_distance": (sum(float(v) for v in distances) / len(distances)) if distances else None,
                "avg_read_precision": _avg_detail(bonus_details, "hit_precision"),
                "avg_node_recall": _avg_detail(bonus_details, "hit_recall"),
                "avg_hit_f1": _avg_detail(bonus_details, "hit_f1"),
                "chain_graph_coverage": _rate_detail(bonus_details, "chain_graph_covered"),
                "chain_hit_rate": _rate_detail(bonus_details, "chain_hit"),
                "anchor_hit_rate": _rate_detail(bonus_details, "anchor_hit"),
                "root_hit_rate": _rate_detail(bonus_details, "root_hit"),
                "avg_chain_node_recall": _avg_detail(bonus_details, "chain_node_recall"),
                "avg_chain_node_precision": _avg([_chain_node_precision(detail) for detail in bonus_details]),
                "avg_chain_node_f1": _avg([_chain_node_f1(detail) for detail in bonus_details]),
                "avg_chain_read_precision": _avg_detail(bonus_details, "chain_read_precision"),
                "avg_first_anchor_step": _avg_detail(bonus_details, "first_anchor_step"),
                "avg_first_root_step": _avg_detail(bonus_details, "first_root_step"),
                "avg_steps_anchor_to_root": _avg_detail(bonus_details, "steps_anchor_to_root"),
                "anchor_before_root_rate": _rate_detail(bonus_details, "anchor_before_root"),
                "avg_order_score": _avg_detail(order_details, "order_score"),
                "reverse_order_rate": _rate(
                    [
                        _bool_int((_detail_number(detail, "order_score") or 0) < 0)
                        for detail in order_details
                        if _detail_number(detail, "order_score") is not None
                    ]
                ),
                "miracle_rate": _rate(
                    [
                        _bool_int(bool(detail.get("miracle_step")) or bool(detail.get("block_miracle_step")))
                        for detail in order_metric_details
                        if detail.get("miracle_step") is not None or detail.get("block_miracle_step") is not None
                    ]
                ),
                "avg_miracle_severity": _avg_detail(order_metric_details, "miracle_severity"),
                "avg_block_order_score": _avg_detail(block_order_details, "block_order_score"),
                "block_reverse_order_rate": _rate(
                    [
                        _bool_int((_detail_number(detail, "block_order_score") or 0) < 0)
                        for detail in block_order_details
                        if _detail_number(detail, "block_order_score") is not None
                    ]
                ),
                "block_miracle_rate": _rate(
                    [
                        _bool_int(detail.get("block_miracle_step"))
                        for detail in order_metric_details
                        if detail.get("block_miracle_step") is not None
                    ]
                ),
                "avg_block_efficiency": _avg_detail(bonus_details, "block_efficiency"),
                "avg_blocks_per_trace": (total_blocks / len(bonus_details)) if bonus_details else None,
                "block_achieve_rate": (_sum_detail(bonus_details, "n_achieving_blocks") / scored_read_blocks)
                if scored_read_blocks
                else None,
                "block_waste_rate": (_sum_detail(bonus_details, "n_wasted_blocks") / scored_read_blocks)
                if scored_read_blocks
                else None,
                "block_loop_rate": (_sum_detail(bonus_details, "n_loop_blocks") / total_blocks) if total_blocks else None,
                "achieving_block_step_share": (_sum_detail(bonus_details, "n_achieving_block_steps") / scored_block_steps)
                if scored_block_steps
                else None,
                "wasted_block_step_share": (_sum_detail(bonus_details, "n_wasted_block_steps") / scored_block_steps)
                if scored_block_steps
                else None,
                "loop_block_step_share": (_sum_detail(bonus_details, "n_loop_block_steps") / block_steps) if block_steps else None,
                "loop_trace_rate": _rate_detail([detail.get("bad_patterns") or {} for detail in details], "has_loop"),
                "error_spiral_rate": _rate_detail([detail.get("bad_patterns") or {} for detail in details], "error_spiral"),
                "not_chain_evaluable_reasons": _detail_distribution(
                    [detail for detail in details if not detail.get("chain_evaluable")],
                    "not_chain_evaluable_reason",
                ),
                "chain_bad_patterns": _chain_bad_distribution(details),
                "avg_turns": _avg([row["turns"] for row in metric_rows]),
                "avg_tool_calls": _avg([row["tool_calls"] for row in metric_rows]),
                "avg_wall_time": _avg([row["wall_time"] for row in metric_rows]),
                "avg_input_tokens": _avg([row["input_tokens"] for row in metric_rows]),
                "avg_output_tokens": _avg([row["output_tokens"] for row in metric_rows]),
                "avg_reasoning_tokens": _avg([row["reasoning_tokens"] for row in metric_rows]),
                "cache_hit_rate": (cache_hit / (input_tokens + cache_hit)) if cache_hit and (input_tokens + cache_hit) else None,
                "cache_write_rate": (cache_write / (input_tokens + cache_write)) if cache_write and (input_tokens + cache_write) else None,
                "total_cache_write_tokens": cache_write if cache_write else None,
                "total_cost": sum(float(row["cost"] or 0) for row in metric_rows) if metric_rows else None,
            }
        )
    return out
