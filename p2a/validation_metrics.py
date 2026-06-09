"""Live P2A validation metrics for SWE-bench dashboard logging."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from p2a.core import BonusMapStore
from p2a.eval_fault_localization import score_record, summarize, write_jsonl

P2A_VALIDATION_METRICS = (
    "bonus_map_coverage",
    "call_graph_coverage",
    "read_rate",
    "graph_hit_rate_over_call_graphs",
    "ground_truth_hit_rate_over_call_graphs",
    "near_hit_rate_over_call_graphs",
    "avg_min_distance_on_hits",
    "avg_best_positive_multiplier_on_hits",
)


def _as_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    return value


def _row_value(values: Any, idx: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        return _as_python(values[idx])
    except (IndexError, KeyError, TypeError):
        return default


def _record_instance_id(record: dict[str, Any], extra_fields: dict[str, Any]) -> str | None:
    for key in ("instance_id", "uid", "task_id"):
        value = record.get(key)
        if isinstance(value, str) and value and ("__" in value or key == "instance_id"):
            return value
    value = extra_fields.get("instance_id")
    return value if isinstance(value, str) and value else None


def validation_records_from_batch(
    batch: Any,
    *,
    output_texts: list[str],
    scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build scorer records from a validation DataProto after generation."""
    non_tensor = getattr(batch, "non_tensor_batch", {})
    records = []
    for idx, response_text in enumerate(output_texts):
        extra_fields = _row_value(non_tensor.get("extra_fields"), idx, {})
        if not isinstance(extra_fields, dict):
            extra_fields = {}

        record = {
            "uid": _row_value(non_tensor.get("uid"), idx),
            "instance_id": _row_value(non_tensor.get("instance_id"), idx),
            "data_source": _row_value(non_tensor.get("data_source"), idx, "unknown") or "unknown",
            "response_text": _row_value(non_tensor.get("response_text"), idx, response_text) or response_text,
            "p2a_step_traces": _row_value(non_tensor.get("p2a_step_traces"), idx),
            "extra_fields": dict(extra_fields),
            "score": scores[idx] if scores and idx < len(scores) else None,
        }
        if record["p2a_step_traces"] is None:
            record.pop("p2a_step_traces")
        instance_id = _record_instance_id(record, record["extra_fields"])
        if instance_id:
            record["instance_id"] = instance_id
            record["extra_fields"].setdefault("instance_id", instance_id)
        records.append(record)
    return records


def score_validation_records(
    records: Iterable[dict[str, Any]],
    *,
    bonus_maps: BonusMapStore,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
) -> list[dict[str, Any]]:
    details = []
    for index, record in enumerate(records):
        detail = score_record(
            record,
            index=index,
            bonus_maps=bonus_maps,
            tracking_mode=tracking_mode,
            near_threshold=near_threshold,
            m_max=m_max,
        )
        detail["data_source"] = record.get("data_source") or "unknown"
        details.append(detail)
    return details


def flatten_validation_metrics(
    details: list[dict[str, Any]],
    *,
    bonus_map_dir: str,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
) -> dict[str, float]:
    """Flatten scored details into val-p2a/<data_source>/<metric> keys."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail.get("data_source") or "unknown")].append(detail)

    metrics: dict[str, float] = {}
    for data_source, group in sorted(grouped.items()):
        summary = summarize(
            group,
            source=Path("live_validation"),
            bonus_map_dir=Path(bonus_map_dir),
            tracking_mode=tracking_mode,
            near_threshold=near_threshold,
            m_max=m_max,
        )
        values = {**summary["rates"], **summary["averages"]}
        prefix = f"val-p2a/{data_source}"
        for name in P2A_VALIDATION_METRICS:
            value = values.get(name)
            if value is not None:
                metrics[f"{prefix}/{name}"] = float(value)
        metrics[f"{prefix}/n_records"] = float(summary["counts"].get("n_records", 0))
        metrics[f"{prefix}/n_with_bonus_map"] = float(summary["counts"].get("n_with_bonus_map", 0))
        metrics[f"{prefix}/n_with_call_graph"] = float(summary["counts"].get("n_with_call_graph", 0))
    return metrics


def compute_validation_p2a_metrics(
    records: list[dict[str, Any]],
    *,
    bonus_map_dir: str,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
    details_out: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    bonus_maps = BonusMapStore(bonus_map_dir)
    details = score_validation_records(
        records,
        bonus_maps=bonus_maps,
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )
    metrics = flatten_validation_metrics(
        details,
        bonus_map_dir=bonus_map_dir,
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )
    if details_out:
        write_jsonl(Path(details_out), details)
    return metrics, details
