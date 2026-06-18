"""Text dashboard rendering for the unified evaluation cache."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from p2a.eval_cache import aggregate_model_metrics, ensure_db


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "   - "
    return f"{value * 100:5.1f}"


def _fmt_num(value: float | None, *, width: int = 6, digits: int = 1) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    return f"{value:{width}.{digits}f}"


def _fmt_tok(value: float | None) -> str:
    if value is None:
        return "    -"
    if value < 1000:
        return f"{int(value):5d}"
    if value < 1_000_000:
        return f"{value / 1000:4.1f}k"
    return f"{value / 1_000_000:4.1f}m"


def _trunc(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) > width:
        return value[: max(width - 1, 0)] + ("~" if width > 1 else "")
    return value.ljust(width)


def _bar(done: int, target: int, width: int = 10) -> str:
    if target <= 0:
        return "." * width
    filled = min(width, round(width * done / target))
    return "#" * filled + "." * (width - filled)


def _flags(row: dict[str, Any]) -> str:
    flags = []
    if row.get("errors"):
        flags.append(f"{row['errors']}err")
    if row.get("done") and row.get("avg_turns") and row["avg_turns"] >= 16:
        flags.append("near-cap")
    if row.get("p2a_read_rate") == 0:
        flags.append("empty-read")
    return " ".join(flags)


def render_model_dashboard(rows: list[dict[str, Any]], *, term_cols: int | None = None) -> list[str]:
    term_cols = term_cols or shutil.get_terminal_size(fallback=(140, 30)).columns
    if not rows:
        return ["No evaluation rows found in the DB."]

    total_done = sum(int(row["done"]) for row in rows)
    total_target = sum(int(row["target"]) for row in rows)
    lines = [
        _trunc(f"progress {total_done}/{total_target} ({(total_done / total_target * 100) if total_target else 0:.1f}%)", term_cols),
        _trunc(
            "rates are percentages: res=resolved, p2a=file-read, cg=call-graph hit, gt=ground-truth hit, near=near-hit",
            term_cols,
        ),
    ]

    fixed = 10 + 9 + 6 * 5 + 7 * 4 + 6 * 5 + 2
    model_w = max(18, min(40, term_cols - fixed))
    header = (
        f"{_trunc('model', model_w)} "
        f"{'progress':<10} {'done/N':>7} "
        f"{'res':>5} {'rew':>5} {'p2a':>5} {'cg':>5} {'gt':>5} {'near':>5} "
        f"{'dist':>6} {'turn':>6} {'tools':>6} {'wall':>6} "
        f"{'in':>5} {'out':>5} {'rsn':>5} {'hit':>5} {'wr':>5} flags"
    )
    lines.append(_trunc(header, term_cols))
    lines.append("-" * min(term_cols, len(header)))

    ranked = sorted(
        rows,
        key=lambda row: (
            -(row.get("resolved_rate") if row.get("resolved_rate") is not None else -1),
            -(row.get("ground_truth_hit_rate") if row.get("ground_truth_hit_rate") is not None else -1),
            str(row.get("model_label") or ""),
        ),
    )
    for row in ranked:
        done = int(row["done"])
        target = int(row["target"])
        line = (
            f"{_trunc(str(row['model_label']), model_w)} "
            f"{_bar(done, target):<10} {done:>3}/{target:<3} "
            f"{_fmt_pct(row.get('resolved_rate'))} "
            f"{_fmt_pct(row.get('reward_rate'))} "
            f"{_fmt_pct(row.get('p2a_read_rate'))} "
            f"{_fmt_pct(row.get('call_graph_hit_rate'))} "
            f"{_fmt_pct(row.get('ground_truth_hit_rate'))} "
            f"{_fmt_pct(row.get('near_hit_rate'))} "
            f"{_fmt_num(row.get('avg_min_distance'), width=6, digits=2)} "
            f"{_fmt_num(row.get('avg_turns'), width=6, digits=1)} "
            f"{_fmt_num(row.get('avg_tool_calls'), width=6, digits=1)} "
            f"{_fmt_num(row.get('avg_wall_time'), width=6, digits=1)} "
            f"{_fmt_tok(row.get('avg_input_tokens'))} "
            f"{_fmt_tok(row.get('avg_output_tokens'))} "
            f"{_fmt_tok(row.get('avg_reasoning_tokens'))} "
            f"{_fmt_pct(row.get('cache_hit_rate'))} "
            f"{_fmt_pct(row.get('cache_write_rate'))} "
            f"{_flags(row)}"
        )
        lines.append(_trunc(line, term_cols))
    return lines


def render_db_dashboard(
    db_path: Path,
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
    term_cols: int | None = None,
) -> list[str]:
    with ensure_db(db_path) as conn:
        rows = aggregate_model_metrics(
            conn,
            experiment_id=experiment_id,
            provider_source=provider_source,
            dataset=dataset,
        )
    return render_model_dashboard(rows, term_cols=term_cols)
