"""Compatibility wrapper for the unified P2A HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from p2a.dashboard_adapter import DashboardRequest, build_dashboard_snapshot
from p2a.dashboard_server import write_static_dashboard
from p2a.eval_fault_localization import _json_default, write_jsonl


def build_dashboard(
    rollouts: Path,
    bonus_map_dir: Path,
    out_dir: Path,
    *,
    tracking_mode: str = "view_and_bash",
    near_threshold: float = 0.5,
    m_max: float = 3.0,
) -> dict[str, Path]:
    """Build a static snapshot with the same HTML used by the live dashboard."""
    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            rollouts=(rollouts,),
            bonus_map_dir=bonus_map_dir,
            tracking_mode=tracking_mode,
            near_threshold=near_threshold,
            m_max=m_max,
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "details.jsonl"
    summary_path = out_dir / "summary.json"
    write_jsonl(details_path, snapshot["details"])
    summary_path.write_text(json.dumps(snapshot["summary"], indent=2, default=_json_default) + "\n", encoding="utf-8")
    paths = write_static_dashboard(out_dir, snapshot)
    return {"details": details_path, "summary": summary_path, "html": paths["html"]}
