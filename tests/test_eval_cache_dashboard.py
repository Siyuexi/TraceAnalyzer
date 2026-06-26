import json
import pickle
import sqlite3

import pytest

from p2a.dashboard_adapter import DashboardRequest, _case_filter_model_metrics, _normalize_detail, build_dashboard_snapshot, read_dashboard_log
from p2a import dashboard_server
from p2a.dashboard_server import write_static_dashboard
from p2a.eval_cache import aggregate_model_metrics, ensure_db, upsert_experiment, upsert_planned_cells, upsert_rollout_record


def _rollout(instance_id: str, *, resolved: bool = True):
    return {
        "run_id": f"run-{instance_id}",
        "instance_id": instance_id,
        "data_source": "swebench-hard",
        "model": "dummy-model",
        "extra_info": {
            "data_source": "swebench-hard",
            "tools_kwargs": {
                "reward": {
                    "metadata": {
                        "problem_statement": f"Problem for {instance_id}",
                        "patch": f"diff --git a/a.py b/a.py\n+fixed {instance_id}\n",
                    }
                }
            },
        },
        "messages": [{"role": "user", "content": "fix"}],
        "trajectory": [{"step_idx": 1, "exit_reason": "finished"}],
        "p2a_step_traces": [
            {
                "step_idx": 1,
                "response_text": "inspect",
                "tool_calls": [{"function": {"name": "execute_bash", "arguments": {"command": "cat /testbed/a.py"}}}],
                "tool_results": [],
            }
        ],
        "response_text": "done",
        "reward": 1.0 if resolved else 0.0,
        "resolved": resolved,
        "wall_time": 2.5,
        "token_usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "cache_hit_tokens": 50,
            "cache_write_tokens": 10,
            "cost": 0.01,
        },
    }


def _set_dataset(record: dict, dataset: str) -> dict:
    record["data_source"] = dataset
    record["dataset"] = dataset
    if isinstance(record.get("extra_info"), dict):
        record["extra_info"]["data_source"] = dataset
    return record


def _detail(instance_id: str, *, case_type: str | None = None):
    detail = {
        "record_index": 0,
        "instance_id": instance_id,
        "data_source": "swebench-hard",
        "has_bonus_map": True,
        "has_step_traces": True,
        "n_reads": 1,
        "hit_call_graph": True,
        "hit_ground_truth": True,
        "hit_near": True,
        "min_distance": 0.0,
        "hit_precision": 1.0,
        "hit_recall": 1.0,
        "hit_f1": 1.0,
        "order_score": 1.0,
        "order_defined": True,
        "miracle_step": False,
        "miracle_severity": 0,
        "block_order_score": 1.0,
        "block_order_defined": True,
        "block_miracle_step": False,
        "block_miracle_severity": 0,
        "path_evaluable": True,
        "chain_evaluable": True,
        "not_path_evaluable_reason": None,
        "not_chain_evaluable_reason": None,
        "path_covered": True,
        "chain_graph_covered": True,
        "path_hit": True,
        "chain_hit": True,
        "anchor_hit": True,
        "root_hit": True,
        "path_node_recall": 1.0,
        "chain_node_recall": 1.0,
        "path_read_precision": 1.0,
        "chain_read_precision": 1.0,
        "first_anchor_step": 0,
        "first_root_step": 0,
        "steps_anchor_to_root": 0,
        "anchor_before_root": True,
        "bad_patterns": {"has_loop": False, "error_spiral": False},
        "path_pattern_flags": {
            "missed_anchor": False,
            "missed_root_after_anchor": False,
            "root_before_anchor": False,
            "chain_stall": False,
            "chain_read_loop": False,
            "off_chain_read_spree": False,
            "error_spiral_on_chain": False,
        },
        "step_details": [
            {
                "step_index": 1,
                "trace_index": 0,
                "family": "view",
                "target_path": "a.py",
                "n_reads": 1,
                "reads": [{"file_path": "a.py", "start_line": 1, "end_line": 999999}],
                "hit_nodes": [{"key": "a.py::root", "node_role": "root_cause"}],
                "min_distance": 0.0,
            }
        ],
        "purpose_blocks": [
            {
                "block_index": 0,
                "family": "view",
                "target_path": "a.py",
                "step_indices": [0],
                "achieved": True,
                "wasted": False,
                "loop": False,
                "n_steps": 1,
                "outcome_defined": True,
                "first_hit_step": 0,
                "min_distance": 0.0,
            }
        ],
        "n_blocks": 1,
        "n_scored_read_blocks": 1,
        "n_achieving_blocks": 1,
        "n_wasted_blocks": 0,
        "n_loop_blocks": 0,
        "n_block_steps": 1,
        "n_scored_read_block_steps": 1,
        "n_achieving_block_steps": 1,
        "n_wasted_block_steps": 0,
        "n_loop_block_steps": 0,
        "block_efficiency": 1.0,
        "path_projection": {
            "anchors": ["a.py::root"],
            "roots": ["a.py::root"],
            "context_nodes": [],
            "graph_context_nodes": [],
            "path_nodes": [
                {
                    "key": "a.py::root",
                    "file_path": "a.py",
                    "start_line": 1,
                    "end_line": 20,
                    "normalized_distance": 0.0,
                    "node_role": "root_cause",
                    "hit": True,
                    "first_step": 0,
                }
            ],
            "path_edges": [],
            "context_edges": [],
            "graph_context_edges": [],
        },
        "chain_bad_patterns": {
            "missed_anchor": False,
            "missed_root_after_anchor": False,
            "root_before_anchor": False,
            "chain_stall": False,
            "chain_read_loop": False,
            "off_chain_read_spree": False,
            "error_spiral_on_chain": False,
        },
        "chain_projection": {
            "anchors": ["a.py::root"],
            "roots": ["a.py::root"],
            "context_nodes": [],
            "chain_nodes": [
                {
                    "key": "a.py::root",
                    "file_path": "a.py",
                    "start_line": 1,
                    "end_line": 20,
                    "normalized_distance": 0.0,
                    "node_role": "root_cause",
                    "hit": True,
                    "first_step": 0,
                }
            ],
            "chain_edges": [],
            "context_edges": [],
        },
    }
    if case_type is not None:
        detail["bonus_case_type"] = case_type
        detail["path_case_kind"] = case_type
        detail["chain_case_kind"] = case_type
    return detail


def _bonus_map(instance_id: str):
    return {
        "instance_id": instance_id,
        "case_type": "direct",
        "traceable": True,
        "selected_issue_anchor_nodes": ["a.py::root"],
        "root_cause_nodes": ["a.py::root"],
        "reward_path_edges": [],
        "call_graph_edges": [],
        "call_graph_nodes": {
            "a.py::root": {
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 20,
                "normalized_distance": 0.0,
                "rewardable": True,
                "node_role": "root_cause",
                "source": "def root():\n    return 1",
            }
        },
    }


def _standard_order_detail(instance_id: str):
    detail = _detail(instance_id, case_type="standard")
    path_projection = {
        "anchors": ["a.py::symptom"],
        "roots": ["a.py::root"],
        "context_nodes": [],
        "graph_context_nodes": [],
        "path_nodes": [
            {
                "key": "a.py::symptom",
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 10,
                "normalized_distance": 1.0,
                "node_role": "symptom",
                "hit": True,
                "first_step": 0,
            },
            {
                "key": "a.py::root",
                "file_path": "a.py",
                "start_line": 11,
                "end_line": 20,
                "normalized_distance": 0.0,
                "node_role": "root_cause",
                "hit": True,
                "first_step": 1,
            },
        ],
        "path_edges": [{"caller": "a.py::symptom", "callee": "a.py::root"}],
        "context_edges": [],
        "graph_context_edges": [],
    }
    detail["path_projection"] = path_projection
    detail["chain_projection"] = {
        "anchors": path_projection["anchors"],
        "roots": path_projection["roots"],
        "context_nodes": path_projection["context_nodes"],
        "chain_nodes": path_projection["path_nodes"],
        "chain_edges": [{"caller": "a.py::symptom", "callee": "a.py::root"}],
        "context_edges": path_projection["context_edges"],
    }
    return detail


def test_normalize_detail_mirrors_path_fields_over_legacy_defaults():
    detail = {
        "instance_id": "case-path-only",
        "path_evaluable": True,
        "not_path_evaluable_reason": None,
        "path_case_kind": "direct",
        "path_covered": True,
        "path_hit": True,
        "path_node_recall": 1.0,
        "path_read_precision": 1.0,
        "n_path_nodes": 1,
        "n_hit_path_nodes": 1,
        "path_pattern_flags": {"missed_anchor": False, "path_read_loop": True},
        "path_projection": {
            "anchors": ["a.py::root"],
            "roots": ["a.py::root"],
            "path_nodes": [{"key": "a.py::root", "hit": True}],
            "path_edges": [],
            "context_nodes": [],
            "context_edges": [],
        },
    }

    normalized = _normalize_detail(detail)

    assert normalized["chain_evaluable"] is True
    assert normalized["not_chain_evaluable_reason"] is None
    assert normalized["chain_case_kind"] == "direct"
    assert normalized["chain_graph_covered"] is True
    assert normalized["chain_hit"] is True
    assert normalized["chain_node_recall"] == 1.0
    assert normalized["chain_read_precision"] == 1.0
    assert normalized["n_chain_nodes"] == 1
    assert normalized["n_hit_chain_nodes"] == 1
    assert normalized["chain_projection"] == normalized["path_projection"]
    assert normalized["chain_projection"]["chain_nodes"] == [{"key": "a.py::root", "hit": True}]
    assert normalized["chain_bad_patterns"]["chain_read_loop"] is True


def test_eval_cache_upserts_cells_without_duplicates(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="openai_compatible",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="openai_compatible",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        first = _rollout("case-1")
        second = _rollout("case-1")
        second["run_id"] = "run-case-1-retry"
        for record in (first, second):
            upsert_rollout_record(
                conn,
                experiment_id="exp",
                provider_source="openai_compatible",
                model_api_name="dummy-model",
                model_label="dummy",
                dataset="swebench-hard",
                record=record,
                detail=_detail("case-1"),
            )
        conn.commit()

        cell_count = conn.execute("SELECT COUNT(*) FROM run_cells").fetchone()[0]
        raw_count = conn.execute("SELECT COUNT(*) FROM raw_rollouts").fetchone()[0]
        metrics_count = conn.execute("SELECT COUNT(*) FROM quantitative_metrics").fetchone()[0]

    assert cell_count == 1
    assert raw_count == 1
    assert metrics_count == 1


def test_eval_cache_migrates_v1_cells_to_rollout_schema(tmp_path):
    db = tmp_path / "traces.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE run_cells (
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
        INSERT INTO run_cells(
          experiment_id, provider_source, model_api_name, model_label, dataset,
          instance_id, status, created_at, updated_at
        )
        VALUES ('exp', 'internal_api', 'dummy-model', 'dummy', 'swebench-hard', 'case-1', 'done', 'now', 'now');
        """
    )
    conn.commit()
    conn.close()

    with ensure_db(db) as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(run_cells)").fetchall()}
        assert {"rollout_index", "rollout_id"} <= columns
        row = migrated.execute("SELECT instance_id, rollout_index, status FROM run_cells").fetchone()
        assert dict(row) == {"instance_id": "case-1", "rollout_index": 0, "status": "done"}
        upsert_planned_cells(
            migrated,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
            rollouts_per_instance=2,
        )
        migrated.commit()
        rows = migrated.execute("SELECT rollout_index, status FROM run_cells ORDER BY rollout_index").fetchall()
        assert [(row["rollout_index"], row["status"]) for row in rows] == [(0, "done"), (1, "pending")]


def test_unified_dashboard_snapshot_includes_db_model_metrics(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=_detail("case-1", case_type="direct"),
        )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")
        raw_row = conn.execute("SELECT issue_description, golden_patch FROM raw_rollouts").fetchone()
        metrics_row = conn.execute(
            """
            SELECT p2a_read, call_graph_hit, ground_truth_hit, near_hit, min_distance, metrics_json
            FROM quantitative_metrics
            """
        ).fetchone()

    assert rows[0]["model_label"] == "dummy"
    assert raw_row["issue_description"] == "Problem for case-1"
    assert raw_row["golden_patch"] == "diff --git a/a.py b/a.py\n+fixed case-1"
    assert metrics_row["p2a_read"] is None
    assert metrics_row["call_graph_hit"] is None
    assert metrics_row["ground_truth_hit"] is None
    assert metrics_row["near_hit"] is None
    assert metrics_row["min_distance"] is None
    assert "detail" not in json.loads(metrics_row["metrics_json"])
    assert rows[0]["target"] == 2
    assert rows[0]["done"] == 1
    assert rows[0]["resolved_rate"] == 1.0
    assert rows[0]["p2a_read_rate"] is None
    assert rows[0]["avg_read_precision"] is None
    assert rows[0]["avg_node_recall"] is None
    assert rows[0]["avg_path_node_precision"] is None
    assert rows[0]["avg_chain_node_precision"] is None
    assert rows[0]["avg_hit_f1"] is None
    assert rows[0]["anchor_hit_rate"] is None
    assert rows[0]["root_hit_rate"] is None
    assert rows[0]["block_achieve_rate"] is None
    assert rows[0]["cache_hit_rate"] == 50 / 150

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir))
    assert snapshot["schema_version"] == "p2a_unified_dashboard_v1"
    assert snapshot["datasets"][0]["dataset"] == "swebench-hard"
    assert snapshot["eval_cells"][0]["experiment_id"] == "exp"
    assert snapshot["experiments"][0]["experiment_id"] == "exp"
    assert snapshot["experiments"][0]["source_kind"] == "third_party_api"
    assert snapshot["model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["model_metrics"][0]["target"] == 2
    assert snapshot["model_metrics"][0]["avg_read_precision"] == 1.0
    assert snapshot["model_metrics"][0]["avg_path_node_precision"] == 1.0
    assert snapshot["model_metrics"][0]["avg_chain_node_precision"] == 1.0
    assert snapshot["path_metric_detail_count"] == 1
    assert snapshot["dynamic_traceable_detail_count"] == 1
    assert snapshot["path_metric_model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["path_metric_model_metrics"][0]["target"] == 1
    assert snapshot["path_metric_model_metrics"][0]["avg_tool_calls"] == 1.0
    assert snapshot["path_metric_model_metrics"][0]["cache_hit_rate"] == 50 / 150
    assert snapshot["dynamic_traceable_model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["dynamic_traceable_model_metrics"][0]["target"] == 1
    assert snapshot["dynamic_traceable_model_metrics"][0]["avg_tool_calls"] == 1.0
    assert snapshot["dynamic_traceable_model_metrics"][0]["cache_hit_rate"] == 50 / 150
    assert snapshot["summary"]["counts"]["n_records"] == 1
    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["issue_description"] == "Problem for case-1"
    assert snapshot["details"][0]["golden_patch"] == "diff --git a/a.py b/a.py\n+fixed case-1"
    assert snapshot["details"][0]["experiment_key"] == snapshot["experiments"][0]["experiment_key"]
    assert snapshot["details"][0]["step_inspection"][0]["tool_names"] == ["execute_bash"]
    assert snapshot["details"][0]["step_inspection"][0]["action_family"] == "read"
    assert snapshot["details"][0]["step_inspection"][0]["chat_text"] == "inspect"
    assert snapshot["details"][0]["step_inspection"][0]["parsed_tool_calls"] == [
        {"name": "execute_bash", "arguments": [{"key": "command", "value": "cat /testbed/a.py"}]}
    ]
    assert snapshot["details"][0]["step_inspection"][0]["recovered_reads"][0]["file_path"] == "a.py"


def test_eval_cache_aggregates_pass_at_n_and_avg_at_n(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2"],
            rollouts_per_instance=2,
        )
        for instance_id, resolved_values in {"case-1": [False, True], "case-2": [False, False]}.items():
            for rollout_index, resolved in enumerate(resolved_values):
                record = _rollout(instance_id, resolved=resolved)
                record["run_id"] = f"run-{instance_id}-{rollout_index}"
                record["rollout_index"] = rollout_index
                record["wall_time"] = 1.0 + rollout_index
                upsert_rollout_record(
                    conn,
                    experiment_id="exp",
                    provider_source="internal_api",
                    model_api_name="dummy-model",
                    model_label="dummy",
                    dataset="swebench-hard",
                    record=record,
                )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")

    row = rows[0]
    assert row["target"] == 2
    assert row["target_rollouts"] == 4
    assert row["done"] == 2
    assert row["done_rollouts"] == 4
    assert row["rollouts_per_instance"] == 2
    assert row["pass_at"] == {"1": 0.0, "2": 0.5}
    assert row["pass_at_n"] == 0.5
    assert row["avg_at"]["1"]["resolved_rate"] == 0.0
    assert row["avg_at"]["2"]["resolved_rate"] == 0.25
    assert row["resolved_rate"] == 0.25
    assert row["resolved_rate_std"] == 0.25
    assert row["avg_wall_time"] == 1.5
    assert row["avg_wall_time_std"] == 0.0

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp"))

    metric = snapshot["model_metrics"][0]
    assert metric["pass_at_n"] == 0.5
    assert metric["avg_at"]["1"]["resolved_rate"] == 0.0
    assert metric["resolved_rate"] == 0.25
    assert metric["rollouts_per_instance"] == 2
    details = sorted((detail["instance_id"], detail["rollout_index"]) for detail in snapshot["details"])
    assert details == [("case-1", 0), ("case-1", 1), ("case-2", 0), ("case-2", 1)]


def test_swebench_pro_dashboard_mixes_p2a_and_resolution_only_cells(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-python.json").write_text(json.dumps(_bonus_map("case-python")), encoding="utf-8")

    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            dataset="swebench-pro",
            config_snapshot={"dataset": "swebench-pro"},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-pro",
            instance_ids=["case-python", "case-go"],
        )

        py_record = _set_dataset(_rollout("case-python", resolved=True), "swebench-pro")
        py_record["repo_language"] = "python"
        upsert_rollout_record(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-pro",
            record=py_record,
        )

        go_record = _set_dataset(_rollout("case-go", resolved=False), "swebench-pro")
        go_record["repo_language"] = "go"
        upsert_rollout_record(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-pro",
            record=go_record,
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp-pro",
            dataset="swebench-pro",
            bonus_map_dir=bonus_dir,
        )
    )
    details = {detail["instance_id"]: detail for detail in snapshot["details"]}

    assert snapshot["datasets"][0]["dataset"] == "swebench-pro"
    assert snapshot["model_metrics"][0]["target"] == 2
    assert snapshot["model_metrics"][0]["resolved_rate"] == 0.5
    assert snapshot["path_metric_detail_count"] == 1
    assert snapshot["dynamic_traceable_detail_count"] == 1

    assert details["case-python"]["has_bonus_map"] is True
    assert details["case-python"]["chain_case_kind"] == "direct"
    assert details["case-python"]["hit_ground_truth"] is True
    assert details["case-python"]["resolved"] is True

    assert details["case-go"]["has_bonus_map"] is False
    assert details["case-go"]["not_chain_evaluable_reason"] == "missing_bonus_map"
    assert details["case-go"]["resolved"] is False
    assert details["case-go"]["data_source"] == "swebench-pro"

    paths = write_static_dashboard(tmp_path / "swebench-pro-dashboard", snapshot)
    html = paths["html"].read_text(encoding="utf-8")
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" in html
    assert "swebench-pro" in html


def test_dashboard_fills_issue_and_patch_from_dataset_parquet_when_db_raw_lacks_metadata(tmp_path):
    pd = pytest.importorskip("pandas")
    data_file = tmp_path / "swe_bench_verified_hard.parquet"
    frame = pd.DataFrame(
        [
            {
                "instance_id": "case-1",
                "problem_statement": "Issue from parquet",
                "patch": "diff --git a/a.py b/a.py\n+from parquet\n",
            }
        ]
    )
    try:
        frame.to_parquet(data_file)
    except Exception as exc:  # noqa: BLE001 - optional parquet engines vary by environment
        pytest.skip(f"parquet engine unavailable: {exc}")

    db = tmp_path / "traces.sqlite"
    record = _rollout("case-1", resolved=True)
    record.pop("extra_info", None)
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=record,
            detail=_detail("case-1", case_type="direct"),
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, dataset="swebench-hard", data_file=data_file))

    assert snapshot["details"][0]["issue_description"] == "Issue from parquet"
    assert snapshot["details"][0]["golden_patch"] == "diff --git a/a.py b/a.py\n+from parquet"


def test_model_metrics_ignore_other_case_bonus_map_fields(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=_standard_order_detail("case-1"),
        )
        other = _standard_order_detail("case-2")
        other["bonus_case_type"] = "missing_bonus_map"
        other["path_case_kind"] = "missing_bonus_map"
        other["chain_case_kind"] = "missing_bonus_map"
        other["path_evaluable"] = False
        other["chain_evaluable"] = False
        other["hit_precision"] = 0.0
        other["hit_recall"] = 0.0
        other["hit_f1"] = 0.0
        other["order_score"] = -1.0
        other["miracle_step"] = True
        other["block_order_score"] = -1.0
        other["block_miracle_step"] = True
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-2", resolved=False),
            detail=other,
        )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")

    assert rows[0]["done"] == 2
    assert rows[0]["avg_read_precision"] is None
    assert rows[0]["avg_node_recall"] is None
    assert rows[0]["avg_order_score"] is None
    assert rows[0]["reverse_order_rate"] is None
    assert rows[0]["miracle_rate"] is None


def test_case_filter_metrics_bucket_non_evaluable_standard_as_others():
    detail = _standard_order_detail("case-1")
    detail["path_evaluable"] = False
    detail["chain_evaluable"] = False
    detail["not_path_evaluable_reason"] = "missing_anchor"
    detail["not_chain_evaluable_reason"] = "missing_anchor"

    rows = _case_filter_model_metrics([detail])

    assert rows["standard"] == []
    assert rows["direct,standard"] == []
    assert rows["others"][0]["target"] == 1
    assert rows["others"][0]["not_path_evaluable_reasons"] == {"missing_anchor": 1}
    assert rows["others"][0]["not_chain_evaluable_reasons"] == {"missing_anchor": 1}


def test_dashboard_does_not_reinfer_miracle_from_stored_first_hit_steps(tmp_path):
    db = tmp_path / "traces.sqlite"
    detail = _standard_order_detail("case-1")
    detail["first_root_step"] = 1
    detail["first_anchor_step"] = 3
    detail["miracle_step"] = False
    detail["miracle_severity"] = 0
    detail["block_miracle_step"] = False
    detail["block_miracle_severity"] = 0
    detail["step_details"] = [
        {
            "step_index": 2,
            "trace_index": 1,
            "family": "read",
            "target_path": "a.py",
            "n_reads": 1,
            "hit_nodes": [
                {"key": "a.py::symptom", "node_role": "symptom"},
                {"key": "a.py::root", "node_role": "root_cause"},
            ],
        }
    ]
    detail["purpose_blocks"] = [
        {
            "block_index": 0,
            "family": "read",
            "trace_indices": [1],
            "step_indices": [2],
            "achieved": True,
            "outcome_defined": True,
            "n_steps": 1,
        }
    ]

    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=detail,
        )
        cell_id = conn.execute("SELECT id FROM run_cells WHERE instance_id = ?", ("case-1",)).fetchone()["id"]
        conn.execute(
            "UPDATE quantitative_metrics SET metrics_json = ? WHERE cell_id = ?",
            (json.dumps({"detail": detail}), cell_id),
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp"))

    assert snapshot["details"][0]["miracle_step"] is False
    assert snapshot["details"][0]["block_miracle_step"] is False
    assert snapshot["details"][0]["first_anchor_step"] == 2
    assert snapshot["details"][0]["first_root_step"] == 2
    assert snapshot["model_metrics"][0]["miracle_rate"] == 0.0
    assert snapshot["path_metric_model_metrics"][0]["miracle_rate"] == 0.0
    assert snapshot["dynamic_traceable_model_metrics"][0]["miracle_rate"] == 0.0


def test_dashboard_infers_default_bonus_map_dir_and_rescores_db_raw_rollouts(tmp_path):
    artifact_root = tmp_path / "data"
    db = artifact_root / "evals" / "traces.sqlite"
    bonus_dir = artifact_root / "bonus_maps" / "swebench-hard"
    bonus_dir.mkdir(parents=True)
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    stale_detail = _detail("case-1", case_type="direct")
    stale_detail["path_projection"]["path_nodes"][0].pop("source", None)
    stale_detail["path_projection"]["path_nodes"][0]["source_preview"] = "truncated\n..."
    stale_detail["chain_projection"]["chain_nodes"][0].pop("source", None)
    stale_detail["chain_projection"]["chain_nodes"][0]["source_preview"] = "truncated\n..."

    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=stale_detail,
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, dataset="swebench-hard"))

    assert {"kind": "bonus_map_dir", "path": str(bonus_dir), "dataset": "swebench-hard", "mode": "inferred"} in snapshot["sources"]
    root = snapshot["details"][0]["path_projection"]["path_nodes"][0]
    assert root["source"] == "def root():\n    return 1"
    assert not root["source"].endswith("...")


def test_dashboard_reads_node_source_from_bonus_map_for_stored_details(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    stale_detail = _detail("case-1", case_type="direct")
    stale_detail["path_projection"]["path_nodes"][0].pop("source", None)
    stale_detail["path_projection"]["path_nodes"][0]["source_preview"] = "truncated\n..."
    stale_detail["chain_projection"]["chain_nodes"][0].pop("source", None)
    stale_detail["chain_projection"]["chain_nodes"][0]["source_preview"] = "truncated\n..."
    details_file = tmp_path / "details.jsonl"
    details_file.write_text(json.dumps(stale_detail) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(details=(details_file,), bonus_map_dir=bonus_dir))

    root = snapshot["details"][0]["path_projection"]["path_nodes"][0]
    assert root["source"] == "def root():\n    return 1"
    assert root["source_preview"] == "def root():\n    return 1"


def test_dashboard_node_source_uses_bonus_map_candidate_filenames(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    instance_id = "repo__1234567890"
    (bonus_dir / "repo__12345678.json").write_text(json.dumps(_bonus_map(instance_id)), encoding="utf-8")
    stale_detail = _detail(instance_id, case_type="direct")
    stale_detail["path_projection"]["path_nodes"][0].pop("source", None)
    stale_detail["path_projection"]["path_nodes"][0]["source_preview"] = "truncated\n..."
    stale_detail["chain_projection"]["chain_nodes"][0].pop("source", None)
    stale_detail["chain_projection"]["chain_nodes"][0]["source_preview"] = "truncated\n..."
    details_file = tmp_path / "details.jsonl"
    details_file.write_text(json.dumps(stale_detail) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(details=(details_file,), bonus_map_dir=bonus_dir))

    root = snapshot["details"][0]["path_projection"]["path_nodes"][0]
    assert root["source"] == "def root():\n    return 1"
    assert root["source_preview"] == "def root():\n    return 1"


def test_dashboard_step_inspection_splits_local_think_and_xml_tool_call(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")
    record["p2a_step_traces"] = [
        {
            "step_idx": 1,
            "response_text": (
                "<think>inspect the suspicious file</think>\n"
                "I will open the target.\n"
                "<function=file_editor>\n"
                "<parameter=command>view</parameter>\n"
                "<parameter=path>/testbed/a.py</parameter>\n"
                "<parameter=view_range>[1, 20]</parameter>\n"
                "<parameter=concise>false</parameter>\n"
                "</function>"
            ),
            "tool_calls": [],
            "tool_results": [],
        }
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(record) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))
    step = snapshot["details"][0]["step_inspection"][0]

    assert step["reasoning_text"] == "inspect the suspicious file"
    assert step["chat_text"] == "I will open the target."
    assert step["tool_names"] == ["file_editor"]
    assert step["action_family"] == "read"
    assert step["parsed_tool_calls"] == [
        {
            "name": "file_editor",
            "arguments": [
                {"key": "command", "value": "view"},
                {"key": "path", "value": "/testbed/a.py"},
                {"key": "view_range", "value": [1, 20]},
                {"key": "concise", "value": False},
            ],
        }
    ]


def test_dashboard_step_inspection_marks_root_edits_and_execution_errors(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")
    record["p2a_step_traces"] = [
        {
            "step_idx": 1,
            "response_text": "patch root",
            "tool_calls": [
                {
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": {
                            "command": "str_replace",
                            "path": "/testbed/a.py",
                            "old_str": "return 0",
                            "new_str": "return 1",
                        },
                    }
                }
            ],
            "tool_results": [{"observation": "Traceback: command failed"}],
        }
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(record) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))
    detail = snapshot["details"][0]
    step = detail["step_inspection"][0]

    assert detail["edited_root_cause"] is True
    assert step["action_family"] == "edit"
    assert step["write_actions"] == [{"file_path": "a.py", "start_line": 1, "end_line": 999999, "command": "str_replace"}]
    assert step["edited_root_cause"] is True
    assert step["execution_error"] is True
    assert step["status"] == "error"


def test_dashboard_step_inspection_does_not_mark_source_text_errors_as_execution_failure(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")
    record["p2a_step_traces"] = [
        {
            "step_idx": 1,
            "response_text": "",
            "reasoning_content": "find the expression code",
            "text_blocks": [{"type": "text", "value": "I will read the file."}],
            "tool_calls": [
                {
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": {"command": "view", "path": "/testbed/a.py", "view_range": [1, 20]},
                    }
                }
            ],
            "tool_results": [
                {
                    "status": "ok",
                    "observation": "Observation:\nclass FieldError(Exception):\n    pass\n",
                }
            ],
        }
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(record) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))
    step = snapshot["details"][0]["step_inspection"][0]

    assert step["reasoning_text"] == "find the expression code"
    assert step["chat_text"] == "I will read the file."
    assert step["execution_error"] is False
    assert step["status"] == "ok"


def test_dashboard_snapshot_uses_readonly_db_connection_under_writer_lock(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
            detail=_detail("case-1"),
        )
        conn.commit()

    writer = sqlite3.connect(db, timeout=0.1)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE run_cells SET updated_at = updated_at")
        snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp"))
    finally:
        writer.rollback()
        writer.close()

    assert snapshot["model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["details"][0]["instance_id"] == "case-1"


def test_unified_dashboard_keeps_experiments_separate_in_overview(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        for exp_id, model, case_id in (("exp-a", "model-a", "case-a"), ("exp-b", "model-b", "case-b")):
            upsert_experiment(
                conn,
                experiment_id=exp_id,
                provider_source="internal_api",
                dataset="swebench-hard",
                config_snapshot={"experiment": exp_id},
            )
            upsert_planned_cells(
                conn,
                experiment_id=exp_id,
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                instance_ids=[case_id],
            )
            record = _rollout(case_id)
            record["model"] = model
            upsert_rollout_record(
                conn,
                experiment_id=exp_id,
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                record=record,
                detail=_detail(case_id),
            )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))

    experiment_keys = {row["experiment_key"] for row in snapshot["experiments"]}
    detail_keys = {row["experiment_key"] for row in snapshot["details"]}
    assert len(snapshot["experiments"]) == 2
    assert len(experiment_keys) == 2
    assert detail_keys == experiment_keys
    assert {row["experiment_id"] for row in snapshot["model_metrics"]} == {"exp-a", "exp-b"}


def test_dashboard_rescores_each_dataset_with_inferred_bonus_maps(tmp_path):
    artifact_root = tmp_path / "data"
    db = artifact_root / "evals" / "traces.sqlite"
    datasets = ("swebench-hard", "r2e-gym-subset")
    for dataset in datasets:
        bonus_dir = artifact_root / "bonus_maps" / dataset
        bonus_dir.mkdir(parents=True)
        (bonus_dir / f"case-{dataset}.json").write_text(json.dumps(_bonus_map(f"case-{dataset}")), encoding="utf-8")

    with ensure_db(db) as conn:
        for dataset in datasets:
            instance_id = f"case-{dataset}"
            upsert_experiment(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                dataset=dataset,
                config_snapshot={"dataset": dataset},
            )
            upsert_planned_cells(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name="dummy-model",
                model_label="dummy",
                dataset=dataset,
                instance_ids=[instance_id],
            )
            stale_detail = _detail(instance_id, case_type="direct")
            stale_detail["data_source"] = dataset
            stale_detail["n_reads"] = 0
            upsert_rollout_record(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name="dummy-model",
                model_label="dummy",
                dataset=dataset,
                record=_set_dataset(_rollout(instance_id), dataset),
                detail=stale_detail,
            )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))

    assert {item["dataset"] for item in snapshot["sources"] if item["kind"] == "bonus_map_dir"} == set(datasets)
    assert {detail["data_source"] for detail in snapshot["details"]} == set(datasets)
    assert all(detail["n_reads"] > 0 for detail in snapshot["details"])


def test_local_training_eval_cells_include_run_step(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    for instance_id in ("case-1", "case-2"):
        (bonus_dir / f"{instance_id}.json").write_text(json.dumps(_bonus_map(instance_id)), encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    records = []
    for step, instance_id in ((10, "case-1"), (20, "case-2")):
        record = _rollout(instance_id)
        record["run_step"] = step
        record["model"] = "trainer-model"
        record["model_label"] = "trainer-model"
        records.append(record)
    rollouts.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))

    assert len(snapshot["eval_cells"]) == 2
    assert {cell["run_step"] for cell in snapshot["eval_cells"]} == {10, 20}
    assert len({cell["eval_cell_key"] for cell in snapshot["eval_cells"]}) == 2


def test_dashboard_dataset_distributions_deduplicate_instances_across_models(tmp_path):
    db = tmp_path / "traces.sqlite"
    instance_ids = [f"case-{index:02d}" for index in range(45)]
    with ensure_db(db) as conn:
        for model_index in range(5):
            model = f"model-{model_index}"
            upsert_experiment(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                dataset="swebench-hard",
                config_snapshot={"experiment": "exp"},
            )
            upsert_planned_cells(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                instance_ids=instance_ids,
            )
            for case_id in instance_ids:
                record = _rollout(case_id)
                record["model"] = model
                upsert_rollout_record(
                    conn,
                    experiment_id="exp",
                    provider_source="internal_api",
                    model_api_name=model,
                    model_label=model,
                    dataset="swebench-hard",
                    record=record,
                    detail=_detail(case_id),
                )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))

    assert snapshot["datasets"] == [
        {
            "dataset": "swebench-hard",
            "n_instances": 45,
            "n_eval_cells": 5,
            "n_trajectories": 225,
            "models": [f"model-{index}" for index in range(5)],
            "source_kinds": ["third_party_api"],
        }
    ]
    dist = snapshot["summary"]["distributions_by_dataset"]["swebench-hard"]
    assert dist["n_instances"] == 45
    assert dist["distributions"]["case_types"] == {"missing_bonus_map": 45}


def test_dashboard_db_runs_carry_explicit_eval_cell_links(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        for model in ("model-a", "model-b"):
            run_dir = tmp_path / "runs" / model
            run_dir.mkdir(parents=True)
            rollouts_path = run_dir / "rollouts.jsonl"
            rollouts_path.write_text("{}\n", encoding="utf-8")
            (run_dir / "run.log").write_text(f"run for {model}\n", encoding="utf-8")
            upsert_experiment(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                dataset="swebench-hard",
                config_snapshot={"experiment": "exp"},
            )
            upsert_planned_cells(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                instance_ids=[f"case-{model[-1]}"],
            )
            record = _rollout(f"case-{model[-1]}")
            record["model"] = model
            upsert_rollout_record(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                record=record,
                detail=_detail(record["instance_id"]),
                artifact_rollouts=rollouts_path,
            )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))
    runs_by_model = {run["model_labels"][0]: run for run in snapshot["runs"]}

    assert set(runs_by_model) == {"model-a", "model-b"}
    assert len(runs_by_model["model-a"]["eval_cell_keys"]) == 1
    assert runs_by_model["model-a"]["eval_cell_keys"][0].endswith("model-a::model-a")
    assert runs_by_model["model-b"]["eval_cell_keys"][0].endswith("model-b::model-b")


def test_unified_dashboard_handles_empty_db(tmp_path):
    db = tmp_path / "empty.sqlite"
    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))

    assert snapshot["model_metrics"] == []
    assert snapshot["summary"]["counts"]["n_records"] == 0
    assert not db.exists()


def test_unified_dashboard_loads_local_uni_agent_run_dir(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")

    run_root = tmp_path / "runs"
    run_dir = run_root / "run-case-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("Beginning environment startup\nSTEP 1\n", encoding="utf-8")
    (run_dir / "interaction_result.json").write_text(
        json.dumps({"messages": [], "trajectory": [], "execution_time": 1.0, "metrics": {}, "reward_score": 1.0}),
        encoding="utf-8",
    )
    with (run_dir / "rollout_cache.pkl").open("wb") as handle:
        pickle.dump(
            {
                "extra_fields": {
                    "instance_id": "case-1",
                    "data_source": "local",
                    "p2a_step_traces": [
                        {
                            "step_idx": 1,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "execute_bash",
                                        "arguments": {"command": "sed -n '1,20p' /testbed/a.py"},
                                    }
                                }
                            ],
                        }
                    ],
                }
            },
            handle,
        )

    snapshot = build_dashboard_snapshot(DashboardRequest(log_dir=run_root, bonus_map_dir=bonus_dir))

    assert snapshot["runs"][0]["status"] == "completed"
    assert snapshot["experiments"][0]["source_kind"] == "local_inference"
    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["root_hit"] is True
    assert snapshot["details"][0]["step_inspection"][0]["tool_names"] == ["execute_bash"]
    assert snapshot["summary"]["counts"]["n_records"] == 1


def test_unified_dashboard_details_mode_matches_rollout_mode(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")

    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(_rollout("case-1")) + "\n", encoding="utf-8")
    rollout_snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))

    details_dir = tmp_path / "eval_details"
    details_dir.mkdir()
    (details_dir / "validation_step_1.jsonl").write_text(
        "\n".join(json.dumps(item) for item in rollout_snapshot["details"]) + "\n",
        encoding="utf-8",
    )
    details_snapshot = build_dashboard_snapshot(DashboardRequest(details=(details_dir,), bonus_map_dir=bonus_dir))

    assert details_snapshot["summary"]["counts"]["n_records"] == rollout_snapshot["summary"]["counts"]["n_records"]
    assert details_snapshot["summary"]["rates"]["root_hit_rate"] == rollout_snapshot["summary"]["rates"]["root_hit_rate"]
    assert details_snapshot["details"][0]["instance_id"] == rollout_snapshot["details"][0]["instance_id"]


def test_unified_static_dashboard_writes_html_snapshot_and_assets(tmp_path):
    snapshot = build_dashboard_snapshot(DashboardRequest())
    paths = write_static_dashboard(tmp_path / "dashboard", snapshot)

    html = paths["html"].read_text(encoding="utf-8")
    app = paths["app"].read_text(encoding="utf-8")
    assert "P2A unified dashboard" in html
    assert "Datasets and eval cells" in html
    assert "trace-inspector" in html
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" in html
    assert 'href="styles.css?v=' in html
    assert 'src="app.js?v=' in html
    assert "selectedExperimentKey" in app
    assert "selectedEvalCellKey" in app
    assert "renderGraph" in app
    assert "step_inspection" in app
    assert "detail-toggle" in app
    assert paths["snapshot"].exists()
    assert paths["app"].exists()
    assert paths["css"].exists()


def test_static_dashboard_escapes_embedded_snapshot_script_end(tmp_path):
    out_dir = tmp_path / "dashboard"
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "details": [{"issue_description": "</script><script>alert(1)</script>"}],
        "summary": {"counts": {"n_records": 1}},
    }

    paths = write_static_dashboard(out_dir, snapshot)

    html = paths["html"].read_text(encoding="utf-8")
    assert html.count("</script>") == 2
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert "</script><script>alert(1)</script>" not in html


def test_dashboard_serve_mode_does_not_prebuild_snapshot(monkeypatch, tmp_path):
    def fail_build(_request):
        raise AssertionError("serve mode must load snapshots lazily")

    served = {}

    def fake_serve(request, *, host, port):
        served["db_path"] = request.db_path
        served["host"] = host
        served["port"] = port

    monkeypatch.setattr(dashboard_server, "build_dashboard_snapshot", fail_build)
    monkeypatch.setattr(dashboard_server, "serve_dashboard", fake_serve)

    result = dashboard_server.main(["--db", str(tmp_path / "locked.sqlite")])

    assert result == 0
    assert served == {"db_path": tmp_path / "locked.sqlite", "host": "0.0.0.0", "port": 8766}


def test_live_dashboard_root_embeds_initial_snapshot(monkeypatch):
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "datasets": [{"dataset": "swebench-hard"}],
        "eval_cells": [{"eval_cell_key": "cell", "dataset": "swebench-hard"}],
        "experiments": [{"experiment_key": "cell", "dataset": "swebench-hard"}],
        "model_metrics": [],
        "runs": [],
        "details": [],
        "summary": {"counts": {"n_records": 0}},
        "sources": [],
    }

    monkeypatch.setattr(dashboard_server, "build_dashboard_snapshot", lambda _request: snapshot)
    handler_type = dashboard_server.make_handler(DashboardRequest())
    handler = object.__new__(handler_type)
    payloads = []
    handler.path = "/"
    handler._send_bytes = lambda payload, content_type, **_kwargs: payloads.append((payload, content_type))

    handler.do_GET()

    html = payloads[0][0].decode("utf-8")
    assert payloads[0][1] == "text/html; charset=utf-8"
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" in html
    assert "swebench-hard" in html


def test_dashboard_log_reader_rejects_paths_outside_run_dir(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("inside", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")

    ok = read_dashboard_log(DashboardRequest(log_dir=tmp_path), run_id="run-1", source="run.log")
    assert ok["text"] == "inside"

    with pytest.raises(FileNotFoundError):
        read_dashboard_log(DashboardRequest(log_dir=tmp_path), run_id="run-1", source="../secret.txt")
    with pytest.raises(FileNotFoundError):
        read_dashboard_log(DashboardRequest(log_dir=tmp_path), run_id="run-1", source="/etc/passwd")


def test_dashboard_response_write_ignores_client_disconnect():
    handler_type = dashboard_server.make_handler(DashboardRequest())
    handler = object.__new__(handler_type)
    calls = []

    class ClosedWfile:
        def write(self, _payload):
            raise BrokenPipeError("client closed")

    handler.wfile = ClosedWfile()
    handler.close_connection = False
    handler.send_response = lambda status: calls.append(("status", status))
    handler.send_header = lambda key, value: calls.append(("header", key, value))
    handler.end_headers = lambda: calls.append(("end",))

    handler._send_bytes(b"payload", "application/json")

    assert handler.close_connection is True
    assert calls[0][0] == "status"
    assert ("header", "Cache-Control", "no-store, max-age=0") in calls
    assert ("header", "Pragma", "no-cache") in calls
