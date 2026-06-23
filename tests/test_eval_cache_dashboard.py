import json
import pickle

from p2a.dashboard_adapter import DashboardRequest, build_dashboard_snapshot
from p2a.dashboard_server import write_static_dashboard
from p2a.eval_cache import aggregate_model_metrics, ensure_db, upsert_experiment, upsert_planned_cells, upsert_rollout_record


def _rollout(instance_id: str, *, resolved: bool = True):
    return {
        "run_id": f"run-{instance_id}",
        "instance_id": instance_id,
        "data_source": "swebench-hard",
        "model": "dummy-model",
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


def _detail(instance_id: str):
    return {
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
        "chain_evaluable": False,
        "not_chain_evaluable_reason": "legacy_detail",
        "chain_hit": False,
        "anchor_hit": False,
        "root_hit": False,
        "bad_patterns": {},
        "chain_bad_patterns": {},
        "step_details": [],
        "purpose_blocks": [],
    }


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


def test_unified_dashboard_snapshot_includes_db_model_metrics(tmp_path):
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
            detail=_detail("case-1"),
        )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")

    assert rows[0]["model_label"] == "dummy"
    assert rows[0]["target"] == 2
    assert rows[0]["done"] == 1
    assert rows[0]["resolved_rate"] == 1.0
    assert rows[0]["p2a_read_rate"] == 1.0
    assert rows[0]["cache_hit_rate"] == 50 / 150

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp"))
    assert snapshot["schema_version"] == "p2a_unified_dashboard_v1"
    assert snapshot["model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["model_metrics"][0]["target"] == 2
    assert snapshot["summary"]["counts"]["n_records"] == 1
    assert snapshot["details"][0]["instance_id"] == "case-1"


def test_unified_dashboard_handles_empty_db(tmp_path):
    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=tmp_path / "empty.sqlite"))

    assert snapshot["model_metrics"] == []
    assert snapshot["summary"]["counts"]["n_records"] == 0


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
    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["root_hit"] is True
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
    assert "P2A unified dashboard" in html
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" in html
    assert paths["snapshot"].exists()
    assert paths["app"].exists()
    assert paths["css"].exists()
