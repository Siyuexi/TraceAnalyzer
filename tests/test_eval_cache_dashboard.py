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
        "chain_evaluable": True,
        "not_chain_evaluable_reason": None,
        "chain_graph_covered": True,
        "chain_hit": True,
        "anchor_hit": True,
        "root_hit": True,
        "chain_node_recall": 1.0,
        "chain_read_precision": 1.0,
        "first_anchor_step": 0,
        "first_root_step": 0,
        "steps_anchor_to_root": 0,
        "anchor_before_root": True,
        "bad_patterns": {"has_loop": False, "error_spiral": False},
        "chain_bad_patterns": {
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
    assert rows[0]["avg_read_precision"] == 1.0
    assert rows[0]["avg_node_recall"] == 1.0
    assert rows[0]["avg_hit_f1"] == 1.0
    assert rows[0]["anchor_hit_rate"] == 1.0
    assert rows[0]["root_hit_rate"] == 1.0
    assert rows[0]["block_achieve_rate"] == 1.0
    assert rows[0]["cache_hit_rate"] == 50 / 150

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp"))
    assert snapshot["schema_version"] == "p2a_unified_dashboard_v1"
    assert snapshot["datasets"][0]["dataset"] == "swebench-hard"
    assert snapshot["eval_cells"][0]["experiment_id"] == "exp"
    assert snapshot["experiments"][0]["experiment_id"] == "exp"
    assert snapshot["experiments"][0]["source_kind"] == "third_party_api"
    assert snapshot["model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["model_metrics"][0]["target"] == 2
    assert snapshot["model_metrics"][0]["avg_read_precision"] == 1.0
    assert snapshot["summary"]["counts"]["n_records"] == 1
    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["experiment_key"] == snapshot["experiments"][0]["experiment_key"]
    assert snapshot["details"][0]["step_inspection"][0]["tool_names"] == ["execute_bash"]
    assert snapshot["details"][0]["step_inspection"][0]["action_family"] == "read"
    assert snapshot["details"][0]["step_inspection"][0]["recovered_reads"][0]["file_path"] == "a.py"


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
    assert "selectedExperimentKey" in app
    assert "selectedEvalCellKey" in app
    assert "renderGraph" in app
    assert "step_inspection" in app
    assert "<details class" not in app
    assert paths["snapshot"].exists()
    assert paths["app"].exists()
    assert paths["css"].exists()
