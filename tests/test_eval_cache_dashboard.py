from p2a.eval_cache import aggregate_model_metrics, ensure_db, upsert_experiment, upsert_planned_cells, upsert_rollout_record
from p2a.eval_dashboard import render_db_dashboard, render_model_dashboard


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
        "instance_id": instance_id,
        "n_reads": 1,
        "hit_call_graph": True,
        "hit_ground_truth": True,
        "hit_near": True,
        "min_distance": 0.0,
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


def test_eval_cache_aggregates_model_metrics_and_dashboard(tmp_path):
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

    rendered = "\n".join(render_db_dashboard(db, experiment_id="exp", term_cols=120))
    assert "dummy" in rendered
    assert "progress 1/2" in rendered
    assert "p2a" in rendered


def test_dashboard_handles_empty_db(tmp_path):
    db = tmp_path / "empty.sqlite"
    lines = render_db_dashboard(db, term_cols=80)

    assert lines == ["No evaluation rows found in the DB."]


def test_render_model_dashboard_orders_models_by_resolved_rate():
    lines = render_model_dashboard(
        [
            {"model_label": "worse", "target": 1, "done": 1, "errors": 0, "resolved_rate": 0.0},
            {"model_label": "better", "target": 1, "done": 1, "errors": 0, "resolved_rate": 1.0},
        ],
        term_cols=100,
    )

    assert "\n".join(lines).find("better") < "\n".join(lines).find("worse")
