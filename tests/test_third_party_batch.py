import asyncio
import json
from pathlib import Path

import pytest

from p2a.third_party_batch import (
    SYSTEM_ERROR_STATUS,
    _system_error_summary,
    load_batch_config,
    resolve_bonus_map_dir,
    run_batch,
    sanitized_config_snapshot,
)
from p2a.third_party_eval import run_batch as run_eval_batch


def test_load_batch_config_defaults_and_dummy_models(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("P2A_SHARED_ROOT", str(shared_root))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(artifacts_root))
    config = load_batch_config(Path("config/third_party_batch.example.yaml"))

    assert config.provider["source"] == "openai_compatible"
    assert config.dataset_name == "swebench-hard"
    assert config.experiment_id == "public-swebench-hard-demo"
    assert config.stage == "smoke"
    assert config.limit == 500
    assert config.rollouts_per_instance == 1
    assert config.per_instance_parallelism == 1
    assert [model.api_name for model in config.models] == [
        "dummy-model-a",
        "dummy-model-b",
    ]
    assert config.db_path == artifacts_root / "evals" / "traces.sqlite"
    assert config.artifacts_dir == artifacts_root / "third_party"


def test_sanitized_config_snapshot_redacts_secret_like_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("P2A_SHARED_ROOT", str(tmp_path / "shared"))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    path = tmp_path / "batch.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
model:
  api_key: should-not-survive
dataset:
  name: swebench-hard
experiment:
  id: demo
models:
  - api_name: dummy-model
storage:
  db: data/evals/traces.sqlite
""",
        encoding="utf-8",
    )
    config = load_batch_config(path)
    snapshot = sanitized_config_snapshot(config)

    assert snapshot["config"]["model"]["api_key"] == "<redacted>"


def test_existing_bonus_map_dir_can_be_used_without_precompute(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("P2A_SHARED_ROOT", str(shared_root))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(artifacts_root))
    path = tmp_path / "batch.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
dataset:
  name: swebench-hard
models:
  - api_name: dummy-model
storage:
  precompute_maps: false
  bonus_map_dir: data/eval_bonus_maps/swebench-hard
""",
        encoding="utf-8",
    )

    config = load_batch_config(path)

    assert config.precompute_maps is False
    assert config.bonus_map_dir == artifacts_root / "eval_bonus_maps" / "swebench-hard"


def test_default_bonus_map_precompute_uses_artifact_root(monkeypatch, tmp_path):
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(artifacts_root))
    path = tmp_path / "batch.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
dataset:
  name: swebench-hard
models:
  - api_name: dummy-model
""",
        encoding="utf-8",
    )
    config = load_batch_config(path)
    calls = []

    def fake_run_setup(args, **_kwargs):
        calls.append(args)
        return args[-1]

    monkeypatch.setattr("p2a.third_party_batch._run_setup", fake_run_setup)

    out = resolve_bonus_map_dir(config, tmp_path / "data.parquet", env={})

    assert out == artifacts_root / "bonus_maps" / "swebench-hard"
    assert calls[0][-1] == str(artifacts_root / "bonus_maps" / "swebench-hard")


def test_batch_config_accepts_swebench_pro_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("P2A_SHARED_ROOT", str(tmp_path / "shared"))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    path = tmp_path / "batch.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
dataset:
  name: swe-bench-pro
models:
  - api_name: dummy-model
storage:
  precompute_maps: false
""",
        encoding="utf-8",
    )

    config = load_batch_config(path)

    assert config.dataset_name == "swebench-pro"


def test_batch_config_parses_rollout_controls(monkeypatch, tmp_path):
    monkeypatch.setenv("P2A_SHARED_ROOT", str(tmp_path / "shared"))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    path = tmp_path / "batch.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
dataset:
  name: swebench-hard
experiment:
  rollouts_per_instance: 8
  per_instance_parallelism: 2
models:
  - api_name: dummy-model
storage:
  precompute_maps: false
""",
        encoding="utf-8",
    )

    config = load_batch_config(path)

    assert config.rollouts_per_instance == 8
    assert config.per_instance_parallelism == 2


def test_eval_batch_expands_rollouts_per_instance(monkeypatch):
    calls = []

    async def fake_run_one(row, *, rollout_index, **_kwargs):
        calls.append((row["instance_id"], rollout_index))
        return {"instance_id": row["instance_id"], "rollout_index": rollout_index}

    monkeypatch.setattr("p2a.third_party_eval.run_one", fake_run_one)

    rows = [{"instance_id": "case-1"}, {"instance_id": "case-2"}]
    records = asyncio.run(
        run_eval_batch(
            rows,
            model_cfg={"model_name": "dummy"},
            agent_cfg={},
            n_parallel=4,
            rollouts_per_instance=3,
            per_instance_parallelism=2,
        )
    )

    assert sorted(calls) == [
        ("case-1", 0),
        ("case-1", 1),
        ("case-1", 2),
        ("case-2", 0),
        ("case-2", 1),
        ("case-2", 2),
    ]
    assert len(records) == 6


def test_batch_config_requires_explicit_models(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
dataset:
  name: swebench-hard
models: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models must be a non-empty list"):
        load_batch_config(path)


def test_system_error_summary_recognizes_all_system_error_rollouts(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        json.dumps(
            {
                "instance_id": "case-1",
                "error": "InvalidStatus: server rejected WebSocket connection: HTTP 403",
                "error_kind": "arl_shell_forbidden",
                "error_stage": "env_start",
                "system_error": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _system_error_summary(rollouts)

    assert summary is not None
    assert summary["n_records"] == 1
    assert summary["error_kinds"] == {"arl_shell_forbidden": 1}
    assert summary["error_stages"] == {"env_start": 1}


def test_run_batch_stops_after_smoke_system_error(monkeypatch, tmp_path):
    monkeypatch.setenv("P2A_SHARED_ROOT", str(tmp_path / "shared"))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    path = tmp_path / "batch.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
dataset:
  name: swebench-hard
experiment:
  id: demo
  stage: both
models:
  - api_name: dummy-model
storage:
  precompute_maps: false
""",
        encoding="utf-8",
    )
    config = load_batch_config(path)
    phases = []

    async def fake_run_model_phase(**kwargs):
        phases.append(kwargs["phase"])
        return {
            "model": kwargs["model"].label,
            "phase": kwargs["phase"],
            "status": SYSTEM_ERROR_STATUS,
        }

    monkeypatch.setattr("p2a.third_party_batch.check_provider_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("p2a.third_party_batch.resolve_data_file", lambda *_args, **_kwargs: tmp_path / "data.parquet")
    monkeypatch.setattr("p2a.third_party_batch.resolve_bonus_map_dir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("p2a.third_party_batch.run_model_phase", fake_run_model_phase)

    results = asyncio.run(run_batch(config, env={}))

    assert phases == ["smoke"]
    assert results[0]["status"] == SYSTEM_ERROR_STATUS


def test_run_batch_tracks_missing_rollout_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("P2A_SHARED_ROOT", str(tmp_path / "shared"))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    path = tmp_path / "batch.yaml"
    path.write_text(
        """
provider:
  source: openai_compatible
dataset:
  name: swebench-hard
experiment:
  id: demo
  stage: full
  limit: 1
  rollouts_per_instance: 2
models:
  - api_name: dummy-model
storage:
  precompute_maps: false
""",
        encoding="utf-8",
    )
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"instance_id": "case-1", "data_source": "swebench-hard"}) + "\n", encoding="utf-8")
    config = load_batch_config(path)
    seen = []

    async def fake_run_subprocess(command, **_kwargs):
        seen.extend(item for index, item in enumerate(command) if index and command[index - 1] == "--rollout-job")
        run_dir = Path(command[command.index("--out") + 1]).parent
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "rollouts.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"run_id": "r0", "instance_id": "case-1", "rollout_index": 0, "data_source": "swebench-hard"}),
                    json.dumps({"run_id": "r1", "instance_id": "case-1", "rollout_index": 1, "data_source": "swebench-hard"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr("p2a.third_party_batch.check_provider_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("p2a.third_party_batch.resolve_data_file", lambda *_args, **_kwargs: data)
    monkeypatch.setattr("p2a.third_party_batch.resolve_bonus_map_dir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("p2a.third_party_batch._run_subprocess", fake_run_subprocess)

    results = asyncio.run(run_batch(config, env={}))

    assert seen == ["case-1:0", "case-1:1"]
    assert results[0]["n_ingested"] == 2
