import asyncio
import json
from pathlib import Path

import pytest

from p2a.third_party_batch import SYSTEM_ERROR_STATUS, _system_error_summary, load_batch_config, run_batch, sanitized_config_snapshot


def test_load_batch_config_loads_public_internal_api_example(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("P2A_SHARED_ROOT", str(shared_root))
    monkeypatch.setenv("P2A_ARTIFACTS_DIR", str(artifacts_root))
    config = load_batch_config(Path("config/third_party_batch.example.yaml"))

    assert config.provider["source"] == "internal_api"
    assert config.provider["api_module"] == ".secrets/internal_api_eval.py"
    assert config.dataset_name == "swebench-hard"
    assert config.experiment_id == "public-swebench-hard-demo"
    assert config.stage == "smoke"
    assert config.limit is None
    assert config.max_turns == 100
    assert [model.api_name for model in config.models] == [
        "deepseek-v4-flash-passthrough",
    ]
    assert config.models[0].overrides["sampling_params"]["max_completion_tokens"] == 384000
    assert config.db_path == artifacts_root / "evals" / "traces.sqlite"
    assert config.artifacts_dir == artifacts_root / "third_party"


def test_sanitized_config_snapshot_redacts_secret_like_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("P2A_SHARED_ROOT", str(tmp_path / "shared"))
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
  bonus_map_dir: data/bonus_maps/swebench-hard
""",
        encoding="utf-8",
    )

    config = load_batch_config(path)

    assert config.precompute_maps is False
    assert config.bonus_map_dir == artifacts_root / "bonus_maps" / "swebench-hard"


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
