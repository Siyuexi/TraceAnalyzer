from pathlib import Path

import pytest

from p2a.third_party_batch import load_batch_config, sanitized_config_snapshot


def test_load_batch_config_defaults_and_dummy_models():
    config = load_batch_config(Path("config/third_party_batch.example.yaml"))

    assert config.provider["source"] == "openai_compatible"
    assert config.dataset_name == "swebench-hard"
    assert config.experiment_id == "public-swebench-hard-demo"
    assert config.stage == "smoke"
    assert config.limit == 500
    assert [model.api_name for model in config.models] == [
        "dummy-model-a",
        "dummy-model-b",
    ]
    assert config.db_path.as_posix() == "data/evals/traces.sqlite"


def test_sanitized_config_snapshot_redacts_secret_like_keys(tmp_path):
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
