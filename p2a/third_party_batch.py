"""Batch orchestration for API-backed Uni-Agent/P2A evaluation."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from p2a.api_providers import check_provider_available, normalize_provider_config, provider_source
from p2a.eval_cache import (
    DONE_STATUS,
    ERROR_STATUS,
    completed_instance_ids,
    ensure_db,
    ingest_artifacts,
    mark_cells_running,
    upsert_experiment,
    upsert_planned_cells,
    utc_now,
)
from p2a.third_party_eval import _instance_id, _load_rows, _select_rows, parse_limit_arg


SUPPORTED_DATASETS = {"swebench-hard", "swebench-verified", "r2e-gym-subset"}
REDACT_KEYS = ("api_key", "apikey", "token", "secret", "password", "authorization")


@dataclass(frozen=True)
class BatchModel:
    api_name: str
    label: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class BatchConfig:
    path: Path
    raw: dict[str, Any]
    provider: dict[str, Any]
    dataset_name: str
    dataset_file: Path | None
    experiment_id: str
    stage: str
    limit: int | None
    offset: int
    max_turns: int
    run_timeout: str | None
    per_model_concurrency: int
    model_parallelism: int
    db_path: Path
    artifacts_dir: Path
    precompute_maps: bool
    bonus_map_dir: Path | None
    models: list[BatchModel]


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _parse_limit(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        if value < 0:
            raise ValueError("experiment.limit must be non-negative, or use 'all'")
        return value
    if isinstance(value, str):
        return parse_limit_arg(value)
    raise ValueError("experiment.limit must be an integer or 'all'")


def _canonical_dataset(name: str) -> str:
    aliases = {
        "hard": "swebench-hard",
        "swe-bench-hard": "swebench-hard",
        "verified": "swebench-verified",
        "swe-bench-verified": "swebench-verified",
        "r2e": "r2e-gym-subset",
        "r2e-gym": "r2e-gym-subset",
    }
    canonical = aliases.get(name, name)
    if canonical not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"dataset.name must be one of: {supported}")
    return canonical


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in REDACT_KEYS):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def sanitized_config_snapshot(config: BatchConfig) -> dict[str, Any]:
    return {
        "schema": "p2a_third_party_batch_v1",
        "source_config": str(config.path),
        "captured_at": utc_now(),
        "config": _redact(config.raw),
    }


def load_batch_config(path: Path) -> BatchConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    provider_cfg = normalize_provider_config(_as_mapping(payload.get("provider"), name="provider"))
    dataset_cfg = _as_mapping(payload.get("dataset"), name="dataset")
    experiment_cfg = _as_mapping(payload.get("experiment"), name="experiment")
    storage_cfg = _as_mapping(payload.get("storage"), name="storage")

    dataset_name = _canonical_dataset(str(dataset_cfg.get("name") or "swebench-hard"))
    dataset_file = Path(dataset_cfg["file"]).expanduser() if dataset_cfg.get("file") else None
    stage = str(experiment_cfg.get("stage") or "smoke")
    if stage not in {"smoke", "full", "both"}:
        raise ValueError("experiment.stage must be smoke, full, or both")

    models_payload = payload.get("models")
    if not isinstance(models_payload, list) or not models_payload:
        raise ValueError("models must be a non-empty list")
    models = []
    for index, item in enumerate(models_payload):
        if not isinstance(item, dict):
            raise ValueError(f"models[{index}] must be a mapping")
        api_name = str(item.get("api_name") or item.get("model") or "").strip()
        if not api_name:
            raise ValueError(f"models[{index}].api_name is required")
        label = str(item.get("label") or api_name).strip()
        overrides = {k: copy.deepcopy(v) for k, v in item.items() if k not in {"api_name", "model", "label"}}
        models.append(BatchModel(api_name=api_name, label=label, overrides=overrides))

    artifacts_dir = Path(storage_cfg.get("artifacts_dir") or "data/third_party").expanduser()
    return BatchConfig(
        path=path,
        raw=payload,
        provider=provider_cfg,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        experiment_id=str(experiment_cfg.get("id") or path.stem),
        stage=stage,
        limit=_parse_limit(experiment_cfg.get("limit"), default=500),
        offset=int(experiment_cfg.get("offset") or 0),
        max_turns=int(experiment_cfg.get("max_turns") or 20),
        run_timeout=str(experiment_cfg["run_timeout"]) if experiment_cfg.get("run_timeout") else None,
        per_model_concurrency=max(1, int(experiment_cfg.get("per_model_concurrency") or 1)),
        model_parallelism=max(1, int(experiment_cfg.get("model_parallelism") or len(models))),
        db_path=Path(storage_cfg.get("db") or "data/evals/traces.sqlite").expanduser(),
        artifacts_dir=artifacts_dir,
        precompute_maps=bool(storage_cfg.get("precompute_maps", True)),
        bonus_map_dir=Path(storage_cfg["bonus_map_dir"]).expanduser() if storage_cfg.get("bonus_map_dir") else None,
        models=models,
    )


def _phase_specs(config: BatchConfig) -> list[tuple[str, int | None]]:
    if config.stage == "smoke":
        return [("smoke", 1)]
    if config.stage == "full":
        return [("full", config.limit)]
    return [("smoke", 1), ("full", config.limit)]


def _duration_seconds(value: str | None) -> float | None:
    if not value or value == "0":
        return None
    stripped = value.strip().lower()
    unit = stripped[-1]
    if unit in {"s", "m", "h"}:
        amount = float(stripped[:-1])
        return amount * {"s": 1, "m": 60, "h": 3600}[unit]
    return float(stripped)


def _safe_slug(value: str) -> str:
    slug = value.replace("/", "_").replace(":", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in slug)


def _run_setup(args: list[str], *, env: dict[str, str]) -> str:
    import subprocess

    proc = subprocess.run(
        ["bash", "scripts/setup.sh", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip() or f"scripts/setup.sh {' '.join(args)} failed")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def resolve_data_file(config: BatchConfig, *, env: dict[str, str]) -> Path:
    if config.dataset_file is not None:
        return config.dataset_file
    return Path(_run_setup(["data", config.dataset_name], env=env))


def resolve_bonus_map_dir(config: BatchConfig, data_file: Path, *, env: dict[str, str]) -> Path | None:
    if not config.precompute_maps:
        return None
    if config.bonus_map_dir is not None:
        output_dir = config.bonus_map_dir
    else:
        output_dir = config.artifacts_dir / config.experiment_id / "bonus_maps" / config.dataset_name
    setup_env = dict(env)
    if config.limit is not None:
        setup_env.setdefault("P2A_SETUP_BONUS_LIMIT", str(config.limit))
    setup_env.setdefault("P2A_SETUP_BONUS_OFFSET", str(config.offset))
    return Path(_run_setup(["maps", config.dataset_name, str(data_file), str(output_dir)], env=setup_env))


def selected_instance_ids(data_file: Path, *, limit: int | None, offset: int) -> list[str]:
    rows = _select_rows(_load_rows(data_file), limit=limit, offset=offset, instance_ids=None)
    instance_ids = []
    for row in rows:
        instance_id = _instance_id(row)
        if not instance_id:
            raise ValueError(f"selected row at offset {offset} has no instance_id")
        instance_ids.append(instance_id)
    return instance_ids


def _model_eval_config(config: BatchConfig, model: BatchModel, run_dir: Path) -> Path:
    payload: dict[str, Any] = {}
    for key in ("agent", "analysis"):
        if key in config.raw:
            payload[key] = copy.deepcopy(config.raw[key])
    payload["provider"] = copy.deepcopy(config.provider)

    model_cfg = copy.deepcopy(config.raw.get("model") or {})
    model_cfg.update(copy.deepcopy(model.overrides.get("model", {})))
    for key, value in model.overrides.items():
        if key != "model":
            model_cfg[key] = copy.deepcopy(value)
    model_cfg["model_name"] = model.api_name
    model_cfg["api_name"] = model.api_name
    payload["model"] = model_cfg

    config_path = run_dir / "third_party_eval.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _base_command(
    *,
    eval_config: Path,
    data_file: Path,
    run_dir: Path,
    missing_ids: list[str],
    config: BatchConfig,
    bonus_map_dir: Path | None,
) -> list[str]:
    command = [
        "bash",
        "scripts/third_party_eval.sh",
        "--config",
        str(eval_config),
        "--data",
        str(data_file),
        "--out",
        str(run_dir / "rollouts.jsonl"),
        "--limit",
        "all",
        "--offset",
        "0",
        "--n-parallel",
        str(config.per_model_concurrency),
        "--max-turns",
        str(config.max_turns),
        "--summary-out",
        str(run_dir / "summary.json"),
        "--details-out",
        str(run_dir / "details.jsonl"),
        "--report-out",
        str(run_dir / "report.md"),
    ]
    if bonus_map_dir is not None:
        command.extend(["--bonus-map-dir", str(bonus_map_dir)])
    for instance_id in missing_ids:
        command.extend(["--instance-id", instance_id])
    return command


async def _run_subprocess(command: list[str], *, env: dict[str, str], timeout_s: float | None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        stdout, _ = await proc.communicate()
        return 124, stdout.decode("utf-8", errors="replace")
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


def _mark_missing_error(
    db_path: Path,
    *,
    config: BatchConfig,
    model: BatchModel,
    instance_ids: list[str],
    error: str,
) -> None:
    with ensure_db(db_path) as conn:
        now = utc_now()
        conn.executemany(
            """
            UPDATE run_cells
            SET status = ?, attempts = attempts + 1, ended_at = ?, error = ?, updated_at = ?
            WHERE experiment_id = ?
              AND provider_source = ?
              AND model_api_name = ?
              AND dataset = ?
              AND instance_id = ?
            """,
            [
                (
                    ERROR_STATUS,
                    now,
                    error[-4000:],
                    now,
                    config.experiment_id,
                    provider_source(config.provider),
                    model.api_name,
                    config.dataset_name,
                    instance_id,
                )
                for instance_id in instance_ids
            ],
        )
        conn.commit()


async def run_model_phase(
    *,
    config: BatchConfig,
    model: BatchModel,
    phase: str,
    phase_limit: int | None,
    data_file: Path,
    bonus_map_dir: Path | None,
    env: dict[str, str],
) -> dict[str, Any]:
    source = provider_source(config.provider)
    target_ids = selected_instance_ids(data_file, limit=phase_limit, offset=config.offset)
    with ensure_db(config.db_path) as conn:
        upsert_experiment(
            conn,
            experiment_id=config.experiment_id,
            provider_source=source,
            dataset=config.dataset_name,
            config_snapshot=sanitized_config_snapshot(config),
        )
        upsert_planned_cells(
            conn,
            experiment_id=config.experiment_id,
            provider_source=source,
            model_api_name=model.api_name,
            model_label=model.label,
            dataset=config.dataset_name,
            instance_ids=target_ids,
        )
        done_ids = completed_instance_ids(
            conn,
            experiment_id=config.experiment_id,
            provider_source=source,
            model_api_name=model.api_name,
            dataset=config.dataset_name,
        )
        missing_ids = [instance_id for instance_id in target_ids if instance_id not in done_ids]
        mark_cells_running(
            conn,
            experiment_id=config.experiment_id,
            provider_source=source,
            model_api_name=model.api_name,
            dataset=config.dataset_name,
            instance_ids=missing_ids,
        )
        conn.commit()

    if not missing_ids:
        return {"model": model.label, "phase": phase, "status": "skipped", "n_missing": 0}

    run_dir = config.artifacts_dir / config.experiment_id / phase / config.dataset_name / _safe_slug(model.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_config = _model_eval_config(config, model, run_dir)
    command = _base_command(
        eval_config=eval_config,
        data_file=data_file,
        run_dir=run_dir,
        missing_ids=missing_ids,
        config=config,
        bonus_map_dir=bonus_map_dir,
    )
    model_env = dict(env)
    model_env["P2A_THIRD_PARTY_MODEL"] = model.api_name

    returncode, output = await _run_subprocess(command, env=model_env, timeout_s=_duration_seconds(config.run_timeout))
    log_path = run_dir / "run.log"
    log_path.write_text(output, encoding="utf-8")
    if returncode != 0:
        _mark_missing_error(
            config.db_path,
            config=config,
            model=model,
            instance_ids=missing_ids,
            error=f"third_party_eval exited {returncode}; see {log_path}",
        )
        return {
            "model": model.label,
            "phase": phase,
            "status": "error",
            "returncode": returncode,
            "log": str(log_path),
            "n_missing": len(missing_ids),
        }

    with ensure_db(config.db_path) as conn:
        n_ingested = ingest_artifacts(
            conn,
            experiment_id=config.experiment_id,
            provider_source=source,
            model_api_name=model.api_name,
            model_label=model.label,
            dataset=config.dataset_name,
            rollouts_path=run_dir / "rollouts.jsonl",
            details_path=run_dir / "details.jsonl",
        )
        conn.commit()
    return {
        "model": model.label,
        "phase": phase,
        "status": DONE_STATUS,
        "n_missing": len(missing_ids),
        "n_ingested": n_ingested,
        "run_dir": str(run_dir),
    }


async def run_batch(config: BatchConfig, *, env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    run_env = dict(os.environ if env is None else env)
    check_provider_available(config.provider, repo_root=Path.cwd())
    data_file = resolve_data_file(config, env=run_env)
    bonus_map_dir = resolve_bonus_map_dir(config, data_file, env=run_env)
    results = []
    semaphore = asyncio.Semaphore(config.model_parallelism)

    async def guarded(model: BatchModel, phase: str, limit: int | None) -> dict[str, Any]:
        async with semaphore:
            print(f"[batch] {phase}: {model.label} ({model.api_name})", flush=True)
            result = await run_model_phase(
                config=config,
                model=model,
                phase=phase,
                phase_limit=limit,
                data_file=data_file,
                bonus_map_dir=bonus_map_dir,
                env=run_env,
            )
            print(f"[batch] {phase}: {model.label} -> {result['status']}", flush=True)
            return result

    for phase, limit in _phase_specs(config):
        phase_results = await asyncio.gather(*(guarded(model, phase, limit) for model in config.models))
        results.extend(phase_results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--batch", dest="config", type=Path, required=True)
    parser.add_argument("--db", type=Path, help="Override storage.db from the batch config")
    args = parser.parse_args()

    config = load_batch_config(args.config)
    if args.db is not None:
        config = BatchConfig(**{**config.__dict__, "db_path": args.db})
    results = asyncio.run(run_batch(config))
    print(json.dumps({"db": str(config.db_path), "results": results}, indent=2))
    return 0 if all(result["status"] in {DONE_STATUS, "skipped"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
