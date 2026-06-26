"""Offline OpenAI-compatible Uni-Agent rollout harness for P2A analysis."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import shlex
import time
from typing import Any
import uuid

import yaml

from p2a.api_providers import make_chat_model, normalize_provider_config, provider_source
from p2a.core import BonusMapStore
from p2a.eval_cache import ensure_db, ingest_artifacts, upsert_experiment
from p2a.eval_fault_localization import (
    _json_default,
    iter_records,
    score_record,
    summarize,
    write_jsonl,
)
from p2a.precompute.uni_agent_sandbox import build_agent_env_config, extract_tools_kwargs


DEFAULT_CONFIG = {
    "experiment": {
        "rollouts_per_instance": 1,
        "per_instance_parallelism": 1,
    },
    "model": {
        "base_url_env": "P2A_THIRD_PARTY_BASE_URL",
        "api_key_env": "P2A_THIRD_PARTY_API_KEY",
        "model_name_env": "P2A_THIRD_PARTY_MODEL",
        "base_url": "",
        "model_name": "",
        "timeout": 300,
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    },
    "agent": {
        "deployment": "arl",
        "tool_parser": "qwen3_coder",
        "tools": [
            {"name": "str_replace_editor"},
            {"name": "execute_bash"},
            {"name": "submit"},
        ],
        "interaction": {
            "action_timeout": 300,
            "timeout_budget": 3,
            "max_turns": 100,
        },
        "tool_install_timeout": 300,
        "skip_tool_install_commands": [],
        "reward_eval_timeout": 600,
        "log_dir": "/tmp/p2a_third_party_eval",
    },
    "analysis": {
        "tracking_mode": "view_and_bash",
        "near_threshold": 0.5,
        "m_max": 3.0,
    },
}
_REDACT_KEYS = ("api_key", "apikey", "token", "secret", "password", "authorization")
SYSTEM_ERROR_KINDS = {
    "arl_config_missing",
    "arl_shell_forbidden",
    "arl_shell_unavailable",
    "arl_gateway_unreachable",
    "image_pull_failed",
    "network_error",
    "runtime_timeout",
}


def classify_error(error: BaseException | str | None) -> str | None:
    if error is None:
        return None
    text = str(error)
    lowered = text.lower()
    if "arl_gateway_url" in lowered:
        return "arl_config_missing"
    if "websocket" in lowered and ("403" in lowered or "forbidden" in lowered or "rejected" in lowered):
        return "arl_shell_forbidden"
    if "interactive shell" in lowered or "/shell" in lowered:
        return "arl_shell_unavailable"
    if any(token in lowered for token in ("imagepullbackoff", "errimagepull", "pull access denied")):
        return "image_pull_failed"
    if "arl" in lowered and any(token in lowered for token in ("connection", "connect", "gateway", "timeout", "refused")):
        return "arl_gateway_unreachable"
    if any(token in lowered for token in ("network", "connection refused", "connection reset", "temporary failure")):
        return "network_error"
    if "timeout" in lowered or "timed out" in lowered:
        return "runtime_timeout"
    return "runtime_error"


def is_system_error_kind(kind: str | None) -> bool:
    return kind in SYSTEM_ERROR_KINDS


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _REDACT_KEYS):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_config(item)
        return redacted
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return _deep_merge(DEFAULT_CONFIG, payload)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    model_cfg = config.setdefault("model", {})
    if getattr(args, "base_url", None):
        model_cfg["base_url"] = args.base_url
    if getattr(args, "model_name", None):
        model_cfg["model_name"] = args.model_name
    if getattr(args, "model_timeout", None) is not None:
        model_cfg["timeout"] = args.model_timeout

    sampling_cfg = model_cfg.setdefault("sampling_params", {})
    if getattr(args, "max_tokens", None) is not None:
        sampling_cfg["max_tokens"] = args.max_tokens
    if getattr(args, "temperature", None) is not None:
        sampling_cfg["temperature"] = args.temperature

    agent_cfg = config.setdefault("agent", {})
    interaction_cfg = agent_cfg.setdefault("interaction", {})
    if getattr(args, "max_turns", None) is not None:
        interaction_cfg["max_turns"] = args.max_turns
    if getattr(args, "action_timeout", None) is not None:
        interaction_cfg["action_timeout"] = args.action_timeout
    if getattr(args, "reward_eval_timeout", None) is not None:
        agent_cfg["reward_eval_timeout"] = args.reward_eval_timeout
    if getattr(args, "tool_install_timeout", None) is not None:
        agent_cfg["tool_install_timeout"] = args.tool_install_timeout
    if getattr(args, "skip_tool_install", None):
        agent_cfg["skip_tool_install_commands"] = args.skip_tool_install
    experiment_cfg = config.setdefault("experiment", {})
    if getattr(args, "rollouts_per_instance", None) is not None:
        experiment_cfg["rollouts_per_instance"] = args.rollouts_per_instance
    if getattr(args, "per_instance_parallelism", None) is not None:
        experiment_cfg["per_instance_parallelism"] = args.per_instance_parallelism
    return config


def _env_or_value(config: dict[str, Any], key: str, env_key: str | None = None) -> str:
    env_name = str(config.get(env_key or f"{key}_env") or "")
    if env_name:
        value = os.getenv(env_name)
        if value:
            return value
    value = config.get(key)
    return str(value or "")


def resolve_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config.get("model") or {})
    provider_cfg = normalize_provider_config(config.get("provider"))
    base_url = _env_or_value(model_cfg, "base_url")
    api_key = _env_or_value(model_cfg, "api_key")
    model_name = _env_or_value(model_cfg, "model_name")
    if provider_cfg["source"] == "openai_compatible" and not base_url:
        raise ValueError("model.base_url is required, either directly or via model.base_url_env")
    if provider_cfg["source"] == "openai_compatible" and not api_key:
        raise ValueError("model.api_key is required via model.api_key_env; do not commit API keys")
    if not model_name:
        raise ValueError("model.model_name is required, either directly or via model.model_name_env")
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
        "timeout": model_cfg.get("timeout", 300),
        "sampling_params": dict(model_cfg.get("sampling_params") or {}),
    }


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return value


def _as_jsonable(value: Any) -> Any:
    value = _maybe_json(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_as_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _as_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return {str(k): _as_jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        records = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("records", [])
    return [_as_jsonable(record) for record in records]


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    offset: int,
    instance_ids: set[str] | None,
) -> list[dict[str, Any]]:
    if instance_ids:
        rows = [row for row in rows if _instance_id(row) in instance_ids]
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def parse_limit_arg(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--limit must be a non-negative integer or 'all'") from exc
    if limit < 0:
        raise argparse.ArgumentTypeError("--limit must be a non-negative integer or 'all'")
    return limit


def _positive_int(value: Any, *, default: int, name: str) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def parse_rollout_job(value: str) -> tuple[str, int]:
    instance_id, sep, raw_index = str(value).rpartition(":")
    if not sep or not instance_id:
        raise argparse.ArgumentTypeError("--rollout-job must be formatted as <instance_id>:<rollout_index>")
    try:
        rollout_index = int(raw_index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rollout-job rollout_index must be an integer") from exc
    if rollout_index < 0:
        raise argparse.ArgumentTypeError("--rollout-job rollout_index must be non-negative")
    return instance_id, rollout_index


def _extra_info(row: dict[str, Any]) -> dict[str, Any]:
    value = _maybe_json(row.get("extra_info"))
    return value if isinstance(value, dict) else {}


def _instance_id(row: dict[str, Any]) -> str | None:
    for value in (
        row.get("instance_id"),
        _extra_info(row).get("instance_id"),
        (extract_tools_kwargs(row).get("reward") or {}).get("metadata", {}).get("instance_id"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _data_source(row: dict[str, Any]) -> str:
    for value in (row.get("data_source"), _extra_info(row).get("data_source")):
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _prompt(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = _as_jsonable(_maybe_json(row.get("prompt")))
    if isinstance(value, list) and value:
        return [_as_jsonable(item) for item in value if isinstance(item, dict)]
    raise ValueError(f"Row {_instance_id(row) or '<unknown>'} has no usable prompt list")


def _make_env(row: dict[str, Any], *, instance_id: str, deployment: str):
    from env.deployment import make_env_config
    from uni_agent.interaction import AgentEnv, AgentEnvConfig

    env_dict = build_agent_env_config(row, instance_id=instance_id, deployment=deployment)
    if env_dict["deployment"].get("type") == "arl":
        env_dict["deployment"]["require_interactive_shell"] = True
        env_config = make_env_config(
            env_dict["deployment"],
            env_variables=env_dict.get("env_variables"),
            post_setup_cmd=env_dict.get("post_setup_cmd"),
            tool_install_dir=env_dict.get("tool_install_dir", "/usr/local/bin"),
        )
    else:
        env_config = AgentEnvConfig(**env_dict)
    return AgentEnv(run_id=f"p2a-third-party-{uuid.uuid4()}", env_config=env_config)


def _make_tools(agent_cfg: dict[str, Any]):
    from uni_agent.interaction import ToolsManager, ToolsManagerConfig

    return ToolsManager(
        ToolsManagerConfig(
            tools=agent_cfg.get("tools") or DEFAULT_CONFIG["agent"]["tools"],
            parser=agent_cfg.get("tool_parser", "qwen3_coder"),
        )
    )


def _make_model(model_cfg: dict[str, Any], provider_cfg: dict[str, Any] | None = None):
    return make_chat_model(model_cfg, provider_cfg)


def _make_interaction(*, run_id: str, env: Any, model: Any, tools_manager: Any, messages: list[dict], agent_cfg: dict):
    from uni_agent.interaction import AgentInteraction

    interaction_cfg = dict(agent_cfg.get("interaction") or {})
    return AgentInteraction(
        run_id=run_id,
        env=env,
        model=model,
        tools_manager=tools_manager,
        messages=messages,
        **interaction_cfg,
    )


def _make_reward(row: dict[str, Any], *, run_id: str, env: Any, agent_cfg: dict[str, Any]):
    from uni_agent.reward import load_reward_spec

    import p2a.reward_specs  # noqa: F401 - registers P2A-local Uni-Agent reward specs

    reward_cfg = dict(extract_tools_kwargs(row).get("reward") or {})
    if not reward_cfg:
        return None
    reward_cfg["run_id"] = run_id
    reward_cfg["env"] = env
    reward_cfg["eval_timeout"] = agent_cfg.get("reward_eval_timeout", reward_cfg.get("eval_timeout", 600))
    return load_reward_spec(reward_cfg)


async def _install_tools(
    env: Any,
    tools: list[Any],
    *,
    timeout: int | float,
    skip_install_commands: list[str] | set[str],
) -> None:
    skip_install_commands = set(skip_install_commands)
    install_dir = env.tool_install_dir
    await env.communicate(f"export PATH={shlex.quote(install_dir.as_posix())}:$PATH", timeout=timeout, check="raise")
    for tool in tools:
        tool_name = tool.name
        if tool.copy_to_remote:
            local_tool_path = tool.local_path
            if local_tool_path is None or not local_tool_path.is_file():
                raise FileNotFoundError(f"Tool {tool_name} has no local executable at {local_tool_path!r}")
            container_tool_path = install_dir / tool_name
            await env.copy_to_container(src=local_tool_path, tgt=container_tool_path)
            await env.communicate(f"chmod +x {container_tool_path.as_posix()}", timeout=timeout, check="raise")
        install_cmd = None if tool_name in skip_install_commands else tool.get_install_command()
        if install_cmd:
            await env.communicate(install_cmd, timeout=timeout, check="raise")
        await env.communicate(
            f"which {tool_name}",
            timeout=timeout,
            check="raise",
            error_msg=f"Failed to install tool {tool_name}",
        )


def build_step_traces(interaction_result: dict[str, Any]) -> list[dict[str, Any]]:
    messages = interaction_result.get("messages") or []
    assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    traces = []
    for idx, step in enumerate(interaction_result.get("trajectory") or []):
        step_data = _as_jsonable(step)
        message = assistant_messages[idx] if idx < len(assistant_messages) else {}
        traces.append(
            {
                "step_idx": int(step_data.get("step_idx", idx + 1)),
                "response_text": step_data.get("response") or message.get("content") or "",
                "thought": step_data.get("thought") or "",
                "tool_calls": _as_jsonable(message.get("tool_calls") or []),
                "tool_results": _as_jsonable(step_data.get("tool_results") or []),
                "exit_reason": step_data.get("exit_reason"),
            }
        )
    return traces


def build_dump_record(
    row: dict[str, Any],
    *,
    run_id: str,
    model_name: str,
    base_url: str,
    rollout_index: int = 0,
    interaction_result: dict[str, Any] | None,
    reward_score: Any,
    reward_details: Any,
    error: str | None = None,
    error_kind: str | None = None,
    error_stage: str | None = None,
) -> dict[str, Any]:
    instance_id = _instance_id(row)
    extra_info = _extra_info(row)
    if instance_id:
        extra_info.setdefault("instance_id", instance_id)
    extra_info.setdefault("data_source", _data_source(row))

    trajectory = _as_jsonable((interaction_result or {}).get("trajectory") or [])
    messages = _as_jsonable((interaction_result or {}).get("messages") or [])
    traces = build_step_traces(interaction_result or {})
    responses = [trace["response_text"] for trace in traces if trace.get("response_text")]
    termination_reason = trajectory[-1].get("exit_reason") if trajectory else "error" if error else "unknown"

    return {
        "schema_version": "p2a_third_party_rollout_v1",
        "run_id": run_id,
        "rollout_index": rollout_index,
        "rollout_id": f"{instance_id}:{rollout_index}" if instance_id is not None else str(rollout_index),
        "instance_id": instance_id,
        "data_source": _data_source(row),
        "model": model_name,
        "base_url": base_url,
        "messages": messages,
        "trajectory": trajectory,
        "p2a_step_traces": traces,
        "response_text": "\n".join(responses),
        "reward": reward_score,
        "reward_details": _as_jsonable(reward_details),
        "resolved": bool(reward_details.get("resolved")) if isinstance(reward_details, dict) else None,
        "termination_reason": termination_reason,
        "execution_time": (interaction_result or {}).get("execution_time"),
        "metrics": _as_jsonable((interaction_result or {}).get("rollout_cache", {}).get("metrics", {})),
        "token_usage": _as_jsonable((interaction_result or {}).get("rollout_cache", {}).get("token_usage", {})),
        "extra_info": extra_info,
        "error": error,
        "error_kind": error_kind,
        "error_stage": error_stage,
        "system_error": is_system_error_kind(error_kind),
    }


async def run_provider_smoke(model_cfg: dict[str, Any], provider_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    model = _make_model(
        {
            **model_cfg,
            "sampling_params": {
                **dict(model_cfg.get("sampling_params") or {}),
                "max_tokens": min(int((model_cfg.get("sampling_params") or {}).get("max_tokens", 16)), 16),
            },
        },
        provider_cfg,
    )
    messages = [
        {"role": "system", "content": "You are a terse smoke-test assistant."},
        {"role": "user", "content": "Reply with exactly: ok"},
    ]
    rollout_cache = await model.prepare_rollout_cache(messages)
    text, tool_calls, _cache, generation_info = await model.query(messages, rollout_cache)
    return {
        "ok": True,
        "has_text": bool(text.strip()),
        "text_preview": text.strip()[:80],
        "n_tool_calls": len(tool_calls),
        "generation_info": generation_info,
    }


async def run_one(
    row: dict[str, Any],
    *,
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    provider_cfg: dict[str, Any] | None = None,
    rollout_index: int = 0,
) -> dict[str, Any]:
    instance_id = _instance_id(row)
    run_id = f"p2a-third-party-{uuid.uuid4()}"
    env = None
    interaction_result: dict[str, Any] | None = None
    reward_score = None
    reward_details = None
    error = None
    error_kind = None
    error_stage = None
    t0 = time.perf_counter()

    async def execute_rollout() -> None:
        nonlocal env, error_stage, interaction_result, reward_details, reward_score, run_id
        if not instance_id:
            raise ValueError("sample row does not carry instance_id")
        error_stage = "env_config"
        env = _make_env(row, instance_id=instance_id, deployment=agent_cfg.get("deployment", "arl"))
        run_id = getattr(getattr(env, "deployment", None), "run_id", run_id)
        error_stage = "tool_config"
        tools_manager = _make_tools(agent_cfg)
        error_stage = "model_config"
        model = _make_model(model_cfg, provider_cfg)
        model.set_tools_schemas(tools_manager.tools_schemas)
        error_stage = "interaction_config"
        interaction = _make_interaction(
            run_id=run_id,
            env=env,
            model=model,
            tools_manager=tools_manager,
            messages=_prompt(row),
            agent_cfg=agent_cfg,
        )
        error_stage = "reward_config"
        reward_spec = _make_reward(row, run_id=run_id, env=env, agent_cfg=agent_cfg)
        error_stage = "env_start"
        await env.start()
        error_stage = "tool_install"
        await _install_tools(
            env,
            tools_manager.tools,
            timeout=agent_cfg.get("tool_install_timeout", 300),
            skip_install_commands=agent_cfg.get("skip_tool_install_commands") or [],
        )
        error_stage = "interaction"
        interaction_result = await interaction.run()
        if reward_spec is not None:
            error_stage = "reward"
            reward_score, reward_details = await reward_spec.compute_reward(interaction_result=interaction_result)
        error_stage = None

    try:
        await execute_rollout()
    except Exception as exc:  # noqa: BLE001 - dump errors per instance and keep the batch moving
        error = f"{type(exc).__name__}: {exc}"
        error_kind = classify_error(exc)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
    record = build_dump_record(
        row,
        run_id=run_id,
        model_name=model_cfg["model_name"],
        base_url=model_cfg.get("base_url") or provider_source(provider_cfg),
        rollout_index=rollout_index,
        interaction_result=interaction_result,
        reward_score=reward_score,
        reward_details=reward_details,
        error=error,
        error_kind=error_kind,
        error_stage=error_stage,
    )
    record["wall_time"] = time.perf_counter() - t0
    return record


async def run_batch(
    rows: list[dict[str, Any]],
    *,
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    n_parallel: int,
    provider_cfg: dict[str, Any] | None = None,
    rollouts_per_instance: int = 1,
    per_instance_parallelism: int = 1,
    rollout_jobs: list[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(n_parallel, 1))
    row_by_id = {_instance_id(row): row for row in rows}
    if rollout_jobs:
        jobs = [(row_by_id[instance_id], rollout_index) for instance_id, rollout_index in rollout_jobs if instance_id in row_by_id]
    else:
        jobs = [(row, rollout_index) for row in rows for rollout_index in range(max(1, int(rollouts_per_instance or 1)))]
    instance_semaphores: dict[str, asyncio.Semaphore] = {}

    def instance_semaphore(row: dict[str, Any]) -> asyncio.Semaphore:
        instance_id = _instance_id(row) or f"row-{id(row)}"
        if instance_id not in instance_semaphores:
            instance_semaphores[instance_id] = asyncio.Semaphore(max(1, int(per_instance_parallelism or 1)))
        return instance_semaphores[instance_id]

    async def guarded(row: dict[str, Any], rollout_index: int) -> dict[str, Any]:
        async with semaphore:
            async with instance_semaphore(row):
                return await run_one(
                    row,
                    model_cfg=model_cfg,
                    agent_cfg=agent_cfg,
                    provider_cfg=provider_cfg,
                    rollout_index=rollout_index,
                )

    return await asyncio.gather(*(guarded(row, rollout_index) for row, rollout_index in jobs))


def write_analysis(
    *,
    rollouts: Path,
    bonus_map_dir: Path,
    summary_out: Path | None,
    details_out: Path | None,
    report_out: Path | None,
    analysis_cfg: dict[str, Any],
) -> None:
    bonus_maps = BonusMapStore(str(bonus_map_dir))
    details = [
        score_record(
            record,
            index=index,
            bonus_maps=bonus_maps,
            tracking_mode=analysis_cfg.get("tracking_mode", "view_and_bash"),
            near_threshold=float(analysis_cfg.get("near_threshold", 0.5)),
            m_max=float(analysis_cfg.get("m_max", 3.0)),
        )
        for index, record in enumerate(iter_records(rollouts))
    ]
    summary = summarize(
        details,
        source=rollouts,
        bonus_map_dir=bonus_map_dir,
        tracking_mode=analysis_cfg.get("tracking_mode", "view_and_bash"),
        near_threshold=float(analysis_cfg.get("near_threshold", 0.5)),
        m_max=float(analysis_cfg.get("m_max", 3.0)),
    )
    if details_out:
        write_jsonl(details_out, details)
    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(format_report(summary, details), encoding="utf-8")


def cache_rollouts(
    *,
    db_path: Path,
    experiment_id: str,
    provider_source_name: str,
    model_api_name: str,
    model_label: str,
    dataset_name: str,
    config_snapshot: dict[str, Any],
    rollouts_path: Path,
    details_path: Path | None = None,
) -> int:
    with ensure_db(db_path) as conn:
        upsert_experiment(
            conn,
            experiment_id=experiment_id,
            provider_source=provider_source_name,
            dataset=dataset_name,
            config_snapshot=config_snapshot,
        )
        n_ingested = ingest_artifacts(
            conn,
            experiment_id=experiment_id,
            provider_source=provider_source_name,
            model_api_name=model_api_name,
            model_label=model_label,
            dataset=dataset_name,
            rollouts_path=rollouts_path,
            details_path=details_path,
        )
        conn.commit()
    return n_ingested


def format_report(summary: dict[str, Any], details: list[dict[str, Any]]) -> str:
    rates = summary.get("rates", {})
    averages = summary.get("averages", {})
    rows = []
    for item in details[:50]:
        rows.append(
            "| {instance} | {reads} | {graph} | {gt} | {dist} | {first_gt} |".format(
                instance=item.get("instance_id") or "-",
                reads=item.get("n_reads", 0),
                graph="yes" if item.get("hit_call_graph") else "no",
                gt="yes" if item.get("hit_ground_truth") else "no",
                dist=item.get("min_distance") if item.get("min_distance") is not None else "-",
                first_gt=item.get("first_ground_truth_step") if item.get("first_ground_truth_step") is not None else "-",
            )
        )
    return "\n".join(
        [
            "# Third-Party P2A Localization Baseline",
            "",
            f"- Records: {summary.get('counts', {}).get('n_records', 0)}",
            f"- Bonus-map coverage: {rates.get('bonus_map_coverage')}",
            f"- Call-graph coverage: {rates.get('call_graph_coverage')}",
            f"- Read rate: {rates.get('read_rate')}",
            f"- Graph hit rate: {rates.get('graph_hit_rate_over_call_graphs')}",
            f"- Ground-truth hit rate: {rates.get('ground_truth_hit_rate_over_call_graphs')}",
            f"- Near-hit rate: {rates.get('near_hit_rate_over_call_graphs')}",
            f"- Average min distance on hits: {averages.get('avg_min_distance_on_hits')}",
            "",
            "| Instance | Reads | Graph hit | GT hit | Min distance | First GT step |",
            "|---|---:|---|---|---:|---:|",
            *rows,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/third_party_eval.deepseek.example.yaml"))
    parser.add_argument("--data", type=Path, help="Input parquet/json/jsonl with Uni-Agent prompt and extra_info fields")
    parser.add_argument("--out", type=Path, default=Path("outputs/third_party_rollouts.jsonl"))
    parser.add_argument("--limit", type=parse_limit_arg, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--rollouts-per-instance", type=int, default=None)
    parser.add_argument("--per-instance-parallelism", type=int, default=None)
    parser.add_argument(
        "--rollout-job",
        action="append",
        type=parse_rollout_job,
        default=[],
        help="Run one explicit rollout job formatted as <instance_id>:<rollout_index>; may be repeated",
    )
    parser.add_argument("--provider-smoke-only", action="store_true")
    parser.add_argument("--base-url", help="Override model.base_url from config/env")
    parser.add_argument("--model-name", help="Override model.model_name from config/env")
    parser.add_argument("--model-timeout", type=float, help="Override model.timeout")
    parser.add_argument("--max-tokens", type=int, help="Override model.sampling_params.max_tokens")
    parser.add_argument("--temperature", type=float, help="Override model.sampling_params.temperature")
    parser.add_argument("--max-turns", type=int, help="Override agent.interaction.max_turns")
    parser.add_argument("--action-timeout", type=float, help="Override agent.interaction.action_timeout")
    parser.add_argument("--tool-install-timeout", type=float, help="Override agent.tool_install_timeout")
    parser.add_argument(
        "--skip-tool-install",
        action="append",
        default=[],
        help="Skip the extra install command for a copied tool name; may be repeated",
    )
    parser.add_argument("--reward-eval-timeout", type=float, help="Override agent.reward_eval_timeout")
    parser.add_argument("--bonus-map-dir", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--details-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--cache-db", type=Path, default=None, help="Optional unified SQLite cache to upsert rollouts/metrics into")
    parser.add_argument("--experiment-id", default=None, help="Experiment id used with --cache-db")
    parser.add_argument("--dataset-name", default=None, help="Dataset key used with --cache-db")
    parser.add_argument("--model-label", default=None, help="Display label used with --cache-db")
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args)
    provider_cfg = normalize_provider_config(config.get("provider"))
    model_cfg = resolve_model_config(config)
    agent_cfg = dict(config.get("agent") or {})
    experiment_cfg = dict(config.get("experiment") or {})
    rollouts_per_instance = _positive_int(
        experiment_cfg.get("rollouts_per_instance", experiment_cfg.get("rollout_n")),
        default=1,
        name="experiment.rollouts_per_instance",
    )
    per_instance_parallelism = _positive_int(
        experiment_cfg.get("per_instance_parallelism", experiment_cfg.get("rollout_parallelism")),
        default=1,
        name="experiment.per_instance_parallelism",
    )

    if args.provider_smoke_only:
        smoke = asyncio.run(run_provider_smoke(model_cfg, provider_cfg))
        print(json.dumps(smoke, indent=2, default=_json_default))
        return 0

    if args.data is None:
        raise ValueError("--data is required unless --provider-smoke-only is set")
    rows = _select_rows(
        _load_rows(args.data),
        limit=args.limit,
        offset=args.offset,
        instance_ids=set(args.instance_id) if args.instance_id else {item[0] for item in args.rollout_job} if args.rollout_job else None,
    )
    if not rows:
        raise ValueError("No rows selected")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = asyncio.run(
        run_batch(
            rows,
            model_cfg=model_cfg,
            agent_cfg=agent_cfg,
            n_parallel=args.n_parallel,
            provider_cfg=provider_cfg,
            rollouts_per_instance=rollouts_per_instance,
            per_instance_parallelism=per_instance_parallelism,
            rollout_jobs=args.rollout_job or None,
        )
    )
    write_jsonl(args.out, records)
    print(json.dumps({"rollouts": str(args.out), "n_records": len(records)}, indent=2))

    if args.bonus_map_dir:
        details_path = args.details_out or args.out.with_suffix(".details.jsonl")
        write_analysis(
            rollouts=args.out,
            bonus_map_dir=args.bonus_map_dir,
            summary_out=args.summary_out or args.out.with_suffix(".summary.json"),
            details_out=details_path,
            report_out=args.report_out or args.out.with_suffix(".report.md"),
            analysis_cfg=dict(config.get("analysis") or {}),
        )
    else:
        details_path = None
    if args.cache_db:
        dataset_name = args.dataset_name or _data_source(rows[0])
        n_cached = cache_rollouts(
            db_path=args.cache_db,
            experiment_id=args.experiment_id or f"third-party-{dataset_name}",
            provider_source_name=provider_source(provider_cfg),
            model_api_name=model_cfg["model_name"],
            model_label=args.model_label or model_cfg["model_name"],
            dataset_name=dataset_name,
            config_snapshot=_redact_config(config),
            rollouts_path=args.out,
            details_path=details_path,
        )
        print(json.dumps({"cache_db": str(args.cache_db), "n_cached": n_cached}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
