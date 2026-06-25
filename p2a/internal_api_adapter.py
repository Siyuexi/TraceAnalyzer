"""Tracked Uni-Agent chat-model adapter for the private internal API.

This module owns the runtime request/response path for ``provider.source:
internal_api``. The private API client, credentials, and model lists remain in
``provider.api_module`` or ``P2A_INTERNAL_API_MODULE`` (default
``.secrets/internal_api_eval.py``).
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any
import uuid

from p2a.internal_api_native import (
    normalize_model_name,
    parse_internal_response,
    to_prompt_history,
    update_save_id_from_response,
    uses_openai_responses_api,
)


DEFAULT_API_MODULE = ".secrets/internal_api_eval.py"
TOKEN_PARAM_KEYS = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "output_seq_len",
    "maxOutputTokens",
)
MODEL_PARAM_MAP_SUFFIXES = ("_params", "_extra_body_map")
MODEL_PARAM_MAP_KEYS = {"special_models", "model_param_models"}


class InternalApiAdapterError(RuntimeError):
    """Raised when the private internal API module cannot be used."""


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: _without_none(item)
            for key, item in value.items()
            if item is not None
        }
        return {key: item for key, item in cleaned.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value if item is not None]
    return value


def _normalized_sampling_params(params: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = _without_none(params or {})
    return copy.deepcopy(cleaned) if isinstance(cleaned, dict) else {}


def _merge_token_params(target: dict[str, Any], params: dict[str, Any]) -> None:
    token_values = {key: params[key] for key in TOKEN_PARAM_KEYS if key in params}
    if not token_values:
        return

    if set(token_values) == {"max_tokens"}:
        existing_keys = [key for key in TOKEN_PARAM_KEYS if key in target]
        if existing_keys:
            for key in existing_keys:
                target[key] = copy.deepcopy(token_values["max_tokens"])
            return

    for key, value in token_values.items():
        target[key] = copy.deepcopy(value)


def _merge_sampling_dict(target: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(target)
    for key, value in params.items():
        if key in TOKEN_PARAM_KEYS:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_sampling_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    _merge_token_params(merged, params)
    return merged


def _models_for_param_map(configs: dict[str, Any], map_key: str) -> set[str]:
    prefixes = []
    if map_key.endswith("_extra_body_map"):
        prefixes.append(map_key.removesuffix("_extra_body_map"))
    if map_key.endswith("_params"):
        prefixes.append(map_key.removesuffix("_params"))

    models: set[str] = set()
    for prefix in prefixes:
        values = configs.get(f"{prefix}_models") or []
        if isinstance(values, dict):
            models.update(str(key) for key in values)
        else:
            models.update(str(item) for item in values)
    return models


def _apply_sampling_to_model_configs(
    configs: dict[str, Any],
    model_name: str,
    params: dict[str, Any],
) -> None:
    if not params:
        return

    for key, value in configs.items():
        if not isinstance(value, dict):
            continue
        if key.endswith("_effort_map") and "reasoning_effort" in params:
            prefix = key.removesuffix("_effort_map")
            route_models = configs.get(f"{prefix}_models") or []
            if model_name in value or model_name in route_models:
                value[model_name] = copy.deepcopy(params["reasoning_effort"])
            continue
        if not (key in MODEL_PARAM_MAP_KEYS or key.endswith(MODEL_PARAM_MAP_SUFFIXES)):
            continue
        if model_name not in value and model_name not in _models_for_param_map(configs, key):
            continue
        current = value.get(model_name) or {}
        if isinstance(current, dict):
            value[model_name] = _merge_sampling_dict(current, params)


def _merge_sampling_into_request_body(body: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    if not params:
        return body
    return _merge_sampling_dict(body, params)


def resolve_api_module_path(
    provider_cfg: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> Path:
    cfg = provider_cfg or {}
    configured = (
        cfg.get("api_module")
        or os.getenv("P2A_INTERNAL_API_MODULE")
        or DEFAULT_API_MODULE
    )
    path = Path(str(configured)).expanduser()
    if path.is_absolute():
        return path
    return (repo_root or Path.cwd()) / path


def load_api_module(
    provider_cfg: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> ModuleType:
    path = resolve_api_module_path(provider_cfg, repo_root=repo_root)
    if not path.is_file():
        raise InternalApiAdapterError(
            "internal_api provider requires private API module at "
            f"{path}. Set provider.api_module or P2A_INTERNAL_API_MODULE."
        )

    module_name = f"p2a_internal_api_eval_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise InternalApiAdapterError(
            f"Could not import internal API module from {path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("Api", "HOST", "user_name", "user_token"):
        if not hasattr(module, name):
            raise InternalApiAdapterError(
                f"internal API module {path} is missing required name {name!r}"
            )

    module.AGENT_MAX_API_RETRIES = int(
        os.environ.get("P2A_INTERNAL_API_MAX_RETRIES", "2")
    )
    module.AGENT_API_ERROR_SLEEP_SECONDS = int(
        os.environ.get("P2A_INTERNAL_API_RETRY_SLEEP", "1")
    )
    module.AGENT_EMPTY_RESPONSE_SLEEP_SECONDS = int(
        os.environ.get("P2A_INTERNAL_API_EMPTY_SLEEP", "1")
    )
    return module


def check_available(
    provider_cfg: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> None:
    load_api_module(provider_cfg, repo_root=repo_root)


class InternalApiChatModel:
    """Chat-model protocol implementation consumed by Uni-Agent."""

    def __init__(
        self,
        *,
        model_name: str,
        api_module: ModuleType,
        sampling_params: dict[str, Any] | None = None,
        timeout: int | float = 300,
        **_: Any,
    ):
        self.model_name = model_name
        self.api_module = api_module
        self.sampling_params = _normalized_sampling_params(sampling_params)
        self.timeout = timeout
        self.tools_schemas: list[dict] | None = None
        self.logger = logging.getLogger("p2a.internal_api_adapter")
        self.api = api_module.Api(
            api_module.HOST,
            api_module.user_name,
            api_module.user_token,
        )
        self._apply_sampling_params()
        self._wrap_request_body_sampling()
        if hasattr(self.api, "timeout"):
            self.api.timeout = timeout
        retry_config = getattr(self.api, "set_retry_config", None)
        if callable(retry_config):
            retry_config(
                max_retries=int(os.environ.get("P2A_INTERNAL_API_MAX_RETRIES", "2")),
                retry_delay=int(os.environ.get("P2A_INTERNAL_API_RETRY_SLEEP", "1")),
                backoff_factor=1.5,
                enable_logging=True,
            )

    def _apply_sampling_params(self) -> None:
        configs = getattr(self.api, "MODEL_CONFIGS", None)
        if not isinstance(configs, dict):
            return
        self.api.MODEL_CONFIGS = copy.deepcopy(configs)
        _apply_sampling_to_model_configs(
            self.api.MODEL_CONFIGS,
            normalize_model_name(self.api, self.model_name),
            self.sampling_params,
        )

    def _wrap_request_body_sampling(self) -> None:
        request = getattr(self.api, "_make_request_with_retry", None)
        if not callable(request):
            return

        def wrapped_request(method: str, url: str, *args: Any, **kwargs: Any):
            body = kwargs.get("json")
            if isinstance(body, dict):
                kwargs["json"] = _merge_sampling_into_request_body(
                    body,
                    self.sampling_params,
                )
            return request(method, url, *args, **kwargs)

        self.api._make_request_with_retry = wrapped_request

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        return {
            "metrics": {},
            "token_usage": {},
            "internal_api_save_id": {},
            "internal_api_assistant_metadata": {},
        }

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, Any]],
        rollout_cache: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return rollout_cache

    async def query(
        self,
        messages: list[dict[str, Any]],
        rollout_cache: dict[str, Any] | None,
        **_: Any,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, Any]]:
        rollout_cache = rollout_cache or {}
        save_id = rollout_cache.setdefault("internal_api_save_id", {})
        assistant_metadata = rollout_cache.setdefault(
            "internal_api_assistant_metadata", {}
        )
        model_name = normalize_model_name(self.api, self.model_name)
        prompt, history = to_prompt_history(
            messages,
            model_name=model_name,
            api_module=self.api_module,
            save_id=save_id,
            assistant_metadata_by_index=assistant_metadata,
        )
        tools = self.tools_schemas or []
        chain_active = uses_openai_responses_api(model_name, self.api_module) and bool(
            save_id.get("response_id")
        )
        call_kwargs: dict[str, Any] = {
            "history": history,
            "tools": tools,
            "tools_mode": bool(tools),
            "history_process": True,
            "save_id": save_id,
        }

        try:
            response_obj = await asyncio.to_thread(
                self.api.call_data_eval,
                model_name,
                prompt,
                **call_kwargs,
            )
        except Exception:
            if chain_active:
                save_id.clear()
            raise

        try:
            payload = response_obj.json()
        except Exception as exc:
            if chain_active:
                save_id.clear()
            response_text = getattr(response_obj, "text", response_obj)
            raise RuntimeError(
                f"Internal API returned non-JSON response: {response_text!r}"
            ) from exc

        status = getattr(response_obj, "status_code", 200)
        if status and status >= 400:
            if chain_active:
                save_id.clear()
            raise RuntimeError(
                f"Internal API HTTP {status}: {payload.get('msg') or payload}"
            )
        if payload.get("code", 0) != 0:
            if chain_active:
                save_id.clear()
            raise RuntimeError(
                f"Internal API error code={payload.get('code')} "
                f"msg={payload.get('msg')}"
            )

        headers = getattr(response_obj, "headers", {}) or {}
        if not isinstance(headers, dict):
            headers = dict(headers)
        update_save_id_from_response(save_id, payload)
        rollout_cache["internal_api_save_id"] = save_id
        content, tool_calls, generation_info = parse_internal_response(payload, headers)
        metadata = {
            key: generation_info[key]
            for key in ("reasoning_content", "reasoning_blocks", "text_blocks")
            if generation_info.get(key)
        }
        if metadata:
            assistant_metadata[str(len(messages))] = metadata
        return content, tool_calls, rollout_cache, generation_info


def make_model(
    model_cfg: dict[str, Any],
    provider_cfg: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
):
    api_module = load_api_module(provider_cfg, repo_root=repo_root)
    return InternalApiChatModel(**model_cfg, api_module=api_module)
