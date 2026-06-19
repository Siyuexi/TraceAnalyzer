"""Tracked Uni-Agent chat-model adapter for the private internal API.

This module owns the runtime request/response path for ``provider.source:
internal_api``. The private API client, credentials, and model lists remain in
``provider.api_module`` or ``P2A_INTERNAL_API_MODULE`` (default
``.secrets/internal_api_eval.py``).
"""

from __future__ import annotations

import asyncio
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


class InternalApiAdapterError(RuntimeError):
    """Raised when the private internal API module cannot be used."""


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
        self.sampling_params = sampling_params or {}
        self.timeout = timeout
        self.tools_schemas: list[dict] | None = None
        self.logger = logging.getLogger("p2a.internal_api_adapter")
        self.api = api_module.Api(
            api_module.HOST,
            api_module.user_name,
            api_module.user_token,
        )
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
