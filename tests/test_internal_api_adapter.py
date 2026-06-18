import asyncio

import pytest

import p2a.internal_api_adapter as internal_api_adapter
from p2a.api_providers import (
    ProviderLoadError,
    check_provider_available,
    load_internal_adapter,
    make_chat_model,
    normalize_provider_config,
)


def _write_fake_api_module(path):
    path.write_text(
        """
class Response:
    status_code = 200
    headers = {"x-usage-prompt-tokens": "2"}

    def json(self):
        return {
            "code": 0,
            "request_id": "resp_1",
            "answer": [
                {
                    "type": "text",
                    "value": "inspect",
                },
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {"command": "view", "path": "/testbed/a.py"},
                            },
                        }
                    ],
                },
            ],
            "cost_info": {"completion_tokens": 3, "cost": 0.01},
        }


class Api:
    MODEL_CONFIGS = {
        "passthrough_models": ["deepseek-v4-flash-passthrough"],
        "passthrough_chat_completions_models": {"deepseek-v4-flash-passthrough"},
    }

    def __init__(self, host, user_name, user_token):
        self.host = host
        self.user_name = user_name
        self.user_token = user_token
        self.calls = []

    def set_retry_config(self, **kwargs):
        self.retry_config = kwargs

    def _normalize_model_name(self, model_name):
        return model_name

    def call_data_eval(self, model_name, prompt, **kwargs):
        self.calls.append({"model_name": model_name, "prompt": prompt, **kwargs})
        return Response()


HOST = "http://internal.example"
user_name = "demo-user"
user_token = "demo-token"
""",
        encoding="utf-8",
    )


def test_internal_api_defaults_to_tracked_adapter():
    cfg = normalize_provider_config({"source": "internal_api"})

    assert "adapter" not in cfg
    assert cfg["api_module"] == ".secrets/internal_api_eval.py"
    assert load_internal_adapter(cfg).__name__ == "p2a.internal_api_adapter"


def test_internal_api_default_checks_private_api_module(tmp_path):
    cfg = {"source": "internal_api", "api_module": "missing_internal_api_eval.py"}

    with pytest.raises(
        ProviderLoadError,
        match="internal_api provider requires private API module",
    ):
        check_provider_available(cfg, repo_root=tmp_path)


def test_internal_api_custom_adapter_missing_fails_clearly(tmp_path):
    missing = tmp_path / "missing_adapter.py"

    with pytest.raises(ProviderLoadError, match="custom adapter was not found"):
        check_provider_available(
            {"source": "internal_api", "adapter": str(missing)},
            repo_root=tmp_path,
        )


def test_tracked_internal_api_adapter_queries_private_api_module(tmp_path, monkeypatch):
    api_module = tmp_path / "internal_api_eval.py"
    _write_fake_api_module(api_module)

    async def call_now(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(internal_api_adapter.asyncio, "to_thread", call_now)
    model = make_chat_model(
        {
            "model_name": "deepseek-v4-flash-passthrough",
            "sampling_params": {"max_tokens": 16},
        },
        {"source": "internal_api", "api_module": api_module.name},
        repo_root=tmp_path,
    )
    model.set_tools_schemas(
        [{"type": "function", "function": {"name": "str_replace_editor"}}]
    )

    async def _query():
        cache = await model.prepare_rollout_cache(
            [{"role": "user", "content": "Fix the bug."}]
        )
        return await model.query(
            [{"role": "user", "content": "Fix the bug."}],
            cache,
        )

    content, tool_calls, cache, info = asyncio.run(_query())

    assert content == "inspect"
    assert tool_calls[0]["function"]["name"] == "str_replace_editor"
    assert cache["internal_api_save_id"]["response_id"] == "resp_1"
    assert cache["token_usage"]["input_tokens"] == 2.0
    assert cache["token_usage"]["output_tokens"] == 3
    assert cache["token_usage"]["cost"] == 0.01
    call = model.inner.api.calls[0]
    assert call["prompt"] == "Fix the bug."
    assert call["history"] == []
    assert call["tools"] == [
        {"type": "function", "function": {"name": "str_replace_editor"}}
    ]
