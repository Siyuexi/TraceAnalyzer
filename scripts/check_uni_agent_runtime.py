"""Check the Uni-Agent Megatron runtime expected by the P2A launcher."""

from __future__ import annotations

import importlib
import json
import os
import socket
import sys
import traceback
from importlib import metadata


def package_version(distribution: str, module: str | None = None) -> dict[str, str | bool]:
    module = module or distribution
    result: dict[str, str | bool] = {"ok": False, "version": "unknown"}
    try:
        imported = importlib.import_module(module)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - diagnostics should preserve the real failure
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    version = getattr(imported, "__version__", None)
    if version is None:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = "unknown"
    result["version"] = version
    return result


def main() -> int:
    report: dict[str, object] = {
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "CUDA_PATH": os.environ.get("CUDA_PATH"),
            "UV_PROJECT_ENVIRONMENT": os.environ.get("UV_PROJECT_ENVIRONMENT"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "NVTE_FRAMEWORK": os.environ.get("NVTE_FRAMEWORK"),
        },
        "packages": {
            "torch": package_version("torch"),
            "vllm": package_version("vllm"),
            "flash_attn": package_version("flash-attn", "flash_attn"),
            "transformer_engine": package_version("transformer-engine", "transformer_engine"),
            "transformer_engine.pytorch": package_version("transformer-engine", "transformer_engine.pytorch"),
            "megatron.core": package_version("megatron-core", "megatron.core"),
            "mbridge": package_version("mbridge"),
            "ray": package_version("ray"),
        },
        "megatron_backend": {"ok": False},
    }

    try:
        import torch

        report["torch_cuda"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        report["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from verl.workers.engine import EngineRegistry, MegatronEngine

        registered = sorted(EngineRegistry._engines.get("language_model", {}))
        if MegatronEngine is None or "megatron" not in registered:
            importlib.import_module("verl.workers.engine.megatron")
            registered = sorted(EngineRegistry._engines.get("language_model", {}))
        report["megatron_backend"] = {
            "ok": "megatron" in registered,
            "language_model_backends": registered,
        }
    except Exception as exc:  # noqa: BLE001
        report["megatron_backend"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    packages = report["packages"]
    required = (
        packages["transformer_engine.pytorch"]["ok"]
        and packages["torch"]["ok"]
        and report["megatron_backend"]["ok"]
    )
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
