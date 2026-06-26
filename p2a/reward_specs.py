"""P2A-local Uni-Agent reward specs.

Importing this module registers reward specs without modifying the Uni-Agent
submodule.
"""

from __future__ import annotations

import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

from p2a.datasets import last_nonempty_line, parse_string_list, swebench_pro_repo_path


def _script_from_metadata_or_dir(metadata: dict[str, Any], *, name: str, key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value

    scripts_dir = (
        metadata.get("swebench_pro_scripts_dir")
        or os.getenv("P2A_SWEBENCH_PRO_SCRIPTS_DIR")
        or os.getenv("SWEBENCH_PRO_SCRIPTS_DIR")
    )
    instance_id = metadata.get("instance_id")
    if scripts_dir and instance_id:
        path = Path(str(scripts_dir)).expanduser() / str(instance_id) / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"SWE-Bench-Pro {name} is required for {instance_id!r}; build the parquet "
        "with scripts/build_data.py swebench-pro --scripts-dir <SWE-bench_Pro-os/run_scripts> "
        "or set P2A_SWEBENCH_PRO_SCRIPTS_DIR."
    )


def _restore_tests_command(metadata: dict[str, Any]) -> str:
    value = metadata.get("swebench_pro_restore_tests_cmd")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return last_nonempty_line(metadata.get("before_repo_set_cmd")) or "true"


def _repo_path(metadata: dict[str, Any]) -> str:
    return swebench_pro_repo_path(metadata)


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _ordered_unique(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _reward_test_args(metadata: dict[str, Any]) -> list[str]:
    f2p = parse_string_list(metadata.get("FAIL_TO_PASS") or metadata.get("fail_to_pass"))
    p2p = parse_string_list(metadata.get("PASS_TO_PASS") or metadata.get("pass_to_pass"))
    if f2p or p2p:
        return _ordered_unique([*f2p, *p2p])
    return _ordered_unique(parse_string_list(metadata.get("selected_test_files_to_run")))


@register_reward_spec("swe_bench_pro")
class SWEBenchProRewardSpec(AbstractRewardSpec):
    """SWE-Bench-Pro verifier using the official per-instance run script/parser."""

    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 1800):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.eval_timeout = eval_timeout
        self.logger = get_logger("reward_spec", run_id=run_id)

    @auto_await
    async def apply_gold_patch(self) -> None:
        await self._apply_patch(str(self.metadata.get("patch") or ""))

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[bool, dict]:
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }
        try:
            run_script = _script_from_metadata_or_dir(self.metadata, name="run_script.sh", key="run_tests")
            parser_script = _script_from_metadata_or_dir(
                self.metadata,
                name="parser.py",
                key="swebench_pro_parser",
            )
            paths = await self._write_eval_files(run_script=run_script, parser_script=parser_script)
            eval_script = self._build_eval_script(paths)
            await self.env.write_file(paths["eval_script"], eval_script)

            t0 = time.perf_counter()
            await self.env.communicate(
                f"bash {shlex.quote(str(paths['eval_script']))} 2>&1 | cat",
                timeout=self.eval_timeout,
                check="ignore",
            )
            result["eval_execution_time"] = time.perf_counter() - t0
            result["eval_completed"] = True

            output = _safe_json_loads(await self.env.read_file(paths["output"]))
            stdout = await self._read_optional(paths["stdout"])
            stderr = await self._read_optional(paths["stderr"])
            eval_report = self._grade(output)
            eval_report["stdout_tail"] = stdout[-4000:]
            eval_report["stderr_tail"] = stderr[-4000:]
            result["eval_report"] = eval_report
            result["resolved"] = bool(eval_report["resolved"])
            self.logger.info(
                f"SWE-Bench-Pro eval: instance={self.metadata.get('instance_id')} "
                f"resolved={result['resolved']} tests={eval_report.get('n_tests')} "
                f"time={result['eval_execution_time']:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - reward failures are captured per instance
            result["error"] = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"Failed to evaluate SWE-Bench-Pro instance: {exc}")
        return result["resolved"], result

    async def _write_eval_files(self, *, run_script: str, parser_script: str) -> dict[str, Path]:
        base = Path(f"/tmp/p2a_swebench_pro_{uuid.uuid4().hex}")
        paths = {
            "run_script": base.with_name(base.name + "_run.sh"),
            "parser": base.with_name(base.name + "_parser.py"),
            "eval_script": base.with_name(base.name + "_eval.sh"),
            "stdout": base.with_name(base.name + "_stdout.log"),
            "stderr": base.with_name(base.name + "_stderr.log"),
            "output": base.with_name(base.name + "_output.json"),
        }
        await self.env.write_file(paths["run_script"], run_script if run_script.endswith("\n") else f"{run_script}\n")
        await self.env.write_file(paths["parser"], parser_script if parser_script.endswith("\n") else f"{parser_script}\n")
        return paths

    def _build_eval_script(self, paths: dict[str, Path]) -> str:
        selected = _reward_test_args(self.metadata)
        selected_arg = " ".join(shlex.quote(item) for item in selected)
        restore_cmd = _restore_tests_command(self.metadata)
        repo_path = _repo_path(self.metadata)
        quoted_repo = shlex.quote(repo_path)
        output_path = shlex.quote(str(paths["output"]))
        stderr_path = shlex.quote(str(paths["stderr"]))
        return "\n".join(
            [
                "#!/bin/bash",
                "set -uo pipefail",
                f"cd {quoted_repo} || {{ printf '{{\"tests\":[],\"restore_error\":\"cd_failed\"}}\\n' > {output_path}; exit 0; }}",
                f"git config --global --add safe.directory {quoted_repo} >/dev/null 2>&1 || true",
                "restore_status=0",
                "(",
                "set -e",
                restore_cmd,
                f") >> {stderr_path} 2>&1 || restore_status=$?",
                'if [ "$restore_status" -ne 0 ]; then',
                f"  printf '{{\"tests\":[],\"restore_error\":\"restore_failed\",\"restore_status\":%s}}\\n' \"$restore_status\" > {output_path}",
                "  exit 0",
                "fi",
                f"chmod +x {shlex.quote(str(paths['run_script']))}",
                f"bash {shlex.quote(str(paths['run_script']))} {selected_arg} > {shlex.quote(str(paths['stdout']))} 2> {shlex.quote(str(paths['stderr']))}",
                "test_status=$?",
                "parser_python=\"$(command -v python3 || command -v python || true)\"",
                "if [ -z \"$parser_python\" ]; then",
                f"  printf '%s\\n' '{{\"tests\":[],\"parser_error\":\"python_not_found\"}}' > {output_path}",
                "else",
                f"  \"$parser_python\" {shlex.quote(str(paths['parser']))} {shlex.quote(str(paths['stdout']))} {shlex.quote(str(paths['stderr']))} {shlex.quote(str(paths['output']))} || "
                f"printf '%s\\n' '{{\"tests\":[],\"parser_error\":\"parser_failed\"}}' > {output_path}",
                "fi",
                "exit 0",
                "",
            ]
        )

    def _grade(self, output: dict[str, Any]) -> dict[str, Any]:
        tests = output.get("tests") if isinstance(output, dict) else None
        tests = tests if isinstance(tests, list) else []
        passed = {
            str(item.get("name"))
            for item in tests
            if isinstance(item, dict) and item.get("status") == "PASSED" and item.get("name") is not None
        }
        f2p = set(parse_string_list(self.metadata.get("FAIL_TO_PASS") or self.metadata.get("fail_to_pass")))
        p2p = set(parse_string_list(self.metadata.get("PASS_TO_PASS") or self.metadata.get("pass_to_pass")))
        missing = sorted((f2p | p2p) - passed)
        report = {
            "resolved": bool(f2p) and not missing,
            "found_eval_status": bool(tests),
            "n_tests": len(tests),
            "n_passed": len(passed),
            "FAIL_TO_PASS": {
                "success": sorted(f2p & passed),
                "failure": sorted(f2p - passed),
            },
            "PASS_TO_PASS": {
                "success": sorted(p2p & passed),
                "failure": sorted(p2p - passed),
            },
            "missing": missing,
        }
        for key in ("restore_error", "restore_status", "parser_error"):
            if key in output:
                report[key] = output[key]
        return report

    async def _read_optional(self, path: Path) -> str:
        try:
            return await self.env.read_file(path)
        except Exception:
            return ""

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        if not patch.strip():
            return
        patch_path = Path(f"/tmp/p2a_swebench_pro_gold_{uuid.uuid4().hex}.diff")
        await self.env.write_file(patch_path, patch)
        repo_path = shlex.quote(_repo_path(self.metadata))
        patch_arg = shlex.quote(patch_path.as_posix())
        commands = [
            f"cd {repo_path} && git apply --whitespace=fix {patch_arg}",
            f"cd {repo_path} && git apply --reject --whitespace=nowarn {patch_arg}",
            f"cd {repo_path} && patch --batch --fuzz=5 -p1 -i {patch_arg}",
        ]
        last_error: Exception | None = None
        for command in commands:
            try:
                await self.env.communicate(command, check="raise")
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError("Failed to apply patch with any command") from last_error
