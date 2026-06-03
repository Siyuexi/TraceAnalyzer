#!/usr/bin/env python3
"""Probe one R2E-Gym case on Uni-Agent/veFaaS before spending on full bonus maps."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys, parse_log_fn

from p2a.precompute.precompute_bonus_maps import make_instance_id, normalize_task
from p2a.precompute.uni_agent_sandbox import create_uni_agent_sandbox, extract_reward_metadata


def _load_task(parquet_path: Path, *, instance_id: str | None, row_index: int) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    if instance_id:
        for _, row in df.iterrows():
            task = normalize_task(row.to_dict())
            if make_instance_id(task) == instance_id:
                return task
        raise ValueError(f"instance_id not found in {parquet_path}: {instance_id}")
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index out of bounds: {row_index} for {len(df)} rows")
    return normalize_task(df.iloc[row_index].to_dict())


def _run_git_apply_check(env, patch_text: str) -> dict[str, Any]:
    patch_b64 = base64.b64encode(patch_text.encode()).decode()
    env._execute_raw(f"printf '%s' '{patch_b64}' | base64 -d > /tmp/_p2a_probe_patch.diff", timeout=30)
    output, _, exit_code = env._execute_raw(
        "cd /testbed && git apply --check --whitespace=nowarn /tmp/_p2a_probe_patch.diff",
        timeout=120,
    )
    return {"git_apply_check_exit": exit_code, "git_apply_check_output_tail": output[-1000:]}


def _apply_gold_patch(env, patch_text: str) -> dict[str, Any]:
    patch_b64 = base64.b64encode(patch_text.encode()).decode()
    env._execute_raw(f"printf '%s' '{patch_b64}' | base64 -d > /tmp/_p2a_probe_gold_patch.diff", timeout=30)
    commands = [
        "cd /testbed && git apply --whitespace=fix /tmp/_p2a_probe_gold_patch.diff",
        "cd /testbed && git apply --reject --whitespace=nowarn /tmp/_p2a_probe_gold_patch.diff",
        "cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/_p2a_probe_gold_patch.diff",
    ]
    attempts = []
    for command in commands:
        output, _, exit_code = env._execute_raw(command, timeout=120)
        attempts.append({"command": command, "exit": exit_code, "output_tail": output[-1000:]})
        if exit_code == 0:
            return {"gold_patch_apply_exit": 0, "gold_patch_apply_attempts": attempts}
    return {"gold_patch_apply_exit": attempts[-1]["exit"] if attempts else None, "gold_patch_apply_attempts": attempts}


def _compute_reward_like_uni_agent(env, metadata: dict[str, Any], *, eval_timeout: int) -> dict[str, Any]:
    from rllm.environments.swe.trace import _read_sandbox_file

    stdout_path = "/tmp/_p2a_probe_reward_stdout.txt"
    stderr_path = "/tmp/_p2a_probe_reward_stderr.txt"
    exit_path = "/tmp/_p2a_probe_reward_exit.txt"
    env._execute_raw(f"rm -f {stdout_path} {stderr_path} {exit_path}", timeout=30)
    wrapper = (
        "export PY_COLORS=0 NO_COLOR=1 TERM=dumb; "
        f"bash /root/run_tests.sh > {stdout_path} 2> {stderr_path}; "
        f"code=$?; printf '%s\\n' \"$code\" > {exit_path}; true"
    )
    _, _, wrapper_exit = env._execute_raw(wrapper, timeout=eval_timeout)
    stdout, stdout_exit = _read_sandbox_file(env, stdout_path)
    stderr, stderr_exit = _read_sandbox_file(env, stderr_path)
    exit_text, exit_read_exit = _read_sandbox_file(env, exit_path)
    try:
        test_exit = int(exit_text.strip().splitlines()[-1])
    except (ValueError, IndexError):
        test_exit = None

    output = re.sub(r"\x1b\[[0-9;]*m|\r", "", f"{stdout}\n{stderr}")
    parsed_status = parse_log_fn(metadata["repo"])(output)
    parsed_status = decolor_dict_keys(parsed_status)
    expected_status = decolor_dict_keys(json.loads(metadata["expected_output_json"]))
    parsed_status = {k.split(" - ")[0]: parsed_status[k] for k in sorted(parsed_status.keys())}
    expected_status = {k.split(" - ")[0]: expected_status[k] for k in sorted(expected_status.keys())}
    if len(parsed_status) != len(expected_status):
        resolved = False
    else:
        resolved = True
        for key, status in parsed_status.items():
            if not key:
                continue
            if key not in expected_status or status != expected_status[key]:
                resolved = False
                break
    return {
        "resolved": bool(resolved),
        "eval_completed": stdout_exit == 0 or stderr_exit == 0,
        "found_eval_status": bool(parsed_status),
        "parsed_status_count": len(parsed_status),
        "expected_status_count": len(expected_status),
        "test_exit": test_exit,
        "wrapper_exit": wrapper_exit,
        "stdout_read_exit": stdout_exit,
        "stderr_read_exit": stderr_exit,
        "exit_read_exit": exit_read_exit,
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
    }


def probe_case(task: dict[str, Any], *, eval_timeout: int) -> dict[str, Any]:
    instance_id = make_instance_id(task)
    metadata = dict(extract_reward_metadata(task))
    patch_text = str(metadata.get("patch") or task.get("patch") or "")
    if not patch_text.strip():
        raise ValueError(f"{instance_id} has no patch in reward metadata")

    env = create_uni_agent_sandbox(task, instance_id=instance_id)
    try:
        env.start()
        checkout = env.checkout_buggy_commit(task, instance_id=instance_id)
        apply_check = _run_git_apply_check(env, patch_text)

        before = _compute_reward_like_uni_agent(env, metadata, eval_timeout=eval_timeout)
        patch_apply = _apply_gold_patch(env, patch_text)
        after = _compute_reward_like_uni_agent(env, metadata, eval_timeout=eval_timeout)
        semantic_pass = bool(
            checkout.get("buggy_checkout_exit") == 0
            and apply_check["git_apply_check_exit"] == 0
            and patch_apply["gold_patch_apply_exit"] == 0
            and before["eval_completed"]
            and after["eval_completed"]
            and before["found_eval_status"]
            and after["found_eval_status"]
            and not before["resolved"]
            and after["resolved"]
        )
        return {
            "instance_id": instance_id,
            "repo": metadata.get("repo"),
            "old_commit_hash": metadata.get("old_commit_hash") or metadata.get("commit_hash"),
            "new_commit_hash": metadata.get("new_commit_hash"),
            **checkout,
            **apply_check,
            **patch_apply,
            "before_gold_patch": before,
            "after_gold_patch": after,
            "semantic_pass": semantic_pass,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--eval-timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    task = _load_task(args.parquet, instance_id=args.instance_id, row_index=args.row_index)
    result = probe_case(task, eval_timeout=args.eval_timeout)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not result["semantic_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
