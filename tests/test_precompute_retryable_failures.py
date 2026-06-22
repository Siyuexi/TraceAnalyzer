import json
import sys
from unittest.mock import patch

import pandas as pd

from p2a.precompute import precompute_bonus_maps as bonus_maps


PROCESS_ARGS = (
    0,
    {"instance_id": "demo__1", "repo": "demo", "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n"},
    None,
    "dynamic",
    None,
    None,
    100_000,
    "uni_agent",
)


def _process_args(output_dir):
    args = list(PROCESS_ARGS)
    args[2] = str(output_dir)
    return tuple(args)


def test_retryable_existing_exception_map_is_not_complete(tmp_path):
    path = tmp_path / "demo__1.json"
    path.write_text(
        json.dumps({"instance_id": "demo__1", "case_type": "no_trace", "reason_code": "exception"}),
        encoding="utf-8",
    )

    assert not bonus_maps._existing_bonus_map_is_complete(path)

    path.write_text(
        json.dumps({"instance_id": "demo__1", "case_type": "no_trace", "reason_code": "no_trace_exit_one"}),
        encoding="utf-8",
    )

    assert bonus_maps._existing_bonus_map_is_complete(path)


def test_process_one_records_retryable_failure_without_writing_bonus_map(tmp_path):
    old_map = tmp_path / "demo__1.json"
    old_map.write_text(
        json.dumps({"instance_id": "demo__1", "case_type": "no_trace", "reason_code": "exception"}),
        encoding="utf-8",
    )
    result = bonus_maps._make_result(
        "demo__1",
        "precompute_failed",
        [],
        [],
        error=True,
        reason_code="precompute_exception",
        diagnostics={
            "precompute_failure": True,
            "failure_kind": "arl_grpc_unavailable",
            "exception_type": "RuntimeError",
            "exception_message": "gRPC Execute failed: connection refused",
        },
    )

    with patch.object(bonus_maps, "compute_dynamic_bonus_map", return_value=result):
        outcome = bonus_maps._process_one(_process_args(tmp_path))

    assert outcome["case_type"] == "precompute_failed"
    assert outcome["failure"]["instance_id"] == "demo__1"
    assert outcome["failure"]["failure_kind"] == "arl_grpc_unavailable"
    assert not old_map.exists()


def test_process_one_still_writes_semantic_error_bonus_map(tmp_path):
    result = bonus_maps._make_result(
        "demo__1",
        "no_trace",
        [],
        [],
        error=True,
        reason_code="no_trace_exit_one",
        diagnostics={"test_exit": 1},
    )

    with patch.object(bonus_maps, "compute_dynamic_bonus_map", return_value=result):
        outcome = bonus_maps._process_one(_process_args(tmp_path))

    output_path = tmp_path / "demo__1.json"
    assert outcome["failure"] is None
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["case_type"] == "no_trace"
    assert saved["reason_code"] == "no_trace_exit_one"


def test_dynamic_exception_result_is_retryable_precompute_failure():
    modified = [
        {
            "name": "demo",
            "qualified_name": "demo",
            "file_path": "pkg/demo.py",
            "start_line": 1,
            "end_line": 2,
        }
    ]

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def start(self):
            raise RuntimeError(
                'sandbox post-setup failed exit=1: gRPC Execute failed: rpc error: code = Unavailable '
                'desc = connection error: desc = "transport: Error while dialing: dial tcp 10.0.0.1:9090: '
                'connect: connection refused"'
            )

        def close(self):
            self.closed = True

    env = FakeEnv()
    task = {
        "instance_id": "demo__1",
        "repo": "demo",
        "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
    }

    with (
        patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
        patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
        patch(
            "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
            return_value=env,
        ),
    ):
        result = bonus_maps.compute_dynamic_bonus_map(task)

    assert result["case_type"] == "precompute_failed"
    assert result["reason_code"] == "precompute_exception"
    assert result["precompute_failure"] is True
    assert result["failure_kind"] == "arl_grpc_unavailable"
    assert "connection refused" in result["exception_message"]
    assert env.closed


def test_main_exits_nonzero_when_retryable_failures_leave_missing_maps(tmp_path):
    result = bonus_maps._make_result(
        "demo__1",
        "precompute_failed",
        [],
        [],
        error=True,
        reason_code="precompute_exception",
        diagnostics={
            "precompute_failure": True,
            "failure_kind": "arl_grpc_unavailable",
            "exception_type": "RuntimeError",
            "exception_message": "gRPC Execute failed: connection refused",
        },
    )
    argv = [
        "precompute_bonus_maps.py",
        "fake.parquet",
        "--output_dir",
        str(tmp_path),
        "--mode",
        "dynamic",
        "--n_parallel",
        "1",
        "--no_skip_filter",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.object(
            bonus_maps.pd,
            "read_parquet",
            return_value=pd.DataFrame(
                [{"instance_id": "demo__1", "repo": "demo", "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n"}]
            ),
        ),
        patch.object(bonus_maps, "compute_dynamic_bonus_map", return_value=result),
    ):
        exit_code = bonus_maps.main()

    assert exit_code == 1
    assert not (tmp_path / "demo__1.json").exists()
    manifest = tmp_path / bonus_maps.FAILURE_MANIFEST_NAME
    assert manifest.exists()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert records[0]["instance_id"] == "demo__1"
    assert records[0]["failure_kind"] == "arl_grpc_unavailable"
