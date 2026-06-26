import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from p2a.datasets import assert_training_data_sources_allowed, canonical_dataset, parse_string_list


ROOT = Path(__file__).resolve().parents[1]


def _load_build_data_module():
    spec = importlib.util.spec_from_file_location("build_data", ROOT / "scripts" / "build_data.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(repo_language="python"):
    return {
        "instance_id": "instance_qutebrowser__qutebrowser-f91ace96223cac8161c16dd061907e138fe85111-vabc",
        "repo": "qutebrowser/qutebrowser",
        "repo_language": repo_language,
        "dockerhub_tag": "qutebrowser.qutebrowser-qutebrowser__qutebrowser-f91ace96223cac8161c16dd061907e138fe85111-vabc",
        "base_commit": "ebfe9b7aa0c4ba9d451f993e08955004aaec4345",
        "before_repo_set_cmd": "\n".join(
            [
                "git reset --hard ebfe9b7aa0c4ba9d451f993e08955004aaec4345",
                "git clean -fd",
                "git checkout ebfe9b7aa0c4ba9d451f993e08955004aaec4345",
                "git checkout f91ace -- tests/unit/utils/test_qtlog.py",
            ]
        ),
        "fail_to_pass": "['tests/unit/utils/test_qtlog.py::TestHideQtWarning::test_unfiltered']",
        "pass_to_pass": '["tests/unit/utils/test_log.py::test_stub"]',
        "selected_test_files_to_run": '["tests/unit/utils/test_qtlog.py::TestHideQtWarning::test_unfiltered"]',
        "patch": "diff --git a/qutebrowser/utils/qtlog.py b/qutebrowser/utils/qtlog.py\n",
        "test_patch": "diff --git a/tests/unit/utils/test_qtlog.py b/tests/unit/utils/test_qtlog.py\n",
        "problem_statement": "Hide Qt warnings by prefix.",
        "requirements": "Keep unmatched warnings visible.",
        "interface": "Type: Function\nName: hide_qt_warning",
        "issue_categories": "",
        "issue_specificity": "",
    }


def test_parse_string_list_accepts_python_repr_and_json():
    assert parse_string_list("['a', 'b']") == ["a", "b"]
    assert parse_string_list('["a", "b"]') == ["a", "b"]
    assert parse_string_list(["a", "b"]) == ["a", "b"]


def test_canonical_dataset_accepts_swebench_pro_alias():
    assert canonical_dataset("swe-bench-pro") == "swebench-pro"
    assert canonical_dataset("pro") == "swebench-pro"


def test_dashboard_registers_swebench_pro_parquet():
    from p2a.dashboard_adapter import DATASET_PARQUET_FILENAMES

    assert DATASET_PARQUET_FILENAMES["swebench-pro"] == ("swe_bench_pro.parquet",)


def test_training_guard_rejects_swebench_pro_parquet(tmp_path):
    path = tmp_path / "swe_bench_pro.parquet"
    pd.DataFrame([{"data_source": "swebench-pro"}]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Eval-only dataset cannot be used"):
        assert_training_data_sources_allowed(path)


def test_training_guard_scans_all_rows(tmp_path):
    path = tmp_path / "mixed_training.parquet"
    rows = [{"data_source": "r2e-gym-subset"} for _ in range(25)]
    rows.append({"data_source": "swebench-pro"})
    pd.DataFrame(rows).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="swebench-pro"):
        assert_training_data_sources_allowed(path)


def test_cmd_swebench_pro_builds_python_subset_with_scripts(monkeypatch, tmp_path):
    build_data = _load_build_data_module()

    scripts_dir = tmp_path / "run_scripts"
    instance_dir = scripts_dir / _sample()["instance_id"]
    instance_dir.mkdir(parents=True)
    (instance_dir / "run_script.sh").write_text("pytest \"$@\"\n", encoding="utf-8")
    (instance_dir / "parser.py").write_text("print('parse')\n", encoding="utf-8")
    out = tmp_path / "swe_bench_pro.parquet"

    monkeypatch.setattr("p2a.hf_assets.load_shared_dataset", lambda *_args, **_kwargs: [_sample(), _sample("go")])
    rc = build_data.cmd_swebench_pro(SimpleNamespace(out=str(out), language="python", scripts_dir=str(scripts_dir)))

    assert rc == 0
    rows = pd.read_parquet(out).to_dict(orient="records")
    assert len(rows) == 1
    row = rows[0]
    assert row["data_source"] == "swebench-pro"
    assert row["repo_language"] == "python"
    assert json.loads(row["FAIL_TO_PASS"]) == ["tests/unit/utils/test_qtlog.py::TestHideQtWarning::test_unfiltered"]
    assert json.loads(row["PASS_TO_PASS"]) == ["tests/unit/utils/test_log.py::test_stub"]
    assert row["run_tests"] == "pytest \"$@\"\n"
    assert row["swebench_pro_parser"] == "print('parse')\n"
    assert "Interface:" in row["prompt"][1]["content"]
    assert "<uploaded_files>\n/app\n</uploaded_files>" in row["prompt"][1]["content"]
    assert "repository in the /app directory" in row["prompt"][1]["content"]
    assert "/app/reproduce_issue.py" in row["prompt"][1]["content"]
    assert "/testbed" not in row["prompt"][1]["content"]
    tools = row["extra_info"]["tools_kwargs"]
    assert tools["reward"]["name"] == "swe_bench_pro"
    assert tools["env"]["deployment"]["image"].startswith("pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:")
    assert tools["env"]["post_setup_cmd"].splitlines()[:2] == ["set -e", "cd /app"]
    assert tools["reward"]["metadata"]["swebench_pro_repo_path"] == "/app"
    assert tools["reward"]["metadata"]["swebench_pro_restore_tests_cmd"] == "git checkout f91ace -- tests/unit/utils/test_qtlog.py"


def test_swebench_pro_reward_grades_f2p_and_p2p():
    from p2a.reward_specs import SWEBenchProRewardSpec

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "FAIL_TO_PASS": json.dumps(["test_a"]),
        "PASS_TO_PASS": json.dumps(["test_b"]),
    }

    report = spec._grade(
        {
            "tests": [
                {"name": "test_a", "status": "PASSED"},
                {"name": "test_b", "status": "FAILED"},
            ]
        }
    )

    assert report["resolved"] is False
    assert report["FAIL_TO_PASS"]["success"] == ["test_a"]
    assert report["PASS_TO_PASS"]["failure"] == ["test_b"]


def test_swebench_pro_reward_eval_script_uses_app_repo_path():
    from p2a.reward_specs import SWEBenchProRewardSpec

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "selected_test_files_to_run": json.dumps(["tests/test_demo.py::test_demo"]),
        "swebench_pro_repo_path": "/app",
    }

    script = spec._build_eval_script(
        {
            "run_script": Path("/tmp/run.sh"),
            "parser": Path("/tmp/parser.py"),
            "stdout": Path("/tmp/stdout.log"),
            "stderr": Path("/tmp/stderr.log"),
            "output": Path("/tmp/output.json"),
        }
    )

    assert "cd /app" in script
    assert "safe.directory /app" in script
    assert 'parser_python="$(command -v python3 || command -v python || true)"' in script
    assert "python_not_found" in script
    assert "parser_failed" in script
    assert "/testbed" not in script


def test_swebench_pro_reward_eval_script_stops_when_restore_fails(tmp_path):
    from p2a.reward_specs import SWEBenchProRewardSpec

    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "ran-tests"
    run_script = tmp_path / "run.sh"
    parser = tmp_path / "parser.py"
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    output = tmp_path / "output.json"
    eval_script = tmp_path / "eval.sh"
    run_script.write_text(f"#!/bin/bash\ntouch {marker}\n", encoding="utf-8")
    parser.write_text("raise SystemExit('parser should not run')\n", encoding="utf-8")

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "swebench_pro_repo_path": str(repo),
        "swebench_pro_restore_tests_cmd": "echo restore failed >&2; false",
    }
    eval_script.write_text(
        spec._build_eval_script(
            {
                "run_script": run_script,
                "parser": parser,
                "stdout": stdout,
                "stderr": stderr,
                "output": output,
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(["bash", str(eval_script)], check=False, text=True, capture_output=True)

    assert proc.returncode == 0
    assert not marker.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "tests": [],
        "restore_error": "restore_failed",
        "restore_status": 1,
    }
    assert "restore failed" in stderr.read_text(encoding="utf-8")


def test_swebench_pro_sandbox_execute_does_not_activate_verified_conda():
    from p2a.precompute.uni_agent_sandbox import UniAgentSandboxAdapter

    calls = []

    class FakeRuntime:
        async def execute(self, command):
            calls.append(command.command)
            return SimpleNamespace(stdout="", stderr="", exit_code=0)

    adapter = UniAgentSandboxAdapter(
        SimpleNamespace(deployment=SimpleNamespace(runtime=FakeRuntime())),
        swebench_pro=True,
        repo_path="/app",
    )

    adapter._execute_raw("echo ok")

    assert adapter.repo_path == "/app"
    assert calls == ["echo ok"]


def test_swebench_pro_task_detection_does_not_match_generic_python_docker_rows():
    from p2a.precompute.uni_agent_sandbox import _is_swebench_pro_task

    assert _is_swebench_pro_task(
        {
            "data_source": "r2e-gym-subset",
            "repo_language": "python",
            "dockerhub_tag": "example-tag",
            "before_repo_set_cmd": "git checkout abc123",
        }
    ) is False
    assert _is_swebench_pro_task(
        {
            "extra_info": {
                "tools_kwargs": {
                    "reward": {
                        "metadata": {
                            "docker_image": "jefzda/sweap-images:example-tag",
                            "repo_path": "/app",
                        }
                    }
                }
            }
        }
    ) is True
