#!/usr/bin/env python3
"""Precompute P2A bonus maps for SWE dataset instances.

Static mode (no sandbox needed):
    Extracts patched callables from AST diff, assigns d=0 to all.

Dynamic mode (requires sandbox):
    Runs full trace pipeline → builds call graph → assigns hop distances.

Classification decision tree (evaluated top-to-bottom, first match wins):

    Static layer (AST diff of old vs new content):
      newly_created  – all GT callables only exist in new_file_content
      no_callable    – patch has no callable-level changes

    Dynamic layer (instrument → run tests → parse traces):
      no_trace       – 0 traces captured after instrumentation (error=True)
      no_gt          – traces exist but none contain a GT callable (error=True)
      all_pass       – GT traces exist but all tests pass on buggy code (error=True)
      no_f2p         – GT traces exist, tests fail, but F2P filter removed all (error=True)
      standard       – F2P→GT call chain with intermediate nodes (traceable=True)
      direct         – F2P→GT call chain, test calls GT directly (traceable=True)

Usage:
    python -m utils.p2a.precompute_bonus_maps \\
        data/swe/R2E_Gym_Subset.parquet \\
        --output_dir data/swe/bonus_maps --mode dynamic --n_parallel 50
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import sys
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from p2a.precompute._path_compat import ensure_paths

ensure_paths()

from rllm.environments.swe.trace import (
    TRACE_FILE_PATH,
    _is_test_file,
    extract_callables_from_ast,
    find_modified_callables_from_task,
    make_instance_id as _legacy_make_instance_id,
    normalize_task as _legacy_normalize_task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_NAMES = frozenset(
    {
        "setUp",
        "tearDown",
        "setUpClass",
        "tearDownClass",
        "asyncSetUp",
        "asyncTearDown",
    }
)

BONUS_MAP_SCHEMA_VERSION = 3
TRACE_SIDECAR_FORMAT = "p2a_trace_sidecar_v1"
TEST_STDOUT_PATH = "/tmp/_swe_test_stdout.txt"
TEST_STDERR_PATH = "/tmp/_swe_test_stderr.txt"
TEST_EXIT_PATH = "/tmp/_swe_test_exit.txt"


_PRODUCER_METADATA: dict | None = None


def _looks_like_bare_hash(value: str) -> bool:
    return len(value) >= 20 and all(ch in "0123456789abcdef" for ch in value)


def normalize_task(task: dict) -> dict:
    """Normalize legacy rows and Uni-Agent ``extra_info.tools_kwargs`` rows."""
    task = _legacy_normalize_task(task)
    extra = task.get("extra_info")
    if isinstance(extra, str) and extra.strip():
        try:
            extra = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra = None
    if not isinstance(extra, dict):
        extra = {}

    tools_kwargs = task.get("tools_kwargs")
    if not isinstance(tools_kwargs, dict):
        maybe_tools = extra.get("tools_kwargs")
        tools_kwargs = maybe_tools if isinstance(maybe_tools, dict) else {}

    reward = tools_kwargs.get("reward") if isinstance(tools_kwargs.get("reward"), dict) else {}
    metadata = reward.get("metadata") if isinstance(reward.get("metadata"), dict) else {}

    merged = {**metadata, **task}
    if tools_kwargs:
        merged["tools_kwargs"] = tools_kwargs
    if extra:
        merged["extra_info"] = extra
    if "repo" in merged and "repo_name" not in merged:
        merged["repo_name"] = merged["repo"]
    return merged


def make_instance_id(task: dict) -> str:
    """Return the instance id used by Uni-Agent R2E parquet rows.

    The old rLLM helper used an 8-char R2E commit prefix. Uni-Agent's
    ``r2e_gym_subset_filtered.py`` uses 10 chars, and trainer lookup keys must
    match the parquet metadata exactly.
    """
    task = normalize_task(task)
    iid = task.get("instance_id")
    if isinstance(iid, str) and iid and not _looks_like_bare_hash(iid):
        return iid

    repo = task.get("repo_name") or task.get("repo")
    commit = task.get("new_commit_hash") or task.get("commit_hash") or task.get("base_commit")
    if isinstance(repo, str) and repo and isinstance(commit, str) and commit:
        return f"{repo}__{commit[:10]}"

    legacy = _legacy_make_instance_id(task)
    if isinstance(legacy, str) and legacy:
        return legacy
    return "unknown"


def _producer_metadata() -> dict:
    """Return stable metadata for cache provenance."""
    global _PRODUCER_METADATA
    if _PRODUCER_METADATA is not None:
        return dict(_PRODUCER_METADATA)

    repo_root = Path(__file__).resolve().parents[2]
    commit = None
    branch = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip() or None
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip() or None
    except OSError:
        pass

    _PRODUCER_METADATA = {
        "schema_version": BONUS_MAP_SCHEMA_VERSION,
        "producer": "utils.p2a.precompute_bonus_maps",
        "producer_commit": commit,
        "producer_branch": branch,
    }
    return dict(_PRODUCER_METADATA)


def _with_metadata(result: dict, *, reason_code: str | None = None, diagnostics: dict | None = None) -> dict:
    """Attach schema/provenance and compact debug diagnostics."""
    enriched = {**_producer_metadata(), **result}
    enriched["reason_code"] = reason_code or result.get("case_type")
    if diagnostics:
        enriched.update(diagnostics)
    return enriched


def _base_diagnostics(
    *,
    test_exit: int | None = None,
    total_trace_entries: int | None = None,
    raw_gt_trace_count: int | None = None,
    f2p_test_funcs: set[str] | None = None,
    f2p_trace_count: int | None = None,
    instrumented_callables_count: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    test_output_capture: str | None = None,
    import_targets: list[dict] | None = None,
) -> dict:
    return {
        "test_exit": test_exit,
        "total_trace_entries": total_trace_entries,
        "raw_gt_trace_count": raw_gt_trace_count,
        "f2p_test_funcs": sorted(f2p_test_funcs) if f2p_test_funcs is not None else None,
        "f2p_trace_count": f2p_trace_count,
        "instrumented_callables_count": instrumented_callables_count,
        "stdout_chars": len(stdout) if stdout is not None else None,
        "stderr_chars": len(stderr) if stderr is not None else None,
        "test_output_capture": test_output_capture,
        "import_targets": import_targets,
    }


def _tail_text(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[-max_chars:]


def _slice_traces(traces: list[list[dict]] | None, max_traces: int | None) -> tuple[list[list[dict]], bool]:
    if not traces:
        return [], False
    if max_traces is None or max_traces < 0 or len(traces) <= max_traces:
        return traces, False
    return traces[:max_traces], True


def _write_trace_sidecar(
    *,
    instance_id: str,
    sidecar_dir: str | None,
    raw_traces: list[list[dict]] | None,
    raw_gt_traces: list[list[dict]] | None,
    f2p_traces: list[list[dict]] | None = None,
    aggregated_f2p_traces: list[list[dict]] | None = None,
    diagnostics: dict | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    max_sidecar_traces: int | None = None,
    max_output_chars: int = 100_000,
) -> dict:
    """Write bulky trace/debug data to a gzip sidecar, not the training JSON."""
    if not sidecar_dir:
        return {}

    sidecar_root = Path(sidecar_dir)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_root / f"{instance_id}.traces.json.gz"

    trace_sets = {
        "raw_traces": raw_traces or [],
        "raw_gt_traces": raw_gt_traces or [],
        "f2p_traces": f2p_traces or [],
        "aggregated_f2p_traces": aggregated_f2p_traces or [],
    }
    saved_trace_sets = {}
    counts = {}
    for name, traces in trace_sets.items():
        saved, truncated = _slice_traces(traces, max_sidecar_traces)
        saved_trace_sets[name] = saved
        counts[name] = {
            "total": len(traces),
            "saved": len(saved),
            "truncated": truncated,
        }

    payload = {
        "schema_version": BONUS_MAP_SCHEMA_VERSION,
        "sidecar_format": TRACE_SIDECAR_FORMAT,
        "instance_id": instance_id,
        "counts": counts,
        "diagnostics": diagnostics or {},
        "stdout_chars": len(stdout) if stdout is not None else None,
        "stderr_chars": len(stderr) if stderr is not None else None,
        "stdout_tail": _tail_text(stdout, max_output_chars),
        "stderr_tail": _tail_text(stderr, max_output_chars),
        **saved_trace_sets,
    }
    with gzip.open(sidecar_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    return {
        "trace_sidecar_path": str(sidecar_path),
        "trace_sidecar_format": TRACE_SIDECAR_FORMAT,
        "trace_sidecar_counts": counts,
    }


def find_newly_created_callables(task: dict) -> list[dict]:
    """Find callables that only exist in new_file_content (added by the fix).

    These are pure additions — the callable doesn't exist in old_file_content
    at all — so they cannot be instrumented on the pre-fix (buggy) code.
    """
    task = normalize_task(task)
    for key in ("parsed_commit_content", "parsed_commit"):
        raw = task.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(raw, dict):
            continue
        file_diffs = raw.get("file_diffs", [])
        if not file_diffs:
            continue

        newly_created: list[dict] = []
        for fd in file_diffs:
            path = fd.get("header", {}).get("file", {}).get("path", "")
            if not path or not path.endswith(".py"):
                continue
            old_src = fd.get("old_file_content") or ""
            new_src = fd.get("new_file_content") or ""
            if not new_src:
                continue

            new_callables = extract_callables_from_ast(new_src, path)
            if old_src:
                old_callables = extract_callables_from_ast(old_src, path)
            else:
                old_callables = {}

            for qname, info in new_callables.items():
                if qname not in old_callables:
                    newly_created.append(info.to_dict())
        return newly_created

    return []


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_PARAMETRIZE_SUFFIX_RE = re.compile(r"\[.*\]$")
_TEST_FUNC_RE = re.compile(r"(?<![A-Za-z0-9_])(test[A-Za-z0-9_]*)(?=$|[^A-Za-z0-9_])")
_PYTEST_STATUS_RE = re.compile(r"\b(PASSED|FAILED|ERROR)\b")


def _strip_parametrize(name: str) -> str:
    """Strip pytest parametrize suffix: ``test_foo[True]`` → ``test_foo``."""
    return _PARAMETRIZE_SUFFIX_RE.sub("", name)


def _normalize_test_func_name(name: str) -> str:
    """Return the bare pytest test function name from a nodeid/qualname."""
    cleaned = _ANSI_ESCAPE_RE.sub("", str(name or "")).strip()
    cleaned = cleaned.split(" - ", 1)[0]
    matches = _TEST_FUNC_RE.findall(cleaned)
    if matches:
        return _strip_parametrize(matches[-1])
    bare = cleaned.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    return _strip_parametrize(bare)


def _parse_pytest_status_lines(raw_output: str) -> dict[str, str]:
    """Parse pytest ``-vv`` status lines when the short summary is absent."""
    statuses: dict[str, str] = {}
    for raw_line in raw_output.splitlines():
        line = _ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not line or "::" not in line:
            continue

        # Verbose progress: ``path.py::test_x[param] FAILED [ 10%]``.
        match = _PYTEST_STATUS_RE.search(line)
        if match:
            status = match.group(1)
            if line.startswith(status):
                remainder = line[len(status) :].strip()
                node = remainder.split(" - ", 1)[0].strip()
            else:
                node = line[: match.start()].strip()
            if "::" in node:
                statuses[node] = status
    return statuses


def _get_f2p_test_funcs(task: dict, raw_output: str, swebench_verified: bool) -> set[str] | None:
    """Identify fail-to-pass (F2P) test function names.

    F2P = tests that FAIL on buggy code and PASS after the developer's fix.

    For SWE-Bench Verified: uses the ``FAIL_TO_PASS`` field from the task.
    For R2E-Gym: parses pytest output for FAILED tests on buggy code.

    Returns:
        set[str]: bare test function names (may be empty if no tests failed).
        None: only when we genuinely cannot parse the test output.

    Note: parametrize suffixes (``[param1-param2]``) are stripped so that
    ``test_foo[True]`` matches trace frames that only contain ``test_foo``.
    """
    if swebench_verified:
        f2p_raw = task.get("FAIL_TO_PASS")
        if f2p_raw:
            if isinstance(f2p_raw, str):
                try:
                    f2p_list = json.loads(f2p_raw)
                except (json.JSONDecodeError, TypeError):
                    f2p_list = [f2p_raw]
            elif isinstance(f2p_raw, list):
                f2p_list = f2p_raw
            else:
                return None
            funcs = set()
            for t in f2p_list:
                funcs.add(_normalize_test_func_name(str(t)))
            return funcs  # may be empty
        return None
    else:
        from rllm.environments.swe.reward import decolor_dict_keys, parse_log_pytest

        test_status = decolor_dict_keys(parse_log_pytest(raw_output))
        if not test_status:
            test_status = _parse_pytest_status_lines(raw_output)
        if not test_status:
            return None  # genuinely can't parse
        failed_funcs = set()
        for name, status in test_status.items():
            if status in ("FAILED", "ERROR"):
                bare = _normalize_test_func_name(name)
                if bare:
                    failed_funcs.add(bare)
        return failed_funcs  # empty set = parsed OK but no failures


def _filter_traces_to_f2p(traces: list[list[dict]], f2p_test_funcs: set[str]) -> list[list[dict]]:
    """Keep only traces whose call chain originates from an F2P test function.

    A trace originates from an F2P test if ANY test-file frame has a
    func_name in *f2p_test_funcs*, or is a fixture (setUp/tearDown) that
    runs for every test including F2P ones.
    """
    if not f2p_test_funcs:
        return []

    filtered = []
    for trace in traces:
        keep = False
        for frame in trace:
            file_path = frame.get("file_path", "")
            if not _is_test_file(file_path):
                continue
            func_name = frame.get("func_name", "")
            bare_func_name = _normalize_test_func_name(func_name)
            if bare_func_name in f2p_test_funcs:
                keep = True
                break
            if bare_func_name in _FIXTURE_NAMES:
                keep = True
                break
        if keep:
            filtered.append(trace)
    return filtered


def _test_func_names_from_callables(callables: list[dict]) -> set[str]:
    """Extract bare test function names from callable metadata."""
    funcs = set()
    for item in callables:
        if not _is_test_file(item.get("file_path", "")):
            continue
        qname = str(item.get("qualified_name") or item.get("name") or "")
        if not qname:
            continue
        bare = _normalize_test_func_name(qname)
        if bare and (bare.startswith("test") or bare in _FIXTURE_NAMES):
            funcs.add(bare)
    return funcs


def _test_func_names_from_traces(traces: list[list[dict]]) -> list[str]:
    """Return sorted bare test-frame names observed in traces."""
    funcs = set()
    for trace in traces:
        for frame in trace:
            if not _is_test_file(frame.get("file_path", "")):
                continue
            name = str(frame.get("func_name") or frame.get("qualified_name") or "")
            bare = _normalize_test_func_name(name)
            if bare:
                funcs.add(bare)
    return sorted(funcs)


def _run_tests_with_file_capture(env, test_script: str, timeout: int = 300) -> tuple[str, str, int | None, dict]:
    """Run the test script while avoiding ARL gateway stdout truncation.

    The ARL gateway can truncate large command stdout.  Pytest's short summary
    lives near the end of output, so dynamic bonus-map construction must capture
    stdout/stderr to sandbox files and read them back with the chunked helper.
    """
    from rllm.environments.swe.trace import _read_sandbox_file

    env._run(f"rm -f {TEST_STDOUT_PATH} {TEST_STDERR_PATH} {TEST_EXIT_PATH}")
    quoted_script = shlex.quote(test_script)
    cmd = (
        "export PY_COLORS=0 NO_COLOR=1 TERM=dumb; "
        'export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -rA --color=no -vv"; '
        f"bash {quoted_script} > {TEST_STDOUT_PATH} 2> {TEST_STDERR_PATH}; "
        f"code=$?; printf '%s\\n' \"$code\" > {TEST_EXIT_PATH}; true"
    )
    wrapper_stdout, wrapper_stderr, wrapper_exit = env._execute_raw(cmd, timeout=timeout)

    stdout, stdout_exit = _read_sandbox_file(env, TEST_STDOUT_PATH)
    stderr, stderr_exit = _read_sandbox_file(env, TEST_STDERR_PATH)
    exit_text, exit_read_code = _read_sandbox_file(env, TEST_EXIT_PATH)
    exit_parse_failed = False
    try:
        test_exit = int(exit_text.strip().splitlines()[-1])
    except (ValueError, IndexError):
        exit_parse_failed = True
        test_exit = wrapper_exit

    if stdout_exit != 0 and wrapper_stdout:
        stdout = wrapper_stdout
    if stderr_exit != 0 and wrapper_stderr:
        stderr = wrapper_stderr

    all_three_read_failed = stdout_exit != 0 and stderr_exit != 0 and exit_read_code != 0
    if all_three_read_failed and exit_parse_failed and wrapper_exit == 0:
        # A successful wrapper exit only means the outer shell step returned,
        # not that the test process passed.  Treat this as unknown so the
        # decision tree cannot misclassify it as all_pass/no_trace_exit_zero.
        test_exit = None

    capture = {
        "stdout_read_exit": stdout_exit,
        "stderr_read_exit": stderr_exit,
        "exit_read_exit": exit_read_code,
        "wrapper_exit": wrapper_exit,
        "exit_parse_failed": exit_parse_failed,
        "all_three_read_failed": all_three_read_failed,
        "trusted_test_exit": test_exit is not None and not all_three_read_failed,
    }
    return stdout, stderr, test_exit, capture


def _module_name_from_file_path(file_path: str) -> str | None:
    """Best-effort Python module name for import-target diagnostics."""
    if not file_path.endswith(".py"):
        return None
    module = file_path[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module or None


def _detect_import_targets(env, patched_callables: list[dict]) -> list[dict]:
    """Compare patched file paths with Python's actual import targets."""
    modules = []
    seen = set()
    for item in patched_callables:
        file_path = item.get("file_path", "")
        module = _module_name_from_file_path(file_path)
        if module and module not in seen:
            seen.add(module)
            modules.append({"module": module, "file_path": file_path})
    if not modules:
        return []

    payload = json.dumps(modules)
    script = (
        "python - <<'PY'\n"
        "import importlib.util, json\n"
        f"mods = json.loads({payload!r})\n"
        "out = []\n"
        "for item in mods:\n"
        "    mod = item['module']\n"
        "    try:\n"
        "        spec = importlib.util.find_spec(mod)\n"
        "        origin = getattr(spec, 'origin', None) if spec else None\n"
        "    except Exception as exc:\n"
        "        origin = f'<error:{type(exc).__name__}:{exc}>'\n"
        "    out.append({**item, 'import_path': origin})\n"
        "print(json.dumps(out))\n"
        "PY"
    )
    stdout, _stderr, exit_code = env._execute_raw(script, timeout=60)
    if exit_code != 0:
        return [{"module": item["module"], "file_path": item["file_path"], "import_path": None, "import_probe_error": exit_code} for item in modules]
    try:
        targets = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return [{"module": item["module"], "file_path": item["file_path"], "import_path": None, "import_probe_error": "parse_failed"} for item in modules]

    repo_prefix = env.repo_path.rstrip("/") + "/"
    for item in targets:
        import_path = item.get("import_path")
        expected = repo_prefix + item.get("file_path", "")
        item["matches_repo_path"] = bool(import_path and import_path == expected)
        item["expected_repo_path"] = expected
    return targets


def _capture_failed(capture_diag: dict | None) -> bool:
    return bool(capture_diag and capture_diag.get("all_three_read_failed"))


def _no_trace_reason_code(test_exit: int | None, capture_diag: dict | None = None) -> str:
    if _capture_failed(capture_diag):
        return "test_setup_failure"
    if test_exit in {2, 4, 5}:
        return "test_setup_failure"
    if test_exit == 0:
        return "no_trace_exit_zero"
    if test_exit == 1:
        return "no_trace_exit_one"
    return "no_trace"


# ---------------------------------------------------------------------------
# Static bonus map
# ---------------------------------------------------------------------------


def compute_static_bonus_map(task: dict) -> dict:
    """Compute a static bonus map (patched callables only, all d=0)."""
    task = normalize_task(task)
    instance_id = make_instance_id(task)
    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)

    # --- Decision tree: static layer ---
    if not all_modified:
        if newly_created:
            case_type = "newly_created"
        else:
            case_type = "no_callable"
        return _with_metadata(
            {
                "instance_id": instance_id,
                "case_type": case_type,
                "traceable": False,
                "error": False,
                "patched_callables": [],
                "newly_created_callables": newly_created,
                "call_graph_nodes": {},
                "hop_max": 0,
            },
            reason_code=case_type,
        )

    # Static mode: every patched callable is at d=0, no test info
    call_graph_nodes = {}
    for mc in all_modified:
        node_key = f"{mc['file_path']}::{mc['qualified_name']}"
        call_graph_nodes[node_key] = {
            "file_path": mc["file_path"],
            "start_line": mc["start_line"],
            "end_line": mc["end_line"],
            "hop_distance": 0,
            "normalized_distance": 0.0,
            "observed_in_trace": False,
        }

    return _with_metadata(
        {
            "instance_id": instance_id,
            "case_type": "static",
            "traceable": False,
            "error": False,
            "patched_callables": all_modified,
            "newly_created_callables": newly_created,
            "call_graph_nodes": call_graph_nodes,
            "hop_max": 0,
        },
        reason_code="static_mode",
    )


# ---------------------------------------------------------------------------
# Dynamic bonus map — decision tree
# ---------------------------------------------------------------------------


def compute_dynamic_bonus_map(
    task: dict,
    *,
    trace_sidecar_dir: str | None = None,
    max_sidecar_traces: int | None = None,
    max_sidecar_output_chars: int = 100_000,
    sandbox_backend: str | None = None,
) -> dict:
    """Compute a dynamic bonus map using the full trace pipeline.

    Implements the decision tree:
      1. newly_created / no_callable  (static layer)
      2. no_trace → no_gt → all_pass / no_f2p → standard / direct  (dynamic layer)
    """
    from rllm.environments.swe.trace import (
        aggregate_traces,
        build_call_graph_from_traces,
        instrument_sandbox,
        parse_fault_traces_from_file,
    )

    task = normalize_task(task)
    instance_id = make_instance_id(task)

    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)
    backend = (sandbox_backend or os.environ.get("P2A_SANDBOX_BACKEND") or "uni_agent").lower()
    env_diag: dict = {"sandbox_backend": backend}
    env = None
    has_structured_callable_source = bool(task.get("parsed_commit_content") or task.get("parsed_commit"))

    if not all_modified and (newly_created or has_structured_callable_source or not task.get("patch") or backend == "legacy"):
        if newly_created:
            case_type = "newly_created"
        else:
            case_type = "no_callable"
        return _make_result(
            instance_id,
            case_type,
            [],
            newly_created,
            error=False,
            reason_code=case_type,
            diagnostics=env_diag,
        )

    try:
        if backend == "uni_agent":
            from p2a.precompute.uni_agent_sandbox import (
                create_uni_agent_sandbox,
                find_changed_callables_via_patch,
            )

            env = create_uni_agent_sandbox(task, instance_id=instance_id)
            env.start()
            env_diag.update(env.checkout_buggy_commit(task, instance_id=instance_id))

            # Uni-Agent R2E parquet keeps reward.metadata.patch but drops the
            # parsed old/new file contents. Recover callable diffs from the
            # actual buggy sandbox plus the golden patch.
            if not all_modified and task.get("patch"):
                patch_modified, patch_newly_created, patch_diag = find_changed_callables_via_patch(env, task)
                all_modified = patch_modified
                newly_created = patch_newly_created or newly_created
                env_diag.update(patch_diag)
        elif backend == "legacy":
            from rllm.environments.swe.swe import SWEEnv

            env = SWEEnv.from_dict(
                {
                    **task,
                    "experiment_id": os.environ.get("ARL_EXPERIMENT_ID", "bonus-maps"),
                }
            )
            env.reset()
        else:
            raise ValueError(f"Unsupported sandbox_backend={backend!r}; expected uni_agent or legacy")

        # ── Static layer ──────────────────────────────────────────────────
        if not all_modified:
            if newly_created:
                case_type = "newly_created"
            else:
                case_type = "no_callable"
            return _make_result(
                instance_id,
                case_type,
                [],
                newly_created,
                error=False,
                reason_code=case_type,
                diagnostics=env_diag,
            )

        # ── Dynamic layer: instrument → run → parse ───────────────────────

        instrumented_callables = instrument_sandbox(env, all_modified)
        if not instrumented_callables:
            print(f"  [{instance_id}] static_fallback: instrumentation produced 0 callables")
            return _make_result(
                instance_id,
                "static_fallback",
                all_modified,
                newly_created,
                error=True,
                reason_code="instrumentation_empty",
                diagnostics={**env_diag, **_base_diagnostics(instrumented_callables_count=0)},
            )

        # Clear stale trace file, run tests
        env._run(f"rm -f {TRACE_FILE_PATH}")

        test_script = "/run_tests.sh" if env.swebench_verified else f"{env.alt_path}/run_tests.sh"
        if not env.swebench_verified:
            env._run(f"sed -i '/pytest/{{/-rA/!s/pytest/pytest -rA/}}' {test_script}")
        stdout, stderr, test_exit, capture_diag = _run_tests_with_file_capture(env, test_script, timeout=300)
        raw_output = f"{stdout}\n{stderr}" if stderr else stdout

        # ── Decision node: NO_TRACE ──────────────────────────────────
        # Parse the raw tracer output once.  raw_traces_all is for diagnostics;
        # raw_traces keeps the historical meaning: traces that entered GT.
        raw_traces_all = parse_fault_traces_from_file(
            env,
            instrumented_callables,
            env.repo_path,
            env.alt_path,
            require_patched=False,
        )
        raw_traces = [trace for trace in raw_traces_all if any(frame.get("is_patched") for frame in trace)]

        # Also check the raw trace file for total entry count (including
        # traces without GT) to distinguish no_trace from no_gt.
        trace_file_out, _, tf_exit = env._execute_raw(f"wc -l < {TRACE_FILE_PATH} 2>/dev/null || echo 0")
        try:
            trace_file_line_count = int(trace_file_out.strip())
        except (ValueError, AttributeError):
            trace_file_line_count = 0
        parsed_trace_count = len(raw_traces_all)
        total_trace_entries = max(trace_file_line_count, parsed_trace_count)
        common_diag = _base_diagnostics(
            test_exit=test_exit,
            total_trace_entries=total_trace_entries,
            raw_gt_trace_count=len(raw_traces),
            instrumented_callables_count=len(instrumented_callables),
            stdout=stdout,
            stderr=stderr,
            test_output_capture="file",
        )
        common_diag.update(env_diag)
        common_diag["trace_file_line_count"] = trace_file_line_count
        common_diag["test_output_capture_detail"] = capture_diag
        common_diag["parsed_trace_count"] = parsed_trace_count
        common_diag["raw_gt_test_funcs"] = _test_func_names_from_traces(raw_traces)
        common_diag.update(
            _write_trace_sidecar(
                instance_id=instance_id,
                sidecar_dir=trace_sidecar_dir,
                raw_traces=raw_traces_all,
                raw_gt_traces=raw_traces,
                diagnostics=common_diag,
                stdout=stdout,
                stderr=stderr,
                max_sidecar_traces=max_sidecar_traces,
                max_output_chars=max_sidecar_output_chars,
            )
        )

        if parsed_trace_count == 0:
            import_targets = _detect_import_targets(env, all_modified)
            common_diag["import_targets"] = import_targets
            no_trace_reason = _no_trace_reason_code(test_exit, capture_diag)
            if test_exit == 0 and not _capture_failed(capture_diag):
                print(f"  [{instance_id}] all_pass: tests passed and 0 trace entries. instrumented={len(instrumented_callables)}")
                return _make_result(
                    instance_id,
                    "all_pass",
                    all_modified,
                    newly_created,
                    error=True,
                    reason_code="buggy_version_passes_no_trace",
                    diagnostics=common_diag,
                )
            print(f"  [{instance_id}] no_trace: 0 trace entries. test_exit={test_exit}, instrumented={len(instrumented_callables)}")
            return _make_result(
                instance_id,
                "no_trace",
                all_modified,
                newly_created,
                error=True,
                reason_code=no_trace_reason,
                diagnostics=common_diag,
            )

        # ── Decision node: NO_GT ─────────────────────────────────────
        # total_trace_entries > 0 but raw_traces (filtered to GT) == 0
        if not raw_traces:
            print(f"  [{instance_id}] no_gt: {total_trace_entries} trace entries but 0 contain GT callables. test_exit={test_exit}")
            return _make_result(
                instance_id,
                "no_gt",
                all_modified,
                newly_created,
                error=True,
                reason_code="patched_never_invoked",
                diagnostics=common_diag,
            )

        # ── Decision node: NO_F2P / ALL_PASS ──────────────────────────
        f2p_test_funcs = _get_f2p_test_funcs(task, raw_output, env.swebench_verified)

        if f2p_test_funcs is None:
            # Can't parse test output at all
            parse_diag = dict(common_diag)
            parse_diag["f2p_test_funcs"] = None
            case_type = "all_pass" if test_exit == 0 else "no_f2p"
            reason_code = "buggy_version_passes" if test_exit == 0 else "f2p_parse_failed"
            print(f"  [{instance_id}] {case_type}: f2p_test_funcs=None (parse failed). Dropping all {len(raw_traces)} traces. test_exit={test_exit}")
            parse_diag.update(
                _write_trace_sidecar(
                    instance_id=instance_id,
                    sidecar_dir=trace_sidecar_dir,
                    raw_traces=raw_traces_all,
                    raw_gt_traces=raw_traces,
                    diagnostics=parse_diag,
                    stdout=stdout,
                    stderr=stderr,
                    max_sidecar_traces=max_sidecar_traces,
                    max_output_chars=max_sidecar_output_chars,
                )
            )
            return _make_result(
                instance_id,
                case_type,
                all_modified,
                newly_created,
                error=True,
                reason_code=reason_code,
                diagnostics=parse_diag,
            )

        if len(f2p_test_funcs) == 0:
            # Tests parsed OK but no failing tests were identified.  This is
            # only a true all_pass if the buggy test run also exited cleanly.
            all_pass_diag = dict(common_diag)
            all_pass_diag["f2p_test_funcs"] = []
            case_type = "all_pass" if test_exit == 0 else "no_f2p"
            reason_code = "buggy_version_passes" if test_exit == 0 else "test_collection_error_no_failed_nodeids"
            print(f"  [{instance_id}] {case_type}: 0 parsed test failures. test_exit={test_exit}")
            all_pass_diag.update(
                _write_trace_sidecar(
                    instance_id=instance_id,
                    sidecar_dir=trace_sidecar_dir,
                    raw_traces=raw_traces_all,
                    raw_gt_traces=raw_traces,
                    diagnostics=all_pass_diag,
                    stdout=stdout,
                    stderr=stderr,
                    max_sidecar_traces=max_sidecar_traces,
                    max_output_chars=max_sidecar_output_chars,
                )
            )
            return _make_result(
                instance_id,
                case_type,
                all_modified,
                newly_created,
                error=True,
                reason_code=reason_code,
                diagnostics=all_pass_diag,
            )

        f2p_traces = _filter_traces_to_f2p(raw_traces, f2p_test_funcs)
        f2p_diag = dict(common_diag)
        f2p_diag["f2p_test_funcs"] = sorted(f2p_test_funcs)
        f2p_diag["f2p_trace_count"] = len(f2p_traces)
        f2p_diag["f2p_recovery_source"] = None
        f2p_diag["f2p_recovery_test_funcs"] = None
        print(f"  [{instance_id}] F2P filter: {len(raw_traces)} → {len(f2p_traces)} (F2P funcs: {f2p_test_funcs})")

        if not f2p_traces and not env.swebench_verified:
            added_test_funcs = _test_func_names_from_callables(newly_created)
            recovered_traces = _filter_traces_to_f2p(raw_traces, added_test_funcs)
            if recovered_traces:
                print(f"  [{instance_id}] F2P recovery via added tests: {len(raw_traces)} → {len(recovered_traces)} (added tests: {added_test_funcs})")
                f2p_traces = recovered_traces
                f2p_diag["f2p_trace_count"] = len(f2p_traces)
                f2p_diag["f2p_recovery_source"] = "newly_created_tests"
                f2p_diag["f2p_recovery_test_funcs"] = sorted(added_test_funcs)

        if not f2p_traces:
            print(f"  [{instance_id}] no_f2p: F2P filter removed all traces")
            f2p_diag.update(
                _write_trace_sidecar(
                    instance_id=instance_id,
                    sidecar_dir=trace_sidecar_dir,
                    raw_traces=raw_traces_all,
                    raw_gt_traces=raw_traces,
                    f2p_traces=f2p_traces,
                    diagnostics=f2p_diag,
                    stdout=stdout,
                    stderr=stderr,
                    max_sidecar_traces=max_sidecar_traces,
                    max_output_chars=max_sidecar_output_chars,
                )
            )
            return _make_result(
                instance_id,
                "no_f2p",
                all_modified,
                newly_created,
                error=True,
                reason_code="f2p_filter_dropped",
                diagnostics=f2p_diag,
            )

        # ── Build call graph from F2P+GT traces ──────────────────────
        traces = aggregate_traces(f2p_traces)

        def _read_file(rel_path: str) -> str:
            from rllm.environments.swe.trace import _read_sandbox_file

            content, exit_code = _read_sandbox_file(env, f"{env.repo_path}/{rel_path}")
            return content if exit_code == 0 else ""

        result = build_call_graph_from_traces(traces, all_modified, file_reader=_read_file)

        # ── Decision node: STANDARD vs DIRECT ────────────────────────
        nodes = result.get("call_graph_nodes", {})
        n_test_entries = sum(1 for v in nodes.values() if _is_test_file(v.get("file_path", "")))
        n_intermediate = sum(1 for v in nodes.values() if not _is_test_file(v.get("file_path", "")) and v.get("normalized_distance", 0) > 0)

        if n_test_entries > 0 and n_intermediate > 0:
            case_type = "standard"
        else:
            case_type = "direct"

        result["instance_id"] = instance_id
        result["case_type"] = case_type
        result["traceable"] = True
        result["error"] = False
        result["newly_created_callables"] = newly_created
        f2p_diag.update(
            _write_trace_sidecar(
                instance_id=instance_id,
                sidecar_dir=trace_sidecar_dir,
                raw_traces=raw_traces_all,
                raw_gt_traces=raw_traces,
                f2p_traces=f2p_traces,
                aggregated_f2p_traces=traces,
                diagnostics=f2p_diag,
                stdout=stdout,
                stderr=stderr,
                max_sidecar_traces=max_sidecar_traces,
                max_output_chars=max_sidecar_output_chars,
            )
        )
        return _with_metadata(result, reason_code=case_type, diagnostics=f2p_diag)

    except Exception as e:
        print(f"  [WARN] Dynamic tracing failed for {instance_id}: {e}")
        traceback.print_exc()
        exception_diag = _base_diagnostics()
        exception_diag.update(env_diag)
        return _make_result(
            instance_id,
            "no_trace",
            all_modified,
            newly_created,
            error=True,
            reason_code="exception",
            diagnostics=exception_diag,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _make_result(
    instance_id: str,
    case_type: str,
    patched_callables: list[dict],
    newly_created_callables: list[dict],
    *,
    error: bool = False,
    reason_code: str | None = None,
    diagnostics: dict | None = None,
) -> dict:
    """Build a result dict for untraceable cases."""
    return _with_metadata(
        {
            "instance_id": instance_id,
            "case_type": case_type,
            "traceable": False,
            "error": error,
            "patched_callables": patched_callables,
            "newly_created_callables": newly_created_callables,
            "call_graph_nodes": {},
            "hop_max": 0,
        },
        reason_code=reason_code or case_type,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Parallel processing & CLI
# ---------------------------------------------------------------------------


def _process_one(args):
    """Worker function for parallel processing."""
    (
        idx,
        task_json,
        output_dir,
        mode,
        trace_sidecar_dir,
        max_sidecar_traces,
        max_sidecar_output_chars,
        sandbox_backend,
    ) = args
    try:
        task = json.loads(task_json) if isinstance(task_json, str) else dict(task_json)

        if mode == "static":
            result = compute_static_bonus_map(task)
        else:
            result = compute_dynamic_bonus_map(
                task,
                trace_sidecar_dir=trace_sidecar_dir,
                max_sidecar_traces=max_sidecar_traces,
                max_sidecar_output_chars=max_sidecar_output_chars,
                sandbox_backend=sandbox_backend,
            )

        instance_id = result["instance_id"]
        output_path = os.path.join(output_dir, f"{instance_id}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        case_type = result.get("case_type", "unknown")
        return idx, instance_id, case_type, None
    except Exception as e:
        return idx, "unknown", "error", str(e)


def main():
    parser = argparse.ArgumentParser(description="Precompute P2A bonus maps")
    parser.add_argument("parquet_path", help="Path to dataset parquet file")
    parser.add_argument("--output_dir", required=True, help="Output directory for bonus map JSONs")
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static", help="static: AST diff only. dynamic: full trace pipeline")
    parser.add_argument(
        "--sandbox_backend",
        choices=["uni_agent", "legacy"],
        default=os.environ.get("P2A_SANDBOX_BACKEND", "uni_agent"),
        help="Sandbox backend for dynamic mode. Default: uni_agent. legacy uses old rLLM/ARL only as an explicit fallback.",
    )
    parser.add_argument("--n_parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N instances")
    parser.add_argument(
        "--save_trace_sidecars",
        action="store_true",
        help="Write raw/f2p/aggregated trace debug sidecars as .json.gz files",
    )
    parser.add_argument(
        "--trace_sidecar_dir",
        default=None,
        help="Directory for trace sidecars. Defaults to <output_dir>/traces when --save_trace_sidecars is set.",
    )
    parser.add_argument(
        "--max_sidecar_traces",
        type=int,
        default=None,
        help="Optional max traces saved per trace set in each sidecar. Unset saves all traces.",
    )
    parser.add_argument(
        "--max_sidecar_output_chars",
        type=int,
        default=100_000,
        help="Max stdout/stderr tail chars saved in each trace sidecar.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trace_sidecar_dir = None
    if args.save_trace_sidecars:
        trace_sidecar_dir = args.trace_sidecar_dir or os.path.join(args.output_dir, "traces")
        os.makedirs(trace_sidecar_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet_path)
    if args.limit:
        df = df.head(args.limit)

    print(f"Processing {len(df)} instances from {args.parquet_path}")
    print(f"Mode: {args.mode}, Output: {args.output_dir}, Workers: {args.n_parallel}")
    if args.mode == "dynamic":
        print(f"Dynamic sandbox backend: {args.sandbox_backend}")
    if trace_sidecar_dir:
        print(f"Trace sidecars: {trace_sidecar_dir}")

    work_items = []
    for idx, row in df.iterrows():
        task_payload = row.to_dict()
        work_items.append(
            (
                idx,
                task_payload,
                args.output_dir,
                args.mode,
                trace_sidecar_dir,
                args.max_sidecar_traces,
                args.max_sidecar_output_chars,
                args.sandbox_backend,
            )
        )

    from collections import Counter

    case_counts = Counter()
    error_count = 0
    total = len(work_items)

    if args.n_parallel <= 1:
        for item in work_items:
            idx, instance_id, case_type, error = _process_one(item)
            if error:
                error_count += 1
                print(f"  [{idx}] ERROR: {error}")
            case_counts[case_type] += 1
            done = sum(case_counts.values())
            if done % 100 == 0:
                traceable = case_counts["direct"] + case_counts["standard"]
                print(f"  Progress: {done}/{total} (traceable: {traceable}, errors: {error_count})")
    else:
        with ThreadPoolExecutor(max_workers=args.n_parallel) as executor:
            futures = {executor.submit(_process_one, item): item for item in work_items}
            done_count = 0
            for future in as_completed(futures):
                idx, instance_id, case_type, error = future.result()
                done_count += 1
                if error:
                    error_count += 1
                case_counts[case_type] += 1
                if done_count % 100 == 0:
                    traceable = case_counts["direct"] + case_counts["standard"]
                    print(f"  Progress: {done_count}/{total} (traceable: {traceable}, errors: {error_count})")

    # Summary table
    traceable = case_counts["direct"] + case_counts["standard"]
    sum(case_counts[ct] for ct in ("no_trace", "no_gt", "no_f2p"))

    print(f"\n{'=' * 50}")
    print(f"Summary: {total} instances from {args.parquet_path}")
    print(f"{'=' * 50}")

    print(f"\ntraceable/          {traceable:5d}  ({100 * traceable / total:.1f}%)")
    print(f"  direct             {case_counts['direct']:5d}")
    print(f"  standard           {case_counts['standard']:5d}")

    print(f"\nuntraceable/        {total - traceable:5d}  ({100 * (total - traceable) / total:.1f}%)")
    print(f"  newly_created      {case_counts['newly_created']:5d}")
    print(f"  no_callable        {case_counts['no_callable']:5d}")
    if case_counts["static"] or case_counts["static_fallback"]:
        print(f"  static             {case_counts['static']:5d}")
        print(f"  static_fallback    {case_counts['static_fallback']:5d}")
    print(f"  all_pass  (error)  {case_counts['all_pass']:5d}")
    print(f"  no_trace  (error)  {case_counts['no_trace']:5d}")
    print(f"  no_gt     (error)  {case_counts['no_gt']:5d}")
    print(f"  no_f2p    (error)  {case_counts['no_f2p']:5d}")

    if error_count:
        print(f"\nprocess errors       {error_count:5d}")

    print(f"\nBonus maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
