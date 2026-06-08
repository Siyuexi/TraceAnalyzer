"""Non-live regression tests for the ARL Uni-Agent adapter."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from env.deployment import ArlDeployment, ArlDeploymentConfig, make_env_config
from env.images import select_r2e_image


ARL_IMAGE_ENV_KEYS = (
    "P2A_ARL_IMAGE_OVERRIDES_JSON",
)


@contextmanager
def patched_env(updates: dict[str, str] | None = None, *, remove: tuple[str, ...] = ()):
    keys = set(updates or {}) | set(remove)
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in remove:
            os.environ.pop(key, None)
        if updates:
            os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ArlAdapterTests(unittest.TestCase):
    def test_agent_config_pins_qwen3_coder_tool_parser(self) -> None:
        text = Path("env/agent_config_arl.yaml").read_text(encoding="utf-8")
        self.assertIn("_target_: env.agent_loop.ArlUniAgentLoop", text)
        self.assertIn("tool_parser: qwen3_coder", text)

    def test_make_env_config_maps_arl_deployment_without_uni_agent_union(self) -> None:
        env_config = make_env_config(
            {
                "type": "arl",
                "image": "registry.local/r2e:latest",
                "gateway_url": "http://gateway",
                "namespace": "p2a",
                "experiment_id": "exp-1",
                "startup_timeout": 12,
                "max_replicas": 2,
                "unknown_future_field": "ignored",
            },
            env_variables={"PIP_CACHE_DIR": "~/.cache/pip"},
            post_setup_cmd="git checkout abc123",
            tool_install_dir="/tools",
        )

        self.assertIsInstance(env_config.deployment, ArlDeploymentConfig)
        self.assertEqual(env_config.deployment.type, "arl")
        self.assertEqual(env_config.deployment.image, "registry.local/r2e:latest")
        self.assertEqual(env_config.deployment.gateway_url, "http://gateway")
        self.assertEqual(env_config.deployment.namespace, "p2a")
        self.assertEqual(env_config.deployment.experiment_id, "exp-1")
        self.assertEqual(env_config.deployment.startup_timeout, 12)
        self.assertEqual(env_config.deployment.max_replicas, 2)
        self.assertEqual(env_config.env_variables, {"PIP_CACHE_DIR": "~/.cache/pip"})
        self.assertEqual(env_config.post_setup_cmd, "git checkout abc123")
        self.assertEqual(str(env_config.tool_install_dir), "/tools")

    def test_deployment_config_builds_direct_arl_deployment(self) -> None:
        config = ArlDeploymentConfig(image="registry.local/r2e:latest", gateway_url="http://gateway")
        deployment = config.get_deployment(run_id="run-1")

        self.assertIsInstance(deployment, ArlDeployment)
        self.assertEqual(deployment.run_id, "run-1")
        with self.assertRaisesRegex(Exception, "runtime not started"):
            _ = deployment.runtime

    def test_agent_loop_maps_arl_env_without_pydantic_deployment_union(self) -> None:
        import env as env_package

        created = {}

        def fake_agent_env(run_id, env_config):
            created["run_id"] = run_id
            created["env_config"] = env_config
            return SimpleNamespace(run_id=run_id, env_config=env_config)

        fake_agent_loop = ModuleType("uni_agent.agent_loop")

        class FakeUniAgentLoop:
            def _init_env(self, config_dict):
                return SimpleNamespace(super_config=config_dict)

        fake_agent_loop.UniAgentLoop = FakeUniAgentLoop
        fake_interaction = ModuleType("uni_agent.interaction")
        fake_interaction.AgentEnv = fake_agent_env

        config_dict = {
            "deployment": {
                "type": "arl",
                "image": "registry.local/r2e:latest",
                "gateway_url": "http://gateway",
            },
            "env_variables": {"PIP_CACHE_DIR": "~/.cache/pip"},
            "post_setup_cmd": "git checkout abc123",
            "tool_install_dir": "/tools",
        }

        try:
            sys.modules.pop("env.agent_loop", None)
            if hasattr(env_package, "agent_loop"):
                delattr(env_package, "agent_loop")
            with patch.dict(
                sys.modules,
                {
                    "uni_agent.agent_loop": fake_agent_loop,
                    "uni_agent.interaction": fake_interaction,
                },
            ):
                from env import agent_loop as agent_loop_module

                loop = object.__new__(agent_loop_module.ArlUniAgentLoop)
                loop.run_id = "run-1"
                agent_env = loop._init_env(config_dict)
        finally:
            sys.modules.pop("env.agent_loop", None)
            if hasattr(env_package, "agent_loop"):
                delattr(env_package, "agent_loop")

        self.assertIs(agent_env.env_config, created["env_config"])
        self.assertEqual(created["run_id"], "run-1")
        self.assertEqual(agent_env.env_config.deployment.type, "arl")
        self.assertEqual(agent_env.env_config.deployment.image, "registry.local/r2e:latest")
        self.assertEqual(agent_env.env_config.deployment.gateway_url, "http://gateway")
        self.assertEqual(agent_env.env_config.env_variables, {"PIP_CACHE_DIR": "~/.cache/pip"})
        self.assertEqual(agent_env.env_config.post_setup_cmd, "git checkout abc123")

    def test_image_override_and_pair_diag_routing_are_explicit(self) -> None:
        # 1. exact per-instance override wins.
        with patched_env(
            {"P2A_ARL_IMAGE_OVERRIDES_JSON": '{"demo__abc123": "registry.local/custom:tag"}'},
        ):
            self.assertEqual(
                select_r2e_image(instance_id="demo__abc123", docker_image="namanjain12/demo_final"),
                "registry.local/custom:tag",
            )

        # 2. a raw namanjain12 R2E ref maps to its pair-diag mirror.
        with patched_env(remove=ARL_IMAGE_ENV_KEYS):
            self.assertEqual(
                select_r2e_image(
                    instance_id="orange3__abcdef1234",
                    docker_image="namanjain12/orange3_final:abcdef1234",
                ),
                "pair-diag-cn-guangzhou.cr.volces.com/code/orange3_final:abcdef1234",
            )

        # 3. an image already on the pair-diag mirror passes through untouched.
        with patched_env(remove=ARL_IMAGE_ENV_KEYS):
            ref = "pair-diag-cn-guangzhou.cr.volces.com/code/django_final:abcdef1234"
            self.assertEqual(
                select_r2e_image(instance_id="django__abcdef1234", docker_image=ref),
                ref,
            )

    def test_p2a_precompute_arl_config_mapping_uses_overrides(self) -> None:
        from p2a.precompute.uni_agent_sandbox import build_agent_env_config

        task = {
            "docker_image": "namanjain12/demo_final",
            "extra_info": {
                "tools_kwargs": {
                    "reward": {
                        "metadata": {
                            "repo": "demo",
                            "instance_id": "demo__abc123",
                        }
                    }
                }
            },
        }
        env_vars = {
            "ARL_GATEWAY_URL": "http://gateway",
            "ARL_NAMESPACE": "p2a",
            "ARL_EXPERIMENT_ID": "exp-precompute",
            "ARL_TIMEOUT": "12",
            "ARL_STARTUP_TIMEOUT": "34",
            "ARL_MAX_REPLICAS": "3",
            "P2A_ARL_IMAGE_OVERRIDES_JSON": '{"demo__abc123": "registry.local/custom:tag"}',
        }
        with patched_env(env_vars, remove=("ARL_SWEREX_ENDPOINT_HOST", "ARL_SWEREX_COMMAND")):
            config = build_agent_env_config(task, instance_id="demo__abc123", deployment="arl")

        deployment = config["deployment"]
        self.assertEqual(deployment["type"], "arl")
        self.assertEqual(deployment["image"], "registry.local/custom:tag")
        self.assertEqual(deployment["gateway_url"], "http://gateway")
        self.assertEqual(deployment["namespace"], "p2a")
        self.assertEqual(deployment["experiment_id"], "exp-precompute")
        self.assertEqual(deployment["timeout"], 12.0)
        self.assertEqual(deployment["startup_timeout"], 34.0)
        self.assertEqual(deployment["max_replicas"], 3)
        self.assertNotIn("endpoint_host", deployment)
        self.assertNotIn("command", deployment)
        self.assertIn("PIP_CACHE_DIR", config["env_variables"])
        self.assertIn("ln -s /testbed/.venv", config["post_setup_cmd"])

    def test_swebench_prepare_uses_metadata_and_skips_scikit_editable_install(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "scikit-learn/scikit-learn",
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
                            "pytest sklearn/tests",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertEqual(diag["swebench_run_tests_source"], "metadata")
        self.assertIn("changed=True", diag["swebench_test_script_patch_stdout"])
        self.assertIn("skipped_editable=1", diag["swebench_test_script_patch_stdout"])
        self.assertIn("skip editable install; sklearn=", text)
        self.assertNotIn("pip install -v --no-use-pep517 --no-build-isolation -e .", text)

    def test_swebench_prepare_adds_no_build_isolation_for_regular_editable_install(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "django/django",
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "pytest tests",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertIn("changed=True", diag["swebench_test_script_patch_stdout"])
        self.assertIn("skipped_editable=0", diag["swebench_test_script_patch_stdout"])
        self.assertIn("python -m pip install --no-build-isolation -e .", text)

    def test_swebench_prepare_replaces_sphinx_tox_current_env(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "sphinx-doc/sphinx",
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .[test]",
                            "tox --current-env -epy39 -v -- tests/test_domain_cpp.py",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertIn("replaced_tox=1", diag["swebench_test_script_patch_stdout"])
        self.assertIn("python -m pytest -rA --color=no -vv tests/test_domain_cpp.py", text)
        self.assertNotIn("tox --current-env", text)

    def test_swebench_unittest_display_names_target_django_f2p(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _normalize_test_func_name, _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "django/django",
                    "FAIL_TO_PASS": (
                        '["test_cull_delete_when_store_empty (cache.tests.DBCacheTests)", '
                        '"test_zero_values (template_tests.filter_tests.test_floatformat.FunctionTests.test_zero_values)"]'
                    ),
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "git checkout abc123 tests/runtests.py tests/deprecation/test_middleware_mixin.py",
                            "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 cache.tests",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertEqual(
            _normalize_test_func_name("test_cull_delete_when_store_empty (cache.tests.DBCacheTests)"),
            "test_cull_delete_when_store_empty",
        )
        self.assertIn("cache.tests.DBCacheTests.test_cull_delete_when_store_empty", diag["swebench_f2p_django_labels"])
        self.assertIn(
            "template_tests.filter_tests.test_floatformat.FunctionTests.test_zero_values",
            diag["swebench_f2p_django_labels"],
        )
        self.assertNotIn(
            "template_tests.filter_tests.test_floatformat.FunctionTests.test_zero_values.test_zero_values",
            diag["swebench_f2p_django_labels"],
        )
        self.assertIn("targeted_django=1", diag["swebench_test_script_patch_stdout"])
        self.assertIn("cache.tests.DBCacheTests.test_cull_delete_when_store_empty", text)
        self.assertIn("git checkout abc123 tests/runtests.py tests/deprecation/test_middleware_mixin.py", text)
        self.assertNotIn(" cache.tests\n", text)

    def test_trace_instrumentation_caches_tracer_after_future_imports(self) -> None:
        from p2a.trace import instrument_source

        source = (
            '"""demo module"""\n'
            "from __future__ import annotations\n"
            "\n"
            "def modify_sys_path():\n"
            "    return 1\n"
        )
        instrumented = instrument_source(
            source,
            [
                {
                    "name": "modify_sys_path",
                    "qualified_name": "modify_sys_path",
                    "file_path": "pylint/__init__.py",
                    "start_line": 4,
                    "end_line": 5,
                }
            ],
        )

        compile(instrumented, "<instrumented>", "exec")
        self.assertLess(
            instrumented.index("from __future__ import annotations"),
            instrumented.index("import _swe_fault_tracer as _p2a_ft"),
        )
        self.assertLess(
            instrumented.index("import _swe_fault_tracer as _p2a_ft"),
            instrumented.index("def modify_sys_path"),
        )
        self.assertIn('globals().get("_p2a_ft") or __import__("_swe_fault_tracer")', instrumented)


if __name__ == "__main__":
    unittest.main()
