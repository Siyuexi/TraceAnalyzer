"""Non-live regression tests for the ARL Uni-Agent adapter."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main()
