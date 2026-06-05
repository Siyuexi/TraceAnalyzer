"""Uni-Agent loop adapter that supports ARL deployment without submodule edits."""

from __future__ import annotations

from typing import Any

from uni_agent.agent_loop import UniAgentLoop
from uni_agent.interaction import AgentEnv

from .deployment import make_env_config


class ArlUniAgentLoop(UniAgentLoop):
    """UniAgentLoop with one extra deployment path: ``env.deployment.type=arl``."""

    def _init_env(self, config_dict: dict[str, Any]) -> AgentEnv:
        deployment = config_dict.get("deployment")
        if not (isinstance(deployment, dict) and deployment.get("type") == "arl"):
            return super()._init_env(config_dict)

        env_config = make_env_config(
            deployment,
            env_variables=config_dict.get("env_variables"),
            post_setup_cmd=config_dict.get("post_setup_cmd"),
            tool_install_dir=config_dict.get("tool_install_dir", "/usr/local/bin"),
        )
        return AgentEnv(run_id=self.run_id, env_config=env_config)
