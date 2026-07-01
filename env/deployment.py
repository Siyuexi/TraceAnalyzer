"""ARL and Nexus sandbox deployment bridges for Uni-Agent."""

from __future__ import annotations

import asyncio
import functools
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import AbstractRuntime, CreateBashSessionRequest, IsAliveResponse

from uni_agent.async_logging import get_logger


def require_arl_gateway_url(explicit: str | None = None) -> str:
    gateway_url = explicit or os.getenv("ARL_GATEWAY_URL")
    if not gateway_url:
        raise RuntimeError("ARL_GATEWAY_URL is required; set it or source .secrets/ips.sh.")
    return gateway_url

DEFAULT_NEXUS_API_BASE_URL = "http://hyrl-sandbox.prod.woa.com:8052"
def _default_nexus_runtime_image() -> str:
    try:
        from nexus.runtime.providers.docker import get_package_version
        return f"mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{get_package_version()}"
    except Exception:
        return "mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest"


@dataclass
class ArlDeploymentConfig:
    """Config object consumed by ``AgentEnv`` through ``get_deployment``.

    The ARL-aware agent loop builds this config directly and passes it to
    ``AgentEnv``, keeping the Uni-Agent submodule untouched. The runtime itself
    is backed by the external ``arl-env`` SDK plus a local ``AbstractRuntime``
    adapter, not by a SWE-ReX server inside the sandbox.
    """

    image: str
    type: str = "arl"
    gateway_url: str | None = None
    namespace: str = "arl"
    experiment_id: str | None = None
    timeout: float = 600.0
    startup_timeout: float = 240.0
    workspace_dir: str = "/workspace"
    delete_on_stop: bool = True
    max_replicas: int | None = None
    resources: dict[str, Any] | None = field(default=None)
    require_interactive_shell: bool = False
    # Accepted for compatibility with older generated configs. Direct ARL mode
    # does not use a SWE-ReX bootstrap command or endpoint template.
    command: str | None = None
    endpoint_host: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ArlDeploymentConfig:
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in allowed}
        if "image" not in kwargs or not kwargs["image"]:
            raise ValueError("ARL deployment config requires an image")
        return cls(**kwargs)

    def get_deployment(self, run_id: str) -> ArlDeployment:
        return ArlDeployment.from_config(self, run_id=run_id)


class ArlDeployment(AbstractDeployment):
    """Boot an ARL sandbox and expose it through a SWE-ReX runtime interface."""

    def __init__(self, run_id: str, **kwargs: Any) -> None:
        self.run_id = run_id
        self._config = ArlDeploymentConfig.from_mapping(kwargs)
        self.logger = get_logger("arl-deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._session: Any | None = None
        self._runtime: AbstractRuntime | None = None
        self._stopped = False

    @classmethod
    def from_config(cls, config: ArlDeploymentConfig, run_id: str | None = None) -> ArlDeployment:
        return cls(run_id=run_id or str(uuid.uuid4()), **config.__dict__)

    def add_hook(self, hook: DeploymentHook) -> None:
        self._hooks.add_hook(hook)

    @property
    def runtime(self) -> AbstractRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError("ARL runtime not started")
        return self._runtime

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        return await self.runtime.is_alive(timeout=timeout)

    async def _blocking(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    async def start(self, max_retries: int = 5) -> None:
        """Create an ARL SDK session and attach the local runtime adapter.

        ``arl-env`` is intentionally imported lazily so non-live config/tests can
        run in the current lightweight source checkout. Phase 1 provides
        ``env.runtime.ArlRuntime`` and finalizes the SDK constructor details.
        """

        try:
            from arl import ManagedSession  # type: ignore
            from .runtime import ArlRuntime
        except ImportError as exc:
            raise RuntimeError(
                "ARL direct deployment requires arl-env==0.4.1 in the Uni-Agent execution environment "
                "and env.runtime.ArlRuntime in this source tree."
            ) from exc

        gateway_url = require_arl_gateway_url(self._config.gateway_url)
        experiment_id = self._config.experiment_id or os.getenv("ARL_EXPERIMENT_ID", "p2a-uniagent-arl")

        self.logger.info(f"Starting ARL deployment image={self._config.image} gateway={gateway_url}")
        self._hooks.on_custom_step("Creating ARL sandbox")

        last_error: Exception | None = None
        for retry in range(max_retries):
            try:
                session = await self._blocking(
                    ManagedSession,
                    image=self._config.image,
                    experiment_id=experiment_id,
                    namespace=self._config.namespace,
                    gateway_url=gateway_url,
                    timeout=self._config.timeout,
                    resources=self._config.resources,
                    workspace_dir=self._config.workspace_dir,
                    max_replicas=self._config.max_replicas,
                    api_key=os.getenv("ARL_API_KEY") or os.getenv("ARL_TOKEN"),
                )
                # ManagedSession is lazy: the sandbox/pod is provisioned only
                # when create_sandbox() runs (it sets session_id + pool_ref).
                # The interactive-shell runtime needs session_id, so provision
                # here, inside the retry loop, before attaching the adapter.
                info = await self._blocking(session.create_sandbox)
                if not (getattr(session, "session_id", None) or getattr(session, "_session_id", None)):
                    session_id = getattr(info, "id", None)
                    if session_id:
                        setattr(session, "_session_id", session_id)
                if not (getattr(session, "session_id", None) or getattr(session, "_session_id", None)):
                    raise RuntimeError("ARL create_sandbox returned without a session id")
                self._session = session
                break
            except Exception as exc:
                last_error = exc
                sleep_time = min(30, 2**retry)
                self.logger.error(f"ARL ManagedSession creation failed: {exc}; retrying in {sleep_time}s")
                await asyncio.sleep(sleep_time)
        if self._session is None:
            raise RuntimeError(f"Failed to create ARL sandbox after {max_retries} retries: {last_error}") from last_error

        self._hooks.on_custom_step("Attaching ARL runtime adapter")
        self._runtime = ArlRuntime(self._session, run_id=self.run_id, logger=self.logger)
        # Precompute can use one-shot execute calls only. Uni-Agent rollouts need
        # the persistent shell because AgentEnv.communicate() carries cwd/env state.
        eager_shell_timeout = float(os.getenv("ARL_EAGER_SHELL_TIMEOUT", "10"))
        try:
            if self._config.require_interactive_shell or eager_shell_timeout > 0:
                await asyncio.wait_for(
                    self._runtime.create_session(CreateBashSessionRequest()),
                    timeout=eager_shell_timeout if eager_shell_timeout > 0 else self._config.startup_timeout,
                )
        except Exception as exc:  # noqa: BLE001 - shell is reopened lazily by run_in_session
            if self._config.require_interactive_shell:
                raise RuntimeError(f"ARL interactive shell preflight failed: {exc}") from exc
            self.logger.warning(
                f"Eager ARL shell open failed ({exc!r}); will open lazily on first interactive use"
            )
        await self._runtime.is_alive(timeout=self._config.startup_timeout)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.error(f"Failed to close ARL runtime: {exc}")
            self._runtime = None

        if self._session is not None:
            try:
                if self._config.delete_on_stop and hasattr(self._session, "delete_sandbox"):
                    await self._blocking(self._session.delete_sandbox)
            except Exception as exc:
                self.logger.error(f"Failed to delete ARL sandbox: {exc}")
            finally:
                close = getattr(self._session, "close", None)
                if close is not None:
                    await self._blocking(close)
                self._session = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


@dataclass
class NexusDeploymentConfig:
    """Config for Nexus (sandbox v3) deployments.

    Uses the Nexus sandbox v3 HTTP API to start/stop runtime containers and
    wraps the resulting runtime URL in a ``tencent-nexus`` ``RemoteRuntime``
    (aliased as ``HttpRuntime``) plus the local ``NexusRuntime`` swe-rex adapter.
    """

    image: str
    type: str = "nexus"
    api_base_url: str | None = None
    token: str | None = None
    runtime_image: str | None = None
    batch_id: str | None = None
    timeout: float = 1200.0
    startup_timeout: float = 600.0
    environment: dict[str, str] | None = field(default=None)
    resource_spec: dict[str, int] | None = field(default_factory=lambda: {"cpu": 16, "memory": 16})

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> NexusDeploymentConfig:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in allowed}
        if "image" not in kwargs or not kwargs["image"]:
            raise ValueError("Nexus deployment config requires an image")
        return cls(**kwargs)

    def get_deployment(self, run_id: str) -> NexusDeployment:
        return NexusDeployment.from_config(self, run_id=run_id)


class NexusDeployment(AbstractDeployment):
    """Boot a Nexus sandbox via ``GongfengRuntimeProvider`` and expose it as swe-rex runtime.

    Delegates sandbox lifecycle (create, watchdog, queue-wait, stop) entirely to
    the official ``GongfengRuntimeProvider`` from ``tencent-nexus``. We only wrap
    the resulting ``Runtime`` in a ``NexusRuntime`` swe-rex adapter.
    """

    def __init__(self, run_id: str, **kwargs: Any) -> None:
        self.run_id = run_id
        self._config = NexusDeploymentConfig.from_mapping(kwargs)
        self.logger = get_logger("nexus-deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._provider: Any | None = None
        self._nexus_runtime: Any | None = None
        self._runtime: AbstractRuntime | None = None
        self._stopped = False
        self._runtime_name = f"p2a-{run_id}"

    @classmethod
    def from_config(cls, config: NexusDeploymentConfig, run_id: str | None = None) -> NexusDeployment:
        return cls(run_id=run_id or str(uuid.uuid4()), **config.__dict__)

    def add_hook(self, hook: DeploymentHook) -> None:
        self._hooks.add_hook(hook)

    @property
    def runtime(self) -> AbstractRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError("Nexus runtime not started")
        return self._runtime

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        return await self.runtime.is_alive(timeout=timeout)

    async def start(self, max_retries: int = 5) -> None:
        """Create a Nexus sandbox via GongfengRuntimeProvider and attach swe-rex adapter."""
        try:
            from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider
            from .runtime import NexusRuntime
        except ImportError as exc:
            raise RuntimeError(
                "Nexus deployment requires tencent-nexus in the execution environment."
            ) from exc

        api_base = (
            self._config.api_base_url
            or os.getenv("GONGFENG_API_BASE_URL", DEFAULT_NEXUS_API_BASE_URL)
        )
        token = self._config.token or os.getenv("GONGFENG_TOKEN", "")
        runtime_image = (
            self._config.runtime_image
            or os.getenv("GONGFENG_RUNTIME_IMAGE")
            or "current"
        )
        environment = dict(self._config.environment or {})
        environment.setdefault("PYTHON_VERSION", "3.12")

        self.logger.info(f"Starting Nexus deployment image={self._config.image} api={api_base}")
        self._hooks.on_custom_step("Creating Nexus sandbox via GongfengRuntimeProvider")

        self._provider = GongfengRuntimeProvider(
            api_base_url=api_base,
            token=token,
            image=self._config.image,
            runtime_image=runtime_image,
            request_timeout=self._config.timeout,
            resource_spec=self._config.resource_spec,
            purpose="eval",
        )

        last_error: Exception | None = None
        for retry in range(max_retries):
            try:
                self._nexus_runtime = await self._provider.acquire(
                    name=self._runtime_name,
                    environment=environment,
                    runtime_id=f"nexus-{uuid.uuid4().hex}",
                )
                break
            except Exception as exc:
                last_error = exc
                sleep_time = min(30, 2 ** retry)
                self.logger.error(f"Nexus sandbox acquire failed: {exc}; retrying in {sleep_time}s")
                await asyncio.sleep(sleep_time)
        if self._nexus_runtime is None:
            raise RuntimeError(
                f"Failed to acquire Nexus sandbox after {max_retries} retries: {last_error}"
            ) from last_error

        self._hooks.on_custom_step("Attaching Nexus runtime adapter")
        self._runtime = NexusRuntime(
            self._nexus_runtime,
            run_id=self.run_id,
            logger=self.logger,
        )
        try:
            await self._runtime.create_session(CreateBashSessionRequest())
        except Exception as exc:
            self.logger.warning(
                f"Eager Nexus shell open failed ({exc!r}); will open lazily on first interactive use"
            )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.error(f"Failed to close Nexus swe-rex runtime: {exc}")
            self._runtime = None

        if self._provider is not None:
            try:
                await self._provider.release(self._runtime_name)
            except Exception as exc:
                self.logger.error(f"Failed to release Nexus sandbox: {exc}")
            self._provider = None
        self._nexus_runtime = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


def make_env_config(
    deployment: dict[str, Any],
    *,
    env_variables: dict[str, str] | None = None,
    post_setup_cmd: str | None = None,
    tool_install_dir: str | Path = "/usr/local/bin",
):
    """Build the light config object accepted by ``AgentEnv``."""
    deploy_type = deployment.get("type", "arl")

    class _Config:
        def __init__(self) -> None:
            if deploy_type == "nexus":
                config = NexusDeploymentConfig.from_mapping(deployment)
                if env_variables:
                    merged = dict(config.environment or {})
                    merged.update(env_variables)
                    config.environment = merged
                self.deployment = config
            else:
                self.deployment = ArlDeploymentConfig.from_mapping(deployment)
            self.env_variables = env_variables
            self.post_setup_cmd = post_setup_cmd
            self.tool_install_dir = Path(tool_install_dir)

    return _Config()
