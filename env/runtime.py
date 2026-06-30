"""ARL-native runtime adapter implementing swe-rex's ``AbstractRuntime``.

This is the "兼容" layer: uni-agent's ``AgentEnv`` drives a swe-rex
``AbstractRuntime`` (9 methods). We keep that *interface* but back it with the
**ARL SDK** (``arl-env``) directly — no swe-rex server:

  * persistent bash sessions  -> ``arl.interactive_shell_client.InteractiveShellClient``
    (Gateway WebSocket PTY shell). ``cd``/``export``/aliases persist across
    ``run_in_session`` calls because it is one long-lived shell process.
  * one-shot ``execute``       -> ``ManagedSession.execute`` (stateless step API).
  * ``read_file``/``write_file``/``upload`` -> chunked base64 over one-shot
    ``execute`` (the SDK has no generic file-transfer API in 0.3.1; swap this
    layer if a stable upload/download lands upstream).

The SDK calls are synchronous (httpx / websockets.sync); every async method
offloads them to a thread executor so we satisfy the async ``AbstractRuntime``
contract without blocking the event loop.

``InteractiveShellClient`` is imported lazily so this module imports even when
``arl-env`` is not yet installed in the ambient env.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
import os
import re
import shlex
import tarfile
import time
import uuid
import warnings
from typing import Any

from swerex.runtime.abstract import (
    AbstractRuntime,
    BashAction,
    BashInterruptAction,
    BashObservation,
    CloseBashSessionRequest,
    CloseBashSessionResponse,
    CloseResponse,
    Command,
    CommandResponse,
    CreateBashSessionRequest,
    CreateBashSessionResponse,
    IsAliveResponse,
    ReadFileRequest,
    ReadFileResponse,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
)

# base64 chunk size per execute step: keep a single shell command well under any
# arg-length limit. base64 expands ~4/3, so 60k chars ≈ 45 KiB of raw bytes/step.
_B64_CHUNK = 60_000
_READ_POLL = 0.5  # seconds per websocket read while draining a command
_DEFAULT_CMD_TIMEOUT = 60.0

# Transient-error retry. At full-corpus (~4.5k instance) scale the ARL gateway
# occasionally drops a connection mid-call — httpx for the managed-session
# execute path, websockets for the interactive PTY shell. These are recoverable
# by retrying (stateless execute) or reopening the shell, so we treat a bounded
# set of network exceptions as transient rather than failing the instance.
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.5  # seconds; exponential backoff: 1.5, 3.0, ...
_TERMINAL_CONTROL_RE = re.compile(
    r"\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\r"
)


def _strip_terminal_controls(text: str) -> str:
    """Remove PTY/readline control sequences before they reach the agent trace."""
    return _TERMINAL_CONTROL_RE.sub("", text)


def _is_transient_step_failure(output: Any) -> bool:
    """Return true when ARL failed before the user's command could run."""
    if int(getattr(output, "exit_code", 0) or 0) == 0:
        return False
    text = "\n".join(
        str(getattr(output, attr, "") or "")
        for attr in ("stderr", "stdout")
    )
    if "gRPC Execute failed" not in text:
        return False
    return (
        "transport: Error while dialing" in text
        or "connection refused" in text
        or "rpc error: code = Unavailable" in text
    )


def _transient_exc_types() -> tuple[type[BaseException], ...]:
    """Best-effort tuple of transient network errors worth retrying.

    Built defensively so this module imports even when a backend lib is absent.
    """
    types: list[type[BaseException]] = [ConnectionError, TimeoutError, OSError]
    try:
        import httpx

        types += [
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ]
    except Exception:  # noqa: BLE001 - httpx optional at import time
        pass
    try:
        import websockets.exceptions as _wse

        types += [_wse.ConnectionClosed, _wse.ConnectionClosedError, _wse.ConnectionClosedOK]
        for name in ("InvalidStatus", "InvalidStatusCode", "InvalidHandshake"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                exc_type = getattr(_wse, name, None)
            if isinstance(exc_type, type):
                types.append(exc_type)
    except Exception:  # noqa: BLE001 - websockets optional at import time
        pass
    return tuple(dict.fromkeys(types))


_TRANSIENT = _transient_exc_types()


def _extract_gateway_url(session: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    client = getattr(session, "_client", None)
    for attr in ("_base_url", "base_url"):
        url = getattr(client, attr, None)
        if isinstance(url, str) and url:
            return url
    raise ValueError(
        "Cannot determine ARL gateway_url from session; pass gateway_url= explicitly."
    )


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in parameters and value is not None}


class ArlRuntime(AbstractRuntime):
    """swe-rex ``AbstractRuntime`` backed by an ARL ``ManagedSession``."""

    def __init__(
        self,
        session: Any,
        *,
        run_id: str,
        logger: Any | None = None,
        gateway_url: str | None = None,
        api_key: str | None = None,
        startup_commands: list[str] | None = None,
    ) -> None:
        self._session = session
        self.run_id = run_id
        self.logger = logger or logging.getLogger(f"arl-runtime.{run_id}")
        self._gateway_url = _extract_gateway_url(session, gateway_url)
        self._api_key = api_key
        self._shells: dict[str, Any] = {}  # session name -> InteractiveShellClient
        self._startup_commands = [command for command in (startup_commands or []) if command.strip()]
        self._shell_startup_sources: dict[str, list[str]] = {}
        self._closed = False

    # ── async plumbing ────────────────────────────────────────────────────
    async def _blocking(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    @property
    def _arl_session_id(self) -> str:
        sid = getattr(self._session, "session_id", None) or getattr(self._session, "_session_id", None)
        if not sid:
            raise RuntimeError("ARL ManagedSession has no session_id (not created?).")
        return sid

    # ── transient-error retry ─────────────────────────────────────────────
    def _retry_sync(self, what: str, func, *args):
        """Run a synchronous SDK call, retrying transient connection drops."""
        last: BaseException | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return func(*args)
            except _TRANSIENT as exc:
                last = exc
                if attempt >= _RETRY_ATTEMPTS:
                    break
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.warning(
                    "ARL %s transient error (attempt %d/%d), retry in %.1fs: %r",
                    what, attempt, _RETRY_ATTEMPTS, delay, exc,
                )
                time.sleep(delay)
        raise last  # type: ignore[misc]

    def _session_execute(self, steps: list[dict[str, Any]]) -> Any:
        """ManagedSession.execute with transient-drop retry (stateless → safe)."""
        return self._retry_sync("execute", self._session.execute, steps)

    def _execute_steps_sync(self, what: str, steps: list[dict[str, Any]]) -> Any:
        """Execute ARL steps, retrying gateway failures reported as step output."""
        last_resp: Any | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            resp = self._session_execute(steps)
            last_resp = resp
            transient = next(
                (
                    result.output
                    for result in getattr(resp, "results", []) or []
                    if _is_transient_step_failure(getattr(result, "output", None))
                ),
                None,
            )
            if transient is None:
                return resp
            if attempt >= _RETRY_ATTEMPTS:
                break
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            message = (getattr(transient, "stderr", "") or getattr(transient, "stdout", "") or "").strip()
            self.logger.warning(
                "ARL %s transient step failure (attempt %d/%d), retry in %.1fs: %s",
                what, attempt, _RETRY_ATTEMPTS, delay, message[-300:],
            )
            time.sleep(delay)
        return last_resp

    # ── persistent interactive shells ─────────────────────────────────────
    def _open_shell(self, name: str, startup_source: list[str] | None) -> str:
        from arl.interactive_shell_client import InteractiveShellClient

        shell = InteractiveShellClient(
            **_supported_kwargs(
                InteractiveShellClient,
                {"gateway_url": self._gateway_url, "api_key": self._api_key},
            )
        )
        self._retry_sync("interactive shell connect", shell.connect, self._arl_session_id)
        self._shell_startup_sources[name] = list(startup_source or [])
        # Suppress PTY echo/prompt and disable readline bracketed-paste before any
        # command output is captured from the interactive shell.
        shell.send_input(
            "export PS1=''; stty -echo 2>/dev/null || true; "
            "bind 'set enable-bracketed-paste off' 2>/dev/null || true\n"
        )
        self._shells[name] = shell
        output = self._drain_banner(shell)
        for src in startup_source or []:
            startup_output, exit_code, failure = self._run_in_shell_sync(
                name,
                f"source {shlex.quote(src)} 2>/dev/null || true",
                _DEFAULT_CMD_TIMEOUT,
            )
            output += startup_output
            if exit_code != 0 or failure:
                raise RuntimeError(f"ARL shell startup source {src!r} failed ({exit_code=} {failure})")
        for command in self._startup_commands:
            startup_output, exit_code, failure = self._run_in_shell_sync(name, command, _DEFAULT_CMD_TIMEOUT)
            output += startup_output
            if exit_code != 0 or failure:
                raise RuntimeError(f"ARL shell startup command failed ({exit_code=} {failure}): {command!r}")
        return output

    @staticmethod
    def _drain_banner(shell: Any, settle: float = 0.5) -> str:
        """Read whatever the shell emits during startup, until it goes quiet."""
        chunks: list[str] = []
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            msg = shell.read_message(timeout=_READ_POLL)
            if msg is None:
                break
            if msg.type == "output":
                chunks.append(_strip_terminal_controls(msg.data))
                deadline = time.monotonic() + settle  # keep draining while active
        return "".join(chunks)

    def _run_in_shell_sync(self, name: str, command: str, timeout: float) -> tuple[str, int, str]:
        """Send a command + sentinel; drain until the sentinel exit line. Sync."""
        shell = self._shells[name]
        marker = f"__ARL_END_{uuid.uuid4().hex}__"
        # ``$?`` here is the exit status of `command`. The executed echo prints
        # ``marker:<digits>``; even if echo isn't suppressed, the *typed* line
        # contains literal ``$?`` and won't match ``marker:\d+``.
        shell.send_input(command + "\n")
        shell.send_input(f'echo "{marker}:$?"\n')

        pattern = re.compile(rf"{re.escape(marker)}:(\d+)")
        buf = ""
        deadline = time.monotonic() + (timeout or _DEFAULT_CMD_TIMEOUT)
        while True:
            if time.monotonic() > deadline:
                try:
                    shell.send_signal("SIGINT")
                except Exception:
                    pass
                return buf, -1, "timeout"
            msg = shell.read_message(timeout=_READ_POLL)
            if msg is None:
                continue
            if msg.type == "output":
                buf += _strip_terminal_controls(msg.data)
                m = pattern.search(buf)
                if m:
                    exit_code = int(m.group(1))
                    output = buf[: m.start()]
                    # strip a trailing echoed-command line if echo wasn't suppressed
                    if output.endswith("\n"):
                        output = output[:-1]
                    return output, exit_code, ""
            elif msg.type == "exit":
                return _strip_terminal_controls(buf), msg.exit_code, "shell_exited"
            elif msg.type == "error":
                return _strip_terminal_controls(buf), -1, _strip_terminal_controls(msg.data or "shell_error")

    # ── one-shot execute helpers (stateless ManagedSession.execute) ────────
    def _exec_sync(self, shell_cmd: str, timeout: float | None = None) -> Any:
        step: dict[str, Any] = {"name": "arl-exec", "command": ["bash", "-lc", shell_cmd]}
        if timeout:
            step["timeout"] = int(timeout)
        resp = self._execute_steps_sync("execute", [step])
        if not resp.results:
            raise RuntimeError("ARL execute returned no results")
        return resp.results[0].output

    def _write_bytes_sync(self, data: bytes, remote_path: str) -> None:
        b64 = base64.b64encode(data).decode("ascii")
        quoted = shlex.quote(remote_path)
        parent = os.path.dirname(remote_path)
        steps: list[dict[str, Any]] = []
        if parent:
            steps.append({"name": "mkdir", "command": ["bash", "-lc", f"mkdir -p {shlex.quote(parent)}"]})
        # First chunk truncates (`>`), the rest append (`>>`). The `or [0]`
        # guarantees one iteration for empty content -> an empty file is written.
        first = True
        for i in range(0, len(b64), _B64_CHUNK) or [0]:
            chunk = b64[i : i + _B64_CHUNK] if b64 else ""
            redirect = ">" if first else ">>"
            steps.append({
                "name": f"write-{i}",
                "command": ["bash", "-lc", f"printf %s {shlex.quote(chunk)} | base64 -d {redirect} {quoted}"],
            })
            first = False
        resp = self._execute_steps_sync("write_file", steps)
        bad = [r for r in resp.results if r.output.exit_code != 0]
        if bad:
            raise RuntimeError(f"write_file failed: {bad[-1].output.stderr}")

    # ── AbstractRuntime: 9 methods ─────────────────────────────────────────
    async def create_session(self, request: CreateBashSessionRequest) -> CreateBashSessionResponse:
        name = request.session
        banner = await self._blocking(self._open_shell, name, list(request.startup_source or []))
        return CreateBashSessionResponse(output=banner, session_type="bash")

    async def run_in_session(self, action: BashAction | BashInterruptAction) -> BashObservation:
        name = getattr(action, "session", "default")
        if getattr(action, "action_type", "command") == "interrupt":
            shell = self._shells.get(name)
            if shell is not None:
                await self._blocking(shell.send_signal, "SIGINT")
            return BashObservation(output="", exit_code=0, session_type="bash")

        if name not in self._shells:  # auto-create if AgentEnv skipped create_session
            await self._blocking(self._open_shell, name, None)

        timeout = float(getattr(action, "timeout", None) or _DEFAULT_CMD_TIMEOUT)

        def _run_with_reconnect() -> tuple[str, int, str]:
            try:
                return self._run_in_shell_sync(name, action.command, timeout)
            except _TRANSIENT as exc:
                # The PTY websocket dropped. Reopen the shell and retry once.
                # Caveat: shell state (cwd/export) is lost on reopen — gate/setup
                # paths should use the stateless ``execute`` for critical steps.
                self.logger.warning("ARL shell %r dropped (%r); reopening + retry once", name, exc)
                old = self._shells.pop(name, None)
                if old is not None:
                    try:
                        old.close()
                    except Exception:  # noqa: BLE001 - best-effort close
                        pass
                self._open_shell(name, self._shell_startup_sources.get(name))
                return self._run_in_shell_sync(name, action.command, timeout)

        output, exit_code, failure = await self._blocking(_run_with_reconnect)
        return BashObservation(
            output=output,
            exit_code=exit_code,
            failure_reason=failure,
            session_type="bash",
        )

    async def close_session(self, request: CloseBashSessionRequest) -> CloseBashSessionResponse:
        shell = self._shells.pop(request.session, None)
        if shell is not None:
            await self._blocking(shell.close)
        return CloseBashSessionResponse(session_type="bash")

    async def execute(self, command: Command) -> CommandResponse:
        cmd = command.command
        cmd_str = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
        prefix = ""
        if getattr(command, "cwd", ""):
            prefix += f"cd {shlex.quote(command.cwd)} && "
        for k, v in (getattr(command, "env", None) or {}).items():
            prefix += f"export {k}={shlex.quote(str(v))}; "
        output = await self._blocking(self._exec_sync, prefix + cmd_str, getattr(command, "timeout", None))
        return CommandResponse(stdout=output.stdout, stderr=output.stderr, exit_code=output.exit_code)

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        output = await self._blocking(self._exec_sync, f"base64 {shlex.quote(request.path)}")
        if output.exit_code != 0:
            raise FileNotFoundError(f"read_file {request.path!r} failed: {output.stderr}")
        raw = base64.b64decode("".join(output.stdout.split()))
        content = raw.decode(request.encoding or "utf-8", request.errors or "strict")
        return ReadFileResponse(content=content)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        data = request.content.encode("utf-8") if isinstance(request.content, str) else request.content
        await self._blocking(self._write_bytes_sync, data, request.path)
        return WriteFileResponse()

    async def upload(self, request: UploadRequest) -> UploadResponse:
        src, dst = request.source_path, request.target_path

        def _upload_sync() -> None:
            if os.path.isdir(src):
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    tar.add(src, arcname=".")
                staging = f"/tmp/arl_upload_{uuid.uuid4().hex}.tgz"
                self._write_bytes_sync(buf.getvalue(), staging)
                out = self._exec_sync(
                    f"mkdir -p {shlex.quote(dst)} && tar xzf {shlex.quote(staging)} -C {shlex.quote(dst)} "
                    f"&& rm -f {shlex.quote(staging)}"
                )
                if out.exit_code != 0:
                    raise RuntimeError(f"upload(dir) failed: {out.stderr}")
            else:
                with open(src, "rb") as fh:
                    self._write_bytes_sync(fh.read(), dst)

        await self._blocking(_upload_sync)
        return UploadResponse()

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        try:
            output = await self._blocking(self._exec_sync, "echo ok", timeout)
            ok = output.exit_code == 0 and "ok" in output.stdout
            return IsAliveResponse(is_alive=ok, message="" if ok else output.stderr)
        except Exception as exc:  # noqa: BLE001 - report liveness failures, don't raise
            return IsAliveResponse(is_alive=False, message=str(exc))

    async def close(self) -> CloseResponse:
        if self._closed:
            return CloseResponse()
        self._closed = True
        for shell in list(self._shells.values()):
            try:
                await self._blocking(shell.close)
            except Exception as exc:  # noqa: BLE001
                self.logger.debug(f"shell close failed: {exc}")
        self._shells.clear()
        return CloseResponse()
