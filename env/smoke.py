"""Smoke tests for ARL SDK -> Uni-Agent runtime connectivity."""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from swerex.runtime.abstract import BashAction, Command, UploadRequest

from .deployment import ArlDeploymentConfig


async def run_smoke(args: argparse.Namespace) -> int:
    config_kwargs = {
        "image": args.image,
        "gateway_url": args.gateway_url,
        "namespace": args.namespace,
        "experiment_id": args.experiment_id,
        "timeout": args.timeout,
        "startup_timeout": args.startup_timeout,
        "delete_on_stop": not args.keep_sandbox,
        "max_replicas": args.max_replicas,
        "require_interactive_shell": True,
    }
    config = ArlDeploymentConfig(**config_kwargs)
    deployment = config.get_deployment(run_id="arl-smoke")
    await deployment.start(max_retries=args.retries)
    try:
        r1 = await deployment.runtime.run_in_session(BashAction(command="export FOO=bar123", timeout=30))
        r2 = await deployment.runtime.run_in_session(BashAction(command="echo $FOO", timeout=30))
        r3 = await deployment.runtime.run_in_session(BashAction(command="cd /tmp", timeout=30))
        r4 = await deployment.runtime.run_in_session(BashAction(command="pwd", timeout=30))
        if r1.exit_code != 0 or r2.output.strip() != "bar123" or r3.exit_code != 0 or r4.output.strip() != "/tmp":
            raise RuntimeError(
                "persistent session failed: "
                f"export_exit={r1.exit_code} echo={r2.output!r} cd_exit={r3.exit_code} pwd={r4.output!r}"
            )

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "hello.txt"
            src.write_text("hello from arl\n", encoding="utf-8")
            await deployment.runtime.upload(UploadRequest(source_path=str(src), target_path="/tmp/p2a-upload.txt"))
            r5 = await deployment.runtime.execute(Command(command=["cat", "/tmp/p2a-upload.txt"], timeout=30))
            if r5.exit_code != 0 or r5.stdout.strip() != "hello from arl":
                raise RuntimeError(f"upload/readback failed: exit={r5.exit_code} output={r5.stdout!r}")

        alive = await deployment.runtime.is_alive(timeout=10)
        if not alive.is_alive:
            raise RuntimeError(f"is_alive failed: {alive}")

        print("ARL_SDK_SMOKE: ok")
        return 0
    finally:
        await deployment.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Container image to boot in ARL")
    parser.add_argument("--gateway-url", default=os.getenv("ARL_GATEWAY_URL"))
    parser.add_argument("--namespace", default=os.getenv("ARL_NAMESPACE", "default"))
    parser.add_argument("--experiment-id", default=os.getenv("ARL_EXPERIMENT_ID", "p2a-uniagent-arl-smoke"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("ARL_TIMEOUT", "600")))
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=float(os.getenv("ARL_STARTUP_TIMEOUT", os.getenv("ARL_SWEREX_STARTUP_TIMEOUT", "240"))),
    )
    parser.add_argument("--max-replicas", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--keep-sandbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(run_smoke(parse_args())))


if __name__ == "__main__":
    main()
