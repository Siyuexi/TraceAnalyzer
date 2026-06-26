"""List and optionally preflight SWE-Bench-Pro image references.

This script does not mirror images by itself. It turns the Phase-1 parquet into
an auditable source/mirror image list and can verify registry manifests with the
local Docker CLI.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.images import mirror_image  # noqa: E402
from p2a.hf_assets import shared_p2a_data_dir  # noqa: E402


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    extra = _as_dict(row.get("extra_info"))
    tools = _as_dict(extra.get("tools_kwargs"))
    reward = _as_dict(tools.get("reward"))
    metadata = _as_dict(reward.get("metadata"))
    return metadata


def image_records(path: Path, *, repo_language: str = "python", limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if repo_language and "repo_language" in df.columns:
        df = df[df["repo_language"].astype(str).str.lower() == repo_language.lower()]
    if offset:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.head(limit)

    records = []
    for row in df.to_dict(orient="records"):
        metadata = _metadata(row)
        source = row.get("docker_image") or metadata.get("docker_image")
        if not source and row.get("dockerhub_tag"):
            source = f"jefzda/sweap-images:{row['dockerhub_tag']}"
        mirror = row.get("image") or metadata.get("image")
        if not mirror and source:
            mirror = mirror_image(str(source))
        records.append(
            {
                "instance_id": row.get("instance_id") or metadata.get("instance_id"),
                "repo": row.get("repo") or metadata.get("repo"),
                "repo_language": row.get("repo_language") or metadata.get("repo_language"),
                "source_image": source,
                "mirror_image": mirror,
            }
        )
    return records


def manifest_status(image: str, *, timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["docker", "manifest", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "unknown", "error": "docker CLI not found"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"manifest inspect timed out after {timeout}s"}
    if proc.returncode == 0:
        return {"status": "present", "error": ""}
    return {"status": "missing", "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""}


def mirror_commands(records: list[dict[str, Any]]) -> list[str]:
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Login first if needed:",
        "#   docker login pair-diag-cn-guangzhou.cr.volces.com",
        "",
    ]
    seen: set[tuple[str, str]] = set()
    for record in records:
        source = str(record.get("source_image") or "").strip()
        mirror = str(record.get("mirror_image") or "").strip()
        if not source or not mirror:
            continue
        key = (source, mirror)
        if key in seen:
            continue
        seen.add(key)
        instance_id = str(record.get("instance_id") or "")
        commands.extend(
            [
                f"# {instance_id}",
                f"docker pull {shlex.quote(source)}",
                f"docker tag {shlex.quote(source)} {shlex.quote(mirror)}",
                f"docker push {shlex.quote(mirror)}",
                "",
            ]
        )
    return commands


def _default_parquet() -> Path:
    return shared_p2a_data_dir() / "swe_bench_pro.parquet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="?", type=Path, default=_default_parquet())
    parser.add_argument("--repo-language", default="python", help="Filter by repo_language; empty string disables filter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--check-manifests", action="store_true", help="Run docker manifest inspect for source and mirror refs")
    parser.add_argument("--manifest-timeout", type=int, default=30)
    parser.add_argument("--emit-mirror-script", action="store_true", help="Emit docker pull/tag/push commands for the selected refs")
    parser.add_argument("--jsonl", action="store_true", help="Emit JSONL instead of a tab-separated table")
    parser.add_argument("--fail-on-missing-mirror", action="store_true")
    args = parser.parse_args(argv)

    records = image_records(args.parquet, repo_language=args.repo_language, limit=args.limit, offset=args.offset)
    if args.emit_mirror_script:
        print("\n".join(mirror_commands(records)))
        return 0

    missing_mirror = 0
    for record in records:
        if args.check_manifests:
            record["source_manifest"] = manifest_status(str(record["source_image"]), timeout=args.manifest_timeout)
            record["mirror_manifest"] = manifest_status(str(record["mirror_image"]), timeout=args.manifest_timeout)
            missing_mirror += record["mirror_manifest"]["status"] != "present"
        if args.jsonl:
            print(json.dumps(record, sort_keys=True))
        else:
            if record is records[0]:
                fields = ["instance_id", "repo", "repo_language", "source_image", "mirror_image"]
                if args.check_manifests:
                    fields += ["source_status", "mirror_status"]
                print("\t".join(fields))
            row = [
                str(record.get("instance_id") or ""),
                str(record.get("repo") or ""),
                str(record.get("repo_language") or ""),
                str(record.get("source_image") or ""),
                str(record.get("mirror_image") or ""),
            ]
            if args.check_manifests:
                row += [
                    str(record["source_manifest"]["status"]),
                    str(record["mirror_manifest"]["status"]),
                ]
            print("\t".join(row))
    if args.check_manifests:
        print(f"checked={len(records)} missing_mirror={missing_mirror}", file=sys.stderr)
    return 1 if args.fail_on_missing_mirror and missing_mirror else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
