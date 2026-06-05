"""Image routing for ARL-backed R2E/Uni-Agent runs."""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MIRROR_REGISTRY = "pair-diag-cn-guangzhou.cr.volces.com"
DEFAULT_MIRROR_NAMESPACE = "code"
R2E_ENTERPRISE_TEMPLATE = "enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/{instance_number}:latest"
DEFAULT_ENTERPRISE_REPOS = frozenset({"coveragepy", "orange3"})


def repo_from_instance_id(instance_id: str) -> str | None:
    if "__" not in instance_id:
        return None
    repo = instance_id.rsplit("__", 1)[0].strip().lower()
    return repo or None


def suffix_from_instance_id(instance_id: str) -> str | None:
    if "__" not in instance_id:
        return None
    suffix = instance_id.rsplit("__", 1)[1].strip().lower()
    return suffix or None


def enterprise_r2e_image(instance_id: str) -> str | None:
    suffix = suffix_from_instance_id(instance_id)
    if not suffix:
        return None
    return R2E_ENTERPRISE_TEMPLATE.format(instance_number=suffix)


def mirror_image(
    docker_image: str,
    *,
    registry: str | None = None,
    namespace: str | None = None,
) -> str:
    registry = registry or os.getenv("ARL_MIRROR_REGISTRY", DEFAULT_MIRROR_REGISTRY)
    namespace = namespace or os.getenv("ARL_MIRROR_NAMESPACE", DEFAULT_MIRROR_NAMESPACE)
    image_path = docker_image.split("/", 1)[1] if "/" in docker_image else docker_image
    return f"{registry.rstrip('/')}/{namespace.strip('/')}/{image_path}"


def _env_enterprise_repos() -> set[str]:
    raw = os.getenv("P2A_ARL_ENTERPRISE_REPOS")
    if raw is None:
        return set(DEFAULT_ENTERPRISE_REPOS)
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _env_image_overrides() -> dict[str, str]:
    raw = os.getenv("P2A_ARL_IMAGE_OVERRIDES_JSON", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("P2A_ARL_IMAGE_OVERRIDES_JSON must be a JSON object") from exc
    if not isinstance(data, dict):
        raise ValueError("P2A_ARL_IMAGE_OVERRIDES_JSON must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def select_r2e_image(
    *,
    instance_id: str,
    docker_image: str | None = None,
    repo_name: str | None = None,
    prefer_enterprise_repos: set[str] | None = None,
    use_mirror: bool | None = None,
) -> str:
    """Select the ARL image for an R2E instance.

    Correctness overrides are explicit and repo-scoped. By default only the
    repos already verified as bad in the plan (coveragepy/orange3) route to the
    enterprise image; other ``namanjain12`` images route through the ARL mirror.
    A later full-corpus audit can extend ``P2A_ARL_ENTERPRISE_REPOS`` or provide
    exact ``P2A_ARL_IMAGE_OVERRIDES_JSON`` entries.
    """

    overrides = _env_image_overrides()
    if instance_id in overrides:
        return overrides[instance_id]

    repo = (repo_name or repo_from_instance_id(instance_id) or "").lower()
    if repo and repo in (prefer_enterprise_repos or _env_enterprise_repos()):
        enterprise = enterprise_r2e_image(instance_id)
        if enterprise:
            return enterprise

    if docker_image and docker_image.startswith("enterprise-public"):
        return docker_image

    if docker_image:
        mirror_enabled = use_mirror
        if mirror_enabled is None:
            mirror_enabled = os.getenv("P2A_ARL_DISABLE_MIRROR", "").lower() not in {"1", "true", "yes"}
        return mirror_image(docker_image) if mirror_enabled else docker_image

    enterprise = enterprise_r2e_image(instance_id)
    if enterprise:
        return enterprise
    raise ValueError(f"Cannot select ARL image for {instance_id!r}: missing docker_image and bad instance id")


def select_image_for_sample(sample_or_task: dict[str, Any], *, instance_id: str | None = None) -> str:
    """Select an image from a Uni-Agent sample or raw R2E task dict."""

    extra = sample_or_task.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    if not isinstance(extra, dict):
        extra = {}

    tools_kwargs = extra.get("tools_kwargs") if isinstance(extra.get("tools_kwargs"), dict) else {}
    env = tools_kwargs.get("env") if isinstance(tools_kwargs.get("env"), dict) else {}
    deployment = env.get("deployment") if isinstance(env.get("deployment"), dict) else {}
    reward = tools_kwargs.get("reward") if isinstance(tools_kwargs.get("reward"), dict) else {}
    metadata = reward.get("metadata") if isinstance(reward.get("metadata"), dict) else {}

    iid = (
        instance_id
        or sample_or_task.get("instance_id")
        or metadata.get("instance_id")
        or (f"{sample_or_task.get('repo_name')}__{str(sample_or_task.get('commit_hash', ''))[:10]}"
            if sample_or_task.get("repo_name") and sample_or_task.get("commit_hash")
            else None)
    )
    if not iid:
        raise ValueError("Cannot select image: sample has no instance_id")

    docker_image = deployment.get("image") or env.get("image") or sample_or_task.get("docker_image")
    repo_name = sample_or_task.get("repo_name") or metadata.get("repo")
    return select_r2e_image(instance_id=str(iid), docker_image=docker_image, repo_name=repo_name)
