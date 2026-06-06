"""Image routing for ARL-backed R2E/Uni-Agent runs.

R2E instances boot the **pair-diag mirror** of the original R2E ``namanjain12``
images: ``pair-diag-cn-guangzhou.cr.volces.com/code/{repo}_final:{commit}``. This
is the reference the old bonus-map report reproduces on, and the build wrapper
(``scripts/build_data.py r2e``) writes the full pair-diag ref into each row's
``deployment.image`` — so this module normally just passes that through (or
mirrors a raw ``namanjain12`` ref). The 2026-06-05 all-enterprise switch was
reverted: enterprise-public is a separate rebuild with divergent Python for
orange3/coveragepy/numpy. ``P2A_ARL_IMAGE_OVERRIDES_JSON`` pins an exact image
per instance.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MIRROR_REGISTRY = "pair-diag-cn-guangzhou.cr.volces.com"
DEFAULT_MIRROR_NAMESPACE = "code"


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


def mirror_image(docker_image: str) -> str:
    """Map an R2E ``namanjain12/{repo}_final:{commit}`` ref to its pair-diag mirror.

    A ref already on the mirror registry is passed through unchanged.
    """
    registry = os.getenv("ARL_MIRROR_REGISTRY", DEFAULT_MIRROR_REGISTRY)
    namespace = os.getenv("ARL_MIRROR_NAMESPACE", DEFAULT_MIRROR_NAMESPACE)
    if docker_image.startswith(registry):
        return docker_image
    image_path = docker_image.split("/", 1)[1] if "/" in docker_image else docker_image
    return f"{registry.rstrip('/')}/{namespace.strip('/')}/{image_path}"


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
) -> str:
    """Select the ARL image for an R2E instance — the pair-diag mirror.

    Resolution order:
      1. exact per-instance override (``P2A_ARL_IMAGE_OVERRIDES_JSON``);
      2. the ``docker_image`` carried by the row, mapped to its pair-diag mirror
         (a pair-diag ref passes through; a ``namanjain12`` ref is mirrored).
    The build wrapper always writes the pair-diag ref into the row, so (2) is the
    normal path; the full commit tag cannot be reconstructed from instance_id
    alone, so a missing ``docker_image`` is an error rather than a silent guess.
    """

    overrides = _env_image_overrides()
    if instance_id in overrides:
        return overrides[instance_id]

    if docker_image:
        return mirror_image(docker_image)

    raise ValueError(
        f"Cannot select ARL image for {instance_id!r}: no docker_image. The parquet "
        f"must carry the pair-diag image (scripts/build_data.py r2e) or set "
        f"P2A_ARL_IMAGE_OVERRIDES_JSON."
    )


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
    return select_r2e_image(instance_id=str(iid), docker_image=docker_image)
