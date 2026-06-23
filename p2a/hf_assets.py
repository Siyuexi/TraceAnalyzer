"""Shared HuggingFace asset locations for TraceAnalyzer project scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def _shared_src_root() -> Path:
    shared_src = os.environ.get("P2A_SHARED_SRC_ROOT")
    if shared_src:
        return Path(shared_src).expanduser().resolve()
    return SRC_ROOT


def _resolve_shared_relative(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_shared_src_root() / path).resolve()


def shared_root() -> Path:
    override = os.environ.get("P2A_SHARED_ROOT")
    if override:
        return _resolve_shared_relative(override)
    return _shared_src_root().parent.parent


def shared_datasets_dir() -> Path:
    override = os.environ.get("P2A_DATASETS_DIR")
    if override:
        return _resolve_shared_relative(override)
    return shared_root() / "datasets"


def shared_p2a_data_dir() -> Path:
    """Return the shared TraceAnalyzer data directory used by shell setup."""
    override = os.environ.get("DATA")
    if override:
        return _resolve_shared_relative(override)
    return shared_datasets_dir() / "p2a"


def project_artifacts_dir() -> Path:
    """Return the TraceAnalyzer-local artifact directory."""
    override = os.environ.get("P2A_ARTIFACTS_DIR") or os.environ.get("P2A_PROJECT_DATA_DIR")
    if override:
        return _resolve_shared_relative(override)
    return _shared_src_root() / "data"


def project_data_dir() -> Path:
    """Compatibility alias for the TraceAnalyzer-local artifact directory."""
    return project_artifacts_dir()


def shared_models_dir() -> Path:
    override = os.environ.get("P2A_MODELS_DIR")
    if override:
        return _resolve_shared_relative(override)
    return shared_root() / "models"


def shared_bonus_maps_dir() -> Path:
    """The training bonus-map directory (read by training, written by precompute).

    Single source of truth: ``P2A_BONUS_MAP_DIR`` if set, else data/bonus_maps.
    """
    override = os.environ.get("P2A_BONUS_MAP_DIR")
    if override:
        return _resolve_shared_relative(override)
    return project_artifacts_dir() / "bonus_maps"


def hf_repo_basename(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def shared_dataset_path(repo_id: str, split: str | None = None) -> Path:
    base = shared_datasets_dir() / hf_repo_basename(repo_id)
    return base / split if split else base


def shared_model_path(repo_id: str | None = None) -> Path:
    repo = repo_id or os.environ.get("P2A_MODEL_REPO") or DEFAULT_MODEL_REPO
    return shared_models_dir() / hf_repo_basename(repo)


def _load_from_existing_path(path: Path, split: str | None):
    from datasets import DatasetDict, load_dataset, load_from_disk

    if not path.exists():
        return None

    try:
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict) and split:
            return loaded.get(split)
        return loaded
    except Exception:  # noqa: BLE001 - fall through to local dataset-file loading
        pass

    try:
        return load_dataset(str(path), split=split)
    except Exception:  # noqa: BLE001 - caller will download from HF
        return None


def load_shared_dataset(repo_id: str, *, split: str, **kwargs: Any):
    """Load a dataset split from ../../datasets, downloading and saving if absent."""
    from datasets import load_dataset

    split_path = shared_dataset_path(repo_id, split)
    dataset_path = shared_dataset_path(repo_id)

    loaded = _load_from_existing_path(split_path, None)
    if loaded is not None:
        print(f"Loading {repo_id}:{split} from {split_path}", flush=True)
        return loaded

    loaded = _load_from_existing_path(dataset_path, split)
    if loaded is not None:
        print(f"Loading {repo_id}:{split} from {dataset_path}", flush=True)
        return loaded

    root = shared_datasets_dir()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}:{split} into shared dataset cache {split_path}", flush=True)
    dataset = load_dataset(repo_id, split=split, cache_dir=str(root / ".hf_cache"), **kwargs)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(split_path))
    return dataset


def ensure_shared_model(repo_id: str | None = None, local_dir: str | Path | None = None) -> Path:
    """Return a local model path under ../../models, downloading if it is missing."""
    from huggingface_hub import snapshot_download

    repo = repo_id or os.environ.get("P2A_MODEL_REPO") or DEFAULT_MODEL_REPO
    target = Path(local_dir).expanduser().resolve() if local_dir else shared_model_path(repo)

    if target.is_dir() and any(target.iterdir()):
        print(f"Using local model at {target}", flush=True)
        return target

    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo} into shared model cache {target}", flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        cache_dir=shared_models_dir() / ".hf_cache",
    )
    return target
