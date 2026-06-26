"""Dataset keys, aliases, and safety checks for P2A data files."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

R2E_DATA_SOURCE = "r2e-gym-subset"
SWEBENCH_VERIFIED_DATA_SOURCE = "swebench-verified"
SWEBENCH_HARD_DATA_SOURCE = "swebench-hard"
SWEBENCH_PRO_DATA_SOURCE = "swebench-pro"

SUPPORTED_EVAL_DATASETS = {
    R2E_DATA_SOURCE,
    SWEBENCH_VERIFIED_DATA_SOURCE,
    SWEBENCH_HARD_DATA_SOURCE,
    SWEBENCH_PRO_DATA_SOURCE,
}
TRAINING_DATASETS = {R2E_DATA_SOURCE}
EVAL_ONLY_DATASETS = {
    SWEBENCH_VERIFIED_DATA_SOURCE,
    SWEBENCH_HARD_DATA_SOURCE,
    SWEBENCH_PRO_DATA_SOURCE,
}

DATASET_ALIASES = {
    "hard": SWEBENCH_HARD_DATA_SOURCE,
    "swe-bench-hard": SWEBENCH_HARD_DATA_SOURCE,
    "verified": SWEBENCH_VERIFIED_DATA_SOURCE,
    "swe-bench-verified": SWEBENCH_VERIFIED_DATA_SOURCE,
    "r2e": R2E_DATA_SOURCE,
    "r2e-gym": R2E_DATA_SOURCE,
    "pro": SWEBENCH_PRO_DATA_SOURCE,
    "swe-pro": SWEBENCH_PRO_DATA_SOURCE,
    "swebenchpro": SWEBENCH_PRO_DATA_SOURCE,
    "swebench-pro": SWEBENCH_PRO_DATA_SOURCE,
    "swe-bench-pro": SWEBENCH_PRO_DATA_SOURCE,
}


def canonical_dataset(name: str) -> str:
    dataset = DATASET_ALIASES.get(str(name).strip(), str(name).strip())
    if dataset not in SUPPORTED_EVAL_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_EVAL_DATASETS))
        raise ValueError(f"dataset.name must be one of: {supported}")
    return dataset


def parse_string_list(value: Any) -> list[str]:
    """Parse dataset list fields that may be JSON, Python repr, or native lists."""

    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str):
        text = str(value).strip()
        return [text] if text else []

    text = value.strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list | tuple | set):
            return [str(item) for item in parsed if str(item).strip()]
        if parsed is None:
            return []
        parsed_text = str(parsed).strip()
        return [parsed_text] if parsed_text else []
    return [text]


def last_nonempty_line(text: Any) -> str | None:
    """Return the last non-empty, stripped line of ``text`` (or None)."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def swebench_pro_repo_path(*sources: Any) -> str:
    """Resolve the SWE-Bench-Pro repo path, defaulting to ``/app``."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("swebench_pro_repo_path", "repo_path"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "/app"


def _row_data_source(row: dict[str, Any]) -> str:
    for value in (row.get("data_source"), row.get("dataset")):
        if isinstance(value, str) and value:
            return canonical_dataset(value) if value in DATASET_ALIASES else value

    extra = row.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra = {}
    if isinstance(extra, dict):
        value = extra.get("data_source") or extra.get("dataset")
        if isinstance(value, str) and value:
            return canonical_dataset(value) if value in DATASET_ALIASES else value
        tools = extra.get("tools_kwargs")
        reward = tools.get("reward") if isinstance(tools, dict) else None
        metadata = reward.get("metadata") if isinstance(reward, dict) else None
        value = metadata.get("data_source") if isinstance(metadata, dict) else None
        if isinstance(value, str) and value:
            return canonical_dataset(value) if value in DATASET_ALIASES else value
    return ""


def assert_training_data_sources_allowed(paths: str | Path | Iterable[str | Path]) -> None:
    """Reject eval-only datasets when a parquet is used as RL training data."""

    if isinstance(paths, str | Path):
        candidates = [paths]
    else:
        candidates = list(paths)

    disallowed: dict[str, set[str]] = {}
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.exists() or path.suffix.lower() != ".parquet":
            continue
        import pandas as pd

        try:
            df = pd.read_parquet(path, columns=["data_source", "extra_info"])
        except Exception:  # noqa: BLE001 - older/synthetic files may not carry both helper columns
            df = pd.read_parquet(path)
        if "data_source" in df.columns and df["data_source"].notna().all():
            sources = {
                canonical_dataset(value) if value in DATASET_ALIASES else value
                for value in df["data_source"].astype(str).unique()
            }
        else:
            sources = {_row_data_source(row) for row in df.to_dict(orient="records")}
        bad = {source for source in sources if source in EVAL_ONLY_DATASETS}
        if bad:
            disallowed[str(path)] = bad

    if disallowed:
        detail = "; ".join(f"{path}: {', '.join(sorted(sources))}" for path, sources in sorted(disallowed.items()))
        raise ValueError(f"Eval-only dataset cannot be used as RL training data ({detail})")
