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


def ordered_unique_strings(items: Iterable[Any]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def selector_file(selector: Any) -> str:
    """Return the test file part of a pytest selector."""
    return str(selector or "").split("::", 1)[0].strip()


def selector_files(selectors: Iterable[Any]) -> list[str]:
    """Return ordered unique files for pytest file or node-id selectors."""
    return ordered_unique_strings(selector_file(selector) for selector in selectors)


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


def _canonical_or_raw_data_source(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    return canonical_dataset(text) if text in DATASET_ALIASES else text


def _row_data_source(row: dict[str, Any]) -> str:
    sources: list[str] = []
    for value in (row.get("data_source"), row.get("dataset")):
        source = _canonical_or_raw_data_source(value)
        if source:
            sources.append(source)

    extra = row.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra = {}
    if isinstance(extra, dict):
        for value in (extra.get("data_source"), extra.get("dataset")):
            source = _canonical_or_raw_data_source(value)
            if source:
                sources.append(source)
        tools = extra.get("tools_kwargs")
        reward = tools.get("reward") if isinstance(tools, dict) else None
        metadata = reward.get("metadata") if isinstance(reward, dict) else None
        if isinstance(metadata, dict):
            for value in (metadata.get("data_source"), metadata.get("dataset")):
                source = _canonical_or_raw_data_source(value)
                if source:
                    sources.append(source)
    for source in sources:
        if source in EVAL_ONLY_DATASETS:
            return source
    return sources[0] if sources else ""


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
        # Always derive sources through _row_data_source so nested eval-only markers are
        # enforced even when a top-level data_source column is present.
        sources = {_row_data_source(row) for row in df.to_dict(orient="records")}
        bad = {source for source in sources if source in EVAL_ONLY_DATASETS}
        if bad:
            disallowed[str(path)] = bad

    if disallowed:
        detail = "; ".join(f"{path}: {', '.join(sorted(sources))}" for path, sources in sorted(disallowed.items()))
        raise ValueError(f"Eval-only dataset cannot be used as RL training data ({detail})")
