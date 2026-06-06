"""R2E skip-case registry — external adapter, does NOT modify uni-agent.

Instances listed in ``config/bad_instances.json`` (under ``src/``) have F2P/P2P
behavior that does not match the R2E metadata expectation: the buggy revision
must FAIL the F2P test and the fixed revision must PASS the P2P test. When that
contract cannot be reproduced (after the full ``test_startup_fixups`` strategy),
the instance carries no valid outcome reward, so **all ARL-based paths — plain
RL data, P2A training data, and P2A bonus-map precompute — must exclude it.**

This module is pure config access (stdlib only). The dataset is filtered
out-of-band by ``scripts/build_data.py r2e`` (skip-filter step); uni-agent is never patched.
"""

from __future__ import annotations

import json
from pathlib import Path

# src/config/bad_instances.json (this module lives in src/p2a/)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "bad_instances.json"


def load_skip_records(path: str | Path | None = None) -> list[dict]:
    """Return the raw skip entries (``{id, repo, reason, source}``)."""
    cfg = Path(path) if path else DEFAULT_CONFIG
    if not cfg.exists():
        return []
    data = json.loads(cfg.read_text(encoding="utf-8"))
    return list(data.get("skip", []))


def load_skip_ids(path: str | Path | None = None) -> set[str]:
    """Return the set of instance_ids to exclude from training."""
    return {r["id"] for r in load_skip_records(path) if r.get("id")}


def should_skip(instance_id: str, path: str | Path | None = None) -> bool:
    return instance_id in load_skip_ids(path)
