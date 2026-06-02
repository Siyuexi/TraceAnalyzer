"""Import-path compatibility for migrated P2A precompute scripts.

The active source tree is now ``src/`` while the old rLLM implementation is
preserved as ``src-backup/``. Bonus-map construction uses Uni-Agent sandboxes
and still imports the old SWE trace helpers as pure instrumentation/parsing
utilities, so these scripts need all three source roots importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
SRC_BACKUP = PROJECT_ROOT / "src-backup"
UNI_AGENT = SRC_ROOT / "uni-agent"


def ensure_paths() -> None:
    for path in (SRC_ROOT, UNI_AGENT, SRC_BACKUP):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def trace_module_path() -> Path:
    return SRC_BACKUP / "rllm" / "environments" / "swe" / "trace.py"


ensure_paths()
