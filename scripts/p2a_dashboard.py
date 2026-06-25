#!/usr/bin/env python
"""Serve or build the unified P2A HTML dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from p2a.dashboard_server import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
