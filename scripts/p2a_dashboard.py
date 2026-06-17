#!/usr/bin/env python
"""Build a static P2A trajectory dashboard from rollout dumps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from p2a.dashboard import build_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path, help="Rollout dump file or directory containing .jsonl, .json, or .parquet dumps")
    parser.add_argument("--bonus-map-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tracking-mode", choices=("view_only", "view_and_bash"), default="view_and_bash")
    parser.add_argument("--near-threshold", type=float, default=0.5)
    parser.add_argument("--m-max", type=float, default=3.0)
    parser.add_argument("--watch", action="store_true", help="Rebuild the dashboard until interrupted")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between --watch rebuilds")
    args = parser.parse_args()

    while True:
        paths = build_dashboard(
            args.rollouts,
            args.bonus_map_dir,
            args.out_dir,
            tracking_mode=args.tracking_mode,
            near_threshold=args.near_threshold,
            m_max=args.m_max,
        )
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
        if not args.watch:
            break
        time.sleep(max(args.interval, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
