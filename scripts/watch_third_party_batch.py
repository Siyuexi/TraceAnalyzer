#!/usr/bin/env python
"""Watch the unified third-party evaluation DB with per-model metrics."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from p2a.eval_dashboard import render_db_dashboard


_ALT_SCREEN_ON = "\x1b[?1049h"
_ALT_SCREEN_OFF = "\x1b[?1049l"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_CURSOR_HOME = "\x1b[H"
_CLEAR_TO_EOL = "\x1b[K"
_CLEAR_TO_END = "\x1b[J"


def _enter_alt_screen() -> None:
    sys.stdout.write(_ALT_SCREEN_ON)
    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()


def _leave_alt_screen() -> None:
    sys.stdout.write(_SHOW_CURSOR)
    sys.stdout.write(_ALT_SCREEN_OFF)
    sys.stdout.flush()


def _redraw(lines: list[str]) -> None:
    buf = [_CURSOR_HOME]
    for line in lines:
        buf.append(line)
        buf.append(_CLEAR_TO_EOL)
        buf.append("\n")
    buf.append(_CLEAR_TO_END)
    sys.stdout.write("".join(buf))
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/evals/traces.sqlite"))
    parser.add_argument("--experiment-id", help="Filter to one experiment id")
    parser.add_argument("--provider-source", help="Filter to one provider source")
    parser.add_argument("--dataset", help="Filter to one dataset")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("-n", "--interval", type=float, default=5.0, help="Refresh seconds")
    args = parser.parse_args()

    def render() -> list[str]:
        header = (
            f"DB: {args.db}   "
            f"experiment={args.experiment_id or '*'}   "
            f"provider={args.provider_source or '*'}   "
            f"dataset={args.dataset or '*'}   "
            f"updated {time.strftime('%H:%M:%S')}"
        )
        return [
            header,
            "",
            *render_db_dashboard(
                args.db,
                experiment_id=args.experiment_id,
                provider_source=args.provider_source,
                dataset=args.dataset,
            ),
        ]

    if args.once:
        print("\n".join(render()))
        return 0

    _enter_alt_screen()
    signal.signal(signal.SIGTERM, lambda *_: (_leave_alt_screen(), sys.exit(0)))
    try:
        while True:
            _redraw(render())
            time.sleep(max(args.interval, 1.0))
    except KeyboardInterrupt:
        pass
    finally:
        _leave_alt_screen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
