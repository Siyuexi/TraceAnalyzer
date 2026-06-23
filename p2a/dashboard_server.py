"""HTTP/static entry points for the unified P2A HTML dashboard."""

from __future__ import annotations

import argparse
import mimetypes
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from p2a.dashboard_adapter import DashboardRequest, build_dashboard_snapshot, read_dashboard_log, snapshot_to_json


STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"


def _read_static(name: str) -> bytes:
    path = (STATIC_DIR / name).resolve()
    if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve():
        raise FileNotFoundError(name)
    return path.read_bytes()


def _index_html(*, embedded_snapshot: dict[str, Any] | None = None) -> bytes:
    html = _read_static("index.html").decode("utf-8")
    if embedded_snapshot is not None:
        payload = snapshot_to_json(embedded_snapshot)
        html = html.replace(
            "</head>",
            f"<script>window.__P2A_DASHBOARD_SNAPSHOT__ = {payload};</script>\n</head>",
        )
    return html.encode("utf-8")


def write_static_dashboard(out_dir: Path, snapshot: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "index.html"
    snapshot_path = out_dir / "snapshot.json"
    app_path = out_dir / "app.js"
    css_path = out_dir / "styles.css"
    html_path.write_bytes(_index_html(embedded_snapshot=snapshot))
    snapshot_path.write_text(snapshot_to_json(snapshot, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(STATIC_DIR / "app.js", app_path)
    shutil.copyfile(STATIC_DIR / "styles.css", css_path)
    return {"html": html_path, "snapshot": snapshot_path, "app": app_path, "css": css_path}


def make_handler(request: DashboardRequest) -> type[BaseHTTPRequestHandler]:
    class P2ADashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(_index_html(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True, "schema_version": "p2a_unified_dashboard_v1"})
                return
            if parsed.path == "/api/snapshot":
                self._send_json(build_dashboard_snapshot(request))
                return
            if parsed.path == "/api/log":
                params = parse_qs(parsed.query)
                run_id = params.get("run_id", [""])[0]
                source = params.get("source", ["run.log"])[0]
                if not run_id:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing run_id")
                    return
                try:
                    self._send_json(read_dashboard_log(request, run_id=run_id, source=source))
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested log source not found")
                return
            self._serve_static(parsed.path.lstrip("/"))

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _serve_static(self, name: str) -> None:
            try:
                payload = _read_static(name)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime_type, _encoding = mimetypes.guess_type(name)
            self._send_bytes(payload, mime_type or "application/octet-stream")

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send_bytes(snapshot_to_json(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return P2ADashboardHandler


def serve_dashboard(request: DashboardRequest, *, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(request))
    print(f"Serving unified P2A dashboard on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def add_dashboard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("rollouts", nargs="*", type=Path, help="Rollout JSONL/JSON/parquet file or dump directory")
    parser.add_argument("--details", action="append", type=Path, default=[], help="Scored details JSONL/JSON file or directory")
    parser.add_argument("--db", type=Path, default=None, help="Unified eval SQLite DB")
    parser.add_argument("--log-dir", type=Path, default=None, help="Uni-Agent run directory root")
    parser.add_argument("--bonus-map-dir", type=Path, default=None, help="Directory containing <instance_id>.json bonus maps")
    parser.add_argument("--experiment-id", help="Filter DB rows to one experiment")
    parser.add_argument("--provider-source", help="Filter DB rows to one provider source")
    parser.add_argument("--dataset", help="Filter DB rows to one dataset")
    parser.add_argument("--tracking-mode", choices=["view_only", "view_and_bash"], default="view_and_bash")
    parser.add_argument("--near-threshold", type=float, default=0.5)
    parser.add_argument("--m-max", type=float, default=3.0)
    parser.add_argument("--detail-limit", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=None, help="Write a static snapshot instead of serving")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--snapshot-json", type=Path, default=None, help="Write one snapshot JSON and exit")


def request_from_args(args: argparse.Namespace) -> DashboardRequest:
    return DashboardRequest(
        rollouts=tuple(args.rollouts or ()),
        details=tuple(args.details or ()),
        db_path=args.db,
        log_dir=args.log_dir,
        bonus_map_dir=args.bonus_map_dir,
        experiment_id=args.experiment_id,
        provider_source=args.provider_source,
        dataset=args.dataset,
        tracking_mode=args.tracking_mode,
        near_threshold=args.near_threshold,
        m_max=args.m_max,
        detail_limit=args.detail_limit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve or build the unified P2A HTML dashboard.")
    add_dashboard_args(parser)
    args = parser.parse_args(argv)
    request = request_from_args(args)
    snapshot = build_dashboard_snapshot(request)

    if args.snapshot_json:
        args.snapshot_json.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot_json.write_text(snapshot_to_json(snapshot, indent=2) + "\n", encoding="utf-8")
        print(args.snapshot_json)
        return 0
    if args.out_dir:
        paths = write_static_dashboard(args.out_dir, snapshot)
        print(paths["html"])
        return 0
    serve_dashboard(request, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
