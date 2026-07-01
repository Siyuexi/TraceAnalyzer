"""HTTP/static entry points for the unified P2A HTML dashboard."""

from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import secrets
import shutil
import socket
import sqlite3
import threading
import time
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from p2a.dashboard_adapter import (
    DashboardRequest,
    _artifact_root_candidates,
    build_dashboard_snapshot,
    read_dashboard_log,
    snapshot_to_json,
)
from p2a.eval_cache import (
    backup_path_for_delete,
    connect,
    connect_readonly,
    count_run_data_targets,
    delete_confirmation_phrase,
    delete_run_data_targets,
    init_db,
)


STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"


def _read_static(name: str) -> bytes:
    path = (STATIC_DIR / name).resolve()
    if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve():
        raise FileNotFoundError(name)
    return path.read_bytes()


def _static_version() -> str:
    try:
        values = [
            str(int((STATIC_DIR / name).stat().st_mtime_ns))
            for name in ("app.js", "styles.css")
        ]
    except OSError:
        return str(int(time.time_ns()))
    return "-".join(values)


def _index_html(*, embedded_snapshot: dict[str, Any] | None = None) -> bytes:
    html = _read_static("index.html").decode("utf-8")
    version = _static_version()
    html = html.replace('href="styles.css"', f'href="styles.css?v={version}"')
    html = html.replace('src="app.js"', f'src="app.js?v={version}"')
    if embedded_snapshot is not None:
        payload = _script_safe_json(embedded_snapshot)
        html = html.replace(
            "</head>",
            f"<script>window.__P2A_DASHBOARD_SNAPSHOT__ = {payload};</script>\n</head>",
        )
    return html.encode("utf-8")


def _script_safe_json(snapshot: dict[str, Any]) -> str:
    return (
        snapshot_to_json(snapshot)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


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


def _tree_change_token(root: Path, *, file_suffixes: tuple[str, ...] | None = None) -> tuple[Any, ...]:
    try:
        root_stat = root.stat()
    except OSError:
        return ((".", "missing", None, None),)
    root_is_dir = root.is_dir()
    entries: list[Any] = [(".", "dir" if root_is_dir else "file", int(root_stat.st_mtime_ns), int(root_stat.st_size))]
    if not root_is_dir:
        return tuple(entries)
    try:
        children = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    except OSError:
        return tuple(entries)
    for path in children:
        try:
            stat = path.stat()
        except OSError:
            continue
        is_dir = path.is_dir()
        if file_suffixes is not None and not is_dir and path.suffix not in file_suffixes:
            continue
        entries.append((path.relative_to(root).as_posix(), "dir" if is_dir else "file", int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(entries)


def _bonus_map_change_token(request: DashboardRequest) -> tuple[Any, ...]:
    paths: list[Path] = []

    def add(path: Path) -> None:
        expanded = path.expanduser()
        if expanded not in paths:
            paths.append(expanded)

    if request.bonus_map_dir is not None:
        add(request.bonus_map_dir)
    else:
        for root in _artifact_root_candidates(request):
            base = root / "bonus_maps"
            add(base / request.dataset if request.dataset else base)
    return tuple((str(path), _tree_change_token(path, file_suffixes=(".json",))) for path in paths)


def _snapshot_change_token(request: DashboardRequest) -> tuple[Any, ...]:
    parts: list[Any] = []
    for label, paths in (("rollouts", request.rollouts), ("details", request.details)):
        for path in paths:
            try:
                stat = path.stat()
                parts.append((label, str(path), int(stat.st_mtime_ns), int(stat.st_size)))
            except OSError:
                parts.append((label, str(path), None, None))
    if request.log_dir:
        parts.append(("log_dir", str(request.log_dir), _tree_change_token(request.log_dir)))
    if request.data_file:
        try:
            stat = request.data_file.stat()
            parts.append(("data_file", str(request.data_file), int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            parts.append(("data_file", str(request.data_file), None, None))
    if request.db_path:
        try:
            conn = connect_readonly(request.db_path, timeout=1.0)
            try:
                metric_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(quantitative_metrics)").fetchall()}
                fingerprint_sql = "MAX(q.fingerprint)" if "fingerprint" in metric_columns else "NULL"
                row = conn.execute(
                    f"""
                    SELECT
                      COUNT(*) AS n,
                      MAX(c.updated_at) AS max_cell_updated,
                      MAX(q.updated_at) AS max_metric_updated,
                      {fingerprint_sql} AS max_fingerprint
                    FROM run_cells c
                    LEFT JOIN quantitative_metrics q ON q.cell_id = c.id
                    """
                ).fetchone()
                parts.append((
                    "db",
                    str(request.db_path),
                    int(row["n"] or 0),
                    row["max_cell_updated"],
                    row["max_metric_updated"],
                    row["max_fingerprint"],
                    request.experiment_id,
                    request.provider_source,
                    request.dataset,
                ))
            finally:
                conn.close()
        except (FileNotFoundError, sqlite3.Error):
            parts.append(("db", str(request.db_path), "unavailable", request.experiment_id, request.provider_source, request.dataset))
    parts.append(("bonus_maps", _bonus_map_change_token(request)))
    parts.append(("params", request.tracking_mode, request.near_threshold, request.m_max, request.detail_limit))
    return tuple(parts)


def _backup_sqlite_database(db_path: Path, backup_path: Path) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source = connect_readonly(db_path, timeout=30.0)
    destination = sqlite3.connect(backup_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def _load_admin_password(path: Path | None) -> str | None:
    if path is not None:
        try:
            value = path.expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        if value:
            return value
    env_secret = os.environ.get("P2A_DASHBOARD_ADMIN_PASSWORD")
    if env_secret:
        return env_secret.strip()
    candidates = []
    env_file = os.environ.get("P2A_DASHBOARD_ADMIN_SECRET") or os.environ.get("P2A_DASHBOARD_ADMIN_SECRET_FILE")
    if env_file:
        candidates.append(Path(env_file))
    candidates.append(Path.cwd() / ".secrets" / "dashboard_admin.txt")
    for candidate in candidates:
        try:
            value = candidate.expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def make_handler(request: DashboardRequest, *, admin_password: str | None = None) -> type[BaseHTTPRequestHandler]:
    snapshot_cache: dict[str, Any] = {"payload": None, "change_token": None}
    snapshot_lock = threading.Lock()
    admin_tokens: set[str] = set()

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
            if parsed.path == "/api/auth/status":
                self._send_json({"ok": True, "admin_enabled": bool(admin_password), "admin": self._is_admin()})
                return
            if parsed.path == "/api/snapshot":
                params = parse_qs(parsed.query)
                self._send_snapshot(force=params.get("force", [""])[0].lower() in {"1", "true", "yes"})
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
                except sqlite3.OperationalError as exc:
                    status = HTTPStatus.SERVICE_UNAVAILABLE if "locked" in str(exc).lower() else HTTPStatus.INTERNAL_SERVER_ERROR
                    self._send_json(
                        {"ok": False, "error": "database_locked" if status == HTTPStatus.SERVICE_UNAVAILABLE else "sqlite_error", "detail": str(exc)},
                        status=status,
                    )
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested log source not found")
                return
            self._serve_static(parsed.path.lstrip("/"))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/auth/login":
                self._handle_login()
                return
            if parsed.path == "/api/auth/logout":
                self._handle_logout()
                return
            if parsed.path == "/api/delete/preview":
                self._handle_delete_preview()
                return
            if parsed.path == "/api/delete":
                self._handle_delete()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

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

        def _send_snapshot(self, *, force: bool = False) -> None:
            try:
                payload = self._build_or_cached_snapshot(force=force)
            except sqlite3.OperationalError as exc:
                status = HTTPStatus.SERVICE_UNAVAILABLE if "locked" in str(exc).lower() else HTTPStatus.INTERNAL_SERVER_ERROR
                self._send_json(
                    {"ok": False, "error": "database_locked" if status == HTTPStatus.SERVICE_UNAVAILABLE else "sqlite_error", "detail": str(exc)},
                    status=status,
                )
                return
            self._send_json(payload)

        def _build_or_cached_snapshot(self, *, force: bool = False) -> dict[str, Any]:
            change_token = _snapshot_change_token(request)
            cached = snapshot_cache.get("payload")
            if cached is not None and not force and snapshot_cache.get("change_token") == change_token:
                return cached
            if request.db_path is not None and not force:
                started = self._start_background_snapshot_build()
                if cached is not None:
                    reason = "snapshot_refresh_started" if started else "snapshot_build_in_progress"
                    return {**cached, "snapshot_status": {"stale": True, "reason": reason}}
                deferred = build_dashboard_snapshot(replace(request, defer_db_scoring=True))
                deferred["snapshot_status"] = {
                    "stale": False,
                    "reason": "snapshot_warming" if started else "snapshot_build_in_progress",
                }
                return deferred
            acquired = snapshot_lock.acquire(blocking=cached is None)
            if not acquired:
                return {**cached, "snapshot_status": {"stale": True, "reason": "snapshot_build_in_progress"}}
            try:
                current = snapshot_cache.get("payload")
                if current is not None and not force and snapshot_cache.get("change_token") == _snapshot_change_token(request):
                    return current
                payload = build_dashboard_snapshot(request)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and cached is not None:
                    return {**cached, "snapshot_status": {"stale": True, "reason": "database_locked", "detail": str(exc)}}
                raise
            else:
                snapshot_cache["payload"] = payload
                snapshot_cache["change_token"] = _snapshot_change_token(request)
                return payload
            finally:
                snapshot_lock.release()

        def _start_background_snapshot_build(self) -> bool:
            if not snapshot_lock.acquire(blocking=False):
                return False

            def worker() -> None:
                try:
                    payload = build_dashboard_snapshot(request)
                    snapshot_cache["payload"] = payload
                    snapshot_cache["change_token"] = _snapshot_change_token(request)
                except sqlite3.OperationalError:
                    pass
                finally:
                    snapshot_lock.release()

            threading.Thread(target=worker, daemon=True).start()
            return True

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("invalid JSON body")
            return payload if isinstance(payload, dict) else {}

        def _cookie_token(self) -> str | None:
            cookie = self.headers.get("Cookie") or ""
            for item in cookie.split(";"):
                name, _, value = item.strip().partition("=")
                if name == "p2a_admin":
                    return value
            return None

        def _is_admin(self) -> bool:
            token = self._cookie_token()
            return bool(token and token in admin_tokens)

        def _require_admin(self) -> bool:
            if self._is_admin():
                return True
            self._send_json({"ok": False, "error": "forbidden"}, status=HTTPStatus.FORBIDDEN)
            return False

        def _handle_login(self) -> None:
            if not admin_password:
                self._send_json({"ok": False, "error": "admin_not_configured"}, status=HTTPStatus.FORBIDDEN)
                return
            try:
                body = self._read_json_body()
            except ValueError as exc:
                self._send_json({"ok": False, "error": "bad_request", "detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            password = str(body.get("password") or "")
            if not hmac.compare_digest(password, admin_password):
                self._send_json({"ok": False, "error": "invalid_password"}, status=HTTPStatus.FORBIDDEN)
                return
            token = secrets.token_urlsafe(32)
            admin_tokens.add(token)
            self._send_json(
                {"ok": True, "admin_enabled": True, "admin": True},
                headers={"Set-Cookie": f"p2a_admin={token}; Path=/; HttpOnly; SameSite=Strict"},
            )

        def _handle_logout(self) -> None:
            token = self._cookie_token()
            if token:
                admin_tokens.discard(token)
            self._send_json(
                {"ok": True, "admin_enabled": bool(admin_password), "admin": False},
                headers={"Set-Cookie": "p2a_admin=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"},
            )

        def _delete_targets_from_body(self, body: dict[str, Any]) -> list[dict[str, Any]]:
            targets = body.get("targets")
            if isinstance(targets, list):
                return [item for item in targets if isinstance(item, dict)]
            target = body.get("target") if isinstance(body.get("target"), dict) else body
            return [target]

        def _handle_delete_preview(self) -> None:
            if not self._require_admin():
                return
            if request.db_path is None:
                self._send_json({"ok": False, "error": "db_required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                body = self._read_json_body()
                targets = self._delete_targets_from_body(body)
                conn = connect_readonly(request.db_path)
                try:
                    counts = count_run_data_targets(conn, targets)
                finally:
                    conn.close()
            except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
                self._send_json({"ok": False, "error": "bad_request", "detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, "counts": counts, "confirmation_phrase": delete_confirmation_phrase(counts)})

        def _handle_delete(self) -> None:
            if not self._require_admin():
                return
            if request.db_path is None:
                self._send_json({"ok": False, "error": "db_required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                body = self._read_json_body()
                targets = self._delete_targets_from_body(body)
                conn = connect_readonly(request.db_path)
                try:
                    counts = count_run_data_targets(conn, targets)
                finally:
                    conn.close()
                expected = delete_confirmation_phrase(counts)
                if str(body.get("confirmation") or "") != expected:
                    self._send_json({"ok": False, "error": "confirmation_required", "confirmation_phrase": expected, "counts": counts}, status=HTTPStatus.BAD_REQUEST)
                    return
                backup_path = backup_path_for_delete(request.db_path)
                _backup_sqlite_database(request.db_path, backup_path)
                writer = connect(request.db_path)
                try:
                    init_db(writer)
                    deleted = delete_run_data_targets(writer, targets)
                finally:
                    writer.close()
            except (ValueError, FileNotFoundError, sqlite3.Error, OSError) as exc:
                self._send_json({"ok": False, "error": "delete_failed", "detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            snapshot_cache["payload"] = None
            snapshot_cache["change_token"] = None
            self._send_json({"ok": True, "counts": deleted, "backup_path": str(backup_path)})

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
            headers: dict[str, str] | None = None,
        ) -> None:
            self._send_bytes(snapshot_to_json(payload).encode("utf-8"), "application/json; charset=utf-8", status=status, headers=headers)

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            headers: dict[str, str] | None = None,
        ) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Pragma", "no-cache")
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return P2ADashboardHandler


def _server_ip_candidates() -> list[str]:
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        host = value.strip()
        if not host or host.startswith("127.") or host in {"localhost", "0.0.0.0", "::1"}:
            return
        if host not in candidates:
            candidates.append(host)

    for key in ("P2A_DASHBOARD_PUBLIC_HOST", "HOST_IP", "HEAD_IP", "RAY_HEAD_IP", "MASTER_IP"):
        add(os.environ.get(key))

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            add(sock.getsockname()[0])
    except OSError:
        pass

    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            add(item[4][0])
    except OSError:
        pass

    return candidates


def _dashboard_urls(host: str, port: int) -> list[tuple[str, str]]:
    if host in {"0.0.0.0", "::", ""}:
        return [("Network", f"http://{candidate}:{port}") for candidate in _server_ip_candidates()]
    return [("URL", f"http://{host}:{port}")]


def serve_dashboard(request: DashboardRequest, *, host: str, port: int, admin_secret: Path | None = None) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(request, admin_password=_load_admin_password(admin_secret)))
    print("Serving unified P2A dashboard")
    print(f"  Bind: http://{host}:{port}")
    urls = _dashboard_urls(host, port)
    for label, url in urls:
        print(f"  {label}: {url}")
    if host in {"0.0.0.0", "::", ""} and not any(label == "Network" for label, _url in urls):
        print("  Network: unavailable; set P2A_DASHBOARD_PUBLIC_HOST or pass --host <server-ip>")
    print(flush=True)
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
    parser.add_argument("--data-file", type=Path, default=None, help="Dataset parquet used to fill missing issue descriptions and golden patches")
    parser.add_argument("--experiment-id", help="Filter DB rows to one experiment")
    parser.add_argument("--provider-source", help="Filter DB rows to one provider source")
    parser.add_argument("--dataset", help="Filter DB rows to one dataset")
    parser.add_argument("--tracking-mode", choices=["view_only", "view_and_bash"], default="view_and_bash")
    parser.add_argument("--near-threshold", type=float, default=0.5)
    parser.add_argument("--m-max", type=float, default=3.0)
    parser.add_argument("--detail-limit", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=None, help="Write a static snapshot instead of serving")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--admin-secret", type=Path, default=None, help="File containing the dashboard admin password")
    parser.add_argument("--snapshot-json", type=Path, default=None, help="Write one snapshot JSON and exit")


def request_from_args(args: argparse.Namespace) -> DashboardRequest:
    return DashboardRequest(
        rollouts=tuple(args.rollouts or ()),
        details=tuple(args.details or ()),
        db_path=args.db,
        log_dir=args.log_dir,
        bonus_map_dir=args.bonus_map_dir,
        data_file=args.data_file,
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

    if args.snapshot_json:
        snapshot = build_dashboard_snapshot(request)
        args.snapshot_json.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot_json.write_text(snapshot_to_json(snapshot, indent=2) + "\n", encoding="utf-8")
        print(args.snapshot_json)
        return 0
    if args.out_dir:
        snapshot = build_dashboard_snapshot(request)
        paths = write_static_dashboard(args.out_dir, snapshot)
        print(paths["html"])
        return 0
    if args.admin_secret is not None:
        serve_dashboard(request, host=args.host, port=args.port, admin_secret=args.admin_secret)
    else:
        serve_dashboard(request, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
