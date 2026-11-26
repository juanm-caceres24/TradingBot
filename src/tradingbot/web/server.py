"""Zero-dependency HTTP/1.1 + REST API + WebSocket server for the 24/7 Web Dashboard.

Exposes:
- GET  /                                        -> Web Dashboard UI
- GET  /static/<file>                           -> Static CSS/JS assets (path-traversal safe)
- GET  /health                                  -> Liveness probe
- GET  /api/config                              -> Non-secret dashboard display config
- GET  /api/assets                              -> Multi-asset watchlist & live status
- GET  /api/assets/<asset_id>/latest?visible=N  -> Candles, indicators, prediction & analysis
- GET  /api/alerts?since_id=&asset_id=&unacknowledged=&limit= -> Cursor-based Alerts API
- POST /api/alerts/<alert_id>/ack               -> Mark alert acknowledged
- POST /api/actions/poll                        -> Trigger immediate poll cycle (all or 1 asset)
- GET  /ws/live                                 -> RFC 6455 WebSocket stream (all live events)
- GET  /ws/alerts                               -> RFC 6455 WebSocket stream (alerts only)
"""

from __future__ import annotations

import http.server
import json
import mimetypes
import queue
import re
import socket
import struct
import threading
import urllib.parse
from pathlib import Path
from typing import Any

import tradingbot
from tradingbot.runtime import BotOrchestrator
from tradingbot.web.websocket import (
    OPCODE_CLOSE,
    OPCODE_PING,
    WebSocketHub,
    compute_accept_key,
    encode_close,
    encode_pong,
    encode_text,
    read_frame,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_ASSET_LATEST_RE = re.compile(r"^/api/assets/([a-z0-9][a-z0-9_-]{0,63})/latest$")
_ALERT_ACK_RE = re.compile(r"^/api/alerts/([1-9][0-9]*)/ack$")
_MAX_BODY_BYTES = 1_048_576


def _parse_bool_query(value: str, param_name: str) -> bool:
    low = value.strip().lower()
    if low in ("1", "true", "yes"):
        return True
    if low in ("0", "false", "no"):
        return False
    raise ValueError(f"Query parameter '{param_name}' must be boolean, got {value!r}")


def _parse_int_query(value: str, param_name: str, *, min_value: int = 0) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Query parameter '{param_name}' must be an integer, got {value!r}") from exc
    if parsed < min_value:
        raise ValueError(f"Query parameter '{param_name}' must be >= {min_value}, got {parsed}")
    return parsed


class _DashboardHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[http.server.BaseHTTPRequestHandler],
        *,
        orchestrator: BotOrchestrator,
        ws_hub: WebSocketHub,
        static_dir: Path,
    ) -> None:
        self.orchestrator = orchestrator
        self.ws_hub = ws_hub
        self.static_dir = static_dir
        super().__init__(server_address, handler_cls)


class _DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    server: _DashboardHTTPServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep server stdout focused on tradingbot status & alerts

    def _add_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status: int, body: dict[str, Any] | list[Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._add_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self._add_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, rel_path: str) -> None:
        static_root = self.server.static_dir.resolve()
        candidate = (static_root / rel_path).resolve()
        if not candidate.is_relative_to(static_root) or not candidate.is_file():
            self._send_json(404, {"error": "Static asset not found"})
            return
        mime, _ = mimetypes.guess_type(str(candidate))
        if candidate.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif candidate.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif candidate.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        else:
            content_type = mime or "application/octet-stream"
        self._send_bytes(200, content_type, candidate.read_bytes())

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        if length < 0 or length > _MAX_BODY_BYTES:
            raise ValueError(f"Request body exceeds limit ({_MAX_BODY_BYTES} bytes)")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("JSON request body must be an object")
        return decoded

    def _handle_websocket(self, channel: str) -> None:
        upgrade = self.headers.get("Upgrade", "").strip().lower()
        ws_key = self.headers.get("Sec-WebSocket-Key", "").strip()
        if upgrade != "websocket" or not ws_key:
            self._send_json(400, {"error": "Expected valid WebSocket upgrade headers"})
            return

        if channel == "alerts" and not self.server.orchestrator.config.notifications.api_enabled:
            self._send_json(403, {"error": "External alerts API is disabled in configuration"})
            return

        try:
            accept = compute_accept_key(ws_key)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii")

        sock: socket.socket = self.connection
        sock.sendall(handshake)
        # Prevent BaseHTTPRequestHandler from attempting another HTTP loop on this socket.
        self.close_connection = True

        sub = self.server.ws_hub.register(channel)
        stop_writer = threading.Event()
        send_lock = threading.Lock()

        def _safe_send(frame: bytes) -> bool:
            with send_lock:
                try:
                    sock.sendall(frame)
                    return True
                except OSError:
                    return False

        # Send an immediate hello event so clients confirm channel readiness.
        _safe_send(encode_text(json.dumps({"type": "connected", "channel": channel})))

        def _writer_loop() -> None:
            while not stop_writer.is_set():
                try:
                    frame = sub.send_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if frame is None:
                    _safe_send(encode_close(1001))
                    stop_writer.set()
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                if not _safe_send(frame):
                    stop_writer.set()
                    return

        writer_thread = threading.Thread(target=_writer_loop, daemon=True)
        writer_thread.start()

        try:
            sock.settimeout(0.5)
            while not stop_writer.is_set():
                try:
                    frame_res = read_frame(sock, allow_idle_timeout=True)
                except TimeoutError:
                    continue
                except OSError:
                    break
                if frame_res is None:
                    break
                opcode, payload = frame_res
                if opcode == OPCODE_CLOSE:
                    close_code = (
                        struct.unpack("!H", payload[:2])[0]
                        if len(payload) >= 2
                        else 1000
                    )
                    _safe_send(encode_close(close_code))
                    break
                if opcode == OPCODE_PING:
                    _safe_send(encode_pong(payload))
        finally:
            stop_writer.set()
            self.server.ws_hub.unregister(sub)
            writer_thread.join(timeout=1.5)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._add_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        orch = self.server.orchestrator

        if path == "/ws/live":
            self._handle_websocket("live")
            return
        if path == "/ws/alerts":
            self._handle_websocket("alerts")
            return
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return

        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "version": tradingbot.__version__,
                    "assets_count": len(orch.runners),
                    "ws_clients": self.server.ws_hub.client_count(),
                },
            )
            return

        if path == "/api/config":
            self._send_json(
                200,
                {
                    "title": orch.config.web.title,
                    "visible_candles": orch.config.web.visible_candles,
                    "api_enabled": orch.config.notifications.api_enabled,
                    "version": tradingbot.__version__,
                },
            )
            return

        if path == "/api/assets":
            summaries: list[dict[str, Any]] = []
            for asset_id, runner in orch.runners.items():
                snap = orch.state.get(asset_id)
                unread = orch.store.count_alerts(
                    asset_id=asset_id,
                    unacknowledged_only=True,
                )
                if snap is not None:
                    summaries.append(snap.summary_dict(unread_alerts=unread))
                else:
                    a = runner.asset
                    summaries.append(
                        {
                            "asset_id": a.id,
                            "provider": a.provider.name,
                            "symbol": a.market.symbol,
                            "timeframe": a.market.timeframe,
                            "mode": a.runtime.mode,
                            "enabled": a.enabled,
                            "candle_count": 0,
                            "last_price": None,
                            "last_open_time_ms": None,
                            "last_is_closed": None,
                            "action": None,
                            "confidence": None,
                            "expected_move_pct": None,
                            "signal": None,
                            "nn_health": {},
                            "alert_status": {},
                            "unread_alerts": unread,
                            "last_sync_ms": None,
                            "last_error": None,
                        }
                    )
            self._send_json(200, {"assets": summaries})
            return

        asset_match = _ASSET_LATEST_RE.match(path)
        if asset_match:
            asset_id = asset_match.group(1)
            snap = orch.state.get(asset_id)
            if snap is None:
                if asset_id not in orch.runners:
                    self._send_json(404, {"error": f"Unknown asset '{asset_id}'"})
                    return
                a = orch.runners[asset_id].asset
                self._send_json(
                    200,
                    {
                        "asset_id": a.id,
                        "provider": a.provider.name,
                        "symbol": a.market.symbol,
                        "timeframe": a.market.timeframe,
                        "mode": a.runtime.mode,
                        "candles": [],
                        "indicators": {},
                        "indicator_plots": {ind.id: ind.plot for ind in a.indicators},
                        "indicator_types": {ind.id: ind.type for ind in a.indicators},
                        "prediction": None,
                        "nn_health": {},
                        "alert_status": {},
                        "last_sync_ms": None,
                        "last_error": None,
                    },
                )
                return

            try:
                visible = (
                    _parse_int_query(query["visible"], "visible", min_value=5)
                    if "visible" in query
                    else orch.config.web.visible_candles
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            self._send_json(200, snap.detail_dict(visible_candles=visible))
            return

        if path == "/api/alerts":
            if not orch.config.notifications.api_enabled:
                self._send_json(403, {"error": "External alerts API is disabled in configuration"})
                return
            try:
                asset_id_filter = query.get("asset_id") or None
                since_id = (
                    _parse_int_query(query["since_id"], "since_id", min_value=0)
                    if "since_id" in query
                    else None
                )
                unack = (
                    _parse_bool_query(query["unacknowledged"], "unacknowledged")
                    if "unacknowledged" in query
                    else False
                )
                limit = (
                    _parse_int_query(query["limit"], "limit", min_value=1)
                    if "limit" in query
                    else 100
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            alerts = orch.store.get_alerts(
                asset_id=asset_id_filter,
                since_id=since_id,
                unacknowledged_only=unack,
                limit=min(limit, 1000),
            )
            items = []
            for a in alerts:
                d = a.to_dict()
                try:
                    d["payload"] = json.loads(a.payload_json) if a.payload_json else {}
                except json.JSONDecodeError:
                    d["payload"] = {}
                items.append(d)
            last_id = max((a.alert_id or 0) for a in alerts) if alerts else (since_id or 0)
            self._send_json(200, {"alerts": items, "last_id": last_id, "count": len(items)})
            return

        self._send_json(404, {"error": f"Endpoint not found: {path}"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        orch = self.server.orchestrator

        ack_match = _ALERT_ACK_RE.match(path)
        if ack_match:
            if not orch.config.notifications.api_enabled:
                self._send_json(403, {"error": "External alerts API is disabled in configuration"})
                return
            alert_id = int(ack_match.group(1))
            updated = orch.store.acknowledge_alert(alert_id)
            if not updated and not orch.store.alert_exists(alert_id):
                self._send_json(404, {"error": f"Alert {alert_id} not found"})
                return
            self._send_json(200, {"id": alert_id, "acknowledged": True})
            return

        if path == "/api/actions/poll":
            try:
                body = self._read_json_body()
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            target_id = body.get("asset_id")
            if target_id is not None:
                if not isinstance(target_id, str) or target_id not in orch.runners:
                    self._send_json(404, {"error": f"Unknown asset '{target_id}'"})
                    return
                outcomes = [orch.poll_asset(target_id)]
            else:
                outcomes = orch.poll_due_assets(force_all=True)

            self._send_json(
                200,
                {
                    "polled": [
                        {
                            "asset_id": o.asset_id,
                            "candles_received": o.candles_received,
                            "new_closed_candle": o.new_closed_candle,
                            "alert_id": o.alert.alert_id if o.alert is not None else None,
                            "error": o.error,
                        }
                        for o in outcomes
                    ]
                },
            )
            return

        self._send_json(404, {"error": f"Endpoint not found: {path}"})


class DashboardServer:
    """Lifecycle wrapper around the embedded HTTP/REST/WebSocket dashboard server."""

    def __init__(
        self,
        orchestrator: BotOrchestrator,
        *,
        host: str | None = None,
        port: int | None = None,
        static_dir: Path | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._host = host if host is not None else orchestrator.config.web.host
        self._port = port if port is not None else orchestrator.config.web.port
        self._static_dir = static_dir if static_dir is not None else _STATIC_DIR
        self._ws_hub = WebSocketHub()
        self._server: _DashboardHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def bound_address(self) -> tuple[str, int]:
        if self._server is None:
            return (self._host, self._port)
        addr = self._server.server_address
        return (str(addr[0]), int(addr[1]))

    @property
    def base_url(self) -> str:
        host, port = self.bound_address
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        return f"http://{display_host}:{port}"

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            return self.bound_address
        self._orchestrator.event_bus.subscribe(self._ws_hub.publish_event)
        self._server = _DashboardHTTPServer(
            (self._host, self._port),
            _DashboardRequestHandler,
            orchestrator=self._orchestrator,
            ws_hub=self._ws_hub,
            static_dir=self._static_dir,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.25},
            daemon=True,
            name="tradingbot-web",
        )
        self._thread.start()
        return self.bound_address

    def stop(self) -> None:
        self._orchestrator.event_bus.unsubscribe(self._ws_hub.publish_event)
        self._ws_hub.shutdown()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
