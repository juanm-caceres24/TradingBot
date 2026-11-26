"""Loopback HTTP and RFC 6455 WebSocket integration tests for DashboardServer."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import socket
import struct
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from tradingbot.config import parse_config_dict
from tradingbot.models import parse_iso_to_ms
from tradingbot.runtime import BotOrchestrator
from tradingbot.storage import SQLiteCandleStore
from tradingbot.web import DashboardServer
from tradingbot.web.websocket import (
    OPCODE_CLOSE,
    OPCODE_PING,
    OPCODE_PONG,
    OPCODE_TEXT,
    compute_accept_key,
    read_frame,
)
from test_nn_client import _make_mock_nn_server

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"
HOUR_MS = 3_600_000


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read().decode("utf-8"))


def _ws_connect(host: str, port: int, path: str) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=3.0)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(req)

    header_buf = bytearray()
    while b"\r\n\r\n" not in header_buf:
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("EOF during WebSocket handshake")
        header_buf.extend(chunk)

    header_text = header_buf.decode("ascii")
    assert "101 Switching Protocols" in header_text, header_text
    expected_accept = compute_accept_key(key)
    assert f"Sec-WebSocket-Accept: {expected_accept}" in header_text, header_text
    return sock


def _ws_send_masked(sock: socket.socket, opcode: int, payload: bytes = b"") -> None:
    mask = b"\x12\x34\x56\x78"
    header = bytearray([0x80 | (opcode & 0x0F), 0x80 | len(payload)])
    header.extend(mask)
    masked_payload = bytes(payload[i] ^ mask[i & 3] for i in range(len(payload)))
    sock.sendall(bytes(header) + masked_payload)


class TestDashboardWebServerAndAPI(unittest.TestCase):
    def test_rest_api_alerts_cursor_actions_and_websockets(self) -> None:
        bullish_rel: list[float] = []
        for step in range(1, 8):
            base = 0.30 * step
            bullish_rel.extend([base - 0.05, base + 0.35, base - 0.15, base + 0.12])

        srv_c, th_c, port_c = _make_mock_nn_server(341, bullish_rel, latency_us=10)
        srv_s, th_s, port_s = _make_mock_nn_server(341, [0.05, 0.85, 0.10], latency_us=6)

        try:
            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            clock = {"now": start_ms + 120 * HOUR_MS + 10_000}
            mono = {"t": 100.0}

            btc_asset = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            btc_asset["provider"] = {"preset_path": "../providers/synthetic.json"}
            btc_asset["neural_networks"]["candle_predictor"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_c,
                "schema_path": "",
            }
            btc_asset["neural_networks"]["signal_classifier"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_s,
                "schema_path": "",
            }
            btc_asset["alerts"]["consecutive_candles"] = 2
            btc_asset["alerts"]["cooldown_candles"] = 1

            cfg = parse_config_dict(
                {
                    "storage": {"db_path": ":memory:", "wal_mode": False, "busy_timeout_ms": 5000},
                    "web": {"enabled": True, "host": "127.0.0.1", "port": 0, "visible_candles": 70, "title": "TestBot"},
                    "notifications": {"api_enabled": True, "webhooks": []},
                    "assets": [btc_asset],
                },
                base_dir=BTC_1H_ASSET_PATH.parent,
            )

            with SQLiteCandleStore(":memory:") as store:
                orch = BotOrchestrator(
                    cfg,
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                orch.initialize(strict_nn_startup=True)

                web = DashboardServer(orch, host="127.0.0.1", port=0)
                host, port = web.start()
                base = f"http://{host}:{port}"

                try:
                    # 1) Health & config
                    code, health = _http_json(f"{base}/health")
                    self.assertEqual(code, 200)
                    self.assertEqual(health["status"], "ok")
                    self.assertEqual(health["assets_count"], 1)

                    code, pub_cfg = _http_json(f"{base}/api/config")
                    self.assertEqual(code, 200)
                    self.assertEqual(pub_cfg["title"], "TestBot")

                    # 2) Assets watchlist & latest detail
                    code, assets_resp = _http_json(f"{base}/api/assets")
                    self.assertEqual(code, 200)
                    self.assertEqual(len(assets_resp["assets"]), 1)
                    self.assertEqual(assets_resp["assets"][0]["asset_id"], "btc_1h")
                    self.assertEqual(assets_resp["assets"][0]["action"], "LONG")

                    code, latest = _http_json(f"{base}/api/assets/btc_1h/latest?visible=50")
                    self.assertEqual(code, 200)
                    self.assertEqual(len(latest["candles"]), 50)
                    self.assertEqual(len(latest["prediction"]["predicted_candles"]), 7)

                    code, _ = _http_json(f"{base}/api/assets/nonexistent/latest")
                    self.assertEqual(code, 404)

                    # 3) Connect real RFC 6455 WebSockets to /ws/live and /ws/alerts
                    ws_live = _ws_connect(host, port, "/ws/live")
                    ws_alerts = _ws_connect(host, port, "/ws/alerts")
                    try:
                        op1, p1 = read_frame(ws_live) or (0, b"")
                        self.assertEqual(op1, OPCODE_TEXT)
                        self.assertEqual(json.loads(p1.decode())["channel"], "live")

                        op2, p2 = read_frame(ws_alerts) or (0, b"")
                        self.assertEqual(op2, OPCODE_TEXT)
                        self.assertEqual(json.loads(p2.decode())["channel"], "alerts")

                        # Test Ping -> Pong keepalive on /ws/live
                        _ws_send_masked(ws_live, OPCODE_PING, b"hi")
                        op_pong, pong_payload = read_frame(ws_live) or (0, b"")
                        self.assertEqual(op_pong, OPCODE_PONG)
                        self.assertEqual(pong_payload, b"hi")

                        # 4) Advance clock 1 hour & trigger POST /api/actions/poll -> fires Alert #1!
                        clock["now"] += HOUR_MS
                        mono["t"] += 10.0
                        code, poll_resp = _http_json(
                            f"{base}/api/actions/poll",
                            method="POST",
                            body={"asset_id": "btc_1h"},
                        )
                        self.assertEqual(code, 200)
                        self.assertEqual(poll_resp["polled"][0]["alert_id"], 1)

                        # /ws/alerts receives ONLY the alert event
                        op_a, payload_a = read_frame(ws_alerts) or (0, b"")
                        self.assertEqual(op_a, OPCODE_TEXT)
                        alert_msg = json.loads(payload_a.decode())
                        self.assertEqual(alert_msg["type"], "alert")
                        self.assertEqual(alert_msg["alert"]["id"], 1)

                        # /ws/live receives candle_update, prediction, and alert
                        live_types: list[str] = []
                        for _ in range(3):
                            op_l, payload_l = read_frame(ws_live) or (0, b"")
                            self.assertEqual(op_l, OPCODE_TEXT)
                            live_types.append(json.loads(payload_l.decode())["type"])
                        self.assertEqual(set(live_types), {"candle_update", "prediction", "alert"})

                        _ws_send_masked(ws_live, OPCODE_CLOSE, struct.pack("!H", 1000))
                        _ws_send_masked(ws_alerts, OPCODE_CLOSE, struct.pack("!H", 1000))
                    finally:
                        ws_live.close()
                        ws_alerts.close()

                    # 5) External alert consumer cursor simulation: GET /api/alerts?since_id=0 + ACK
                    code, alerts_resp = _http_json(f"{base}/api/alerts?since_id=0&unacknowledged=true")
                    self.assertEqual(code, 200)
                    self.assertEqual(alerts_resp["count"], 1)
                    self.assertEqual(alerts_resp["last_id"], 1)
                    self.assertEqual(alerts_resp["alerts"][0]["action"], "LONG")

                    code, ack_resp = _http_json(f"{base}/api/alerts/1/ack", method="POST")
                    self.assertEqual(code, 200)
                    self.assertTrue(ack_resp["acknowledged"])

                    code, after_ack = _http_json(f"{base}/api/alerts?since_id=0&unacknowledged=true")
                    self.assertEqual(code, 200)
                    self.assertEqual(after_ack["count"], 0)

                    code, after_cursor = _http_json(f"{base}/api/alerts?since_id=1")
                    self.assertEqual(code, 200)
                    self.assertEqual(after_cursor["count"], 0)

                    # 6) Static Web Dashboard files & path-traversal protection
                    for path, expected_mime in (
                        ("/", "text/html"),
                        ("/static/styles.css", "text/css"),
                        ("/static/chart.js", "application/javascript"),
                        ("/static/app.js", "application/javascript"),
                    ):
                        with urllib.request.urlopen(f"{base}{path}", timeout=3.0) as resp:
                            self.assertEqual(resp.status, 200)
                            self.assertIn(expected_mime, resp.headers.get("Content-Type", ""))
                            self.assertGreater(len(resp.read()), 100)

                    code_trav, _ = _http_json(f"{base}/static/../server.py")
                    self.assertEqual(code_trav, 404)
                finally:
                    web.stop()
        finally:
            for srv, th in ((srv_c, th_c), (srv_s, th_s)):
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)

    def test_live_app_starts_and_stops_embedded_web_dashboard(self) -> None:
        live_path = ROOT_DIR / "apps" / "live_app.py"
        spec = importlib.util.spec_from_file_location("live_app_web_test", live_path)
        assert spec is not None and spec.loader is not None
        live_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live_app)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            asset_raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            asset_raw["provider"] = {
                "preset_path": str((ROOT_DIR / "configs" / "providers" / "synthetic.json").resolve())
            }
            asset_raw["runtime"]["poll_interval_sec"] = 0.05
            asset_raw["neural_networks"]["candle_predictor"]["enabled"] = False
            asset_raw["neural_networks"]["signal_classifier"]["enabled"] = False
            asset_file = tmp_path / "btc_1h.json"
            asset_file.write_text(json.dumps(asset_raw), encoding="utf-8")

            bot_cfg_path = tmp_path / "bot.json"
            bot_cfg_path.write_text(
                json.dumps(
                    {
                        "storage": {"db_path": str(tmp_path / "m.db"), "wal_mode": True, "busy_timeout_ms": 5000},
                        "web": {"enabled": True, "host": "127.0.0.1", "port": 0, "visible_candles": 70},
                        "notifications": {"api_enabled": True, "webhooks": []},
                        "assets": [{"config_path": str(asset_file)}],
                    }
                ),
                encoding="utf-8",
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = live_app.main(
                    [
                        str(bot_cfg_path),
                        "--mode",
                        "live_only",
                        "--web-host",
                        "127.0.0.1",
                        "--web-port",
                        "0",
                        "--max-cycles",
                        "1",
                    ]
                )
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("Web Dashboard listening on http://127.0.0.1:", out)
            self.assertIn("stopped cleanly after 1 poll cycle(s)", out)


if __name__ == "__main__":
    unittest.main()
