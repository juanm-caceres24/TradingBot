"""Integration tests for WebhookDispatcher, EventBus, RuntimeState, and BotOrchestrator."""

from __future__ import annotations

import copy
import http.server
import json
import os
import threading
import unittest
from pathlib import Path
from typing import Any

from tradingbot.config import WebhookConfig, parse_config_dict
from tradingbot.models import AlertEvent, parse_iso_to_ms
from tradingbot.runtime import BotOrchestrator, WebhookDispatcher
from tradingbot.storage import SQLiteCandleStore
from test_nn_client import _make_mock_nn_server

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"
HOUR_MS = 3_600_000


def _start_webhook_receiver() -> tuple[http.server.ThreadingHTTPServer, threading.Thread, int, list[dict[str, Any]]]:
    received: list[dict[str, Any]] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            received.append(
                {
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
            )
            if self.path == "/fail":
                self.send_response(500)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"received":true}')

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, thread, port, received


class TestWebhookDispatcher(unittest.TestCase):
    def test_webhook_post_auth_header_and_failure_isolation(self) -> None:
        srv, thread, port, received = _start_webhook_receiver()
        os.environ["TEST_WH_TOKEN"] = "Bearer secret123"
        try:
            dispatcher = WebhookDispatcher(
                [
                    WebhookConfig(
                        name="telegram_ok",
                        url=f"http://127.0.0.1:{port}/notify",
                        enabled=True,
                        timeout_sec=2.0,
                        headers={"X-App": "tradingbot"},
                        auth_token_env="TEST_WH_TOKEN",
                        auth_header="Authorization",
                    ),
                    WebhookConfig(
                        name="broken_500",
                        url=f"http://127.0.0.1:{port}/fail",
                        enabled=True,
                        timeout_sec=2.0,
                    ),
                    WebhookConfig(
                        name="disabled_hook",
                        url=f"http://127.0.0.1:{port}/ignored",
                        enabled=False,
                    ),
                ]
            )
            alert = AlertEvent(
                alert_id=7,
                asset_id="btc_1h",
                provider="synthetic",
                symbol="BTCUSDT",
                timeframe="1h",
                anchor_open_time_ms=10 * HOUR_MS,
                created_at_ms=10 * HOUR_MS + 50,
                action="LONG",
                confidence=0.82,
                consecutive_hits=2,
                rule_name="consecutive_2_bars",
                payload_json='{"risk_reward":2.4}',
            )
            results = dispatcher.dispatch(alert)
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].status_code, 200)
            self.assertFalse(results[1].success)
            self.assertEqual(results[1].status_code, 500)

            self.assertEqual(len(received), 2)
            self.assertEqual(received[0]["headers"]["authorization"], "Bearer secret123")
            self.assertEqual(received[0]["headers"]["x-app"], "tradingbot")
            self.assertEqual(received[0]["body"]["event"], "tradingbot.alert")
            self.assertEqual(received[0]["body"]["alert"]["id"], 7)
            self.assertAlmostEqual(received[0]["body"]["alert"]["payload"]["risk_reward"], 2.4)
        finally:
            os.environ.pop("TEST_WH_TOKEN", None)
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=2.0)


class TestBotOrchestrator(unittest.TestCase):
    def test_sync_live_multi_asset_poll_prediction_alert_and_webhook(self) -> None:
        # 7 bullish relative candles + strong LONG signal
        bullish_rel: list[float] = []
        for step in range(1, 8):
            base = 0.30 * step
            bullish_rel.extend([base - 0.05, base + 0.35, base - 0.15, base + 0.12])

        srv_c, th_c, port_c = _make_mock_nn_server(341, bullish_rel, latency_us=10)
        srv_s, th_s, port_s = _make_mock_nn_server(341, [0.08, 0.84, 0.08], latency_us=6)
        wh_srv, wh_th, wh_port, wh_received = _start_webhook_receiver()

        try:
            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            clock = {"now": start_ms + 120 * HOUR_MS + 10_000}
            mono = {"t": 100.0}

            btc_asset = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            btc_asset["provider"] = {"preset_path": "../providers/synthetic.json"}
            btc_asset["runtime"]["poll_interval_sec"] = 5.0
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
            btc_asset["alerts"]["cooldown_candles"] = 3

            # Second asset running live_only with persist_to_db=False and no NNs
            eth_asset = copy.deepcopy(btc_asset)
            eth_asset["id"] = "eth_1h"
            eth_asset["market"]["symbol"] = "ETHUSDT"
            eth_asset["runtime"]["mode"] = "live_only"
            eth_asset["runtime"]["persist_to_db"] = False
            eth_asset["neural_networks"]["candle_predictor"]["enabled"] = False
            eth_asset["neural_networks"]["signal_classifier"]["enabled"] = False

            app_cfg = parse_config_dict(
                {
                    "storage": {"db_path": ":memory:", "wal_mode": False, "busy_timeout_ms": 5000},
                    "web": {"enabled": False, "host": "127.0.0.1", "port": 8000, "visible_candles": 70},
                    "notifications": {
                        "api_enabled": True,
                        "webhooks": [
                            {
                                "name": "test_hook",
                                "url": f"http://127.0.0.1:{wh_port}/alert",
                                "enabled": True,
                                "timeout_sec": 2.0,
                            }
                        ],
                    },
                    "assets": [btc_asset, eth_asset],
                },
                base_dir=BTC_1H_ASSET_PATH.parent,
            )

            events: list[dict[str, Any]] = []
            with SQLiteCandleStore(":memory:") as store:
                orch = BotOrchestrator(
                    app_cfg,
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                orch.event_bus.subscribe(events.append)

                # Startup: btc_1h syncs closed bars + active bar and gets LONG #1 (streak 1/2)
                init_outcomes = orch.initialize(strict_nn_startup=True)
                self.assertEqual(len(init_outcomes), 2)
                btc_init = init_outcomes[0]
                self.assertIsNotNone(btc_init.prediction)
                self.assertIsNone(btc_init.alert)  # streak = 1/2

                # live_only eth_1h did NOT persist to SQLite but HAS candles in RuntimeState!
                self.assertEqual(store.count_candles("synthetic", "ETHUSDT", "1h"), 0)
                eth_state = orch.state.get("eth_1h")
                self.assertIsNotNone(eth_state)
                assert eth_state is not None
                self.assertGreaterEqual(len(eth_state.candles), 80)

                # Advance simulated clock by 1 full hour -> new closed candle arrives -> streak 2/2 -> ALERT!
                clock["now"] += HOUR_MS
                mono["t"] += 10.0
                poll_outcomes = orch.poll_due_assets()
                btc_poll = [o for o in poll_outcomes if o.asset_id == "btc_1h"][0]
                self.assertTrue(btc_poll.new_closed_candle)
                self.assertIsNotNone(btc_poll.alert)
                assert btc_poll.alert is not None
                self.assertEqual(btc_poll.alert.action, "LONG")
                self.assertEqual(btc_poll.alert.alert_id, 1)
                self.assertEqual(len(btc_poll.webhook_results), 1)
                self.assertTrue(btc_poll.webhook_results[0].success)

                # Verify persisted in SQLite alerts + predictions, delivered via webhook, and broadcast on EventBus
                stored_alerts = store.get_alerts(asset_id="btc_1h", since_id=0)
                self.assertEqual(len(stored_alerts), 1)
                self.assertEqual(stored_alerts[0].action, "LONG")
                self.assertGreaterEqual(
                    len(store.get_predictions("synthetic", "BTCUSDT", "1h", asset_id="btc_1h")),
                    2,
                )
                self.assertEqual(len(wh_received), 1)
                self.assertTrue(any(e["type"] == "alert" for e in events))

                # RuntimeState detail_dict slices cleanly for visible_candles=70
                btc_snap = orch.state.get("btc_1h")
                assert btc_snap is not None
                detail = btc_snap.detail_dict(visible_candles=70)
                self.assertEqual(len(detail["candles"]), 70)
                self.assertEqual(len(detail["indicators"]["sma_20"]["value"]), 70)
                self.assertEqual(detail["prediction"]["analysis"]["action"], "LONG")
        finally:
            for srv, th in ((srv_c, th_c), (srv_s, th_s), (wh_srv, wh_th)):
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
