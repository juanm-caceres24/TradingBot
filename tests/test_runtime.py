"""Integration tests for WebhookDispatcher, EventBus, RuntimeState, and BotOrchestrator."""

from __future__ import annotations

import copy
import http.server
import json
import os
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from tradingbot.config import WebhookConfig, parse_config_dict
from tradingbot.models import AlertEvent, parse_iso_to_ms
from tradingbot.nn_client import NNClientError
from tradingbot.providers import ProviderError
from tradingbot.runtime import BotOrchestrator, EventBus, WebhookDispatcher
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
        # 6 bullish relative candles + strong LONG signal
        bullish_rel: list[float] = []
        for step in range(1, 7):
            base = 0.30 * step
            bullish_rel.extend([base - 0.05, base + 0.35, base - 0.15, base + 0.12])

        srv_c, th_c, port_c = _make_mock_nn_server(264, bullish_rel, latency_us=10)
        srv_s, th_s, port_s = _make_mock_nn_server(264, [0.08, 0.84, 0.08], latency_us=6)
        wh_srv, wh_th, wh_port, wh_received = _start_webhook_receiver()

        try:
            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            clock = {"now": start_ms + 300 * HOUR_MS + 10_000}
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
                self.assertEqual(len(detail["indicators"]["ema_20"]["value"]), 70)
                self.assertEqual(len(detail["indicators"]["sma_200"]["value"]), 70)
                self.assertEqual(detail["prediction"]["analysis"]["action"], "LONG")
        finally:
            for srv, th in ((srv_c, th_c), (srv_s, th_s), (wh_srv, wh_th)):
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)

    def test_event_bus_isolation_and_webhook_edge_cases(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        def _broken(_ev: dict[str, Any]) -> None:
            raise RuntimeError("boom")

        def _good(ev: dict[str, Any]) -> None:
            seen.append(str(ev.get("type")))

        bus.subscribe(_broken)
        bus.subscribe(_good)
        bus.publish({"type": "ping"})
        self.assertEqual(seen, ["ping"])

        bus.unsubscribe(_good)
        bus.publish({"type": "pong"})
        self.assertEqual(seen, ["ping"])

        # Separate webhooks exercise both missing auth env var and unreachable socket branches
        os.environ.pop("MISSING_WH_ENV_TOKEN", None)
        alert = AlertEvent(
            alert_id=1,
            asset_id="btc_1h",
            provider="synthetic",
            symbol="BTCUSDT",
            timeframe="1h",
            anchor_open_time_ms=HOUR_MS,
            created_at_ms=HOUR_MS + 1,
            action="LONG",
            confidence=0.8,
            consecutive_hits=1,
            rule_name="single_bar_threshold",
            payload_json="{}",
        )
        dispatcher = WebhookDispatcher(
            [
                WebhookConfig(
                    name="missing_env_hook",
                    url="http://127.0.0.1:1/never_called",
                    enabled=True,
                    timeout_sec=0.5,
                    auth_token_env="MISSING_WH_ENV_TOKEN",
                ),
                WebhookConfig(
                    name="unreachable_hook",
                    url="http://127.0.0.1:1/unreachable",
                    enabled=True,
                    timeout_sec=0.5,
                ),
            ]
        )
        results = dispatcher.dispatch(alert)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].success)
        self.assertIsNone(results[0].status_code)
        self.assertIn("not set", results[0].error or "")
        self.assertFalse(results[1].success)
        self.assertIsNone(results[1].status_code)
        self.assertTrue(bool(results[1].error))
        self.assertNotIn("not set", results[1].error or "")

    def test_runtime_hardening_gap_error_non_persist_open_candle_restart_and_empty_loop(self) -> None:
        srv_s, th_s, port_s = _make_mock_nn_server(264, [0.05, 0.85, 0.10], latency_us=5)
        try:
            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            clock = {"now": start_ms + 300 * HOUR_MS + 10_000}
            mono = {"t": 100.0}

            def _make_cfg(
                *,
                mode: str = "sync_live",
                persist_to_db: bool = True,
                predict_on_open: bool = False,
                signal_port: int = port_s,
                signal_enabled: bool = True,
                consecutive: int = 1,
                asset_enabled: bool = True,
            ) -> Any:
                btc = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
                btc["enabled"] = asset_enabled
                btc["provider"] = {"preset_path": "../providers/synthetic.json"}
                btc["runtime"]["mode"] = mode
                btc["runtime"]["persist_to_db"] = persist_to_db
                btc["runtime"]["predict_on_open_candle"] = predict_on_open
                btc["runtime"]["poll_interval_sec"] = 1.0
                btc["neural_networks"]["candle_predictor"]["enabled"] = False
                btc["neural_networks"]["signal_classifier"] = {
                    "enabled": signal_enabled,
                    "host": "127.0.0.1",
                    "port": signal_port,
                    "schema_path": "",
                }
                btc["alerts"]["consecutive_candles"] = consecutive
                btc["alerts"]["cooldown_candles"] = 0
                return parse_config_dict(
                    {
                        "storage": {"db_path": ":memory:", "wal_mode": False, "busy_timeout_ms": 5000},
                        "web": {"enabled": False, "host": "127.0.0.1", "port": 8000, "visible_candles": 70},
                        "notifications": {"api_enabled": True, "webhooks": []},
                        "assets": [btc],
                    },
                    base_dir=BTC_1H_ASSET_PATH.parent,
                )

            # 1) sync_live + persist_to_db=False populates RAM buffer without touching SQLite
            with SQLiteCandleStore(":memory:") as store:
                orch = BotOrchestrator(
                    _make_cfg(persist_to_db=False, signal_enabled=False),
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                orch.initialize()
                self.assertEqual(store.count_candles("synthetic", "BTCUSDT", "1h"), 0)
                snap = orch.state.get("btc_1h")
                assert snap is not None
                self.assertGreaterEqual(len(snap.candles), 80)

                # Multi-bar jump (3 hours) works in RAM without SQLite writes
                clock["now"] += 3 * HOUR_MS
                outcome = orch.poll_asset("btc_1h")
                self.assertIsNone(outcome.error)
                self.assertTrue(outcome.new_closed_candle)
                self.assertEqual(store.count_candles("synthetic", "BTCUSDT", "1h"), 0)

            # 2) Multi-bar gap-heal ProviderError is caught cleanly (never crashes poll_once)
            with SQLiteCandleStore(":memory:") as store:
                orch = BotOrchestrator(
                    _make_cfg(persist_to_db=True, signal_enabled=False),
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                orch.initialize()
                clock["now"] += 4 * HOUR_MS

                def _fail_backfill(*_a: Any, **_kw: Any) -> Any:
                    raise ProviderError("Simulated network outage during gap-heal")

                orch._sync_service.backfill_asset = _fail_backfill  # type: ignore[method-assign]
                outcome = orch.poll_asset("btc_1h")
                self.assertIsNotNone(outcome.error)
                self.assertIn("Simulated network outage", outcome.error or "")

            # 3) predict_on_open_candle=True predicts in RAM without saving prediction or emitting alert
            with SQLiteCandleStore(":memory:") as store:
                orch = BotOrchestrator(
                    _make_cfg(predict_on_open=True, consecutive=1),
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                orch.initialize()
                preds_after_init = len(store.get_predictions("synthetic", "BTCUSDT", "1h", asset_id="btc_1h"))
                alerts_after_init = len(store.get_alerts(asset_id="btc_1h"))
                self.assertEqual(alerts_after_init, 1)

                # Poll same open hour (+30s): predicts in RAM, does NOT persist prediction or emit alert
                clock["now"] += 30_000
                o_open = orch.poll_asset("btc_1h")
                self.assertFalse(o_open.new_closed_candle)
                self.assertIsNotNone(o_open.prediction)
                self.assertIsNone(o_open.alert)
                self.assertEqual(
                    len(store.get_predictions("synthetic", "BTCUSDT", "1h", asset_id="btc_1h")),
                    preds_after_init,
                )
                self.assertEqual(len(store.get_alerts(asset_id="btc_1h")), alerts_after_init)

                # 4) Restarting a new orchestrator on the SAME SQLite store seeds AlertEvaluator
                # so the exact same anchor candle does NOT re-emit a duplicate alert!
                orch2 = BotOrchestrator(
                    _make_cfg(predict_on_open=False, consecutive=1),
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                init2 = orch2.initialize()
                self.assertIsNone(init2[0].alert)
                self.assertEqual(len(store.get_alerts(asset_id="btc_1h")), alerts_after_init)

            # 5) Offline NN server: strict_nn_startup=False tolerates outage, True raises NNClientError
            with SQLiteCandleStore(":memory:") as store:
                offline_cfg = _make_cfg(signal_port=1)
                orch_soft = BotOrchestrator(
                    offline_cfg,
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                outcomes_soft = orch_soft.initialize(strict_nn_startup=False)
                self.assertEqual(len(outcomes_soft), 1)
                self.assertIsNotNone(outcomes_soft[0].error)

                orch_strict = BotOrchestrator(
                    offline_cfg,
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                with self.assertRaises(NNClientError):
                    orch_strict.initialize(strict_nn_startup=True)

            # 6) Empty runners run_loop waits via stop_event instead of busy-spinning
            with SQLiteCandleStore(":memory:") as store:
                empty_orch = BotOrchestrator(
                    _make_cfg(asset_enabled=False),
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                self.assertEqual(len(empty_orch.runners), 0)
                t0 = time.monotonic()
                cycles = empty_orch.run_loop(max_cycles=1)
                elapsed = time.monotonic() - t0
                self.assertEqual(cycles, 1)
                self.assertGreaterEqual(elapsed, 0.4)

            # 7) Concurrent poll_asset() during startup initialize() only initializes once
            with SQLiteCandleStore(":memory:") as store:
                race_orch = BotOrchestrator(
                    _make_cfg(signal_enabled=False),
                    store,
                    clock_ms=lambda: clock["now"],
                    sleep_fn=lambda _s: None,
                    monotonic_fn=lambda: mono["t"],
                )
                init_calls = 0
                orig_init_locked = race_orch._initialize_locked
                entered = threading.Event()
                proceed = threading.Event()

                def _slow_init_locked(*, strict_nn_startup: bool = False) -> Any:
                    nonlocal init_calls
                    init_calls += 1
                    entered.set()
                    proceed.wait(timeout=2.0)
                    return orig_init_locked(strict_nn_startup=strict_nn_startup)

                race_orch._initialize_locked = _slow_init_locked  # type: ignore[method-assign]
                t_init = threading.Thread(target=race_orch.initialize)
                t_init.start()
                self.assertTrue(entered.wait(timeout=2.0))
                t_poll = threading.Thread(target=lambda: race_orch.poll_asset("btc_1h"))
                t_poll.start()
                time.sleep(0.05)
                proceed.set()
                t_init.join(timeout=3.0)
                t_poll.join(timeout=3.0)
                self.assertEqual(init_calls, 1)
        finally:
            srv_s.shutdown()
            srv_s.server_close()
            th_s.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
