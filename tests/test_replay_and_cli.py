"""Integration and CLI tests for ReplayEngine, apps/replay_app.py, and apps/live_app.py."""

from __future__ import annotations

import importlib.util
import io
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from tradingbot.config import parse_asset_dict
from tradingbot.models import parse_iso_to_ms
from tradingbot.providers import SyntheticProvider, create_provider
from tradingbot.runtime import EventBus, ReplayEngine
from tradingbot.storage import SQLiteCandleStore
from test_nn_client import _make_mock_nn_server

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"
LIVE_APP_PATH = ROOT_DIR / "apps" / "live_app.py"
REPLAY_APP_PATH = ROOT_DIR / "apps" / "replay_app.py"
HOUR_MS = 3_600_000


def _load_app(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestReplayAndCLIApps(unittest.TestCase):
    def test_replay_engine_metrics_alerts_and_cli_apps(self) -> None:
        # 7 bullish relative OHLC candles + strong LONG probability
        bullish_rel: list[float] = []
        for step in range(1, 8):
            base = 0.25 * step
            bullish_rel.extend([base - 0.05, base + 0.30, base - 0.15, base + 0.10])

        srv_c, th_c, port_c = _make_mock_nn_server(341, bullish_rel, latency_us=9)
        srv_s, th_s, port_s = _make_mock_nn_server(341, [0.10, 0.80, 0.10], latency_us=5)

        try:
            raw_asset = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            raw_asset["provider"] = {
                "preset_path": str((ROOT_DIR / "configs" / "providers" / "synthetic.json").resolve())
            }
            raw_asset["runtime"]["poll_interval_sec"] = 0.05
            raw_asset["neural_networks"]["candle_predictor"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_c,
                "schema_path": "",
            }
            raw_asset["neural_networks"]["signal_classifier"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_s,
                "schema_path": "",
            }
            raw_asset["alerts"]["consecutive_candles"] = 2
            raw_asset["alerts"]["cooldown_candles"] = 3

            asset = parse_asset_dict(raw_asset, base_dir=BTC_1H_ASSET_PATH.parent)
            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            end_ms = start_ms + 95 * HOUR_MS
            provider = create_provider(asset.provider, clock_ms=lambda: end_ms + HOUR_MS)
            assert isinstance(provider, SyntheticProvider)
            candles = provider.fetch_candles("BTCUSDT", "1h", start_ms=start_ms, end_ms=end_ms, limit=95)

            # 1) Direct ReplayEngine test (95 bars -> first_anchor=79 -> 16 bars evaluated)
            bus = EventBus()
            bus_events: list[dict[str, Any]] = []
            bus.subscribe(bus_events.append)

            with SQLiteCandleStore(":memory:") as store:
                engine = ReplayEngine(asset, clock_ms=lambda: end_ms)
                engine.verify_models()
                summary = engine.run(
                    candles,
                    store=store,
                    event_bus=bus,
                    persist_predictions=True,
                    persist_alerts=True,
                )
                self.assertEqual(summary.bars_evaluated, 16)
                self.assertEqual(summary.action_counts["LONG"], 16)
                self.assertGreaterEqual(summary.alerts_triggered, 3)
                self.assertIsNotNone(summary.mean_abs_return_error)
                self.assertGreater(summary.directional_total, 0)
                self.assertEqual(len(store.get_alerts(asset_id="btc_1h")), summary.alerts_triggered)
                self.assertTrue(any(e["type"] == "alert" for e in bus_events))

            # 2) End-to-end replay_app.py and live_app.py CLI smoke tests
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                db_path = tmp_path / "market.db"
                asset_file = tmp_path / "btc_1h.json"
                asset_file.write_text(json.dumps(raw_asset), encoding="utf-8")

                bot_cfg_path = tmp_path / "bot.json"
                bot_cfg_path.write_text(
                    json.dumps(
                        {
                            "storage": {"db_path": str(db_path), "wal_mode": True, "busy_timeout_ms": 5000},
                            "web": {"enabled": False, "host": "127.0.0.1", "port": 8000, "visible_candles": 70},
                            "notifications": {"api_enabled": True, "webhooks": []},
                            "assets": [{"config_path": str(asset_file)}],
                        }
                    ),
                    encoding="utf-8",
                )

                with SQLiteCandleStore(db_path) as store:
                    store.upsert_candles(candles)

                replay_app = _load_app("replay_app", REPLAY_APP_PATH)
                buf_r = io.StringIO()
                with redirect_stdout(buf_r):
                    rc_r = replay_app.main(
                        [str(bot_cfg_path), "--asset", "btc_1h", "--persist-alerts", "--verbose"]
                    )
                self.assertEqual(rc_r, 0)
                self.assertIn("evaluated=16", buf_r.getvalue())
                self.assertIn("ALERT LONG", buf_r.getvalue())

                live_app = _load_app("live_app", LIVE_APP_PATH)
                buf_l = io.StringIO()
                with redirect_stdout(buf_l):
                    rc_l = live_app.main(
                        [
                            str(bot_cfg_path),
                            "--asset",
                            "btc_1h",
                            "--mode",
                            "live_only",
                            "--headless",
                            "--strict-nn",
                            "--max-cycles",
                            "2",
                        ]
                    )
                self.assertEqual(rc_l, 0)
                self.assertIn("stopped cleanly after 2 poll cycle(s)", buf_l.getvalue())

                # 3) CLI error handling: replay --end <= --start, live unknown --asset, occupied port
                err_buf = io.StringIO()
                with redirect_stderr(err_buf):
                    rc_bad_range = replay_app.main(
                        [
                            str(bot_cfg_path),
                            "--asset",
                            "btc_1h",
                            "--start",
                            "2025-01-10",
                            "--end",
                            "2025-01-01",
                        ]
                    )
                self.assertEqual(rc_bad_range, 2)

                with redirect_stderr(err_buf):
                    rc_unknown_asset = live_app.main(
                        [str(bot_cfg_path), "--asset", "nonexistent_asset", "--headless"]
                    )
                self.assertEqual(rc_unknown_asset, 1)

                # Bind a loopback port and verify live_app catches OSError cleanly when web is enabled
                bot_web_cfg_path = tmp_path / "bot_web.json"
                bot_web_cfg_path.write_text(
                    json.dumps(
                        {
                            "storage": {"db_path": str(db_path), "wal_mode": True, "busy_timeout_ms": 5000},
                            "web": {"enabled": True, "host": "127.0.0.1", "port": 0, "visible_candles": 70},
                            "notifications": {"api_enabled": True, "webhooks": []},
                            "assets": [{"config_path": str(asset_file)}],
                        }
                    ),
                    encoding="utf-8",
                )
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
                    blocker.bind(("127.0.0.1", 0))
                    blocker.listen(1)
                    occupied_port = blocker.getsockname()[1]
                    with redirect_stderr(err_buf):
                        rc_busy = live_app.main(
                            [
                                str(bot_web_cfg_path),
                                "--asset",
                                "btc_1h",
                                "--mode",
                                "live_only",
                                "--web-host",
                                "127.0.0.1",
                                "--web-port",
                                str(occupied_port),
                                "--max-cycles",
                                "1",
                            ]
                        )
                    self.assertEqual(rc_busy, 1)
        finally:
            for srv, th in ((srv_c, th_c), (srv_s, th_s)):
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
