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
BTC_1H_ASSET_PATH = ROOT_DIR / "templates" / "configs" / "assets" / "example_asset.json"
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
        # 6 bullish relative OHLC candles + strong LONG probability
        bullish_rel: list[float] = []
        for step in range(1, 7):
            base = 0.25 * step
            bullish_rel.extend([base - 0.05, base + 0.30, base - 0.15, base + 0.10])

        srv_c, th_c, port_c = _make_mock_nn_server(384, bullish_rel, latency_us=9)
        srv_s, th_s, port_s = _make_mock_nn_server(384, [0.10, 0.80, 0.10], latency_us=5)

        try:
            raw_asset = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            raw_asset["provider"] = {
                "preset_path": str((ROOT_DIR / "templates" / "configs" / "providers" / "synthetic.json").resolve())
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
            end_ms = start_ms + 238 * HOUR_MS
            provider = create_provider(asset.provider, clock_ms=lambda: end_ms + HOUR_MS)
            assert isinstance(provider, SyntheticProvider)
            candles = provider.fetch_candles("BTCUSDT", "1h", start_ms=start_ms, end_ms=end_ms, limit=238)

            # 1) Direct ReplayEngine test (238 bars -> first_anchor=222 -> 16 bars evaluated)
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
                # Filtered signal metrics cover the 10 bars with a complete 6-bar horizon (222..231)
                self.assertEqual(summary.signal_trades_evaluated, 10)
                self.assertEqual(summary.signal_long_total, 10)
                self.assertEqual(summary.signal_short_total, 0)
                self.assertEqual(summary.signal_directional_hits, summary.signal_long_hits)
                self.assertIsNotNone(summary.signal_avg_return_pct)
                self.assertEqual(
                    summary.tb_tp_hits + summary.tb_sl_hits + summary.tb_timeouts,
                    summary.signal_trades_evaluated,
                )
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
                self.assertIn("hold_funnel:", buf_r.getvalue())
                self.assertIn("filtered_signals=10", buf_r.getvalue())
                self.assertIn("TB(TP/SL/TO)=", buf_r.getvalue())
                self.assertIn("ALERT LONG", buf_r.getvalue())

                # --signal-only flag runs signal_classifier alone without calling candle_predictor
                buf_so = io.StringIO()
                with redirect_stdout(buf_so):
                    rc_so = replay_app.main(
                        [str(bot_cfg_path), "--asset", "btc_1h", "--signal-only"]
                    )
                self.assertEqual(rc_so, 0)
                self.assertIn("[signal-only]", buf_so.getvalue())
                self.assertIn("directional_acc=N/A", buf_so.getvalue())
                self.assertIn("filtered_signals=10", buf_so.getvalue())

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

                # --signal-only with signal_classifier disabled exits cleanly with code 1
                raw_no_sig = json.loads(json.dumps(raw_asset))
                raw_no_sig["neural_networks"]["signal_classifier"]["enabled"] = False
                no_sig_asset_file = tmp_path / "btc_1h_no_sig.json"
                no_sig_asset_file.write_text(json.dumps(raw_no_sig), encoding="utf-8")
                no_sig_bot_cfg = tmp_path / "bot_no_sig.json"
                no_sig_bot_cfg.write_text(
                    json.dumps(
                        {
                            "storage": {"db_path": str(db_path), "wal_mode": True, "busy_timeout_ms": 5000},
                            "web": {"enabled": False, "host": "127.0.0.1", "port": 8000, "visible_candles": 70},
                            "notifications": {"api_enabled": True, "webhooks": []},
                            "assets": [{"config_path": str(no_sig_asset_file)}],
                        }
                    ),
                    encoding="utf-8",
                )
                err_no_sig = io.StringIO()
                with redirect_stderr(err_no_sig):
                    rc_no_sig = replay_app.main(
                        [str(no_sig_bot_cfg), "--asset", "btc_1h", "--signal-only"]
                    )
                self.assertEqual(rc_no_sig, 1)
                self.assertIn("cannot run with --signal-only", err_no_sig.getvalue())

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

    def test_replay_signal_only_short_triple_barrier_outcomes(self) -> None:
        # Calibrated SHORT edge (HOLD=0.44, LONG=0.16, SHORT=0.40) evaluated in signal-only mode
        srv_s, th_s, port_s = _make_mock_nn_server(384, [0.44, 0.16, 0.40], latency_us=4)
        try:
            raw_asset = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            raw_asset["provider"] = {
                "preset_path": str((ROOT_DIR / "templates" / "configs" / "providers" / "synthetic.json").resolve())
            }
            raw_asset["features"]["triple_barrier"] = {
                "mode": "pct",
                "take_profit_pct": 0.02,
                "stop_loss_pct": 0.01,
            }
            raw_asset["neural_networks"]["candle_predictor"]["enabled"] = False
            raw_asset["neural_networks"]["signal_classifier"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_s,
                "schema_path": "",
            }
            asset = parse_asset_dict(raw_asset, base_dir=BTC_1H_ASSET_PATH.parent)

            # Build 231 deterministic candles (first_anchor=222, horizon=6 -> 3 evaluated trades at 222, 223, 224)
            # Flat base price 100.0 for bars 0..224; then drop to 97.0 (hits SHORT TP 98.0 cleanly)
            from tradingbot.models import Candle

            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            candles: list[Candle] = []
            for i in range(231):
                open_ms = start_ms + i * HOUR_MS
                # Slight deterministic jitter on warmup bars so ATR/RSI/BB are well-defined and finite
                base = 100.0 + (0.2 if i % 2 == 0 else -0.2) if i < 222 else (
                    100.0 if i <= 224 else 97.0
                )
                candles.append(
                    Candle(
                        provider="synthetic",
                        symbol="BTCUSDT",
                        timeframe="1h",
                        open_time_ms=open_ms,
                        close_time_ms=open_ms + HOUR_MS - 1,
                        open=base,
                        high=base + 0.3,
                        low=base - 0.3,
                        close=base,
                        volume=100.0,
                        is_closed=True,
                    )
                )

            engine = ReplayEngine(asset, clock_ms=lambda: start_ms + 232 * HOUR_MS)
            summary = engine.run(candles)
            self.assertEqual(summary.bars_evaluated, 9)
            self.assertEqual(summary.action_counts["SHORT"], 9)
            self.assertIsNone(summary.mean_abs_return_error)  # candle_predictor disabled
            self.assertEqual(summary.directional_total, 0)
            self.assertEqual(summary.signal_trades_evaluated, 3)
            self.assertEqual(summary.signal_short_total, 3)
            self.assertEqual(summary.signal_short_hits, 3)
            self.assertEqual(summary.signal_directional_hits, 3)
            self.assertAlmostEqual(summary.signal_avg_return_pct or 0.0, 0.03, places=6)
            self.assertEqual(summary.tb_tp_hits, 3)
            self.assertEqual(summary.tb_sl_hits, 0)
            self.assertEqual(summary.tb_timeouts, 0)
            self.assertEqual(sum(summary.hold_reasons.values()), 0)

            # Verify hold_reasons funnel counts conflict rejections cleanly when candle confirmation is enabled,
            # while require_candle_confirmation=False (btc_1h default) preserves SHORT actions
            srv_c_up, th_c_up, port_c_up = _make_mock_nn_server(
                384, [0.1, 0.4, 0.0, 0.3] * 6, latency_us=3
            )
            try:
                raw_conflict = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
                raw_conflict["provider"] = raw_asset["provider"]
                raw_conflict["neural_networks"]["candle_predictor"] = {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": port_c_up,
                    "schema_path": "",
                }
                raw_conflict["neural_networks"]["signal_classifier"] = {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": port_s,
                    "schema_path": "",
                }
                # btc_1h default (require_candle_confirmation=False): opposing candles do not veto SHORT
                sig_driven_summary = ReplayEngine(
                    parse_asset_dict(raw_conflict, base_dir=BTC_1H_ASSET_PATH.parent),
                    clock_ms=lambda: start_ms + 232 * HOUR_MS,
                ).run(candles)
                self.assertEqual(sig_driven_summary.action_counts["SHORT"], 9)
                self.assertEqual(sig_driven_summary.hold_reasons["conflict"], 0)

                raw_conflict["analysis"]["require_candle_confirmation"] = True
                conflict_asset = parse_asset_dict(raw_conflict, base_dir=BTC_1H_ASSET_PATH.parent)
                conflict_summary = ReplayEngine(
                    conflict_asset, clock_ms=lambda: start_ms + 232 * HOUR_MS
                ).run(candles)
                self.assertEqual(conflict_summary.action_counts["HOLD"], 9)
                self.assertEqual(conflict_summary.hold_reasons["conflict"], 9)
                self.assertEqual(conflict_summary.hold_reasons["signal_threshold"], 0)
            finally:
                srv_c_up.shutdown()
                srv_c_up.server_close()
                th_c_up.join(timeout=2.0)
        finally:
            srv_s.shutdown()
            srv_s.server_close()
            th_s.join(timeout=2.0)

    def test_replay_tb_whipsaw_timeout_24_7_gap_and_cli_edge_summaries(self) -> None:
        from tradingbot.models import Candle

        srv_long, th_long, port_long = _make_mock_nn_server(384, [0.44, 0.40, 0.16], latency_us=4)
        srv_short, th_short, port_short = _make_mock_nn_server(384, [0.44, 0.16, 0.40], latency_us=4)
        srv_hold, th_hold, port_hold = _make_mock_nn_server(384, [0.80, 0.10, 0.10], latency_us=4)

        def _make_asset(port: int) -> dict[str, Any]:
            raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            raw["provider"] = {
                "preset_path": str((ROOT_DIR / "templates" / "configs" / "providers" / "synthetic.json").resolve())
            }
            raw["features"]["triple_barrier"] = {
                "mode": "pct",
                "take_profit_pct": 0.02,
                "stop_loss_pct": 0.01,
            }
            raw["neural_networks"]["candle_predictor"]["enabled"] = False
            raw["neural_networks"]["signal_classifier"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port,
                "schema_path": "",
            }
            return raw

        def _build_series(
            start_ms: int,
            *,
            whipsaw_bar: int | None = None,
            skip_bar: int | None = None,
        ) -> list[Candle]:
            # 229 bars -> first_anchor=222, horizon=6 -> anchor 222 has horizon 223..228
            out: list[Candle] = []
            for i in range(229):
                if skip_bar is not None and i == skip_bar:
                    continue
                open_ms = start_ms + i * HOUR_MS
                base = 100.0 + (0.2 if i % 2 == 0 else -0.2) if i < 222 else 100.0
                hi = base + 0.3
                lo = base - 0.3
                if whipsaw_bar is not None and i == whipsaw_bar:
                    # Touches both +2% TP (102) and -2% TP/SL (98) on the exact same candle
                    hi = 103.0
                    lo = 97.0
                out.append(
                    Candle(
                        provider="synthetic",
                        symbol="BTCUSDT",
                        timeframe="1h",
                        open_time_ms=open_ms,
                        close_time_ms=open_ms + HOUR_MS - 1,
                        open=base,
                        high=hi,
                        low=lo,
                        close=base,
                        volume=100.0,
                        is_closed=True,
                    )
                )
            return out

        try:
            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            clock = lambda: start_ms + 240 * HOUR_MS

            # 1) Same-bar SL+TP whipsaw resolves conservatively as "sl" for both LONG and SHORT
            whipsaw_candles = _build_series(start_ms, whipsaw_bar=223)
            for port, expected_action in ((port_long, "LONG"), (port_short, "SHORT")):
                asset = parse_asset_dict(_make_asset(port), base_dir=BTC_1H_ASSET_PATH.parent)
                s_whip = ReplayEngine(asset, clock_ms=clock).run(whipsaw_candles)
                self.assertEqual(s_whip.action_counts[expected_action], 7)
                self.assertEqual(s_whip.signal_trades_evaluated, 1)
                self.assertEqual(s_whip.tb_tp_hits, 0)
                self.assertEqual(s_whip.tb_sl_hits, 1)
                self.assertEqual(s_whip.tb_timeouts, 0)

            # 2) Flat horizon never hits TP or SL -> timeout (tb_timeouts=1, tb_resolved=0)
            flat_candles = _build_series(start_ms)
            asset_long = parse_asset_dict(_make_asset(port_long), base_dir=BTC_1H_ASSET_PATH.parent)
            s_to = ReplayEngine(asset_long, clock_ms=clock).run(flat_candles)
            self.assertEqual(s_to.signal_trades_evaluated, 1)
            self.assertEqual(s_to.tb_tp_hits, 0)
            self.assertEqual(s_to.tb_sl_hits, 0)
            self.assertEqual(s_to.tb_timeouts, 1)

            # 3) Timestamp gap inside 24/7 horizon (skip bar 225, add bar 229 so idx+6 exists but shifted)
            gapped = _build_series(start_ms, skip_bar=225)
            extra_ms = start_ms + 229 * HOUR_MS
            gapped.append(
                Candle(
                    provider="synthetic",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    open_time_ms=extra_ms,
                    close_time_ms=extra_ms + HOUR_MS - 1,
                    open=100.0,
                    high=100.3,
                    low=99.7,
                    close=100.0,
                    volume=100.0,
                    is_closed=True,
                )
            )
            s_gap = ReplayEngine(asset_long, clock_ms=clock).run(gapped)
            self.assertEqual(s_gap.signal_trades_evaluated, 0)

            # 4) CLI edge summaries: win_rate=N/A (all timeouts) and filtered_signals=0 (all HOLD)
            replay_app = _load_app("replay_app_edges", REPLAY_APP_PATH)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                db_path = tmp_path / "market.db"
                with SQLiteCandleStore(db_path) as store:
                    store.upsert_candles(flat_candles)

                for port, expected_substr in (
                    (port_long, "TB(TP/SL/TO)=0/0/1 (win_rate=N/A)"),
                    (port_hold, "filtered_signals=0 (no actionable LONG/SHORT trades with complete horizon)"),
                ):
                    asset_file = tmp_path / f"btc_1h_{port}.json"
                    asset_file.write_text(json.dumps(_make_asset(port)), encoding="utf-8")
                    bot_cfg = tmp_path / f"bot_{port}.json"
                    bot_cfg.write_text(
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
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = replay_app.main([str(bot_cfg), "--asset", "btc_1h"])
                    self.assertEqual(rc, 0)
                    self.assertIn(expected_substr, buf.getvalue())
        finally:
            for srv, th in ((srv_long, th_long), (srv_short, th_short), (srv_hold, th_hold)):
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)

    def test_replay_cli_empty_assets_and_unknown_asset_hint(self) -> None:
        replay_app = _load_app("replay_app_empty", REPLAY_APP_PATH)
        empty_tpl = ROOT_DIR / "templates" / "configs" / "bot.json"
        out_buf = io.StringIO()
        with redirect_stdout(out_buf):
            rc = replay_app.main([str(empty_tpl)])
        self.assertEqual(rc, 0)
        self.assertIn("No assets configured", out_buf.getvalue())

        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc_err = replay_app.main([str(empty_tpl), "--asset", "btc_1h"])
        self.assertEqual(rc_err, 1)
        self.assertIn("(none", err_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
