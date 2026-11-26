"""Unit and end-to-end CLI tests for SyncService and apps/backfill_app.py."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from tradingbot.config import AssetConfig, ProviderConfig, parse_asset_dict
from tradingbot.models import Candle, TimeRange, Timeframe, parse_iso_to_ms
from tradingbot.providers import BaseProvider, SyncService
from tradingbot.storage import SQLiteCandleStore

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"
BACKFILL_APP_PATH = ROOT_DIR / "apps" / "backfill_app.py"
HOUR_MS = 3_600_000


def _load_backfill_app_module() -> Any:
    spec = importlib.util.spec_from_file_location("backfill_app", BACKFILL_APP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_asset(
    asset_id: str = "btc_1h",
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    max_candles: int = 50,
    initial_history_from: str = "",
) -> AssetConfig:
    raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
    raw["id"] = asset_id
    raw["market"]["symbol"] = symbol
    raw["market"]["timeframe"] = timeframe
    raw["runtime"]["initial_history_from"] = initial_history_from
    raw["provider"] = {
        "preset_path": "../providers/synthetic.json",
        "max_candles_per_request": max_candles,
    }
    return parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent)


class _SparseGapProvider(BaseProvider):
    """Fake provider returning candles except over designated empty windows."""

    def __init__(self, max_candles: int, empty_windows: list[TimeRange]) -> None:
        self._max_candles = max_candles
        self._empty_windows = empty_windows
        self.call_count = 0

    @property
    def name(self) -> str:
        return "sparse_mock"

    @property
    def max_candles_per_request(self) -> int:
        return self._max_candles

    def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        self.call_count += 1
        tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe.parse(timeframe)
        assert start_ms is not None and end_ms is not None
        out: list[Candle] = []
        ts = tf.align_ceil(start_ms)
        while ts < end_ms and len(out) < (limit or self._max_candles):
            if not any(w.contains(ts) for w in self._empty_windows):
                out.append(
                    Candle(
                        provider=self.name,
                        symbol=symbol,
                        timeframe=tf.label,
                        open_time_ms=ts,
                        close_time_ms=ts + tf.interval_ms - 1,
                        open=100.0,
                        high=102.0,
                        low=99.0,
                        close=101.0,
                        volume=10.0,
                        is_closed=True,
                    )
                )
            ts += tf.interval_ms
        return out


class TestSyncService(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteCandleStore(":memory:")
        self.start = parse_iso_to_ms("2025-01-01T00:00:00Z")

    def tearDown(self) -> None:
        self.store.close()

    def test_paginated_backfill_and_idempotency(self) -> None:
        now_ms = self.start + 200 * HOUR_MS + 5_000
        asset = _make_synthetic_asset("btc_1h", max_candles=50)
        service = SyncService(self.store, clock_ms=lambda: now_ms)

        progress_events: list[tuple[str, int, int, int]] = []
        report = service.backfill_asset(
            asset,
            start_ms=self.start,
            end_ms=self.start + 125 * HOUR_MS,
            progress_cb=lambda aid, i, n, _r, cnt: progress_events.append((aid, i, n, cnt)),
        )

        # 125 candles with page size 50 -> 3 chunks (50, 50, 25)
        self.assertEqual(report.chunks_fetched, 3)
        self.assertEqual(report.candles_upserted, 125)
        self.assertEqual(progress_events, [("btc_1h", 1, 3, 50), ("btc_1h", 2, 3, 50), ("btc_1h", 3, 3, 25)])
        self.assertEqual(self.store.count_candles("synthetic", "BTCUSDT", "1h"), 125)

        # Second backfill over same range is a complete no-op
        report2 = service.backfill_asset(
            asset,
            start_ms=self.start,
            end_ms=self.start + 125 * HOUR_MS,
        )
        self.assertEqual(report2.chunks_fetched, 0)
        self.assertEqual(report2.candles_upserted, 0)

    def test_restart_gap_recovery_internal_tail_and_unclosed_bar(self) -> None:
        asset = _make_synthetic_asset("btc_1h", max_candles=50)
        # Initial run at hour 10 -> fetch 0..10
        now1 = self.start + 10 * HOUR_MS + 1000
        svc1 = SyncService(self.store, clock_ms=lambda: now1)
        svc1.backfill_asset(asset, start_ms=self.start, end_ms=self.start + 10 * HOUR_MS)

        # Simulate internal gap (delete hours 4, 5) + unclosed crash bar at hour 10
        self.store.delete_candles(
            "synthetic", "BTCUSDT", "1h", start_ms=self.start + 4 * HOUR_MS, end_ms=self.start + 6 * HOUR_MS
        )
        self.store.upsert_candles(
            [
                Candle(
                    provider="synthetic",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    open_time_ms=self.start + 10 * HOUR_MS,
                    close_time_ms=self.start + 11 * HOUR_MS - 1,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1.0,
                    is_closed=False,
                )
            ]
        )

        # Restart at hour 13 -> must repair [4h, 6h) AND [10h, 13h) (including unclosed hour 10)
        now2 = self.start + 13 * HOUR_MS + 5_000
        svc2 = SyncService(self.store, clock_ms=lambda: now2)
        report = svc2.sync_asset_to_now(asset, now_ms=now2, include_open_candle=True)

        self.assertEqual(
            report.planned_ranges,
            (
                TimeRange(self.start + 4 * HOUR_MS, self.start + 6 * HOUR_MS),
                TimeRange(self.start + 10 * HOUR_MS, self.start + 13 * HOUR_MS),
            ),
        )
        # Closed bars 0..12 (13 bars) + active open bar 13 -> 14 total stored
        self.assertEqual(self.store.count_candles("synthetic", "BTCUSDT", "1h", closed_only=True), 13)
        self.assertEqual(self.store.count_candles("synthetic", "BTCUSDT", "1h", closed_only=False), 14)

    def test_empty_ranges_recorded_and_force_recheck_overrides(self) -> None:
        now_ms = self.start + 20 * HOUR_MS + 1000
        # Hours 4..6 are market-closed/empty inside a 10-hour window
        empty_window = TimeRange(self.start + 4 * HOUR_MS, self.start + 6 * HOUR_MS)
        fake_provider = _SparseGapProvider(max_candles=50, empty_windows=[empty_window])

        raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        raw["id"] = "nvda_1h"
        raw["market"]["symbol"] = "NVDA"
        raw["market"]["market_schedule"] = "sessions"
        raw["provider"] = {
            "name": "sparse_mock",
            "type": "synthetic",
            "max_candles_per_request": 50,
            "synthetic": {
                "seed": 1,
                "base_price": 100.0,
                "volatility": 0.01,
                "drift": 0.0,
                "base_volume": 10.0,
            },
        }
        asset = parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent)

        service = SyncService(
            self.store,
            provider_factory=lambda _cfg, **_kw: fake_provider,
            clock_ms=lambda: now_ms,
        )

        r1 = service.backfill_asset(asset, start_ms=self.start, end_ms=self.start + 10 * HOUR_MS)
        self.assertEqual(r1.candles_upserted, 8)
        self.assertEqual(r1.empty_ranges_recorded, 1)
        self.assertEqual(fake_provider.call_count, 1)

        # Second sync skips verified empty window -> 0 API calls!
        r2 = service.backfill_asset(asset, start_ms=self.start, end_ms=self.start + 10 * HOUR_MS)
        self.assertEqual(r2.chunks_fetched, 0)
        self.assertEqual(fake_provider.call_count, 1)

        # force_recheck (respect_verified_ranges=False) re-queries the empty window
        r3 = service.backfill_asset(
            asset,
            start_ms=self.start,
            end_ms=self.start + 10 * HOUR_MS,
            respect_verified_ranges=False,
        )
        self.assertEqual(r3.chunks_fetched, 1)
        self.assertEqual(fake_provider.call_count, 2)

    def test_provider_cache_isolates_distinct_overrides_and_weekly_sync_works(self) -> None:
        raw1 = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        raw1["provider"] = {"preset_path": "../providers/synthetic.json", "synthetic": {"seed": 1, "base_price": 1000.0, "volatility": 0.01, "drift": 0.0, "base_volume": 10.0}}
        raw2 = copy.deepcopy(raw1)
        raw2["id"] = "eth_1h"
        raw2["provider"]["synthetic"]["seed"] = 99
        raw2["provider"]["synthetic"]["base_price"] = 5000.0

        a1 = parse_asset_dict(raw1, base_dir=BTC_1H_ASSET_PATH.parent)
        a2 = parse_asset_dict(raw2, base_dir=BTC_1H_ASSET_PATH.parent)
        a1_copy = parse_asset_dict(raw1, base_dir=BTC_1H_ASSET_PATH.parent)

        mon_start = parse_iso_to_ms("2024-01-01T00:00:00Z")
        week_ms = 7 * 86_400_000
        service = SyncService(self.store, clock_ms=lambda: mon_start + 10 * week_ms)
        self.assertIs(service.get_provider(a1.provider), service.get_provider(a1_copy.provider))
        self.assertIsNot(service.get_provider(a1.provider), service.get_provider(a2.provider))

        # Weekly 1w backfill must upsert 5 Monday bars and record 0 false empty ranges
        weekly_asset = _make_synthetic_asset("btc_1w", timeframe="1w", max_candles=50)
        rep = service.backfill_asset(weekly_asset, start_ms=mon_start, end_ms=mon_start + 5 * week_ms)
        self.assertEqual(rep.candles_upserted, 5)
        self.assertEqual(rep.empty_ranges_recorded, 0)
        # Re-running same weekly backfill is a no-op
        rep2 = service.backfill_asset(weekly_asset, start_ms=mon_start, end_ms=mon_start + 5 * week_ms)
        self.assertEqual(rep2.chunks_fetched, 0)


class TestBackfillAppCLI(unittest.TestCase):
    def test_cli_dry_run_backfill_and_sync_now_end_to_end(self) -> None:
        backfill_app = _load_backfill_app_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "test_market.db"

            # Build self-contained temporary multi-asset synthetic config
            syn_preset = json.loads(
                (ROOT_DIR / "configs" / "providers" / "synthetic.json").read_text(encoding="utf-8")
            )
            syn_preset["max_candles_per_request"] = 20
            (tmp_path / "synthetic.json").write_text(json.dumps(syn_preset), encoding="utf-8")

            btc_asset = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            btc_asset["provider"] = {"preset_path": "synthetic.json"}
            (tmp_path / "btc_1h.json").write_text(json.dumps(btc_asset), encoding="utf-8")

            eth_asset = copy.deepcopy(btc_asset)
            eth_asset["id"] = "eth_1h"
            eth_asset["market"]["symbol"] = "ETHUSDT"
            (tmp_path / "eth_1h.json").write_text(json.dumps(eth_asset), encoding="utf-8")

            bot_cfg = {
                "storage": {"db_path": str(db_path), "wal_mode": True, "busy_timeout_ms": 5000},
                "web": {"enabled": False, "host": "127.0.0.1", "port": 8000, "visible_candles": 70},
                "notifications": {"api_enabled": True, "webhooks": []},
                "assets": [
                    {"config_path": "btc_1h.json"},
                    {"config_path": "eth_1h.json", "enabled": False},
                ],
            }
            bot_cfg_path = tmp_path / "bot.json"
            bot_cfg_path.write_text(json.dumps(bot_cfg), encoding="utf-8")

            # 1) Dry run writes nothing
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = backfill_app.main(
                    [str(bot_cfg_path), "--start", "2025-01-01T00:00:00Z", "--end", "2025-01-02T00:00:00Z", "--dry-run"]
                )
            self.assertEqual(rc, 0)
            self.assertIn("[DRY RUN]", buf.getvalue())
            self.assertIn("missing_candles=24", buf.getvalue())

            with SQLiteCandleStore(db_path) as store:
                self.assertEqual(store.count_candles("synthetic", "BTCUSDT", "1h"), 0)

            # 2) Real backfill for enabled asset (btc_1h -> 24 hourly candles)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = backfill_app.main(
                    [str(bot_cfg_path), "--start", "2025-01-01T00:00:00Z", "--end", "2025-01-02T00:00:00Z"]
                )
            self.assertEqual(rc, 0)
            self.assertIn("upserted=24", buf.getvalue())

            # 3) Explicit --asset eth_1h backfills even though disabled in bot.json
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = backfill_app.main(
                    [
                        str(bot_cfg_path),
                        "--asset",
                        "eth_1h",
                        "--start",
                        "2025-01-01T00:00:00Z",
                        "--end",
                        "2025-01-01T12:00:00Z",
                    ]
                )
            self.assertEqual(rc, 0)

            with SQLiteCandleStore(db_path) as store:
                self.assertEqual(store.count_candles("synthetic", "BTCUSDT", "1h"), 24)
                self.assertEqual(store.count_candles("synthetic", "ETHUSDT", "1h"), 12)

            # 4) Controlled CLI error exit codes
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    backfill_app.main([str(bot_cfg_path), "--sync-now", "--start", "2025-01-01"]),
                    2,
                )
                self.assertEqual(
                    backfill_app.main([str(bot_cfg_path), "--start", "2025-01-05", "--end", "2025-01-01"]),
                    2,
                )
                self.assertEqual(
                    backfill_app.main([str(bot_cfg_path), "--asset", "nonexistent", "--sync-now"]),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
