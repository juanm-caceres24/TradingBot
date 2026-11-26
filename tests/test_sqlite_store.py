"""Unit tests for SQLiteCandleStore."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tradingbot.models import Candle, TimeRange
from tradingbot.storage import SQLiteCandleStore

HOUR_MS = 3_600_000


def _make_candle(
    idx: int,
    *,
    provider: str = "binance_spot",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    close_price: float | None = None,
    is_closed: bool = True,
) -> Candle:
    open_ms = idx * HOUR_MS
    close_ms = open_ms + HOUR_MS - 1
    o = 100.0 + idx
    c = close_price if close_price is not None else o + 1.0
    h = max(o, c) + 2.0
    l = min(o, c) - 2.0
    return Candle(
        provider=provider,
        symbol=symbol,
        timeframe=timeframe,
        open_time_ms=open_ms,
        close_time_ms=close_ms,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=10.0 + idx,
        is_closed=is_closed,
    )


class TestSQLiteCandleStore(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteCandleStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_file_backed_wal_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nested" / "dir" / "market.db"
            with SQLiteCandleStore(db_path, wal_mode=True) as store:
                self.assertTrue(db_path.is_file())
                store.upsert_candles([_make_candle(1)])
                self.assertEqual(store.count_candles("binance_spot", "BTCUSDT", "1h"), 1)

    def test_upsert_deduplicates_and_updates_open_to_closed(self) -> None:
        open_bar = _make_candle(10, close_price=111.0, is_closed=False)
        self.assertEqual(self.store.upsert_candles([open_bar]), 1)
        self.assertEqual(self.store.count_candles("binance_spot", "BTCUSDT", "1h", closed_only=True), 0)
        self.assertEqual(self.store.count_candles("binance_spot", "BTCUSDT", "1h", closed_only=False), 1)

        closed_bar = _make_candle(10, close_price=115.0, is_closed=True)
        self.assertEqual(self.store.upsert_candles([closed_bar]), 1)
        self.assertEqual(self.store.count_candles("binance_spot", "BTCUSDT", "1h"), 1)
        loaded = self.store.get_candles("binance_spot", "BTCUSDT", "1h")
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded[0].close, 115.0)
        self.assertTrue(loaded[0].is_closed)

    def test_range_query_latest_bounds_and_series_isolation(self) -> None:
        btc_bars = [_make_candle(i) for i in range(5)]
        eth_bars = [_make_candle(i, symbol="ETHUSDT") for i in range(3)]
        self.store.upsert_candles(btc_bars + eth_bars)

        # Half-open [1h, 4h) -> indices 1, 2, 3
        sliced = self.store.get_candles(
            "binance_spot", "BTCUSDT", "1h", start_ms=1 * HOUR_MS, end_ms=4 * HOUR_MS
        )
        self.assertEqual([c.open_time_ms for c in sliced], [1 * HOUR_MS, 2 * HOUR_MS, 3 * HOUR_MS])

        # Latest 2 candles returned in ascending chronological order
        latest = self.store.get_latest_candles("binance_spot", "BTCUSDT", "1h", limit=2)
        self.assertEqual([c.open_time_ms for c in latest], [3 * HOUR_MS, 4 * HOUR_MS])
        self.assertEqual(self.store.get_latest_candles("binance_spot", "BTCUSDT", "1h", limit=0), [])

        # Bounds and isolation
        self.assertEqual(self.store.get_bounds("binance_spot", "BTCUSDT", "1h"), (0, 4 * HOUR_MS))
        self.assertEqual(self.store.get_bounds("binance_spot", "ETHUSDT", "1h"), (0, 2 * HOUR_MS))
        self.assertIsNone(self.store.get_bounds("binance_spot", "SOLUSDT", "1h"))

        # Delete partial range [1h, 3h)
        deleted = self.store.delete_candles(
            "binance_spot", "BTCUSDT", "1h", start_ms=1 * HOUR_MS, end_ms=3 * HOUR_MS
        )
        self.assertEqual(deleted, 2)
        self.assertEqual(
            self.store.get_open_times("binance_spot", "BTCUSDT", "1h"),
            [0, 3 * HOUR_MS, 4 * HOUR_MS],
        )

    def test_sync_ranges_record_dedup_and_overlap_query(self) -> None:
        r1 = TimeRange(10 * HOUR_MS, 20 * HOUR_MS)
        r2 = TimeRange(30 * HOUR_MS, 40 * HOUR_MS)
        self.store.record_sync_range("binance_spot", "BTCUSDT", "1h", r1, checked_at_ms=1000)
        self.store.record_sync_range("binance_spot", "BTCUSDT", "1h", r1, checked_at_ms=2000)
        self.store.record_sync_range("binance_spot", "BTCUSDT", "1h", r2, checked_at_ms=2000)
        # Empty range ignored safely
        self.store.record_sync_range("binance_spot", "BTCUSDT", "1h", TimeRange(5, 5), checked_at_ms=2000)

        all_ranges = self.store.get_sync_ranges("binance_spot", "BTCUSDT", "1h")
        self.assertEqual(all_ranges, [r1, r2])

        # Query overlapping only r1 (starts inside r1, ends before r2)
        overlapping = self.store.get_sync_ranges(
            "binance_spot", "BTCUSDT", "1h", start_ms=15 * HOUR_MS, end_ms=30 * HOUR_MS
        )
        self.assertEqual(overlapping, [r1])

    def test_prediction_audit_log(self) -> None:
        p1 = json.dumps({"action": "LONG", "confidence": 0.72})
        p2 = json.dumps({"action": "HOLD", "confidence": 0.61})
        id1 = self.store.save_prediction("binance_spot", "BTCUSDT", "1h", 10 * HOUR_MS, 1000, "candle_predictor", p1)
        id2 = self.store.save_prediction("binance_spot", "BTCUSDT", "1h", 11 * HOUR_MS, 2000, "signal_classifier", p2)
        self.assertGreater(id2, id1)

        rows = self.store.get_predictions("binance_spot", "BTCUSDT", "1h", limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model_name"], "signal_classifier")
        self.assertEqual(json.loads(rows[1]["payload_json"])["action"], "LONG")


if __name__ == "__main__":
    unittest.main()
