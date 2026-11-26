"""Unit tests for GapDetector and half-open interval utilities."""

from __future__ import annotations

import unittest

from tradingbot.models import Candle, TimeRange, Timeframe, parse_iso_to_ms
from tradingbot.storage import (
    GapDetector,
    SQLiteCandleStore,
    chunk_range,
    merge_ranges,
    subtract_ranges,
)

DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def _candle_at(open_ms: int, tf: str = "1d", is_closed: bool = True) -> Candle:
    interval = Timeframe.parse(tf).interval_ms
    return Candle(
        provider="binance_spot",
        symbol="BTCUSDT",
        timeframe=tf,
        open_time_ms=open_ms,
        close_time_ms=open_ms + interval - 1,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=50.0,
        is_closed=is_closed,
    )


class TestIntervalUtilities(unittest.TestCase):
    def test_merge_ranges(self) -> None:
        ranges = [
            TimeRange(10, 20),
            TimeRange(0, 5),
            TimeRange(5, 10),   # touches previous & next
            TimeRange(30, 30),  # empty ignored
            TimeRange(25, 35),
            TimeRange(28, 32),  # contained
        ]
        self.assertEqual(merge_ranges(ranges), [TimeRange(0, 20), TimeRange(25, 35)])

    def test_subtract_ranges(self) -> None:
        base = [TimeRange(0, 100)]
        exc = [TimeRange(10, 20), TimeRange(40, 60), TimeRange(90, 120)]
        self.assertEqual(
            subtract_ranges(base, exc),
            [TimeRange(0, 10), TimeRange(20, 40), TimeRange(60, 90)],
        )

    def test_chunk_range_pagination(self) -> None:
        tf = Timeframe.parse("1h")
        full = TimeRange(0, 2500 * HOUR_MS)
        chunks = chunk_range(full, tf, max_candles=1000)
        self.assertEqual(
            chunks,
            [
                TimeRange(0, 1000 * HOUR_MS),
                TimeRange(1000 * HOUR_MS, 2000 * HOUR_MS),
                TimeRange(2000 * HOUR_MS, 2500 * HOUR_MS),
            ],
        )
        self.assertEqual(chunk_range(TimeRange(0, 0), tf, max_candles=100), [])


class TestGapDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteCandleStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_tail_gap_day_1_to_10_restarted_on_day_13(self) -> None:
        # Stored closed candles for days 1..10 (open_time = day 1..10)
        self.store.upsert_candles([_candle_at(d * DAY_MS, "1d") for d in range(1, 11)])
        # Restart midway through day 13 -> latest closed candle is day 12 (end = day 13)
        now_ms = 13 * DAY_MS + 5 * HOUR_MS

        missing = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1d", now_ms
        )
        # Missing closed candles: day 11 and day 12 -> [11*DAY, 13*DAY)
        self.assertEqual(missing, [TimeRange(11 * DAY_MS, 13 * DAY_MS)])
        self.assertEqual(missing[0].candle_count(Timeframe.parse("1d")), 2)

    def test_tail_gap_2025_to_oct_2026(self) -> None:
        start_2025 = parse_iso_to_ms("2025-12-28T00:00:00Z")
        # Stored candles Dec 28..31 2025
        self.store.upsert_candles([_candle_at(start_2025 + i * DAY_MS, "1d") for i in range(4)])

        now_ms = parse_iso_to_ms("2026-10-26T14:30:00Z")
        expected_start = parse_iso_to_ms("2026-01-01T00:00:00Z")
        expected_end = parse_iso_to_ms("2026-10-26T00:00:00Z")

        missing = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1d", now_ms
        )
        self.assertEqual(missing, [TimeRange(expected_start, expected_end)])

    def test_internal_and_tail_gaps_toggle(self) -> None:
        # Days 1..3 and 6..8 stored (days 4..5 missing internally; day 9..10 missing at tail)
        stored_days = [1, 2, 3, 6, 7, 8]
        self.store.upsert_candles([_candle_at(d * DAY_MS, "1d") for d in stored_days])
        now_ms = 11 * DAY_MS + 1000

        with_internal = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1d", now_ms, fill_internal_gaps=True
        )
        self.assertEqual(
            with_internal,
            [TimeRange(4 * DAY_MS, 6 * DAY_MS), TimeRange(9 * DAY_MS, 11 * DAY_MS)],
        )

        tail_only = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1d", now_ms, fill_internal_gaps=False
        )
        self.assertEqual(tail_only, [TimeRange(9 * DAY_MS, 11 * DAY_MS)])

    def test_unclosed_candle_is_treated_as_missing_for_recovery(self) -> None:
        # Hour 1 & 2 closed; Hour 3 crashed while still open (is_closed=False)
        self.store.upsert_candles(
            [
                _candle_at(1 * HOUR_MS, "1h", is_closed=True),
                _candle_at(2 * HOUR_MS, "1h", is_closed=True),
                _candle_at(3 * HOUR_MS, "1h", is_closed=False),
            ]
        )
        now_ms = 5 * HOUR_MS + 12_000
        missing = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1h", now_ms
        )
        # Must re-fetch Hour 3 AND Hour 4 -> [3h, 5h)
        self.assertEqual(missing, [TimeRange(3 * HOUR_MS, 5 * HOUR_MS)])

    def test_verified_sync_ranges_suppress_non_24_7_gaps(self) -> None:
        # Stock-like gap: Friday day 5 present, Mon day 8 present, Sat-Sun [6, 8) verified empty
        self.store.upsert_candles([_candle_at(5 * DAY_MS, "1d"), _candle_at(8 * DAY_MS, "1d")])
        self.store.record_sync_range(
            "binance_spot", "BTCUSDT", "1d", TimeRange(6 * DAY_MS, 8 * DAY_MS), checked_at_ms=9999
        )

        missing = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1d", now_ms=9 * DAY_MS + 500
        )
        self.assertEqual(missing, [])

        # Ignoring verified ranges reveals the raw weekend gap again
        raw_missing = GapDetector.plan_sync_ranges(
            self.store,
            "binance_spot",
            "BTCUSDT",
            "1d",
            now_ms=9 * DAY_MS + 500,
            respect_verified_ranges=False,
        )
        self.assertEqual(raw_missing, [TimeRange(6 * DAY_MS, 8 * DAY_MS)])

    def test_empty_db_plans_initial_history_or_min_window(self) -> None:
        now_ms = 200 * HOUR_MS + 30_000

        # Without initial_history_from -> last 50 closed candles
        default_plan = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1h", now_ms, min_candles_when_empty=50
        )
        self.assertEqual(default_plan, [TimeRange(150 * HOUR_MS, 200 * HOUR_MS)])

        # With unaligned initial_history_from -> aligned ceil to next candle open
        from_plan = GapDetector.plan_sync_ranges(
            self.store,
            "binance_spot",
            "BTCUSDT",
            "1h",
            now_ms,
            initial_history_from_ms=100 * HOUR_MS + 1,
        )
        self.assertEqual(from_plan, [TimeRange(101 * HOUR_MS, 200 * HOUR_MS)])

    def test_head_gap_backfill_when_initial_history_precedes_existing(self) -> None:
        # DB only has hours 50..55, user sets initial_history_from = hour 40
        self.store.upsert_candles([_candle_at(h * HOUR_MS, "1h") for h in range(50, 56)])
        missing = GapDetector.plan_sync_ranges(
            self.store,
            "binance_spot",
            "BTCUSDT",
            "1h",
            now_ms=58 * HOUR_MS + 100,
            initial_history_from_ms=40 * HOUR_MS,
            fill_internal_gaps=True,
        )
        self.assertEqual(
            missing,
            [TimeRange(40 * HOUR_MS, 50 * HOUR_MS), TimeRange(56 * HOUR_MS, 58 * HOUR_MS)],
        )

    def test_weekly_binance_monday_candles_are_recognized_without_false_gaps(self) -> None:
        week_ms = 7 * DAY_MS
        mon0 = parse_iso_to_ms("2024-01-01T00:00:00Z")  # Monday
        mon1 = mon0 + week_ms
        mon2 = mon0 + 2 * week_ms
        mon3 = mon0 + 3 * week_ms
        # Store week 0 and week 2 (week 1 missing)
        self.store.upsert_candles([_candle_at(mon0, "1w"), _candle_at(mon2, "1w")])
        # Query on Wednesday of week 3 -> latest closed weekly bar ends at mon3
        now_wed = mon3 + 2 * DAY_MS
        missing = GapDetector.plan_sync_ranges(
            self.store, "binance_spot", "BTCUSDT", "1w", now_wed
        )
        self.assertEqual(missing, [TimeRange(mon1, mon2)])


if __name__ == "__main__":
    unittest.main()
