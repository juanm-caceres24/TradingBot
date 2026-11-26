"""Unit tests for tradingbot.models."""

from __future__ import annotations

import math
import unittest

from tradingbot.models import (
    AlertEvent,
    Candle,
    PredictedCandle,
    PredictionSnapshot,
    SignalPrediction,
    TimeRange,
    Timeframe,
    TradeAnalysis,
    format_ms_to_iso,
    parse_iso_to_ms,
    validate_asset_id,
)


class TestTimeframeAndTimeUtils(unittest.TestCase):
    def test_parse_valid_timeframes(self) -> None:
        self.assertEqual(Timeframe.parse("1m").interval_ms, 60_000)
        self.assertEqual(Timeframe.parse("15m").interval_ms, 900_000)
        self.assertEqual(Timeframe.parse(" 1H ").label, "1h")
        self.assertEqual(Timeframe.parse("1h").interval_seconds, 3600)
        self.assertEqual(Timeframe.parse("1d").interval_ms, 86_400_000)
        self.assertEqual(Timeframe.parse("1w").interval_ms, 604_800_000)

    def test_parse_invalid_timeframes(self) -> None:
        for bad in ("", "0m", "-1h", "1s", "h", "15x", "1.5h"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Timeframe.parse(bad)

    def test_alignment_helpers(self) -> None:
        tf = Timeframe.parse("1h")
        hour = 3_600_000
        self.assertTrue(tf.is_aligned(0))
        self.assertTrue(tf.is_aligned(5 * hour))
        self.assertFalse(tf.is_aligned(5 * hour + 1))

        self.assertEqual(tf.align_floor(5 * hour + 1234), 5 * hour)
        self.assertEqual(tf.align_ceil(5 * hour), 5 * hour)
        self.assertEqual(tf.align_ceil(5 * hour + 1), 6 * hour)
        self.assertEqual(tf.next_open(5 * hour), 6 * hour)

    def test_iso_roundtrip(self) -> None:
        ts_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
        self.assertEqual(ts_ms, 1_735_689_600_000)
        self.assertEqual(format_ms_to_iso(ts_ms), "2025-01-01T00:00:00Z")
        self.assertEqual(parse_iso_to_ms("2025-01-01"), ts_ms)

        with self.assertRaises(ValueError):
            parse_iso_to_ms("not-a-date")
        with self.assertRaises(ValueError):
            parse_iso_to_ms("1960-01-01T00:00:00Z")

    def test_validate_asset_id(self) -> None:
        self.assertEqual(validate_asset_id("btc_1h"), "btc_1h")
        self.assertEqual(validate_asset_id("sp500-1d"), "sp500-1d")
        for bad in ("", "BTC_1H", "_btc", "btc/1h", "btc 1h"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_asset_id(bad)


class TestTimeRange(unittest.TestCase):
    def test_half_open_semantics(self) -> None:
        tf = Timeframe.parse("1m")
        r = TimeRange(0, 180_000)
        self.assertFalse(r.is_empty)
        self.assertEqual(r.candle_count(tf), 3)
        self.assertTrue(r.contains(0))
        self.assertTrue(r.contains(120_000))
        self.assertFalse(r.contains(180_000))

        adjacent = TimeRange(180_000, 240_000)
        self.assertFalse(r.overlaps(adjacent))
        self.assertTrue(r.touches(adjacent))

    def test_invalid_range(self) -> None:
        with self.assertRaises(ValueError):
            TimeRange(100, 50)
        with self.assertRaises(ValueError):
            TimeRange(-1, 100)
        with self.assertRaises(ValueError):
            TimeRange(True, 100)  # type: ignore[arg-type]


class TestCandlePredictionAndAlertModels(unittest.TestCase):
    def test_valid_candle_and_properties(self) -> None:
        c = Candle(
            provider="binance_spot",
            symbol="BTCUSDT",
            timeframe="1h",
            open_time_ms=0,
            close_time_ms=3_599_999,
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=42.5,
            is_closed=True,
        )
        self.assertTrue(c.is_bullish)
        self.assertAlmostEqual(c.body_size, 5.0)
        self.assertAlmostEqual(c.full_range, 15.0)
        self.assertEqual(c.to_dict()["symbol"], "BTCUSDT")

    def test_invalid_candle_geometry_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Candle("p", "BTCUSDT", "1h", 0, 10, 100.0, 102.0, 95.0, 105.0, 1.0)
        with self.assertRaises(ValueError):
            Candle("p", "BTCUSDT", "1h", 0, 10, 100.0, 110.0, 101.0, 105.0, 1.0)
        with self.assertRaises(ValueError):
            Candle("p", "BTCUSDT", "1h", 0, 10, 100.0, 110.0, 95.0, 105.0, -0.1)
        with self.assertRaises(ValueError):
            Candle("p", "BTCUSDT", "1h", 0, 10, math.nan, 110.0, 95.0, 105.0, 1.0)
        with self.assertRaises(ValueError):
            Candle("p", "BTCUSDT", "1h", 0, 10, 0.0, 110.0, 0.0, 105.0, 1.0)

    def test_prediction_and_alert_models(self) -> None:
        pc = PredictedCandle(step=1, open_time_ms=3_600_000, open=105.0, high=108.0, low=104.0, close=107.0, confidence=0.85)
        sig = SignalPrediction(hold_prob=0.15, long_prob=0.75, short_prob=0.10, latency_us=12)
        self.assertEqual(sig.best_action, "LONG")
        self.assertAlmostEqual(sig.best_probability, 0.75)

        analysis = TradeAnalysis(
            action="LONG",
            expected_move_pct=0.019,
            estimated_tp=108.0,
            estimated_sl=104.0,
            risk_reward=3.0,
            confidence=0.75,
            reason="Bullish forecast aligned with LONG signal",
        )
        snap = PredictionSnapshot(
            asset_id="btc_1h",
            provider="binance_spot",
            symbol="BTCUSDT",
            timeframe="1h",
            anchor_open_time_ms=0,
            created_at_ms=3_600_001,
            predicted_candles=(pc,),
            signal=sig,
            analysis=analysis,
            candle_latency_us=18,
        )
        self.assertEqual(len(snap.predicted_candles), 1)
        self.assertEqual(snap.asset_id, "btc_1h")

        alert = AlertEvent(
            asset_id="btc_1h",
            provider="binance_spot",
            symbol="BTCUSDT",
            timeframe="1h",
            anchor_open_time_ms=3_600_000,
            created_at_ms=3_600_100,
            action="LONG",
            confidence=0.78,
            consecutive_hits=2,
            rule_name="consecutive_signal",
            payload_json='{"tp": 108.0}',
        )
        self.assertEqual(alert.to_dict()["action"], "LONG")
        self.assertFalse(alert.acknowledged)

        with self.assertRaises(ValueError):
            SignalPrediction(hold_prob=0.5, long_prob=0.5, short_prob=0.5)
        with self.assertRaises(ValueError):
            AlertEvent(
                asset_id="btc_1h",
                provider="binance_spot",
                symbol="BTCUSDT",
                timeframe="1h",
                anchor_open_time_ms=0,
                created_at_ms=1,
                action="LONG",
                confidence=1.2,
                consecutive_hits=1,
                rule_name="bad_conf",
                payload_json="{}",
            )
        with self.assertRaises(ValueError):
            AlertEvent(
                asset_id="btc_1h",
                provider="binance_spot",
                symbol="BTCUSDT",
                timeframe="1h",
                anchor_open_time_ms=0,
                created_at_ms=1,
                action="LONG",
                confidence=0.8,
                consecutive_hits=0,
                rule_name="bad_hits",
                payload_json="{}",
            )


if __name__ == "__main__":
    unittest.main()
