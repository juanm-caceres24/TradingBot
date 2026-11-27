"""Analytical unit tests for built-in indicators and the IndicatorEngine registry."""

from __future__ import annotations

import math
import unittest

from tradingbot.config import IndicatorConfig
from tradingbot.indicators import (
    ATRIndicator,
    BaseIndicator,
    BollingerBandsIndicator,
    EMAIndicator,
    HTFContextIndicator,
    INDICATOR_REGISTRY,
    IndicatorEngine,
    IndicatorError,
    RSIIndicator,
    SMAIndicator,
    SqueezeMomentumIndicator,
    VolatilityRegimeIndicator,
    register_indicator,
)
from tradingbot.models import Candle, Timeframe, parse_iso_to_ms

HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def _candles_from_closes(
    closes: list[float],
    *,
    spread: float = 1.0,
) -> list[Candle]:
    bars: list[Candle] = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        hi = max(o, c) + spread
        lo = max(0.01, min(o, c) - spread)
        bars.append(
            Candle(
                provider="synthetic",
                symbol="BTCUSDT",
                timeframe="1h",
                open_time_ms=i * HOUR_MS,
                close_time_ms=(i + 1) * HOUR_MS - 1,
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=100.0,
                is_closed=True,
            )
        )
    return bars


class TestBuiltinIndicators(unittest.TestCase):
    def test_sma_exact_values_and_warmup(self) -> None:
        bars = _candles_from_closes([10.0, 20.0, 30.0, 40.0, 50.0])
        sma = SMAIndicator("sma_3", {"period": 3})
        self.assertEqual(sma.warmup_period(), 2)
        vals = sma.compute(bars)["value"]
        self.assertEqual(vals[:2], [None, None])
        self.assertAlmostEqual(vals[2], 20.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(vals[3], 30.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(vals[4], 40.0)  # type: ignore[arg-type]

    def test_ema_exact_values_and_warmup(self) -> None:
        bars = _candles_from_closes([10.0, 20.0, 30.0, 40.0])
        ema = EMAIndicator("ema_3", {"period": 3})
        self.assertEqual(ema.warmup_period(), 2)
        vals = ema.compute(bars)["value"]
        self.assertEqual(vals[:2], [None, None])
        # Seed = SMA([10, 20, 30]) = 20.0; alpha = 2 / (3 + 1) = 0.5 -> next = 0.5*40 + 0.5*20 = 30.0
        self.assertAlmostEqual(vals[2], 20.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(vals[3], 30.0)  # type: ignore[arg-type]

    def test_rsi_monotone_up_down_and_flat(self) -> None:
        up = _candles_from_closes([10.0, 11.0, 12.0, 13.0, 14.0])
        rsi = RSIIndicator("rsi_3", {"period": 3})
        self.assertEqual(rsi.warmup_period(), 3)
        vals_up = rsi.compute(up)["value"]
        self.assertEqual(vals_up[:3], [None, None, None])
        self.assertAlmostEqual(vals_up[3], 100.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(vals_up[4], 100.0)  # type: ignore[arg-type]

        down = _candles_from_closes([14.0, 13.0, 12.0, 11.0, 10.0])
        vals_down = rsi.compute(down)["value"]
        self.assertAlmostEqual(vals_down[3], 0.0)  # type: ignore[arg-type]

        flat = _candles_from_closes([10.0, 10.0, 10.0, 10.0, 10.0])
        vals_flat = rsi.compute(flat)["value"]
        self.assertAlmostEqual(vals_flat[3], 50.0)  # type: ignore[arg-type]

    def test_atr_exact_wilder_smoothing(self) -> None:
        # Constant close=100, spread=2 -> high=102, low=98 -> TR = 4.0 every bar
        bars = _candles_from_closes([100.0] * 6, spread=2.0)
        atr = ATRIndicator("atr_3", {"period": 3})
        self.assertEqual(atr.warmup_period(), 3)
        vals = atr.compute(bars)["value"]
        self.assertEqual(vals[:3], [None, None, None])
        for v in vals[3:]:
            self.assertAlmostEqual(v, 4.0)  # type: ignore[arg-type]

    def test_bollinger_bands_exact(self) -> None:
        bars = _candles_from_closes([10.0, 12.0, 14.0, 16.0])
        bb = BollingerBandsIndicator("bb_3", {"period": 3, "mult": 2.0})
        out = bb.compute(bars)
        self.assertIsNone(out["middle"][1])
        # Window [10, 12, 14]: mean=12, pop std = sqrt(((4+0+4)/3)) = sqrt(8/3)
        expected_std = math.sqrt(8.0 / 3.0)
        self.assertAlmostEqual(out["middle"][2], 12.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(out["upper"][2], 12.0 + 2.0 * expected_std)  # type: ignore[arg-type]
        self.assertAlmostEqual(out["lower"][2], 12.0 - 2.0 * expected_std)  # type: ignore[arg-type]

    def test_squeeze_momentum_warmup_squeeze_on_and_off(self) -> None:
        sqz = SqueezeMomentumIndicator(
            "sqz_4",
            {"bb_length": 4, "bb_mult": 2.0, "kc_length": 4, "kc_mult": 1.5, "mom_length": 4},
        )
        # warmup = max(3, 3, 2*4 - 2 = 6) = 6
        self.assertEqual(sqz.warmup_period(), 6)

        # Flat closes with wide high-low wicks -> std(close)=0 (BB width=0) inside KC -> squeeze ON (+1.0)
        coiled = _candles_from_closes([100.0] * 10, spread=5.0)
        out_coiled = sqz.compute(coiled)
        self.assertTrue(all(v is None for v in out_coiled["momentum"][:6]))
        self.assertTrue(all(v is None for v in out_coiled["squeeze_state"][:6]))
        self.assertEqual(out_coiled["squeeze_state"][6], 1.0)
        self.assertAlmostEqual(out_coiled["momentum"][6], 0.0)  # type: ignore[arg-type]

        # Strong breakout with tiny wicks -> large close std outside narrow KC -> squeeze OFF (-1.0) & positive momentum
        breakout = _candles_from_closes([100.0, 100.0, 100.0, 100.0, 101.0, 104.0, 110.0, 120.0, 135.0], spread=0.05)
        out_brk = sqz.compute(breakout)
        self.assertEqual(out_brk["squeeze_state"][-1], -1.0)
        self.assertGreater(out_brk["momentum"][-1], 0.0)  # type: ignore[arg-type]

    def test_parameter_validation_and_custom_registration(self) -> None:
        with self.assertRaises(IndicatorError):
            SMAIndicator("bad", {"period": 0})
        with self.assertRaises(IndicatorError):
            SMAIndicator("bad", {"period": 10, "unknown": 1})

        class DummyIndicator(BaseIndicator):
            indicator_type = "dummy_test"

            def _validate_params(self) -> None:
                pass

            def warmup_period(self) -> int:
                return 0

            def output_columns(self) -> tuple[str, ...]:
                return ("flag",)

            def nn_scale_kind(self, column: str) -> str:
                return "bounded"

            def compute(self, candles: list[Candle]) -> dict[str, list[float | None]]:
                return {"flag": [1.0 if c.is_bullish else -1.0 for c in candles]}

        with self.assertRaises(IndicatorError):
            register_indicator("not_a_class", 123)  # type: ignore[arg-type]

        register_indicator("dummy_test", DummyIndicator)
        self.addCleanup(INDICATOR_REGISTRY.pop, "dummy_test", None)
        engine = IndicatorEngine(
            [
                IndicatorConfig(id="sma_3", type="sma", params={"period": 3}, use_in_nn=False, plot="overlay"),
                IndicatorConfig(id="dummy", type="dummy_test", params={}, use_in_nn=True, plot="none"),
            ]
        )
        self.assertEqual(engine.max_warmup(nn_only=False), 2)
        self.assertEqual(engine.max_warmup(nn_only=True), 0)
        res = engine.compute_all(_candles_from_closes([10.0, 11.0, 12.0]), nn_only=True)
        self.assertEqual(list(res.keys()), ["dummy"])
        self.assertEqual(res["dummy"]["flag"], [1.0, 1.0, 1.0])

    def test_volatility_regime_warmup_bounds_and_expansion_detection(self) -> None:
        vr = VolatilityRegimeIndicator(
            "vol_reg", {"fast_period": 3, "slow_period": 10, "rank_window": 10}
        )
        # warmup = max(10, 3 + 10 - 1) = 12
        self.assertEqual(vr.warmup_period(), 12)
        self.assertEqual(vr.output_columns(), ("vol_ratio", "vol_rank"))
        self.assertEqual(vr.nn_scale_kind("vol_ratio"), "bounded")
        self.assertEqual(vr.nn_scale_kind("vol_rank"), "bounded")

        # Constant volatility -> vol_ratio == 0.0, mid-rank vol_rank == 0.0
        flat_bars = _candles_from_closes([100.0] * 20, spread=2.0)
        out_flat = vr.compute(flat_bars)
        self.assertTrue(all(v is None for v in out_flat["vol_ratio"][:12]))
        self.assertTrue(all(v is None for v in out_flat["vol_rank"][:12]))
        for i in range(12, 20):
            self.assertAlmostEqual(out_flat["vol_ratio"][i], 0.0, places=9)  # type: ignore[arg-type]
            self.assertAlmostEqual(out_flat["vol_rank"][i], 0.0, places=9)  # type: ignore[arg-type]

        # Calm regime followed by a sharp volatility burst -> positive vol_ratio & high vol_rank
        burst_bars = _candles_from_closes([100.0] * 16, spread=0.5)
        for i in range(16, 22):
            open_ms = i * HOUR_MS
            burst_bars.append(
                Candle(
                    provider="synthetic",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    open_time_ms=open_ms,
                    close_time_ms=open_ms + HOUR_MS - 1,
                    open=100.0,
                    high=110.0,
                    low=90.0,
                    close=100.0,
                    volume=100.0,
                    is_closed=True,
                )
            )
        out_burst = vr.compute(burst_bars)
        self.assertGreater(out_burst["vol_ratio"][-1], 0.5)  # type: ignore[arg-type]
        self.assertGreater(out_burst["vol_rank"][-1], 0.7)  # type: ignore[arg-type]
        for col in ("vol_ratio", "vol_rank"):
            for v in out_burst[col][12:]:
                assert v is not None
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)

        # Empty input and flat zero-range candles (ATR == 0) must stay finite without NaN/ZeroDivisionError
        self.assertEqual(vr.compute([]), {"vol_ratio": [], "vol_rank": []})
        zero_range_bars = _candles_from_closes([100.0] * 18, spread=0.0)
        out_zero = vr.compute(zero_range_bars)
        self.assertTrue(all(v is None for v in out_zero["vol_ratio"][:12]))
        self.assertTrue(all(v is None for v in out_zero["vol_rank"][:12]))
        for i in range(12, 18):
            self.assertEqual(out_zero["vol_ratio"][i], 0.0)
            self.assertEqual(out_zero["vol_rank"][i], 0.0)

        # Invalid params rejected
        with self.assertRaises(IndicatorError):
            VolatilityRegimeIndicator("bad", {"fast_period": 10, "slow_period": 10, "rank_window": 10})
        with self.assertRaises(IndicatorError):
            VolatilityRegimeIndicator("bad", {"fast_period": 3, "slow_period": 10, "rank_window": 1})

    def test_htf_context_exact_causal_parity_and_validation(self) -> None:
        htf = HTFContextIndicator(
            "htf_4h",
            {
                "base_timeframe": "1h",
                "htf_timeframe": "4h",
                "ema_fast": 3,
                "ema_slow": 6,
                "rsi_period": 4,
            },
        )
        # ratio=4, need_htf=max(6, 5)=6 -> warmup = (6 - 1) * 4 = 20
        self.assertEqual(htf.warmup_period(), 20)
        self.assertEqual(htf.nn_scale_kind("trend"), "price_delta")
        self.assertEqual(htf.nn_scale_kind("close_vs_slow"), "price_delta")
        self.assertEqual(htf.nn_scale_kind("rsi"), "rsi_0_100")

        # Start mid-bucket at 02:00 UTC to verify partial initial bucket handling too
        closes = [
            100.0 + 1.2 * math.sin(i * 0.37) + 0.18 * i + (0.6 if i % 5 == 0 else -0.3)
            for i in range(44)
        ]
        bars: list[Candle] = []
        base_ts = 2 * HOUR_MS
        for i, c in enumerate(closes):
            o = closes[i - 1] if i > 0 else c
            bars.append(
                Candle(
                    provider="synthetic",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    open_time_ms=base_ts + i * HOUR_MS,
                    close_time_ms=base_ts + (i + 1) * HOUR_MS - 1,
                    open=o,
                    high=max(o, c) + 0.5,
                    low=min(o, c) - 0.5,
                    close=c,
                    volume=100.0,
                    is_closed=True,
                )
            )

        out = htf.compute(bars)
        for col in ("trend", "close_vs_slow", "rsi"):
            self.assertTrue(all(v is None for v in out[col][:20]))

        # Brute-force prefix verification: at every i >= warmup, resample ONLY bars[:i+1]
        # and confirm bit-for-bit equality (proves zero future leakage & exact O(N) math)
        ema_f_ref = EMAIndicator("ef", {"period": 3})
        ema_s_ref = EMAIndicator("es", {"period": 6})
        rsi_ref = RSIIndicator("r", {"period": 4})
        for i in range(20, len(bars)):
            htf_bars: list[Candle] = []
            for b in bars[: i + 1]:
                bts = (b.open_time_ms // (4 * HOUR_MS)) * (4 * HOUR_MS)
                if not htf_bars or htf_bars[-1].open_time_ms != bts:
                    htf_bars.append(
                        Candle(
                            provider=b.provider,
                            symbol=b.symbol,
                            timeframe="4h",
                            open_time_ms=bts,
                            close_time_ms=bts + 4 * HOUR_MS - 1,
                            open=b.open,
                            high=b.high,
                            low=b.low,
                            close=b.close,
                            volume=b.volume,
                            is_closed=True,
                        )
                    )
                else:
                    prev = htf_bars[-1]
                    htf_bars[-1] = Candle(
                        provider=prev.provider,
                        symbol=prev.symbol,
                        timeframe="4h",
                        open_time_ms=prev.open_time_ms,
                        close_time_ms=prev.close_time_ms,
                        open=prev.open,
                        high=max(prev.high, b.high),
                        low=min(prev.low, b.low),
                        close=b.close,
                        volume=prev.volume + b.volume,
                        is_closed=True,
                    )
            ef_val = ema_f_ref.compute(htf_bars)["value"][-1]
            es_val = ema_s_ref.compute(htf_bars)["value"][-1]
            r_val = rsi_ref.compute(htf_bars)["value"][-1]
            assert ef_val is not None and es_val is not None and r_val is not None
            self.assertAlmostEqual(out["trend"][i], ef_val - es_val, places=10)  # type: ignore[arg-type]
            self.assertAlmostEqual(out["close_vs_slow"][i], bars[i].close - es_val, places=10)  # type: ignore[arg-type]
            self.assertAlmostEqual(out["rsi"][i], r_val, places=10)  # type: ignore[arg-type]

        # Empty input returns empty series
        self.assertEqual(htf.compute([]), {"trend": [], "close_vs_slow": [], "rsi": []})

        # Invalid params rejected
        for bad_params in (
            {"base_timeframe": "4h", "htf_timeframe": "1h"},
            {"base_timeframe": "4h", "htf_timeframe": "4h"},
            {"base_timeframe": "3h", "htf_timeframe": "4h"},
            {"base_timeframe": "1h", "htf_timeframe": "4h", "ema_fast": 12, "ema_slow": 12},
        ):
            with self.assertRaises(IndicatorError):
                HTFContextIndicator("bad_htf", bad_params)

    def test_htf_context_daily_to_weekly_monday_alignment(self) -> None:
        htf = HTFContextIndicator(
            "htf_1w",
            {
                "base_timeframe": "1d",
                "htf_timeframe": "1w",
                "ema_fast": 2,
                "ema_slow": 4,
                "rsi_period": 3,
            },
        )
        # ratio=7, need_htf=max(4, 4)=4 -> warmup = (4 - 1) * 7 = 21
        self.assertEqual(htf.warmup_period(), 21)

        # Start on Thursday 2025-01-02 UTC so Monday 2025-01-06 UTC (index 4) starts the 2nd week
        # (proves Monday 00:00 UTC alignment rather than Unix-epoch Thursday alignment)
        start_ms = parse_iso_to_ms("2025-01-02T00:00:00Z")
        monday_anchor = parse_iso_to_ms("2024-12-30T00:00:00Z")
        tf_1w = Timeframe.parse("1w")
        closes = [
            100.0 + 2.0 * math.sin(i * 0.45) + 0.4 * i + (1.1 if i % 4 == 0 else -0.5)
            for i in range(36)
        ]
        bars: list[Candle] = []
        for i, c in enumerate(closes):
            o = closes[i - 1] if i > 0 else c
            bars.append(
                Candle(
                    provider="synthetic",
                    symbol="BTCUSDT",
                    timeframe="1d",
                    open_time_ms=start_ms + i * DAY_MS,
                    close_time_ms=start_ms + (i + 1) * DAY_MS - 1,
                    open=o,
                    high=max(o, c) + 0.8,
                    low=min(o, c) - 0.8,
                    close=c,
                    volume=100.0,
                    is_closed=True,
                )
            )

        out = htf.compute(bars)
        for col in ("trend", "close_vs_slow", "rsi"):
            self.assertTrue(all(v is None for v in out[col][:21]))

        ema_f_ref = EMAIndicator("ef", {"period": 2})
        ema_s_ref = EMAIndicator("es", {"period": 4})
        rsi_ref = RSIIndicator("r", {"period": 3})
        for i in range(21, len(bars)):
            weekly_bars: list[Candle] = []
            for b in bars[: i + 1]:
                bts = tf_1w.align_floor(b.open_time_ms)
                self.assertEqual((bts - monday_anchor) % (7 * DAY_MS), 0)
                if not weekly_bars or weekly_bars[-1].open_time_ms != bts:
                    weekly_bars.append(
                        Candle(
                            provider=b.provider,
                            symbol=b.symbol,
                            timeframe="1w",
                            open_time_ms=bts,
                            close_time_ms=bts + 7 * DAY_MS - 1,
                            open=b.open,
                            high=b.high,
                            low=b.low,
                            close=b.close,
                            volume=b.volume,
                            is_closed=True,
                        )
                    )
                else:
                    prev = weekly_bars[-1]
                    weekly_bars[-1] = Candle(
                        provider=prev.provider,
                        symbol=prev.symbol,
                        timeframe="1w",
                        open_time_ms=prev.open_time_ms,
                        close_time_ms=prev.close_time_ms,
                        open=prev.open,
                        high=max(prev.high, b.high),
                        low=min(prev.low, b.low),
                        close=b.close,
                        volume=prev.volume + b.volume,
                        is_closed=True,
                    )
            ef_val = ema_f_ref.compute(weekly_bars)["value"][-1]
            es_val = ema_s_ref.compute(weekly_bars)["value"][-1]
            r_val = rsi_ref.compute(weekly_bars)["value"][-1]
            assert ef_val is not None and es_val is not None and r_val is not None
            self.assertAlmostEqual(out["trend"][i], ef_val - es_val, places=10)  # type: ignore[arg-type]
            self.assertAlmostEqual(out["close_vs_slow"][i], bars[i].close - es_val, places=10)  # type: ignore[arg-type]
            self.assertAlmostEqual(out["rsi"][i], r_val, places=10)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
