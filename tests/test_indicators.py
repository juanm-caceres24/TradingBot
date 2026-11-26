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
    INDICATOR_REGISTRY,
    IndicatorEngine,
    IndicatorError,
    RSIIndicator,
    SMAIndicator,
    SqueezeMomentumIndicator,
    register_indicator,
)
from tradingbot.models import Candle

HOUR_MS = 3_600_000


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


if __name__ == "__main__":
    unittest.main()
