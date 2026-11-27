"""Unit and end-to-end loopback tests for candle reconstruction, TradeAnalyzer, PredictionEngine, and AlertEvaluator."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tradingbot.analysis import (
    AlertEvaluator,
    PredictionEngine,
    TradeAnalyzer,
    reconstruct_predicted_candles,
    snapshot_to_json,
)
from tradingbot.config import AnalysisConfig, TripleBarrierConfig, parse_asset_dict
from tradingbot.features import WindowContext
from tradingbot.models import (
    Candle,
    PredictedCandle,
    PredictionSnapshot,
    SignalPrediction,
    TradeAnalysis,
    parse_iso_to_ms,
)
from tradingbot.providers import SyntheticProvider, create_provider
from test_nn_client import _make_mock_nn_server

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"
HOUR_MS = 3_600_000


def _anchor_candle(close: float = 100.0, open_time_ms: int = 0) -> Candle:
    return Candle(
        provider="binance_spot",
        symbol="BTCUSDT",
        timeframe="1h",
        open_time_ms=open_time_ms,
        close_time_ms=open_time_ms + HOUR_MS - 1,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=50.0,
        is_closed=True,
    )


class TestReconstructionAndTradeAnalyzer(unittest.TestCase):
    def test_reconstruct_predicted_candles_geometry_and_decay(self) -> None:
        anchor = _anchor_candle(100.0, open_time_ms=10 * HOUR_MS)
        ctx = WindowContext(
            anchor_index=79,
            anchor_candle=anchor,
            anchor_close=100.0,
            scale=2.0,
            vector=[],
        )
        # Deliberately pass inverted high/low on step 1 to verify OHLC clamping
        raw_out = [
            0.0, -0.5, 1.5, 1.0,   # step 1: o=100, raw_h=99, raw_l=103, c=102 -> h=103, l=99
            1.0, 2.0, 0.5, 1.5,    # step 2: o=102, h=104, l=101, c=103
        ]
        preds = reconstruct_predicted_candles(raw_out, ctx, "1h")
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0].step, 1)
        self.assertEqual(preds[0].open_time_ms, 11 * HOUR_MS)
        self.assertAlmostEqual(preds[0].open, 100.0)
        self.assertAlmostEqual(preds[0].high, 103.0)
        self.assertAlmostEqual(preds[0].low, 99.0)
        self.assertAlmostEqual(preds[0].close, 102.0)
        self.assertGreater(preds[0].confidence, preds[1].confidence)

    def test_trade_analyzer_dual_candle_only_and_signal_only_modes(self) -> None:
        analyzer = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.60,
                min_risk_reward=1.5,
                min_expected_move_pct=0.01,
            ),
            TripleBarrierConfig(take_profit_pct=0.02, stop_loss_pct=0.01),
        )
        bullish_candles = (
            PredictedCandle(1, HOUR_MS, 100.0, 101.5, 99.5, 101.0, 0.9),
            PredictedCandle(2, 2 * HOUR_MS, 101.0, 103.0, 100.5, 102.5, 0.8),
        )
        long_sig = SignalPrediction(hold_prob=0.15, long_prob=0.75, short_prob=0.10)
        short_sig = SignalPrediction(hold_prob=0.15, long_prob=0.10, short_prob=0.75)

        # Dual cross-confirmed LONG (tp=103, sl=99.5 -> reward=3, risk=0.5 -> RR=6.0)
        res_dual = analyzer.analyze(100.0, predicted_candles=bullish_candles, signal=long_sig)
        self.assertEqual(res_dual.action, "LONG")
        self.assertAlmostEqual(res_dual.confidence, 0.75)
        self.assertGreaterEqual(res_dual.risk_reward or 0.0, 5.0)
        self.assertIn("Cross-confirmed LONG", res_dual.reason)

        # Conflicting signal (SHORT) vs bullish candles -> downgraded to HOLD
        res_conflict = analyzer.analyze(100.0, predicted_candles=bullish_candles, signal=short_sig)
        self.assertEqual(res_conflict.action, "HOLD")
        self.assertIn("Conflicting models", res_conflict.reason)

        # Candle-only bullish -> LONG with heuristic confidence
        res_candle_only = analyzer.analyze(100.0, predicted_candles=bullish_candles, signal=None)
        self.assertEqual(res_candle_only.action, "LONG")
        self.assertGreater(res_candle_only.confidence, 0.7)

        # Signal-only LONG -> uses Triple Barrier TP/SL (RR = 0.02 / 0.01 = 2.0 >= 1.5)
        res_sig_only = analyzer.analyze(100.0, predicted_candles=(), signal=long_sig)
        self.assertEqual(res_sig_only.action, "LONG")
        self.assertAlmostEqual(res_sig_only.estimated_tp or 0.0, 102.0)
        self.assertAlmostEqual(res_sig_only.estimated_sl or 0.0, 99.0)
        self.assertAlmostEqual(res_sig_only.risk_reward or 0.0, 2.0)

        # No models -> neutral HOLD
        res_none = analyzer.analyze(100.0)
        self.assertEqual(res_none.action, "HOLD")

    def test_trade_analyzer_short_rr_and_dual_min_move_rejections(self) -> None:
        analyzer = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.60,
                min_risk_reward=1.5,
                min_expected_move_pct=0.01,
            ),
            TripleBarrierConfig(take_profit_pct=0.02, stop_loss_pct=0.01),
        )
        bearish_candles = (
            PredictedCandle(1, HOUR_MS, 100.0, 100.5, 98.5, 99.0, 0.9),
            PredictedCandle(2, 2 * HOUR_MS, 99.0, 99.5, 97.0, 97.5, 0.8),
        )
        short_sig = SignalPrediction(hold_prob=0.10, long_prob=0.10, short_prob=0.80)

        # Dual cross-confirmed SHORT
        res_dual_short = analyzer.analyze(100.0, predicted_candles=bearish_candles, signal=short_sig)
        self.assertEqual(res_dual_short.action, "SHORT")
        self.assertAlmostEqual(res_dual_short.estimated_tp or 0.0, 97.0)
        self.assertAlmostEqual(res_dual_short.estimated_sl or 0.0, 100.5)
        self.assertGreaterEqual(res_dual_short.risk_reward or 0.0, 5.0)

        # Candle-only SHORT
        res_c_short = analyzer.analyze(100.0, predicted_candles=bearish_candles, signal=None)
        self.assertEqual(res_c_short.action, "SHORT")

        # Signal-only SHORT
        res_s_short = analyzer.analyze(100.0, predicted_candles=(), signal=short_sig)
        self.assertEqual(res_s_short.action, "SHORT")
        self.assertAlmostEqual(res_s_short.estimated_tp or 0.0, 98.0)
        self.assertAlmostEqual(res_s_short.estimated_sl or 0.0, 101.0)

        # ATR-multiple Triple Barrier in Signal-only mode (ATR=2.0, TP=1.5x=+3.0, SL=1.0x=-2.0)
        atr_analyzer = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.60,
                min_risk_reward=1.5,
                min_expected_move_pct=0.01,
            ),
            TripleBarrierConfig(mode="atr_multiple", take_profit_atr=1.5, stop_loss_atr=1.0),
        )
        res_atr = atr_analyzer.analyze(
            100.0,
            predicted_candles=(),
            signal=SignalPrediction(hold_prob=0.10, long_prob=0.80, short_prob=0.10),
            anchor_atr=2.0,
        )
        self.assertEqual(res_atr.action, "LONG")
        self.assertAlmostEqual(res_atr.estimated_tp or 0.0, 103.0)
        self.assertAlmostEqual(res_atr.estimated_sl or 0.0, 98.0)
        self.assertAlmostEqual(res_atr.risk_reward or 0.0, 1.5)
        self.assertAlmostEqual(res_atr.expected_move_pct, 0.03)

        # Extreme ATR on SHORT (take_profit_atr * atr >= anchor_close) and sub-cent asset
        # must floor prices strictly > 0 instead of raising ValueError
        res_atr_short_extreme = atr_analyzer.analyze(
            100.0,
            predicted_candles=(),
            signal=SignalPrediction(hold_prob=0.10, long_prob=0.10, short_prob=0.80),
            anchor_atr=100.0,
        )
        self.assertGreater(res_atr_short_extreme.estimated_tp or 0.0, 0.0)
        self.assertGreater(res_atr_short_extreme.estimated_sl or 0.0, 100.0)

        res_subcent = atr_analyzer.analyze(
            0.0005,
            predicted_candles=(PredictedCandle(1, HOUR_MS, 0.0005, 0.0006, 0.0005, 0.00058, 0.9),),
            signal=SignalPrediction(hold_prob=0.10, long_prob=0.80, short_prob=0.10),
            anchor_atr=0.001,
        )
        self.assertGreater(res_subcent.estimated_sl or 0.0, 0.0)
        self.assertGreater(res_subcent.estimated_tp or 0.0, 0.0)

        # Dual-model rejection when expected_move_pct < min_expected_move_pct despite strong signal & good R:R
        tiny_move_candles = (
            PredictedCandle(1, HOUR_MS, 100.0, 100.6, 99.9, 100.4, 0.9),
        )
        long_sig = SignalPrediction(hold_prob=0.10, long_prob=0.80, short_prob=0.10)
        res_small = analyzer.analyze(100.0, predicted_candles=tiny_move_candles, signal=long_sig)
        self.assertEqual(res_small.action, "HOLD")
        self.assertIn("expected move", res_small.reason)

        # Poor R:R rejection across signal-only and candle-only
        poor_rr_analyzer = TradeAnalyzer(
            AnalysisConfig(min_signal_confidence=0.60, min_risk_reward=3.0, min_expected_move_pct=0.01),
            TripleBarrierConfig(take_profit_pct=0.02, stop_loss_pct=0.01),  # RR=2.0 < 3.0
        )
        res_poor_sig = poor_rr_analyzer.analyze(100.0, predicted_candles=(), signal=long_sig)
        self.assertEqual(res_poor_sig.action, "HOLD")
        self.assertIn("filtered by R:R", res_poor_sig.reason)

        poor_rr_candles = (PredictedCandle(1, HOUR_MS, 100.0, 102.0, 97.0, 101.5, 0.9),)  # RR = 2/3 = 0.67
        res_poor_c = analyzer.analyze(100.0, predicted_candles=poor_rr_candles, signal=None)
        self.assertEqual(res_poor_c.action, "HOLD")
        self.assertIn("filtered by R:R", res_poor_c.reason)

    def test_trade_analyzer_directional_edge_mode(self) -> None:
        edge_analyzer = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.55,
                min_risk_reward=1.5,
                min_expected_move_pct=0.005,
                signal_decision_mode="directional_edge",
                min_directional_prob=0.35,
                min_directional_ratio=1.5,
            ),
            TripleBarrierConfig(mode="atr_multiple", take_profit_atr=1.5, stop_loss_atr=1.0),
        )
        bullish_candles = (
            PredictedCandle(1, HOUR_MS, 100.0, 101.5, 99.6, 100.8, 0.9),
            PredictedCandle(2, 2 * HOUR_MS, 100.8, 102.5, 100.2, 101.2, 0.8),
        )
        bearish_candles = (
            PredictedCandle(1, HOUR_MS, 100.0, 100.4, 98.5, 99.2, 0.9),
            PredictedCandle(2, 2 * HOUR_MS, 99.2, 99.8, 97.5, 98.8, 0.8),
        )

        # Calibrated bullish edge where HOLD is still argmax (0.44 > 0.38 > 0.18, ratio=2.11x >= 1.5x)
        edge_long = SignalPrediction(hold_prob=0.44, long_prob=0.38, short_prob=0.18)
        res_sig_only = edge_analyzer.analyze(100.0, predicted_candles=(), signal=edge_long, anchor_atr=2.0)
        self.assertEqual(res_sig_only.action, "LONG")
        self.assertAlmostEqual(res_sig_only.confidence, 0.38)
        self.assertAlmostEqual(res_sig_only.risk_reward or 0.0, 1.5)
        self.assertIn("edge=38.0%", res_sig_only.reason)

        # Dual-model cross-confirmation accepts matching bullish forecast and rejects bearish conflict
        res_dual_ok = edge_analyzer.analyze(
            100.0, predicted_candles=bullish_candles, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_dual_ok.action, "LONG")
        self.assertAlmostEqual(res_dual_ok.confidence, 0.38)
        self.assertIn("Cross-confirmed LONG", res_dual_ok.reason)

        res_dual_conflict = edge_analyzer.analyze(
            100.0, predicted_candles=bearish_candles, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_dual_conflict.action, "HOLD")
        self.assertIn("Conflicting models", res_dual_conflict.reason)

        # Bearish directional edge with opp_prob == 0.0 -> infinite ratio, no ZeroDivisionError
        edge_short_zero = SignalPrediction(hold_prob=0.60, long_prob=0.0, short_prob=0.40)
        res_zero = edge_analyzer.analyze(
            100.0, predicted_candles=bearish_candles, signal=edge_short_zero, anchor_atr=2.0
        )
        self.assertEqual(res_zero.action, "SHORT")
        self.assertIn("(inf)", res_zero.reason)

        # Rejections: weak ratio (0.36 vs 0.30 = 1.20x < 1.5x), low prob (0.30 vs 0.10), exact tie
        for weak_sig in (
            SignalPrediction(hold_prob=0.34, long_prob=0.36, short_prob=0.30),
            SignalPrediction(hold_prob=0.60, long_prob=0.30, short_prob=0.10),
            SignalPrediction(hold_prob=0.20, long_prob=0.40, short_prob=0.40),
        ):
            res_hold = edge_analyzer.analyze(
                100.0, predicted_candles=bullish_candles, signal=weak_sig, anchor_atr=2.0
            )
            self.assertEqual(res_hold.action, "HOLD")
            self.assertAlmostEqual(res_hold.confidence, weak_sig.hold_prob)

        # Exact directional_edge boundaries (dir_prob == 0.35 and exact 1.50x ratio 0.36/0.24)
        exact_prob_sig = SignalPrediction(hold_prob=0.45, long_prob=0.35, short_prob=0.20)
        res_exact_prob = edge_analyzer.analyze(
            100.0, predicted_candles=bullish_candles, signal=exact_prob_sig, anchor_atr=2.0
        )
        self.assertEqual(res_exact_prob.action, "LONG")
        self.assertAlmostEqual(res_exact_prob.confidence, 0.35)

        exact_ratio_sig = SignalPrediction(hold_prob=0.40, long_prob=0.36, short_prob=0.24)
        res_exact_ratio = edge_analyzer.analyze(
            100.0, predicted_candles=bullish_candles, signal=exact_ratio_sig, anchor_atr=2.0
        )
        self.assertEqual(res_exact_ratio.action, "LONG")
        self.assertAlmostEqual(res_exact_ratio.confidence, 0.36)

        # When directional_edge qualifies but downstream filters reject, confidence remains dir_prob
        # (a) Signal-only filtered by min_expected_move_pct in low-volatility ATR regime
        res_low_vol_sig = edge_analyzer.analyze(
            100.0, predicted_candles=(), signal=edge_long, anchor_atr=0.1
        )
        self.assertEqual(res_low_vol_sig.action, "HOLD")
        self.assertIn("filtered by expected move", res_low_vol_sig.reason)
        self.assertAlmostEqual(res_low_vol_sig.confidence, 0.38)

        # (b) Dual-model filtered by min_expected_move_pct / min_risk_reward preserves dir_prob
        flat_candles = (PredictedCandle(1, HOUR_MS, 100.0, 100.2, 99.9, 100.1, 0.9),)
        res_dual_flat = edge_analyzer.analyze(
            100.0, predicted_candles=flat_candles, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_dual_flat.action, "HOLD")
        self.assertIn("filtered by expected move", res_dual_flat.reason)
        self.assertAlmostEqual(res_dual_flat.confidence, 0.38)

        poor_rr_candles = (PredictedCandle(1, HOUR_MS, 100.0, 101.0, 97.0, 100.8, 0.9),)
        res_dual_rr = edge_analyzer.analyze(
            100.0, predicted_candles=poor_rr_candles, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_dual_rr.action, "HOLD")
        self.assertIn("filtered by R:R", res_dual_rr.reason)
        self.assertAlmostEqual(res_dual_rr.confidence, 0.38)

        # Compare against argmax mode: HOLD=0.80 > 0.55 must NOT emit contradictory "80.0% < 55.0%"
        argmax_analyzer = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.55,
                min_risk_reward=1.5,
                min_expected_move_pct=0.005,
                signal_decision_mode="argmax",
            ),
            TripleBarrierConfig(mode="atr_multiple", take_profit_atr=1.5, stop_loss_atr=1.0),
        )
        self.assertEqual(
            argmax_analyzer.analyze(100.0, predicted_candles=(), signal=edge_long, anchor_atr=2.0).action,
            "HOLD",
        )
        strong_hold = SignalPrediction(hold_prob=0.80, long_prob=0.12, short_prob=0.08)
        res_argmax_hold = argmax_analyzer.analyze(
            100.0, predicted_candles=bullish_candles, signal=strong_hold, anchor_atr=2.0
        )
        self.assertEqual(res_argmax_hold.action, "HOLD")
        self.assertIn("argmax=HOLD 80.0%", res_argmax_hold.reason)
        self.assertNotIn("< 55.0%", res_argmax_hold.reason)

        # Dual-model with symmetric regularized candles (candle R:R ~1.08 < 1.5):
        # tp_sl_source="candles" rejects by R:R, whereas tp_sl_source="triple_barrier" uses ATR barriers (R:R=1.5)
        symmetric_bullish = (
            PredictedCandle(1, HOUR_MS, 100.0, 101.2, 98.9, 100.4, 0.9),
            PredictedCandle(2, 2 * HOUR_MS, 100.4, 101.3, 98.8, 100.6, 0.8),
        )
        res_sym_candles = edge_analyzer.analyze(
            100.0, predicted_candles=symmetric_bullish, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_sym_candles.action, "HOLD")
        self.assertIn("filtered by R:R", res_sym_candles.reason)

        tb_dual_analyzer = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.55,
                min_risk_reward=1.5,
                min_expected_move_pct=0.005,
                signal_decision_mode="directional_edge",
                min_directional_prob=0.35,
                min_directional_ratio=1.5,
                tp_sl_source="triple_barrier",
            ),
            TripleBarrierConfig(mode="atr_multiple", take_profit_atr=1.5, stop_loss_atr=1.0),
        )
        res_sym_tb = tb_dual_analyzer.analyze(
            100.0, predicted_candles=symmetric_bullish, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_sym_tb.action, "LONG")
        self.assertAlmostEqual(res_sym_tb.estimated_tp or 0.0, 103.0)
        self.assertAlmostEqual(res_sym_tb.estimated_sl or 0.0, 98.0)
        self.assertAlmostEqual(res_sym_tb.risk_reward or 0.0, 1.5)
        self.assertIn("TB R:R=1.50", res_sym_tb.reason)

        # Signal-driven mode (require_candle_confirmation=False): opposing bearish candles
        # remain visible on chart without vetoing a qualifying LONG signal
        sig_driven_analyzer = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.55,
                min_risk_reward=1.5,
                min_expected_move_pct=0.005,
                signal_decision_mode="directional_edge",
                min_directional_prob=0.35,
                min_directional_ratio=1.5,
                tp_sl_source="triple_barrier",
                require_candle_confirmation=False,
            ),
            TripleBarrierConfig(mode="atr_multiple", take_profit_atr=1.5, stop_loss_atr=1.0),
        )
        res_sig_driven = sig_driven_analyzer.analyze(
            100.0, predicted_candles=bearish_candles, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_sig_driven.action, "LONG")
        self.assertAlmostEqual(res_sig_driven.confidence, 0.38)
        self.assertAlmostEqual(res_sig_driven.estimated_tp or 0.0, 103.0)
        self.assertAlmostEqual(res_sig_driven.estimated_sl or 0.0, 98.0)
        self.assertAlmostEqual(res_sig_driven.expected_move_pct, 0.03)
        self.assertIn("Signal-driven LONG", res_sig_driven.reason)
        self.assertIn("candle_forecast=-1.20%", res_sig_driven.reason)

        # Signal-driven mode must still enforce min_expected_move_pct (low-vol ATR regime) and min_risk_reward
        res_sig_driven_low_vol = sig_driven_analyzer.analyze(
            100.0, predicted_candles=bearish_candles, signal=edge_long, anchor_atr=0.1
        )
        self.assertEqual(res_sig_driven_low_vol.action, "HOLD")
        self.assertIn("filtered by expected move", res_sig_driven_low_vol.reason)
        self.assertAlmostEqual(res_sig_driven_low_vol.confidence, 0.38)
        self.assertAlmostEqual(res_sig_driven_low_vol.expected_move_pct, 0.0015)

        sig_driven_high_rr = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.55,
                min_risk_reward=2.0,
                min_expected_move_pct=0.005,
                signal_decision_mode="directional_edge",
                min_directional_prob=0.35,
                min_directional_ratio=1.5,
                tp_sl_source="triple_barrier",
                require_candle_confirmation=False,
            ),
            TripleBarrierConfig(mode="atr_multiple", take_profit_atr=1.5, stop_loss_atr=1.0),
        )
        res_sig_driven_rr = sig_driven_high_rr.analyze(
            100.0, predicted_candles=bullish_candles, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_sig_driven_rr.action, "HOLD")
        self.assertIn("filtered by R:R", res_sig_driven_rr.reason)
        self.assertAlmostEqual(res_sig_driven_rr.confidence, 0.38)

        sig_driven_candle_tp = TradeAnalyzer(
            AnalysisConfig(
                min_signal_confidence=0.55,
                min_risk_reward=1.5,
                min_expected_move_pct=0.005,
                signal_decision_mode="directional_edge",
                min_directional_prob=0.35,
                min_directional_ratio=1.5,
                tp_sl_source="candles",
                require_candle_confirmation=False,
            ),
            TripleBarrierConfig(mode="atr_multiple", take_profit_atr=1.5, stop_loss_atr=1.0),
        )
        res_sig_driven_candle_rr = sig_driven_candle_tp.analyze(
            100.0, predicted_candles=poor_rr_candles, signal=edge_long, anchor_atr=2.0
        )
        self.assertEqual(res_sig_driven_candle_rr.action, "HOLD")
        self.assertIn("filtered by R:R", res_sig_driven_candle_rr.reason)
        self.assertAlmostEqual(res_sig_driven_candle_rr.confidence, 0.38)

        # Candle-predictor-only mode (signal=None) with tp_sl_source="triple_barrier" uses TB barriers
        res_candle_only_tb = tb_dual_analyzer.analyze(
            100.0, predicted_candles=symmetric_bullish, signal=None, anchor_atr=2.0
        )
        self.assertEqual(res_candle_only_tb.action, "LONG")
        self.assertAlmostEqual(res_candle_only_tb.estimated_tp or 0.0, 103.0)
        self.assertAlmostEqual(res_candle_only_tb.estimated_sl or 0.0, 98.0)
        self.assertAlmostEqual(res_candle_only_tb.risk_reward or 0.0, 1.5)
        self.assertIn("Candle-forecast only LONG", res_candle_only_tb.reason)

        # Invalid signal_decision_mode / tp_sl_source raise ValueError defensively
        with self.assertRaises(ValueError):
            TradeAnalyzer(
                AnalysisConfig(0.55, 1.5, 0.005, signal_decision_mode="invalid"),
                TripleBarrierConfig(take_profit_pct=0.02, stop_loss_pct=0.01),
            )
        with self.assertRaises(ValueError):
            TradeAnalyzer(
                AnalysisConfig(0.55, 1.5, 0.005, tp_sl_source="invalid"),
                TripleBarrierConfig(take_profit_pct=0.02, stop_loss_pct=0.01),
            )


class TestAlertEvaluator(unittest.TestCase):
    def _make_snapshot(
        self,
        bar_idx: int,
        action: str,
        confidence: float = 0.80,
        move_pct: float = 0.02,
        rr: float | None = 2.0,
    ) -> PredictionSnapshot:
        return PredictionSnapshot(
            asset_id="btc_1h",
            provider="binance_spot",
            symbol="BTCUSDT",
            timeframe="1h",
            anchor_open_time_ms=bar_idx * HOUR_MS,
            created_at_ms=bar_idx * HOUR_MS + 100,
            analysis=TradeAnalysis(
                action=action,
                expected_move_pct=move_pct if action == "LONG" else (-move_pct if action == "SHORT" else 0.0),
                estimated_tp=102.0 if action != "HOLD" else None,
                estimated_sl=99.0 if action != "HOLD" else None,
                risk_reward=rr if action != "HOLD" else None,
                confidence=confidence,
                reason="test",
            ),
        )

    def test_consecutive_candles_cooldown_and_idempotency(self) -> None:
        raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        raw["alerts"] = {
            "enabled": True,
            "actions": ["LONG", "SHORT"],
            "min_confidence": 0.70,
            "min_risk_reward": 1.5,
            "min_expected_move_pct": 0.005,
            "consecutive_candles": 2,
            "cooldown_candles": 3,
        }
        asset = parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent)
        evaluator = AlertEvaluator(asset, clock_ms=lambda: 999_999)

        # Bar 1: LONG #1 (streak 1/2 -> no alert)
        self.assertIsNone(evaluator.evaluate(self._make_snapshot(1, "LONG")))
        # Duplicate poll of Bar 1 -> ignored cleanly (streak stays 1/2)
        self.assertIsNone(evaluator.evaluate(self._make_snapshot(1, "LONG")))
        self.assertEqual(evaluator.status_dict()["streak_count"], 1)

        # Bar 2: LONG #2 (streak 2/2 -> FIRES alert!)
        ev1 = evaluator.evaluate(self._make_snapshot(2, "LONG"))
        self.assertIsNotNone(ev1)
        assert ev1 is not None
        self.assertEqual(ev1.action, "LONG")
        self.assertEqual(ev1.consecutive_hits, 2)
        self.assertEqual(ev1.rule_name, "consecutive_2_bars")

        # Bars 3, 4: still in cooldown (candles_since_alert = 1, 2 < 3 -> suppressed)
        self.assertIsNone(evaluator.evaluate(self._make_snapshot(3, "LONG")))
        self.assertIsNone(evaluator.evaluate(self._make_snapshot(4, "LONG")))

        # Bar 5: cooldown satisfied (3 >= 3), streak continues -> FIRES second alert
        ev2 = evaluator.evaluate(self._make_snapshot(5, "LONG"))
        self.assertIsNotNone(ev2)
        assert ev2 is not None
        self.assertEqual(ev2.consecutive_hits, 5)

        # Bar 6: HOLD resets streak to 0
        self.assertIsNone(evaluator.evaluate(self._make_snapshot(6, "HOLD")))
        self.assertEqual(evaluator.status_dict()["streak_count"], 0)

        # Bar 7..8: flip to SHORT (streak 1 at bar 7, streak 2 at bar 8 -> cooldown 3/3 at bar 8 -> FIRES SHORT!)
        self.assertIsNone(evaluator.evaluate(self._make_snapshot(7, "SHORT")))
        ev3 = evaluator.evaluate(self._make_snapshot(8, "SHORT"))
        self.assertIsNotNone(ev3)
        assert ev3 is not None
        self.assertEqual(ev3.action, "SHORT")
        self.assertEqual(ev3.consecutive_hits, 2)

    def test_alert_evaluator_edge_cases_and_gap_handling(self) -> None:
        raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        raw["alerts"] = {
            "enabled": False,
            "actions": ["LONG", "SHORT"],
            "min_confidence": 0.70,
            "min_risk_reward": 1.5,
            "min_expected_move_pct": 0.01,
            "consecutive_candles": 1,
            "cooldown_candles": 0,
        }
        disabled_eval = AlertEvaluator(parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent))
        self.assertIsNone(disabled_eval.evaluate(self._make_snapshot(1, "LONG")))

        # Enable with consecutive=1 and cooldown=0 -> fires every bar with rule single_bar_threshold
        raw["alerts"]["enabled"] = True
        ev_single = AlertEvaluator(parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent))
        # Threshold rejections: low confidence, low expected move, low R:R, and risk_reward=None
        self.assertIsNone(ev_single.evaluate(self._make_snapshot(1, "LONG", confidence=0.50)))
        self.assertIsNone(ev_single.evaluate(self._make_snapshot(2, "LONG", move_pct=0.002)))
        self.assertIsNone(ev_single.evaluate(self._make_snapshot(3, "LONG", rr=1.0)))
        self.assertIsNone(ev_single.evaluate(self._make_snapshot(4, "LONG", rr=None)))

        a1 = ev_single.evaluate(self._make_snapshot(5, "LONG"))
        self.assertIsNotNone(a1)
        assert a1 is not None
        self.assertEqual(a1.rule_name, "single_bar_threshold")
        # Direct LONG -> SHORT flip with cooldown=0 fires immediately on next bar
        a2 = ev_single.evaluate(self._make_snapshot(6, "SHORT"))
        self.assertIsNotNone(a2)
        assert a2 is not None
        self.assertEqual(a2.action, "SHORT")

        # seed_last_anchor blocks older/equal bars, reset clears all state
        ev_single.seed_last_anchor(10 * HOUR_MS)
        self.assertIsNone(ev_single.evaluate(self._make_snapshot(10, "LONG")))
        ev_single.reset()
        self.assertEqual(ev_single.status_dict()["total_emitted"], 0)
        self.assertIsNotNone(ev_single.evaluate(self._make_snapshot(1, "LONG")))

        # 24/7 market: a missing bar gap breaks a consecutive_candles=2 streak
        raw["alerts"]["consecutive_candles"] = 2
        ev_247 = AlertEvaluator(parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent))
        self.assertIsNone(ev_247.evaluate(self._make_snapshot(1, "LONG")))
        # Jump from bar 1 to bar 3 (missing bar 2) -> streak resets to 1, does NOT fire
        self.assertIsNone(ev_247.evaluate(self._make_snapshot(3, "LONG")))
        self.assertEqual(ev_247.status_dict()["streak_count"], 1)

        # Sessions market: overnight/weekend gap preserves consecutive streak
        raw["market"]["market_schedule"] = "sessions"
        ev_sess = AlertEvaluator(parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent))
        self.assertIsNone(ev_sess.evaluate(self._make_snapshot(1, "LONG")))
        fired = ev_sess.evaluate(self._make_snapshot(10, "LONG"))
        self.assertIsNotNone(fired)
        assert fired is not None
        self.assertEqual(fired.consecutive_hits, 2)


class TestEndToEndPredictionEngine(unittest.TestCase):
    def test_prediction_engine_with_dual_mock_nn_servers_and_alerts(self) -> None:
        # 6 bullish relative OHLC candles (rising +0.25 .. +1.5 ATR above anchor_close)
        bullish_rel: list[float] = []
        for step in range(1, 7):
            base = 0.25 * step
            bullish_rel.extend([base - 0.05, base + 0.30, base - 0.15, base + 0.10])

        srv_c, th_c, port_c = _make_mock_nn_server(384, bullish_rel, latency_us=12)
        srv_s, th_s, port_s = _make_mock_nn_server(384, [0.10, 0.80, 0.10], latency_us=8)
        srv_e, th_e, port_e = _make_mock_nn_server(384, [0.33, 0.67], latency_us=4)

        try:
            raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            raw["provider"] = {"preset_path": "../providers/synthetic.json"}
            raw["neural_networks"]["verify_schema_on_start"] = True
            raw["neural_networks"]["candle_predictor"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_c,
                "schema_path": "",
            }
            raw["neural_networks"]["signal_classifier"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_s,
                "schema_path": "",
            }
            raw["neural_networks"]["extra_models"] = {
                "regime_head": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": port_e,
                    "schema_path": "",
                }
            }
            raw["alerts"]["consecutive_candles"] = 2
            asset = parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent)

            start_ms = parse_iso_to_ms("2025-01-01T00:00:00Z")
            end_ms = start_ms + 240 * HOUR_MS
            provider = create_provider(asset.provider, clock_ms=lambda: end_ms + HOUR_MS)
            assert isinstance(provider, SyntheticProvider)
            candles = provider.fetch_candles("BTCUSDT", "1h", start_ms=start_ms, end_ms=end_ms, limit=240)

            engine = PredictionEngine(asset, clock_ms=lambda: end_ms)
            infos = engine.verify_models()
            self.assertEqual(set(infos.keys()), {"candle_predictor", "signal_classifier", "regime_head"})

            evaluator = AlertEvaluator(asset, clock_ms=lambda: end_ms)
            snap1 = engine.predict(candles[:-1])
            self.assertEqual(len(snap1.predicted_candles), 6)
            self.assertEqual(snap1.extra_outputs.get("regime_head"), (0.33, 0.67))
            assert snap1.analysis is not None
            self.assertEqual(snap1.analysis.action, "LONG")
            self.assertIsNone(evaluator.evaluate(snap1))

            snap2 = engine.predict(candles)
            alert = evaluator.evaluate(snap2)
            self.assertIsNotNone(alert)
            assert alert is not None
            self.assertEqual(alert.action, "LONG")
            self.assertEqual(alert.consecutive_hits, 2)

            # Verify JSON serialization works cleanly including extra_outputs
            encoded = json.loads(snapshot_to_json(snap2))
            self.assertEqual(encoded["asset_id"], "btc_1h")
            self.assertEqual(len(encoded["predicted_candles"]), 6)
            self.assertEqual(encoded["analysis"]["action"], "LONG")
            self.assertEqual(encoded["extra_outputs"]["regime_head"], [0.33, 0.67])
        finally:
            for srv, th in ((srv_c, th_c), (srv_s, th_s), (srv_e, th_e)):
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
