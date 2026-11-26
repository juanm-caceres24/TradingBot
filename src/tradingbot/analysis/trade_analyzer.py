"""Actionable trade analysis combining Candle Predictor and Signal Classifier outputs."""

from __future__ import annotations

from collections.abc import Sequence

from tradingbot.config import AnalysisConfig, TripleBarrierConfig
from tradingbot.models import PredictedCandle, SignalPrediction, TradeAnalysis


class TradeAnalyzer:
    """Evaluates predicted candles and/or multiclass probabilities into a TradeAnalysis."""

    def __init__(
        self,
        analysis_config: AnalysisConfig,
        triple_barrier: TripleBarrierConfig,
    ) -> None:
        self._cfg = analysis_config
        self._tb = triple_barrier

    def _compute_candle_levels(
        self,
        direction: str,
        anchor_close: float,
        candles: Sequence[PredictedCandle],
    ) -> tuple[float | None, float | None, float | None]:
        """Compute (estimated_tp, estimated_sl, risk_reward) from predicted OHLC path."""
        if not candles or direction not in ("LONG", "SHORT"):
            return None, None, None

        max_high = max(c.high for c in candles)
        min_low = min(c.low for c in candles)

        if direction == "LONG":
            tp = max_high if max_high > anchor_close else anchor_close * (1.0 + self._tb.take_profit_pct)
            sl = min_low if min_low < anchor_close else anchor_close * (1.0 - self._tb.stop_loss_pct)
            reward = max(0.0, tp - anchor_close)
            risk = max(1e-9, anchor_close - sl)
            return round(tp, 8), round(sl, 8), round(reward / risk, 4)

        tp = min_low if min_low < anchor_close else anchor_close * (1.0 - self._tb.take_profit_pct)
        sl = max_high if max_high > anchor_close else anchor_close * (1.0 + self._tb.stop_loss_pct)
        reward = max(0.0, anchor_close - tp)
        risk = max(1e-9, sl - anchor_close)
        return round(tp, 8), round(sl, 8), round(reward / risk, 4)

    def _compute_triple_barrier_levels(
        self, direction: str, anchor_close: float
    ) -> tuple[float | None, float | None, float | None]:
        if direction == "LONG":
            tp = anchor_close * (1.0 + self._tb.take_profit_pct)
            sl = anchor_close * (1.0 - self._tb.stop_loss_pct)
            rr = self._tb.take_profit_pct / self._tb.stop_loss_pct
            return round(tp, 8), round(sl, 8), round(rr, 4)
        if direction == "SHORT":
            tp = anchor_close * (1.0 - self._tb.take_profit_pct)
            sl = anchor_close * (1.0 + self._tb.stop_loss_pct)
            rr = self._tb.take_profit_pct / self._tb.stop_loss_pct
            return round(tp, 8), round(sl, 8), round(rr, 4)
        return None, None, None

    @staticmethod
    def _directional_consistency(
        direction: str, anchor_close: float, candles: Sequence[PredictedCandle]
    ) -> float:
        if not candles:
            return 0.0
        if direction == "LONG":
            hits = sum(1 for c in candles if c.close >= anchor_close)
        elif direction == "SHORT":
            hits = sum(1 for c in candles if c.close <= anchor_close)
        else:
            return 0.0
        return hits / float(len(candles))

    def analyze(
        self,
        anchor_close: float,
        *,
        predicted_candles: Sequence[PredictedCandle] = (),
        signal: SignalPrediction | None = None,
    ) -> TradeAnalysis:
        """Synthesize a TradeAnalysis from whichever model outputs are available."""
        has_candles = len(predicted_candles) > 0
        has_signal = signal is not None

        if not has_candles and not has_signal:
            return TradeAnalysis(
                action="HOLD",
                expected_move_pct=0.0,
                estimated_tp=None,
                estimated_sl=None,
                risk_reward=None,
                confidence=0.0,
                reason="No neural network models enabled",
            )

        # Case 1: Signal Classifier only (no Candle Predictor)
        if has_signal and not has_candles:
            assert signal is not None
            best_act = signal.best_action
            best_prob = signal.best_probability
            if best_act == "HOLD" or best_prob < self._cfg.min_signal_confidence:
                return TradeAnalysis(
                    action="HOLD",
                    expected_move_pct=0.0,
                    estimated_tp=None,
                    estimated_sl=None,
                    risk_reward=None,
                    confidence=round(best_prob, 4),
                    reason=(
                        f"Signal-only HOLD (best={best_act} {best_prob:.1%}, "
                        f"min={self._cfg.min_signal_confidence:.1%})"
                    ),
                )
            tp, sl, rr = self._compute_triple_barrier_levels(best_act, anchor_close)
            signed_move = self._tb.take_profit_pct if best_act == "LONG" else -self._tb.take_profit_pct
            if rr is not None and rr < self._cfg.min_risk_reward:
                return TradeAnalysis(
                    action="HOLD",
                    expected_move_pct=round(signed_move, 6),
                    estimated_tp=tp,
                    estimated_sl=sl,
                    risk_reward=rr,
                    confidence=round(best_prob, 4),
                    reason=f"Signal-only {best_act} filtered by R:R ({rr:.2f} < {self._cfg.min_risk_reward:.2f})",
                )
            return TradeAnalysis(
                action=best_act,
                expected_move_pct=round(signed_move, 6),
                estimated_tp=tp,
                estimated_sl=sl,
                risk_reward=rr,
                confidence=round(best_prob, 4),
                reason=f"Signal-only {best_act}: prob={best_prob:.1%}, TB R:R={rr:.2f}",
            )

        # Compute candle forecast summary (shared by candle-only and dual-model modes)
        final_close = predicted_candles[-1].close
        expected_move_pct = (final_close - anchor_close) / anchor_close
        if expected_move_pct >= self._cfg.min_expected_move_pct:
            candle_dir = "LONG"
        elif expected_move_pct <= -self._cfg.min_expected_move_pct:
            candle_dir = "SHORT"
        else:
            candle_dir = "HOLD"

        # Case 2: Candle Predictor only (heuristic confidence)
        if not has_signal:
            if candle_dir == "HOLD":
                return TradeAnalysis(
                    action="HOLD",
                    expected_move_pct=round(expected_move_pct, 6),
                    estimated_tp=None,
                    estimated_sl=None,
                    risk_reward=None,
                    confidence=round(min(0.5, abs(expected_move_pct) / max(self._cfg.min_expected_move_pct, 1e-6) * 0.5), 4),
                    reason=(
                        f"Candle-forecast only HOLD: expected move {expected_move_pct:+.2%} "
                        f"below threshold {self._cfg.min_expected_move_pct:.2%}"
                    ),
                )
            tp, sl, rr = self._compute_candle_levels(candle_dir, anchor_close, predicted_candles)
            consistency = self._directional_consistency(candle_dir, anchor_close, predicted_candles)
            rr_factor = min(1.0, (rr or 0.0) / max(self._cfg.min_risk_reward, 1e-6))
            heuristic_conf = round(max(0.0, min(1.0, 0.6 * consistency + 0.4 * rr_factor)), 4)

            if rr is not None and rr < self._cfg.min_risk_reward:
                return TradeAnalysis(
                    action="HOLD",
                    expected_move_pct=round(expected_move_pct, 6),
                    estimated_tp=tp,
                    estimated_sl=sl,
                    risk_reward=rr,
                    confidence=heuristic_conf,
                    reason=(
                        f"Candle-forecast only {candle_dir} filtered by R:R "
                        f"({rr:.2f} < {self._cfg.min_risk_reward:.2f})"
                    ),
                )
            return TradeAnalysis(
                action=candle_dir,
                expected_move_pct=round(expected_move_pct, 6),
                estimated_tp=tp,
                estimated_sl=sl,
                risk_reward=rr,
                confidence=heuristic_conf,
                reason=(
                    f"Candle-forecast only {candle_dir}: move={expected_move_pct:+.2%}, "
                    f"consistency={consistency:.0%}, R:R={rr:.2f}"
                ),
            )

        # Case 3: Dual-Model Cross-Confirmation (Candle Predictor + Signal Classifier)
        assert signal is not None
        sig_act = signal.best_action
        sig_prob = signal.best_probability

        if sig_act == "HOLD" or sig_prob < self._cfg.min_signal_confidence:
            return TradeAnalysis(
                action="HOLD",
                expected_move_pct=round(expected_move_pct, 6),
                estimated_tp=None,
                estimated_sl=None,
                risk_reward=None,
                confidence=round(sig_prob, 4),
                reason=(
                    f"Signal below threshold ({sig_act} {sig_prob:.1%} < "
                    f"{self._cfg.min_signal_confidence:.1%}); forecast={expected_move_pct:+.2%}"
                ),
            )

        # Reject active directional contradiction between models
        if (sig_act == "LONG" and expected_move_pct < 0.0) or (
            sig_act == "SHORT" and expected_move_pct > 0.0
        ):
            return TradeAnalysis(
                action="HOLD",
                expected_move_pct=round(expected_move_pct, 6),
                estimated_tp=None,
                estimated_sl=None,
                risk_reward=None,
                confidence=round(sig_prob * 0.5, 4),
                reason=(
                    f"Conflicting models: signal={sig_act} ({sig_prob:.1%}) "
                    f"vs candle forecast={expected_move_pct:+.2%}"
                ),
            )

        tp, sl, rr = self._compute_candle_levels(sig_act, anchor_close, predicted_candles)
        if rr is not None and rr < self._cfg.min_risk_reward:
            return TradeAnalysis(
                action="HOLD",
                expected_move_pct=round(expected_move_pct, 6),
                estimated_tp=tp,
                estimated_sl=sl,
                risk_reward=rr,
                confidence=round(sig_prob, 4),
                reason=(
                    f"Cross-confirmed {sig_act} filtered by R:R "
                    f"({rr:.2f} < {self._cfg.min_risk_reward:.2f})"
                ),
            )

        return TradeAnalysis(
            action=sig_act,
            expected_move_pct=round(expected_move_pct, 6),
            estimated_tp=tp,
            estimated_sl=sl,
            risk_reward=rr,
            confidence=round(sig_prob, 4),
            reason=(
                f"Cross-confirmed {sig_act}: signal={sig_prob:.1%}, "
                f"forecast={expected_move_pct:+.2%}, R:R={rr:.2f}"
            ),
        )
