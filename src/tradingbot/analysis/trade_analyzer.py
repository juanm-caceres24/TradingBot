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
        if analysis_config.signal_decision_mode not in ("argmax", "directional_edge"):
            raise ValueError(
                f"Unsupported signal_decision_mode: {analysis_config.signal_decision_mode!r}"
            )
        self._cfg = analysis_config
        self._tb = triple_barrier

    def _select_signal_action(
        self, signal: SignalPrediction
    ) -> tuple[str, float, str, str]:
        """Evaluate a multiclass Softmax signal into (action, confidence, reject_detail, accept_detail).

        - ``argmax``: requires ``signal.best_action != "HOLD"`` and
          ``signal.best_probability >= min_signal_confidence``.
        - ``directional_edge``: compares ``long_prob`` vs ``short_prob`` directly so
          calibrated probabilities under a high HOLD prior are not paralyzed by argmax,
          requiring ``dir_prob >= min_directional_prob`` and
          ``dir_prob >= opp_prob * min_directional_ratio``.
        """
        if self._cfg.signal_decision_mode == "argmax":
            best_act = signal.best_action
            best_prob = signal.best_probability
            if best_act == "HOLD":
                return (
                    "HOLD",
                    best_prob,
                    f"argmax=HOLD {best_prob:.1%}",
                    "",
                )
            if best_prob < self._cfg.min_signal_confidence:
                return (
                    "HOLD",
                    best_prob,
                    f"{best_act} {best_prob:.1%} < {self._cfg.min_signal_confidence:.1%}",
                    "",
                )
            return best_act, best_prob, "", f"prob={best_prob:.1%}"

        # directional_edge mode
        if signal.long_prob > signal.short_prob:
            dir_act, dir_prob, opp_prob = "LONG", signal.long_prob, signal.short_prob
        elif signal.short_prob > signal.long_prob:
            dir_act, dir_prob, opp_prob = "SHORT", signal.short_prob, signal.long_prob
        else:
            return (
                "HOLD",
                signal.hold_prob,
                f"tied directional probs (LONG={signal.long_prob:.1%}, SHORT={signal.short_prob:.1%})",
                "",
            )

        ratio_str = f"{dir_prob / opp_prob:.2f}x" if opp_prob > 1e-12 else "inf"
        meets_prob = (dir_prob + 1e-12) >= self._cfg.min_directional_prob
        meets_ratio = (dir_prob + 1e-12) >= (opp_prob * self._cfg.min_directional_ratio)
        if not meets_prob or not meets_ratio:
            return (
                "HOLD",
                signal.hold_prob,
                (
                    f"edge {dir_act} {dir_prob:.1%} vs {opp_prob:.1%} ({ratio_str}; "
                    f"min_prob={self._cfg.min_directional_prob:.1%}, "
                    f"min_ratio={self._cfg.min_directional_ratio:.2f}x)"
                ),
                "",
            )
        return (
            dir_act,
            dir_prob,
            "",
            f"edge={dir_prob:.1%} vs {opp_prob:.1%} ({ratio_str})",
        )

    def _compute_candle_levels(
        self,
        direction: str,
        anchor_close: float,
        candles: Sequence[PredictedCandle],
        anchor_atr: float | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute (estimated_tp, estimated_sl, risk_reward) from predicted OHLC path."""
        if not candles or direction not in ("LONG", "SHORT"):
            return None, None, None

        max_high = max(c.high for c in candles)
        min_low = min(c.low for c in candles)

        def _fallback_dists() -> tuple[float, float]:
            return self._tb.barrier_distances(anchor_close, atr=anchor_atr)

        if direction == "LONG":
            raw_tp = max_high if max_high > anchor_close else anchor_close + _fallback_dists()[0]
            raw_sl = min_low if min_low < anchor_close else anchor_close - _fallback_dists()[1]
            tp = max(1e-8, round(raw_tp, 8))
            sl = max(1e-8, round(raw_sl, 8))
            reward = max(0.0, tp - anchor_close)
            risk = max(1e-9, anchor_close - sl)
            return tp, sl, round(reward / risk, 4)

        raw_tp = min_low if min_low < anchor_close else anchor_close - _fallback_dists()[0]
        raw_sl = max_high if max_high > anchor_close else anchor_close + _fallback_dists()[1]
        tp = max(1e-8, round(raw_tp, 8))
        sl = max(1e-8, round(raw_sl, 8))
        reward = max(0.0, anchor_close - tp)
        risk = max(1e-9, sl - anchor_close)
        return tp, sl, round(reward / risk, 4)

    def _compute_triple_barrier_levels(
        self,
        direction: str,
        anchor_close: float,
        anchor_atr: float | None = None,
    ) -> tuple[float | None, float | None, float | None, float]:
        if direction not in ("LONG", "SHORT"):
            return None, None, None, 0.0
        tp_dist, sl_dist = self._tb.barrier_distances(anchor_close, atr=anchor_atr)
        if direction == "LONG":
            tp = max(1e-8, round(anchor_close + tp_dist, 8))
            sl = max(1e-8, round(anchor_close - sl_dist, 8))
            reward = max(0.0, tp - anchor_close)
            risk = max(1e-9, anchor_close - sl)
            rr = reward / risk
            move_pct = reward / max(1e-9, anchor_close)
            return tp, sl, round(rr, 4), move_pct
        tp = max(1e-8, round(anchor_close - tp_dist, 8))
        sl = max(1e-8, round(anchor_close + sl_dist, 8))
        reward = max(0.0, anchor_close - tp)
        risk = max(1e-9, sl - anchor_close)
        rr = reward / risk
        move_pct = reward / max(1e-9, anchor_close)
        return tp, sl, round(rr, 4), -move_pct

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
        anchor_atr: float | None = None,
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
            sig_act, sig_prob, reject_detail, accept_detail = self._select_signal_action(signal)
            if sig_act == "HOLD":
                hold_reason = (
                    f"Signal-only HOLD (best={signal.best_action} {signal.best_probability:.1%}, "
                    f"min={self._cfg.min_signal_confidence:.1%})"
                    if self._cfg.signal_decision_mode == "argmax"
                    else f"Signal-only HOLD ({reject_detail})"
                )
                return TradeAnalysis(
                    action="HOLD",
                    expected_move_pct=0.0,
                    estimated_tp=None,
                    estimated_sl=None,
                    risk_reward=None,
                    confidence=round(sig_prob, 4),
                    reason=hold_reason,
                )
            tp, sl, rr, signed_move = self._compute_triple_barrier_levels(
                sig_act, anchor_close, anchor_atr=anchor_atr
            )
            if abs(signed_move) < self._cfg.min_expected_move_pct:
                return TradeAnalysis(
                    action="HOLD",
                    expected_move_pct=round(signed_move, 6),
                    estimated_tp=tp,
                    estimated_sl=sl,
                    risk_reward=rr,
                    confidence=round(sig_prob, 4),
                    reason=(
                        f"Signal-only {sig_act} filtered by expected move "
                        f"({signed_move:+.2%} < {self._cfg.min_expected_move_pct:.2%})"
                    ),
                )
            if rr is not None and rr < self._cfg.min_risk_reward:
                return TradeAnalysis(
                    action="HOLD",
                    expected_move_pct=round(signed_move, 6),
                    estimated_tp=tp,
                    estimated_sl=sl,
                    risk_reward=rr,
                    confidence=round(sig_prob, 4),
                    reason=f"Signal-only {sig_act} filtered by R:R ({rr:.2f} < {self._cfg.min_risk_reward:.2f})",
                )
            return TradeAnalysis(
                action=sig_act,
                expected_move_pct=round(signed_move, 6),
                estimated_tp=tp,
                estimated_sl=sl,
                risk_reward=rr,
                confidence=round(sig_prob, 4),
                reason=f"Signal-only {sig_act}: {accept_detail}, TB R:R={rr:.2f}",
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
            tp, sl, rr = self._compute_candle_levels(
                candle_dir, anchor_close, predicted_candles, anchor_atr=anchor_atr
            )
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
        sig_act, sig_prob, reject_detail, accept_detail = self._select_signal_action(signal)

        if sig_act == "HOLD":
            return TradeAnalysis(
                action="HOLD",
                expected_move_pct=round(expected_move_pct, 6),
                estimated_tp=None,
                estimated_sl=None,
                risk_reward=None,
                confidence=round(sig_prob, 4),
                reason=(
                    f"Signal below threshold ({reject_detail}); "
                    f"forecast={expected_move_pct:+.2%}"
                ),
            )

        # Reject active directional contradiction or flat forecast below threshold
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

        if abs(expected_move_pct) < self._cfg.min_expected_move_pct:
            return TradeAnalysis(
                action="HOLD",
                expected_move_pct=round(expected_move_pct, 6),
                estimated_tp=None,
                estimated_sl=None,
                risk_reward=None,
                confidence=round(sig_prob, 4),
                reason=(
                    f"Cross-confirmed {sig_act} filtered by expected move "
                    f"({expected_move_pct:+.2%} < {self._cfg.min_expected_move_pct:.2%})"
                ),
            )

        tp, sl, rr = self._compute_candle_levels(
            sig_act, anchor_close, predicted_candles, anchor_atr=anchor_atr
        )
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

        dual_sig_tag = (
            f"signal={sig_prob:.1%}"
            if self._cfg.signal_decision_mode == "argmax"
            else accept_detail
        )
        return TradeAnalysis(
            action=sig_act,
            expected_move_pct=round(expected_move_pct, 6),
            estimated_tp=tp,
            estimated_sl=sl,
            risk_reward=rr,
            confidence=round(sig_prob, 4),
            reason=(
                f"Cross-confirmed {sig_act}: {dual_sig_tag}, "
                f"forecast={expected_move_pct:+.2%}, R:R={rr:.2f}"
            ),
        )
