"""Stateful per-asset alert rule evaluator with persistence and cooldown filtering."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from tradingbot.config import AlertRulesConfig, AssetConfig
from tradingbot.models import AlertEvent, PredictionSnapshot
from tradingbot.providers.base import default_clock_ms


class AlertEvaluator:
    """Evaluates closed-candle PredictionSnapshots against per-asset alert rules.

    Filtering features:
    - Action allow-list (e.g., ['LONG', 'SHORT']).
    - Minimum confidence, minimum Risk-Reward ratio, minimum expected move.
    - Consecutive-candle persistence (`consecutive_candles`) to filter single-bar noise.
    - Anti-spam cooldown (`cooldown_candles`) after emitting an alert.
    - Idempotent against duplicate/stale candle timestamps.
    """

    def __init__(
        self,
        asset: AssetConfig,
        *,
        clock_ms: Callable[[], int] = default_clock_ms,
    ) -> None:
        self._asset = asset
        self._rules: AlertRulesConfig = asset.alerts
        self._clock_ms = clock_ms
        self._last_anchor_ms: int | None = None
        self._streak_action: str | None = None
        self._streak_count: int = 0
        # Start ready to emit as soon as the first consecutive streak qualifies.
        self._candles_since_alert: int = self._rules.cooldown_candles
        self._total_emitted: int = 0

    def reset(self) -> None:
        self._last_anchor_ms = None
        self._streak_action = None
        self._streak_count = 0
        self._candles_since_alert = self._rules.cooldown_candles
        self._total_emitted = 0

    def status_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self._asset.id,
            "enabled": self._rules.enabled,
            "streak_action": self._streak_action,
            "streak_count": self._streak_count,
            "required_consecutive": self._rules.consecutive_candles,
            "candles_since_alert": self._candles_since_alert,
            "cooldown_candles": self._rules.cooldown_candles,
            "total_emitted": self._total_emitted,
        }

    def _qualifies(self, snapshot: PredictionSnapshot) -> bool:
        analysis = snapshot.analysis
        if analysis is None:
            return False
        if analysis.action not in self._rules.actions:
            return False
        if analysis.confidence < self._rules.min_confidence:
            return False
        if abs(analysis.expected_move_pct) < self._rules.min_expected_move_pct:
            return False
        if (
            analysis.risk_reward is not None
            and analysis.risk_reward < self._rules.min_risk_reward
        ):
            return False
        return True

    def evaluate(
        self,
        snapshot: PredictionSnapshot,
        *,
        now_ms: int | None = None,
    ) -> AlertEvent | None:
        """Evaluate one closed-candle PredictionSnapshot and return an AlertEvent if triggered."""
        if not self._rules.enabled:
            return None

        # Deduplicate re-polls of the exact same (or older) candle.
        if (
            self._last_anchor_ms is not None
            and snapshot.anchor_open_time_ms <= self._last_anchor_ms
        ):
            return None

        self._last_anchor_ms = snapshot.anchor_open_time_ms
        self._candles_since_alert += 1

        if not self._qualifies(snapshot):
            self._streak_action = None
            self._streak_count = 0
            return None

        assert snapshot.analysis is not None
        action = snapshot.analysis.action
        if action == self._streak_action:
            self._streak_count += 1
        else:
            self._streak_action = action
            self._streak_count = 1

        if self._streak_count < self._rules.consecutive_candles:
            return None
        if self._candles_since_alert < self._rules.cooldown_candles:
            return None

        created_ms = self._clock_ms() if now_ms is None else now_ms
        rule_name = (
            f"consecutive_{self._rules.consecutive_candles}_bars"
            if self._rules.consecutive_candles > 1
            else "single_bar_threshold"
        )
        payload = {
            "action": action,
            "confidence": snapshot.analysis.confidence,
            "expected_move_pct": snapshot.analysis.expected_move_pct,
            "estimated_tp": snapshot.analysis.estimated_tp,
            "estimated_sl": snapshot.analysis.estimated_sl,
            "risk_reward": snapshot.analysis.risk_reward,
            "reason": snapshot.analysis.reason,
            "signal": (
                {
                    "hold": snapshot.signal.hold_prob,
                    "long": snapshot.signal.long_prob,
                    "short": snapshot.signal.short_prob,
                }
                if snapshot.signal is not None
                else None
            ),
        }

        self._candles_since_alert = 0
        self._total_emitted += 1

        return AlertEvent(
            asset_id=self._asset.id,
            provider=snapshot.provider,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            anchor_open_time_ms=snapshot.anchor_open_time_ms,
            created_at_ms=created_ms,
            action=action,
            confidence=snapshot.analysis.confidence,
            consecutive_hits=self._streak_count,
            rule_name=rule_name,
            payload_json=json.dumps(payload, separators=(",", ":")),
            acknowledged=False,
        )
