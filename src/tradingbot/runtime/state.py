"""Thread-safe shared runtime state and lightweight pub/sub EventBus.

Decouples the multi-asset collector/analysis loop from downstream UI and API
consumers (console logging and embedded stdlib HTTP/REST + RFC 6455 WebSocket push).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tradingbot.analysis import snapshot_to_dict
from tradingbot.models import Candle, PredictionSnapshot

EventListener = Callable[[dict[str, Any]], None]


class EventBus:
    """Thread-safe in-process pub/sub bus for live candle, prediction, and alert events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: list[EventListener] = []

    def subscribe(self, listener: EventListener) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: EventListener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Isolate misbehaving subscribers so the collector loop never crashes.
                continue


@dataclass
class AssetRuntimeSnapshot:
    """Latest thread-safe in-memory state of one managed asset."""

    asset_id: str
    provider: str
    symbol: str
    timeframe: str
    mode: str
    enabled: bool
    candles: tuple[Candle, ...] = ()
    indicators: dict[str, dict[str, list[float | None]]] = field(default_factory=dict)
    indicator_plots: dict[str, str] = field(default_factory=dict)
    indicator_types: dict[str, str] = field(default_factory=dict)
    latest_prediction: PredictionSnapshot | None = None
    nn_health: dict[str, bool] = field(default_factory=dict)
    alert_status: dict[str, Any] = field(default_factory=dict)
    last_sync_ms: int | None = None
    last_error: str | None = None

    def summary_dict(self, *, unread_alerts: int = 0) -> dict[str, Any]:
        last_candle = self.candles[-1] if self.candles else None
        analysis = (
            self.latest_prediction.analysis
            if self.latest_prediction is not None
            else None
        )
        signal = (
            self.latest_prediction.signal
            if self.latest_prediction is not None
            else None
        )
        return {
            "asset_id": self.asset_id,
            "provider": self.provider,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "enabled": self.enabled,
            "candle_count": len(self.candles),
            "last_price": last_candle.close if last_candle is not None else None,
            "last_open_time_ms": last_candle.open_time_ms if last_candle is not None else None,
            "last_is_closed": last_candle.is_closed if last_candle is not None else None,
            "action": analysis.action if analysis is not None else None,
            "confidence": analysis.confidence if analysis is not None else None,
            "expected_move_pct": analysis.expected_move_pct if analysis is not None else None,
            "signal": (
                {
                    "hold": signal.hold_prob,
                    "long": signal.long_prob,
                    "short": signal.short_prob,
                }
                if signal is not None
                else None
            ),
            "nn_health": dict(self.nn_health),
            "alert_status": dict(self.alert_status),
            "unread_alerts": unread_alerts,
            "last_sync_ms": self.last_sync_ms,
            "last_error": self.last_error,
        }

    def detail_dict(self, *, visible_candles: int | None = None) -> dict[str, Any]:
        candles_slice = (
            self.candles[-visible_candles:]
            if visible_candles is not None and len(self.candles) > visible_candles
            else self.candles
        )
        offset = len(self.candles) - len(candles_slice)
        sliced_indicators: dict[str, dict[str, list[float | None]]] = {}
        for ind_id, cols in self.indicators.items():
            sliced_indicators[ind_id] = {
                col: list(vals[offset:]) for col, vals in cols.items()
            }
        return {
            "asset_id": self.asset_id,
            "provider": self.provider,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "candles": [c.to_dict() for c in candles_slice],
            "indicators": sliced_indicators,
            "indicator_plots": dict(self.indicator_plots),
            "indicator_types": dict(self.indicator_types),
            "prediction": (
                snapshot_to_dict(self.latest_prediction)
                if self.latest_prediction is not None
                else None
            ),
            "nn_health": dict(self.nn_health),
            "alert_status": dict(self.alert_status),
            "last_sync_ms": self.last_sync_ms,
            "last_error": self.last_error,
        }


class RuntimeState:
    """Thread-safe container storing latest snapshots for all active assets."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._assets: dict[str, AssetRuntimeSnapshot] = {}

    def update(self, snapshot: AssetRuntimeSnapshot) -> None:
        with self._lock:
            self._assets[snapshot.asset_id] = snapshot

    def get(self, asset_id: str) -> AssetRuntimeSnapshot | None:
        with self._lock:
            return self._assets.get(asset_id)

    def all_snapshots(self) -> list[AssetRuntimeSnapshot]:
        with self._lock:
            return list(self._assets.values())
