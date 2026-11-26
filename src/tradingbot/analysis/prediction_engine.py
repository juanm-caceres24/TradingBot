"""End-to-end per-asset prediction facade combining features, NN client, and analysis."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from tradingbot.analysis.reconstruction import reconstruct_predicted_candles
from tradingbot.analysis.trade_analyzer import TradeAnalyzer
from tradingbot.config import AssetConfig
from tradingbot.features import FeaturePipeline
from tradingbot.models import Candle, PredictionSnapshot
from tradingbot.nn_client import AssetNNClient
from tradingbot.providers.base import default_clock_ms


def snapshot_to_dict(snapshot: PredictionSnapshot) -> dict[str, Any]:
    """Serialize a PredictionSnapshot into a JSON-compatible dictionary."""
    return {
        "asset_id": snapshot.asset_id,
        "provider": snapshot.provider,
        "symbol": snapshot.symbol,
        "timeframe": snapshot.timeframe,
        "anchor_open_time_ms": snapshot.anchor_open_time_ms,
        "created_at_ms": snapshot.created_at_ms,
        "candle_latency_us": snapshot.candle_latency_us,
        "predicted_candles": [
            {
                "step": pc.step,
                "open_time_ms": pc.open_time_ms,
                "open": pc.open,
                "high": pc.high,
                "low": pc.low,
                "close": pc.close,
                "confidence": pc.confidence,
            }
            for pc in snapshot.predicted_candles
        ],
        "signal": (
            {
                "hold_prob": snapshot.signal.hold_prob,
                "long_prob": snapshot.signal.long_prob,
                "short_prob": snapshot.signal.short_prob,
                "best_action": snapshot.signal.best_action,
                "best_probability": snapshot.signal.best_probability,
                "latency_us": snapshot.signal.latency_us,
            }
            if snapshot.signal is not None
            else None
        ),
        "analysis": (
            {
                "action": snapshot.analysis.action,
                "expected_move_pct": snapshot.analysis.expected_move_pct,
                "estimated_tp": snapshot.analysis.estimated_tp,
                "estimated_sl": snapshot.analysis.estimated_sl,
                "risk_reward": snapshot.analysis.risk_reward,
                "confidence": snapshot.analysis.confidence,
                "reason": snapshot.analysis.reason,
            }
            if snapshot.analysis is not None
            else None
        ),
        "extra_outputs": {
            name: list(values) for name, values in snapshot.extra_outputs.items()
        },
    }


def snapshot_to_json(snapshot: PredictionSnapshot) -> str:
    """Serialize a PredictionSnapshot into compact JSON for SQLite audit or WebSocket push."""
    return json.dumps(snapshot_to_dict(snapshot), separators=(",", ":"))


class PredictionEngine:
    """Runs feature extraction, NN server inference, candle reconstruction, and trade analysis."""

    def __init__(
        self,
        asset: AssetConfig,
        *,
        nn_client: AssetNNClient | None = None,
        clock_ms: Callable[[], int] = default_clock_ms,
    ) -> None:
        self._asset = asset
        self._pipeline = FeaturePipeline(asset)
        self._nn_client = nn_client if nn_client is not None else AssetNNClient(asset)
        self._analyzer = TradeAnalyzer(asset.analysis, asset.features.triple_barrier)
        self._clock_ms = clock_ms

    @property
    def asset(self) -> AssetConfig:
        return self._asset

    @property
    def pipeline(self) -> FeaturePipeline:
        return self._pipeline

    @property
    def nn_client(self) -> AssetNNClient:
        return self._nn_client

    def verify_models(self) -> dict[str, Any]:
        if not self._asset.neural_networks.verify_schema_on_start:
            return {}
        return self._nn_client.verify_all()

    def predict(
        self,
        candles: Sequence[Candle],
        *,
        now_ms: int | None = None,
    ) -> PredictionSnapshot:
        """Produce a PredictionSnapshot anchored to the last candle in `candles`."""
        ts_ms = self._clock_ms() if now_ms is None else now_ms
        if not self._nn_client.has_any_model:
            if not candles:
                raise ValueError("candles must not be empty")
            anchor = candles[-1]
            analysis = self._analyzer.analyze(anchor.close)
            return PredictionSnapshot(
                asset_id=self._asset.id,
                provider=self._asset.provider.name,
                symbol=self._asset.market.symbol,
                timeframe=self._asset.market.timeframe,
                anchor_open_time_ms=anchor.open_time_ms,
                created_at_ms=ts_ms,
                predicted_candles=(),
                signal=None,
                analysis=analysis,
                candle_latency_us=None,
            )

        ctx = self._pipeline.build_latest_inference_window(candles)
        raw_candles = self._nn_client.predict_candles(ctx.vector)
        signal_pred = self._nn_client.predict_signal(ctx.vector)
        extra_preds = self._nn_client.predict_extra(ctx.vector)

        predicted_candles = (
            reconstruct_predicted_candles(
                raw_candles.output,
                ctx,
                self._asset.market.parsed_timeframe,
            )
            if raw_candles is not None
            else ()
        )
        analysis = self._analyzer.analyze(
            ctx.anchor_close,
            predicted_candles=predicted_candles,
            signal=signal_pred,
        )

        return PredictionSnapshot(
            asset_id=self._asset.id,
            provider=self._asset.provider.name,
            symbol=self._asset.market.symbol,
            timeframe=self._asset.market.timeframe,
            anchor_open_time_ms=ctx.anchor_candle.open_time_ms,
            created_at_ms=ts_ms,
            predicted_candles=predicted_candles,
            signal=signal_pred,
            analysis=analysis,
            candle_latency_us=raw_candles.latency_us if raw_candles is not None else None,
            extra_outputs={name: pred.output for name, pred in extra_preds.items()},
        )
