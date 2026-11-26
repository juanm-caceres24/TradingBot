"""Historical replay engine for fast model and alert-rule evaluation."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tradingbot.analysis import (
    AlertEvaluator,
    TradeAnalyzer,
    reconstruct_predicted_candles,
    snapshot_to_dict,
    snapshot_to_json,
)
from tradingbot.config import AssetConfig
from tradingbot.features import FeaturePipeline
from tradingbot.models import AlertEvent, Candle, PredictionSnapshot
from tradingbot.nn_client import AssetNNClient
from tradingbot.providers.base import default_clock_ms
from tradingbot.runtime.state import EventBus
from tradingbot.runtime.webhooks import WebhookDispatcher
from tradingbot.storage import CandleStore


@dataclass(frozen=True)
class ReplaySummary:
    """Quantitative summary of a historical replay run."""

    asset_id: str
    total_candles: int
    bars_evaluated: int
    predictions_made: int
    alerts_triggered: int
    action_counts: dict[str, int]
    directional_hits: int
    directional_total: int
    mean_abs_return_error: float | None
    alerts: tuple[AlertEvent, ...]


StepCallback = Callable[[int, Candle, PredictionSnapshot, AlertEvent | None], None]


class ReplayEngine:
    """Replays historical closed candles chronologically against NeuralNetwork server(s)."""

    def __init__(
        self,
        asset: AssetConfig,
        *,
        nn_client: AssetNNClient | None = None,
        clock_ms: Callable[[], int] = default_clock_ms,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._asset = asset
        self._pipeline = FeaturePipeline(asset)
        self._nn_client = nn_client if nn_client is not None else AssetNNClient(asset)
        self._analyzer = TradeAnalyzer(asset.analysis, asset.features.triple_barrier)
        self._clock_ms = clock_ms
        self._sleep_fn = sleep_fn

    def verify_models(self) -> None:
        if self._asset.neural_networks.verify_schema_on_start and self._nn_client.has_any_model:
            self._nn_client.verify_all()

    def run(
        self,
        candles: Sequence[Candle],
        *,
        store: CandleStore | None = None,
        event_bus: EventBus | None = None,
        webhook_dispatcher: WebhookDispatcher | None = None,
        persist_predictions: bool = False,
        persist_alerts: bool = False,
        step_delay_sec: float = 0.0,
        step_cb: StepCallback | None = None,
    ) -> ReplaySummary:
        """Step through `candles` chronologically, evaluating predictions and alert rules.

        All built-in indicators are strictly causal (depending only on bars <= i),
        so precomputing indicator arrays once preserves exact live-streaming equivalence
        while keeping multi-thousand-bar replays O(N).
        """
        n = len(candles)
        action_counts = {"HOLD": 0, "LONG": 0, "SHORT": 0}
        if n < self._pipeline.min_candles_for_inference:
            return ReplaySummary(
                asset_id=self._asset.id,
                total_candles=n,
                bars_evaluated=0,
                predictions_made=0,
                alerts_triggered=0,
                action_counts=action_counts,
                directional_hits=0,
                directional_total=0,
                mean_abs_return_error=None,
                alerts=(),
            )

        ind_data = self._pipeline.indicator_engine.compute_all(candles, nn_only=True)
        atr_series = (
            self._pipeline._compute_norm_atr(candles)  # noqa: SLF001
        )

        evaluator = AlertEvaluator(self._asset, clock_ms=self._clock_ms)
        tf = self._asset.market.parsed_timeframe
        horizon = self._asset.features.forecast_horizon
        first_anchor = self._pipeline.max_warmup + self._asset.features.input_window - 1

        bars_evaluated = 0
        predictions_made = 0
        directional_hits = 0
        directional_total = 0
        abs_errors: list[float] = []
        emitted_alerts: list[AlertEvent] = []

        for idx in range(first_anchor, n):
            ctx = self._pipeline.build_input_window(
                candles,
                idx,
                precomputed_indicators=ind_data,
                precomputed_atr=atr_series,
            )
            if ctx is None:
                continue

            bars_evaluated += 1
            now_ms = self._clock_ms()

            raw_candles = self._nn_client.predict_candles(ctx.vector)
            signal_pred = self._nn_client.predict_signal(ctx.vector)
            extra_preds = self._nn_client.predict_extra(ctx.vector)
            predicted_candles = (
                reconstruct_predicted_candles(raw_candles.output, ctx, tf)
                if raw_candles is not None
                else ()
            )
            analysis = self._analyzer.analyze(
                ctx.anchor_close,
                predicted_candles=predicted_candles,
                signal=signal_pred,
            )
            snapshot = PredictionSnapshot(
                asset_id=self._asset.id,
                provider=self._asset.provider.name,
                symbol=self._asset.market.symbol,
                timeframe=tf.label,
                anchor_open_time_ms=ctx.anchor_candle.open_time_ms,
                created_at_ms=now_ms,
                predicted_candles=predicted_candles,
                signal=signal_pred,
                analysis=analysis,
                candle_latency_us=raw_candles.latency_us if raw_candles is not None else None,
                extra_outputs={name: pred.output for name, pred in extra_preds.items()},
            )

            predictions_made += 1
            action_counts[analysis.action] = action_counts.get(analysis.action, 0) + 1

            # Post-hoc accuracy check when actual future bar `idx + horizon` exists and matches horizon time.
            if idx + horizon < n and predicted_candles:
                future_candle = candles[idx + horizon]
                horizon_valid = (
                    self._asset.market.market_schedule != "24/7"
                    or future_candle.open_time_ms
                    == ctx.anchor_candle.open_time_ms + horizon * tf.interval_ms
                )
                if horizon_valid:
                    actual_future_close = future_candle.close
                    actual_ret = (actual_future_close - ctx.anchor_close) / ctx.anchor_close
                    pred_ret = (predicted_candles[-1].close - ctx.anchor_close) / ctx.anchor_close
                    abs_errors.append(abs(pred_ret - actual_ret))
                    if actual_ret != 0.0 and pred_ret != 0.0:
                        directional_total += 1
                        if (actual_ret > 0.0) == (pred_ret > 0.0):
                            directional_hits += 1

            if persist_predictions and store is not None:
                store.save_prediction(
                    snapshot.provider,
                    snapshot.symbol,
                    snapshot.timeframe,
                    snapshot.anchor_open_time_ms,
                    snapshot.created_at_ms,
                    "replay_composite",
                    snapshot_to_json(snapshot),
                    asset_id=self._asset.id,
                )

            if event_bus is not None:
                event_bus.publish(
                    {
                        "type": "prediction",
                        "asset_id": self._asset.id,
                        "prediction": snapshot_to_dict(snapshot),
                    }
                )

            alert = evaluator.evaluate(snapshot, now_ms=now_ms)
            if alert is not None:
                if persist_alerts and store is not None:
                    aid = store.save_alert(alert)
                    alert = dataclasses.replace(alert, alert_id=aid)
                if webhook_dispatcher is not None:
                    webhook_dispatcher.dispatch(alert)
                if event_bus is not None:
                    event_bus.publish(
                        {
                            "type": "alert",
                            "asset_id": self._asset.id,
                            "alert": alert.to_dict(),
                        }
                    )
                emitted_alerts.append(alert)

            if step_cb is not None:
                step_cb(idx, ctx.anchor_candle, snapshot, alert)

            if step_delay_sec > 0.0:
                self._sleep_fn(step_delay_sec)

        mae = (sum(abs_errors) / len(abs_errors)) if abs_errors else None
        return ReplaySummary(
            asset_id=self._asset.id,
            total_candles=n,
            bars_evaluated=bars_evaluated,
            predictions_made=predictions_made,
            alerts_triggered=len(emitted_alerts),
            action_counts=action_counts,
            directional_hits=directional_hits,
            directional_total=directional_total,
            mean_abs_return_error=mae,
            alerts=tuple(emitted_alerts),
        )
