"""Multi-asset continuous runtime orchestrator (`sync_live` and `live_only` modes)."""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tradingbot.analysis import (
    AlertEvaluator,
    PredictionEngine,
    snapshot_to_dict,
    snapshot_to_json,
)
from tradingbot.config import AppConfig, AssetConfig
from tradingbot.features import InsufficientDataError
from tradingbot.models import AlertEvent, Candle, PredictionSnapshot
from tradingbot.nn_client import NNClientError, SchemaMismatchError
from tradingbot.providers import ProviderError, SyncService, create_provider, default_clock_ms
from tradingbot.runtime.state import AssetRuntimeSnapshot, EventBus, RuntimeState
from tradingbot.runtime.webhooks import WebhookDeliveryResult, WebhookDispatcher
from tradingbot.storage import CandleStore


@dataclass(frozen=True)
class PollOutcome:
    """Structured result of one asset polling cycle."""

    asset_id: str
    candles_received: int
    new_closed_candle: bool
    prediction: PredictionSnapshot | None = None
    alert: AlertEvent | None = None
    webhook_results: tuple[WebhookDeliveryResult, ...] = ()
    error: str | None = None


class AssetRunner:
    """Manages sync, live polling, indicators, predictions, and alerts for one AssetConfig."""

    def __init__(
        self,
        asset: AssetConfig,
        *,
        visible_candles: int = 70,
        clock_ms: Callable[[], int] = default_clock_ms,
        prediction_engine: PredictionEngine | None = None,
    ) -> None:
        self._asset = asset
        self._clock_ms = clock_ms
        self._prediction_engine = (
            prediction_engine
            if prediction_engine is not None
            else PredictionEngine(asset, clock_ms=clock_ms)
        )
        self._alert_evaluator = AlertEvaluator(asset, clock_ms=clock_ms)
        self._lock = threading.RLock()
        self._indicator_engine = self._prediction_engine.pipeline.indicator_engine
        self._indicator_plots = {ind.id: ind.plot for ind in asset.indicators}
        self._indicator_types = {ind.id: ind.type for ind in asset.indicators}

        # Retain enough bars in RAM for indicator warmup + NN input window + Web UI zoom (up to 250 bars).
        max_warm = self._indicator_engine.max_warmup(nn_only=False)
        chart_need = max_warm + max(visible_candles, 250)
        nn_need = self._prediction_engine.pipeline.min_candles_for_inference
        self._buffer_limit = max(chart_need, nn_need) + 20

        self._candles: list[Candle] = []
        self._indicators: dict[str, dict[str, list[float | None]]] = {}
        self._latest_prediction: PredictionSnapshot | None = None
        self._last_closed_open_time_ms: int | None = None
        self._last_sync_ms: int | None = None
        self._last_error: str | None = None
        self._nn_health: dict[str, bool] = {}

    @property
    def asset(self) -> AssetConfig:
        return self._asset

    @property
    def buffer_limit(self) -> int:
        return self._buffer_limit

    def _merge_candles_into_buffer(self, incoming: Sequence[Candle]) -> bool:
        """Upsert candles by open_time_ms in chronological order; return True if a newer closed bar arrived."""
        if not incoming:
            return False
        by_time = {c.open_time_ms: c for c in self._candles}
        for c in incoming:
            by_time[c.open_time_ms] = c
        merged = [by_time[k] for k in sorted(by_time.keys())]
        if len(merged) > self._buffer_limit:
            merged = merged[-self._buffer_limit :]
        self._candles = merged

        closed = [c for c in self._candles if c.is_closed]
        if not closed:
            return False
        newest_closed_ts = closed[-1].open_time_ms
        if self._last_closed_open_time_ms is None or newest_closed_ts > self._last_closed_open_time_ms:
            self._last_closed_open_time_ms = newest_closed_ts
            return True
        return False

    def _recompute_indicators(self) -> None:
        self._indicators = self._indicator_engine.compute_all(self._candles, nn_only=False)

    def _build_runtime_snapshot(self) -> AssetRuntimeSnapshot:
        return AssetRuntimeSnapshot(
            asset_id=self._asset.id,
            provider=self._asset.provider.name,
            symbol=self._asset.market.symbol,
            timeframe=self._asset.market.timeframe,
            mode=self._asset.runtime.mode,
            enabled=self._asset.enabled,
            candles=tuple(self._candles),
            indicators={
                ind_id: {col: list(vals) for col, vals in cols.items()}
                for ind_id, cols in self._indicators.items()
            },
            indicator_plots=dict(self._indicator_plots),
            indicator_types=dict(self._indicator_types),
            latest_prediction=self._latest_prediction,
            nn_health=dict(self._nn_health),
            alert_status=self._alert_evaluator.status_dict(),
            last_sync_ms=self._last_sync_ms,
            last_error=self._last_error,
        )

    def _run_prediction_and_alerts(
        self,
        store: CandleStore,
        webhook_dispatcher: WebhookDispatcher,
        event_bus: EventBus,
        *,
        on_closed_bar: bool,
        now_ms: int,
    ) -> tuple[PredictionSnapshot | None, AlertEvent | None, list[WebhookDeliveryResult]]:
        if not self._prediction_engine.nn_client.has_any_model:
            return None, None, []

        source_candles = (
            [c for c in self._candles if c.is_closed]
            if on_closed_bar or not self._asset.runtime.predict_on_open_candle
            else list(self._candles)
        )
        if not source_candles:
            return None, None, []

        try:
            snapshot = self._prediction_engine.predict(source_candles, now_ms=now_ms)
        except InsufficientDataError:
            return None, None, []
        except NNClientError as exc:
            self._last_error = str(exc)
            self._nn_health = self._prediction_engine.nn_client.health_summary()
            return None, None, []

        self._latest_prediction = snapshot
        # Successful predict() already proves every enabled endpoint responded cleanly;
        # avoid extra GET /health round-trips on every candle.
        nn_client = self._prediction_engine.nn_client
        updated_health: dict[str, bool] = {}
        if nn_client.candle_enabled:
            updated_health["candle_predictor"] = True
        if nn_client.signal_enabled:
            updated_health["signal_classifier"] = True
        for name in snapshot.extra_outputs:
            updated_health[name] = True
        self._nn_health = updated_health

        if on_closed_bar and self._asset.runtime.persist_to_db:
            store.save_prediction(
                snapshot.provider,
                snapshot.symbol,
                snapshot.timeframe,
                snapshot.anchor_open_time_ms,
                snapshot.created_at_ms,
                "composite",
                snapshot_to_json(snapshot),
                asset_id=self._asset.id,
            )

        event_bus.publish(
            {
                "type": "prediction",
                "asset_id": self._asset.id,
                "prediction": snapshot_to_dict(snapshot),
            }
        )

        # Alerts are evaluated strictly on closed candles to prevent intra-bar flicker spam.
        if not on_closed_bar:
            return snapshot, None, []

        alert = self._alert_evaluator.evaluate(snapshot, now_ms=now_ms)
        if alert is None:
            return snapshot, None, []

        alert_id = store.save_alert(alert)
        persisted_alert = dataclasses.replace(alert, alert_id=alert_id)
        wh_results = webhook_dispatcher.dispatch(persisted_alert)
        event_bus.publish(
            {
                "type": "alert",
                "asset_id": self._asset.id,
                "alert": persisted_alert.to_dict(),
            }
        )
        return snapshot, persisted_alert, wh_results

    def initialize(
        self,
        sync_service: SyncService,
        store: CandleStore,
        runtime_state: RuntimeState,
        webhook_dispatcher: WebhookDispatcher,
        event_bus: EventBus,
        *,
        strict_nn_startup: bool = False,
        now_ms: int | None = None,
    ) -> PollOutcome:
        """Perform startup model check, initial sync/live-only load, and initial indicators."""
        with self._lock:
            effective_now = self._clock_ms() if now_ms is None else now_ms
            self._last_error = None

            # Seed AlertEvaluator from latest persisted alert so restarting does not re-emit
            # a duplicate alert on the exact same closed bar.
            latest_alerts = store.get_alerts(asset_id=self._asset.id, limit=1)
            if latest_alerts:
                self._alert_evaluator.seed_last_anchor(latest_alerts[0].anchor_open_time_ms)

            # Always fail-fast on genuine SchemaMismatchError; tolerate temporarily offline NN servers
            # unless strict_nn_startup=True.
            if self._prediction_engine.nn_client.has_any_model:
                try:
                    self._prediction_engine.verify_models()
                    self._nn_health = self._prediction_engine.nn_client.health_summary()
                except SchemaMismatchError:
                    raise
                except NNClientError as exc:
                    self._nn_health = self._prediction_engine.nn_client.health_summary()
                    self._last_error = str(exc)
                    if strict_nn_startup:
                        raise

            mode = self._asset.runtime.mode
            received = 0
            try:
                if mode == "sync_live" and self._asset.runtime.persist_to_db:
                    report = sync_service.sync_asset_to_now(
                        self._asset,
                        now_ms=effective_now,
                        min_candles_when_empty=self._buffer_limit,
                        include_open_candle=True,
                    )
                    received = report.candles_upserted
                    loaded = store.get_latest_candles(
                        self._asset.provider.name,
                        self._asset.market.symbol,
                        self._asset.market.timeframe,
                        limit=self._buffer_limit,
                        closed_only=False,
                    )
                    self._merge_candles_into_buffer(loaded)
                elif mode in ("live_only", "sync_live"):
                    # In live_only (or sync_live with persist_to_db=False), load working buffer in RAM.
                    fetched = sync_service.fetch_latest_candles(
                        self._asset,
                        limit=self._buffer_limit,
                        persist=self._asset.runtime.persist_to_db,
                    )
                    received = len(fetched)
                    self._merge_candles_into_buffer(fetched)
                else:
                    loaded = store.get_latest_candles(
                        self._asset.provider.name,
                        self._asset.market.symbol,
                        self._asset.market.timeframe,
                        limit=self._buffer_limit,
                        closed_only=False,
                    )
                    received = len(loaded)
                    self._merge_candles_into_buffer(loaded)
            except ProviderError as exc:
                self._last_error = str(exc)
                runtime_state.update(self._build_runtime_snapshot())
                return PollOutcome(
                    asset_id=self._asset.id,
                    candles_received=0,
                    new_closed_candle=False,
                    error=str(exc),
                )

            self._last_sync_ms = effective_now
            self._recompute_indicators()

            has_closed = any(c.is_closed for c in self._candles)
            prediction, alert, wh_results = self._run_prediction_and_alerts(
                store,
                webhook_dispatcher,
                event_bus,
                on_closed_bar=has_closed,
                now_ms=effective_now,
            )
            runtime_state.update(self._build_runtime_snapshot())
            event_bus.publish(
                {
                    "type": "status",
                    "asset_id": self._asset.id,
                    "status": "initialized",
                    "candle_count": len(self._candles),
                }
            )
            return PollOutcome(
                asset_id=self._asset.id,
                candles_received=received,
                new_closed_candle=has_closed,
                prediction=prediction,
                alert=alert,
                webhook_results=tuple(wh_results),
                error=self._last_error,
            )

    def poll_once(
        self,
        sync_service: SyncService,
        store: CandleStore,
        runtime_state: RuntimeState,
        webhook_dispatcher: WebhookDispatcher,
        event_bus: EventBus,
        *,
        now_ms: int | None = None,
    ) -> PollOutcome:
        """Fetch latest bars, heal unexpected multi-bar gaps, recompute indicators, and predict."""
        with self._lock:
            effective_now = self._clock_ms() if now_ms is None else now_ms
            self._last_error = None
            prev_closed_ts = self._last_closed_open_time_ms
            tf = self._asset.market.parsed_timeframe

            try:
                fetched = sync_service.fetch_latest_candles(
                    self._asset,
                    limit=3,
                    persist=self._asset.runtime.persist_to_db,
                )

                # If a poll paused longer than 1 interval in sync_live mode, heal missed closed bars cleanly.
                closed_fetched = [c for c in fetched if c.is_closed]
                if (
                    self._asset.runtime.mode == "sync_live"
                    and prev_closed_ts is not None
                    and closed_fetched
                    and closed_fetched[-1].open_time_ms - prev_closed_ts > tf.interval_ms
                ):
                    if self._asset.runtime.persist_to_db:
                        sync_service.backfill_asset(
                            self._asset,
                            start_ms=prev_closed_ts + tf.interval_ms,
                            end_ms=closed_fetched[-1].open_time_ms,
                        )
                        reloaded = store.get_latest_candles(
                            self._asset.provider.name,
                            self._asset.market.symbol,
                            tf.label,
                            limit=self._buffer_limit,
                            closed_only=False,
                        )
                        new_closed = self._merge_candles_into_buffer(reloaded)
                    else:
                        missed_bars = int(
                            (closed_fetched[-1].open_time_ms - prev_closed_ts) // tf.interval_ms
                        )
                        healed = sync_service.fetch_latest_candles(
                            self._asset,
                            limit=min(missed_bars + 2, self._buffer_limit),
                            persist=False,
                        )
                        new_closed = self._merge_candles_into_buffer(healed)
                else:
                    new_closed = self._merge_candles_into_buffer(fetched)
            except ProviderError as exc:
                self._last_error = str(exc)
                runtime_state.update(self._build_runtime_snapshot())
                return PollOutcome(
                    asset_id=self._asset.id,
                    candles_received=0,
                    new_closed_candle=False,
                    error=str(exc),
                )

            self._last_sync_ms = effective_now
            self._recompute_indicators()

            if self._candles:
                event_bus.publish(
                    {
                        "type": "candle_update",
                        "asset_id": self._asset.id,
                        "candle": self._candles[-1].to_dict(),
                        "is_new_close": new_closed,
                    }
                )

            prediction: PredictionSnapshot | None = None
            alert: AlertEvent | None = None
            wh_results: list[WebhookDeliveryResult] = []
            if new_closed or self._asset.runtime.predict_on_open_candle:
                prediction, alert, wh_results = self._run_prediction_and_alerts(
                    store,
                    webhook_dispatcher,
                    event_bus,
                    on_closed_bar=new_closed,
                    now_ms=effective_now,
                )

            runtime_state.update(self._build_runtime_snapshot())
            return PollOutcome(
                asset_id=self._asset.id,
                candles_received=len(fetched),
                new_closed_candle=new_closed,
                prediction=prediction,
                alert=alert,
                webhook_results=tuple(wh_results),
                error=self._last_error,
            )


class BotOrchestrator:
    """Coordinates continuous multi-asset live ingestion, predictions, alerts, and webhooks."""

    def __init__(
        self,
        config: AppConfig,
        store: CandleStore,
        *,
        assets: Sequence[AssetConfig] | None = None,
        sync_service: SyncService | None = None,
        webhook_dispatcher: WebhookDispatcher | None = None,
        clock_ms: Callable[[], int] = default_clock_ms,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._store = store
        self._clock_ms = clock_ms
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn

        selected_assets = tuple(assets) if assets is not None else config.enabled_assets
        self._sync_service = (
            sync_service
            if sync_service is not None
            else SyncService(
                store,
                provider_factory=create_provider,
                clock_ms=clock_ms,
                sleep_fn=sleep_fn,
                monotonic_fn=monotonic_fn,
            )
        )
        self._webhooks = (
            webhook_dispatcher
            if webhook_dispatcher is not None
            else WebhookDispatcher(config.notifications.webhooks)
        )
        self._state = RuntimeState()
        self._event_bus = EventBus()
        self._stop_event = threading.Event()

        self._runners: dict[str, AssetRunner] = {
            asset.id: AssetRunner(
                asset,
                visible_candles=config.web.visible_candles,
                clock_ms=clock_ms,
            )
            for asset in selected_assets
        }
        self._next_poll_mono: dict[str, float] = {}
        self._init_lock = threading.Lock()
        self._initialized = False

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def store(self) -> CandleStore:
        return self._store

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def runners(self) -> dict[str, AssetRunner]:
        return self._runners

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _initialize_locked(self, *, strict_nn_startup: bool = False) -> list[PollOutcome]:
        outcomes: list[PollOutcome] = []
        for asset_id, runner in self._runners.items():
            outcome = runner.initialize(
                self._sync_service,
                self._store,
                self._state,
                self._webhooks,
                self._event_bus,
                strict_nn_startup=strict_nn_startup,
            )
            outcomes.append(outcome)
            self._next_poll_mono[asset_id] = (
                self._monotonic_fn() + runner.asset.runtime.poll_interval_sec
            )
        self._initialized = True
        return outcomes

    def initialize(self, *, strict_nn_startup: bool = False) -> list[PollOutcome]:
        with self._init_lock:
            return self._initialize_locked(strict_nn_startup=strict_nn_startup)

    def _ensure_initialized(self, *, strict_nn_startup: bool = False) -> list[PollOutcome] | None:
        """Double-checked lazy initialization so concurrent polls never initialize twice."""
        if self._initialized:
            return None
        with self._init_lock:
            if self._initialized:
                return None
            return self._initialize_locked(strict_nn_startup=strict_nn_startup)

    def poll_asset(self, asset_id: str) -> PollOutcome:
        """Thread-safe single-asset poll ensuring startup initialization completed first."""
        if asset_id not in self._runners:
            raise KeyError(f"Unknown asset '{asset_id}'")
        self._ensure_initialized()
        runner = self._runners[asset_id]
        outcome = runner.poll_once(
            self._sync_service,
            self._store,
            self._state,
            self._webhooks,
            self._event_bus,
        )
        self._next_poll_mono[asset_id] = (
            self._monotonic_fn() + runner.asset.runtime.poll_interval_sec
        )
        return outcome

    def poll_due_assets(self, *, force_all: bool = False) -> list[PollOutcome]:
        """Poll every asset whose poll interval has elapsed (or all if force_all=True)."""
        init_outcomes = self._ensure_initialized()
        if init_outcomes is not None:
            return init_outcomes

        now_mono = self._monotonic_fn()
        outcomes: list[PollOutcome] = []
        for asset_id, runner in self._runners.items():
            due_at = self._next_poll_mono.get(asset_id, 0.0)
            if force_all or now_mono >= due_at:
                outcome = runner.poll_once(
                    self._sync_service,
                    self._store,
                    self._state,
                    self._webhooks,
                    self._event_bus,
                )
                outcomes.append(outcome)
                self._next_poll_mono[asset_id] = (
                    self._monotonic_fn() + runner.asset.runtime.poll_interval_sec
                )
        return outcomes

    def run_loop(
        self,
        *,
        strict_nn_startup: bool = False,
        max_cycles: int | None = None,
        on_outcome: Callable[[PollOutcome], None] | None = None,
    ) -> int:
        """Run continuous multi-asset polling loop until stop() or max_cycles reached."""
        self._stop_event.clear()
        init_outcomes = self.initialize(strict_nn_startup=strict_nn_startup)
        if on_outcome is not None:
            for out in init_outcomes:
                on_outcome(out)

        cycles = 0
        while not self._stop_event.is_set():
            if max_cycles is not None and cycles >= max_cycles:
                break

            if not self._next_poll_mono:
                if self._stop_event.wait(timeout=0.5):
                    break
                cycles += 1
                continue

            now_mono = self._monotonic_fn()
            next_due = min(self._next_poll_mono.values())
            wait_sec = next_due - now_mono
            if wait_sec > 0.0:
                # Wait in small slices so Ctrl+C / stop() exits promptly.
                sleep_step = min(wait_sec, 0.5)
                if self._stop_event.wait(timeout=sleep_step):
                    break
                continue

            outcomes = self.poll_due_assets()
            cycles += 1
            if on_outcome is not None:
                for out in outcomes:
                    on_outcome(out)

        return cycles
