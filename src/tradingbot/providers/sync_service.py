"""Multi-asset historical synchronization and gap-repair service."""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from tradingbot.config import AssetConfig, ProviderConfig
from tradingbot.models import Candle, TimeRange, Timeframe
from tradingbot.providers.base import BaseProvider, default_clock_ms
from tradingbot.providers.factory import create_provider
from tradingbot.storage.base import CandleStore
from tradingbot.storage.gap_detector import GapDetector, chunk_range, subtract_ranges


@dataclass(frozen=True)
class SyncReport:
    """Summary metrics produced by a backfill or sync-to-now operation."""

    asset_id: str
    provider: str
    symbol: str
    timeframe: str
    planned_ranges: tuple[TimeRange, ...]
    chunks_fetched: int
    candles_upserted: int
    empty_ranges_recorded: int
    duration_ms: int


ProgressCallback = Callable[[str, int, int, TimeRange, int], None]


class SyncService:
    """Coordinates GapDetector, market data providers, and CandleStore persistence."""

    def __init__(
        self,
        store: CandleStore,
        *,
        provider_factory: Callable[..., BaseProvider] = create_provider,
        clock_ms: Callable[[], int] = default_clock_ms,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._provider_factory = provider_factory
        self._clock_ms = clock_ms
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        # Cache provider instances by config so multiple assets sharing one exchange
        # (e.g. btc_1h and btc_1w on binance_spot) share its rate-limit throttle.
        self._providers: dict[str, BaseProvider] = {}

    def get_provider(self, config: ProviderConfig) -> BaseProvider:
        key = json.dumps(dataclasses.asdict(config), sort_keys=True, separators=(",", ":"))
        provider = self._providers.get(key)
        if provider is None:
            provider = self._provider_factory(
                config,
                clock_ms=self._clock_ms,
                sleep_fn=self._sleep_fn,
                monotonic_fn=self._monotonic_fn,
            )
            self._providers[key] = provider
        return provider

    @staticmethod
    def default_min_candles_for_asset(asset: AssetConfig) -> int:
        """Compute a sensible initial candle count when the DB has no history yet."""
        max_indicator_hint = 0
        for ind in asset.indicators:
            for val in ind.params.values():
                if isinstance(val, int) and not isinstance(val, bool) and val > max_indicator_hint:
                    max_indicator_hint = val
        return max(100, max_indicator_hint + asset.features.input_window + asset.features.forecast_horizon + 50)

    def plan_backfill(
        self,
        asset: AssetConfig,
        *,
        start_ms: int,
        end_ms: int,
        respect_verified_ranges: bool = True,
    ) -> list[TimeRange]:
        """Compute missing closed-candle ranges inside [start_ms, end_ms)."""
        return GapDetector.find_missing_ranges(
            self._store,
            asset.provider.name,
            asset.market.symbol,
            asset.market.timeframe,
            start_ms,
            end_ms,
            respect_verified_ranges=respect_verified_ranges,
        )

    def plan_sync_to_now(
        self,
        asset: AssetConfig,
        *,
        now_ms: int | None = None,
        min_candles_when_empty: int | None = None,
        respect_verified_ranges: bool = True,
    ) -> list[TimeRange]:
        """Compute missing closed-candle ranges needed to bring an asset up to date."""
        effective_now = self._clock_ms() if now_ms is None else now_ms
        min_bars = (
            min_candles_when_empty
            if min_candles_when_empty is not None
            else self.default_min_candles_for_asset(asset)
        )
        return GapDetector.plan_sync_ranges(
            self._store,
            asset.provider.name,
            asset.market.symbol,
            asset.market.timeframe,
            effective_now,
            initial_history_from_ms=asset.runtime.initial_history_from_ms(),
            min_candles_when_empty=min_bars,
            fill_internal_gaps=asset.runtime.fill_internal_gaps,
            respect_verified_ranges=respect_verified_ranges,
        )

    def _record_empty_subranges(
        self,
        asset: AssetConfig,
        tf: Timeframe,
        chunk: TimeRange,
        fetched: list[Candle],
        *,
        max_candles: int,
        now_ms: int,
    ) -> int:
        """Record empty historical intervals so non-24/7 or maintenance gaps are not re-polled."""
        latest_closed_end = tf.align_floor(now_ms)
        if chunk.end_ms > latest_closed_end:
            return 0

        if not fetched:
            self._store.record_sync_range(
                asset.provider.name,
                asset.market.symbol,
                tf.label,
                chunk,
                checked_at_ms=now_ms,
            )
            return 1

        # Only infer empty sub-gaps inside a non-empty chunk if the response was NOT
        # truncated by the provider's page limit.
        if len(fetched) >= max_candles:
            return 0

        covered = [
            TimeRange(c.open_time_ms, c.open_time_ms + tf.interval_ms)
            for c in fetched
            if c.is_closed and tf.is_aligned(c.open_time_ms)
        ]
        empty_subranges = subtract_ranges([chunk], covered)
        for empty_range in empty_subranges:
            self._store.record_sync_range(
                asset.provider.name,
                asset.market.symbol,
                tf.label,
                empty_range,
                checked_at_ms=now_ms,
            )
        return len(empty_subranges)

    def _execute_ranges(
        self,
        asset: AssetConfig,
        ranges: list[TimeRange],
        *,
        now_ms: int,
        persist: bool,
        dry_run: bool,
        progress_cb: ProgressCallback | None,
    ) -> SyncReport:
        t0 = self._monotonic_fn()
        tf = asset.market.parsed_timeframe
        provider = self.get_provider(asset.provider)

        all_chunks: list[TimeRange] = []
        for missing_range in ranges:
            all_chunks.extend(chunk_range(missing_range, tf, provider.max_candles_per_request))

        if dry_run or not all_chunks:
            elapsed_ms = int((self._monotonic_fn() - t0) * 1000)
            return SyncReport(
                asset_id=asset.id,
                provider=provider.name,
                symbol=asset.market.symbol,
                timeframe=tf.label,
                planned_ranges=tuple(ranges),
                chunks_fetched=0,
                candles_upserted=0,
                empty_ranges_recorded=0,
                duration_ms=max(0, elapsed_ms),
            )

        chunks_fetched = 0
        candles_upserted = 0
        empty_recorded = 0
        total_chunks = len(all_chunks)

        for idx, chunk in enumerate(all_chunks, start=1):
            fetched = provider.fetch_candles(
                asset.market.symbol,
                tf,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                limit=provider.max_candles_per_request,
            )
            chunks_fetched += 1

            if persist:
                if fetched:
                    candles_upserted += self._store.upsert_candles(fetched)
                empty_recorded += self._record_empty_subranges(
                    asset,
                    tf,
                    chunk,
                    fetched,
                    max_candles=provider.max_candles_per_request,
                    now_ms=now_ms,
                )

            if progress_cb is not None:
                progress_cb(asset.id, idx, total_chunks, chunk, len(fetched))

        elapsed_ms = int((self._monotonic_fn() - t0) * 1000)
        return SyncReport(
            asset_id=asset.id,
            provider=provider.name,
            symbol=asset.market.symbol,
            timeframe=tf.label,
            planned_ranges=tuple(ranges),
            chunks_fetched=chunks_fetched,
            candles_upserted=candles_upserted,
            empty_ranges_recorded=empty_recorded,
            duration_ms=max(0, elapsed_ms),
        )

    def backfill_asset(
        self,
        asset: AssetConfig,
        *,
        start_ms: int,
        end_ms: int,
        respect_verified_ranges: bool = True,
        dry_run: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> SyncReport:
        """Backfill missing closed candles for `asset` strictly within [start_ms, end_ms)."""
        now_ms = self._clock_ms()
        capped_end = min(end_ms, asset.market.parsed_timeframe.align_floor(now_ms))
        ranges = self.plan_backfill(
            asset,
            start_ms=start_ms,
            end_ms=capped_end,
            respect_verified_ranges=respect_verified_ranges,
        )
        return self._execute_ranges(
            asset,
            ranges,
            now_ms=now_ms,
            persist=True,
            dry_run=dry_run,
            progress_cb=progress_cb,
        )

    def sync_asset_to_now(
        self,
        asset: AssetConfig,
        *,
        now_ms: int | None = None,
        min_candles_when_empty: int | None = None,
        respect_verified_ranges: bool = True,
        include_open_candle: bool = False,
        dry_run: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> SyncReport:
        """Detect and repair all missing history up to `now_ms`, optionally fetching the active candle."""
        effective_now = self._clock_ms() if now_ms is None else now_ms
        ranges = self.plan_sync_to_now(
            asset,
            now_ms=effective_now,
            min_candles_when_empty=min_candles_when_empty,
            respect_verified_ranges=respect_verified_ranges,
        )
        report = self._execute_ranges(
            asset,
            ranges,
            now_ms=effective_now,
            persist=asset.runtime.persist_to_db,
            dry_run=dry_run,
            progress_cb=progress_cb,
        )

        if include_open_candle and not dry_run:
            latest = self.fetch_latest_candles(asset, limit=2)
            upserted = report.candles_upserted + (
                len(latest) if asset.runtime.persist_to_db else 0
            )
            return SyncReport(
                asset_id=report.asset_id,
                provider=report.provider,
                symbol=report.symbol,
                timeframe=report.timeframe,
                planned_ranges=report.planned_ranges,
                chunks_fetched=report.chunks_fetched + 1,
                candles_upserted=upserted,
                empty_ranges_recorded=report.empty_ranges_recorded,
                duration_ms=report.duration_ms,
            )

        return report

    def fetch_latest_candles(
        self,
        asset: AssetConfig,
        *,
        limit: int = 2,
        persist: bool | None = None,
    ) -> list[Candle]:
        """Fetch the most recent candles (closed + active) for live polling."""
        provider = self.get_provider(asset.provider)
        candles = provider.fetch_candles(
            asset.market.symbol,
            asset.market.parsed_timeframe,
            limit=limit,
        )
        should_persist = asset.runtime.persist_to_db if persist is None else persist
        if should_persist and candles:
            self._store.upsert_candles(candles)
        return candles
