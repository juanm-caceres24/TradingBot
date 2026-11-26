"""Gap detection, interval arithmetic, and sync range planning."""

from __future__ import annotations

from collections.abc import Sequence

from tradingbot.models import TimeRange, Timeframe
from tradingbot.storage.base import CandleStore


def merge_ranges(ranges: Sequence[TimeRange]) -> list[TimeRange]:
    """Sort and coalesce overlapping or adjacent non-empty half-open ranges."""
    non_empty = sorted(r for r in ranges if not r.is_empty)
    if not non_empty:
        return []

    merged: list[TimeRange] = [non_empty[0]]
    for current in non_empty[1:]:
        prev = merged[-1]
        if current.start_ms <= prev.end_ms:
            merged[-1] = TimeRange(prev.start_ms, max(prev.end_ms, current.end_ms))
        else:
            merged.append(current)
    return merged


def subtract_ranges(
    base_ranges: Sequence[TimeRange],
    excluded_ranges: Sequence[TimeRange],
) -> list[TimeRange]:
    """Return disjoint ranges in `base_ranges` not covered by `excluded_ranges`."""
    bases = merge_ranges(base_ranges)
    excludes = merge_ranges(excluded_ranges)
    if not bases or not excludes:
        return bases

    result: list[TimeRange] = []
    exc_idx = 0
    num_exc = len(excludes)

    for base in bases:
        cursor = base.start_ms
        while exc_idx < num_exc and excludes[exc_idx].end_ms <= cursor:
            exc_idx += 1

        j = exc_idx
        while j < num_exc and excludes[j].start_ms < base.end_ms:
            exc = excludes[j]
            if exc.start_ms > cursor:
                result.append(TimeRange(cursor, exc.start_ms))
            cursor = max(cursor, exc.end_ms)
            if cursor >= base.end_ms:
                break
            j += 1

        if cursor < base.end_ms:
            result.append(TimeRange(cursor, base.end_ms))

    return result


def chunk_range(
    time_range: TimeRange,
    timeframe: Timeframe | int,
    max_candles: int,
) -> list[TimeRange]:
    """Split a half-open range into API-friendly paginated sub-ranges."""
    if isinstance(max_candles, bool) or not isinstance(max_candles, int) or max_candles < 1:
        raise ValueError(f"max_candles must be >= 1, got {max_candles!r}")
    interval_ms = timeframe.interval_ms if isinstance(timeframe, Timeframe) else timeframe
    if isinstance(interval_ms, bool) or not isinstance(interval_ms, int) or interval_ms <= 0:
        raise ValueError(f"interval_ms must be > 0, got {interval_ms!r}")

    if time_range.is_empty:
        return []

    step_ms = interval_ms * max_candles
    chunks: list[TimeRange] = []
    cursor = time_range.start_ms
    while cursor < time_range.end_ms:
        chunk_end = min(cursor + step_ms, time_range.end_ms)
        chunks.append(TimeRange(cursor, chunk_end))
        cursor = chunk_end
    return chunks


class GapDetector:
    """Identifies missing candle ranges against a CandleStore."""

    @staticmethod
    def find_missing_ranges(
        store: CandleStore,
        provider: str,
        symbol: str,
        timeframe: Timeframe | str,
        start_ms: int,
        end_ms: int,
        *,
        respect_verified_ranges: bool = True,
    ) -> list[TimeRange]:
        """Find missing closed-candle ranges inside [align_ceil(start_ms), align_floor(end_ms)).

        Unclosed candles in the store are treated as incomplete so restarts recover
        their finalized OHLCV values cleanly.
        """
        tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe.parse(timeframe)
        if end_ms <= start_ms:
            return []

        aligned_start = tf.align_ceil(start_ms)
        aligned_end = tf.align_floor(end_ms)
        if aligned_start >= aligned_end:
            return []

        existing_times = store.get_open_times(
            provider,
            symbol,
            tf.label,
            start_ms=aligned_start,
            end_ms=aligned_end,
            closed_only=True,
        )

        missing: list[TimeRange] = []
        cursor = aligned_start
        interval = tf.interval_ms

        for ts in existing_times:
            # Guard against stray unaligned timestamps if injected manually.
            if not tf.is_aligned(ts):
                continue
            if ts > cursor:
                missing.append(TimeRange(cursor, ts))
            if ts + interval > cursor:
                cursor = ts + interval

        if cursor < aligned_end:
            missing.append(TimeRange(cursor, aligned_end))

        if respect_verified_ranges and missing:
            verified = store.get_sync_ranges(
                provider,
                symbol,
                tf.label,
                start_ms=aligned_start,
                end_ms=aligned_end,
            )
            missing = subtract_ranges(missing, verified)

        return merge_ranges(missing)

    @staticmethod
    def plan_sync_ranges(
        store: CandleStore,
        provider: str,
        symbol: str,
        timeframe: Timeframe | str,
        now_ms: int,
        *,
        initial_history_from_ms: int | None = None,
        min_candles_when_empty: int = 100,
        fill_internal_gaps: bool = True,
        respect_verified_ranges: bool = True,
    ) -> list[TimeRange]:
        """Compute missing closed-candle ranges needed to bring storage up to `now_ms`.

        Scenarios handled:
        1. Empty DB with `initial_history_from_ms`: fetch [align_ceil(initial), latest_closed_end).
        2. Empty DB without `initial_history_from_ms`: fetch last `min_candles_when_empty` candles.
        3. Populated DB with `fill_internal_gaps=True`: repair any head gap (if
           `initial_history_from_ms` precedes stored history), all internal gaps, and the tail gap.
        4. Populated DB with `fill_internal_gaps=False`: fetch only the tail gap after the
           latest closed candle up to `latest_closed_end`.
        """
        tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe.parse(timeframe)
        if isinstance(min_candles_when_empty, bool) or not isinstance(min_candles_when_empty, int) or min_candles_when_empty < 1:
            raise ValueError(f"min_candles_when_empty must be >= 1, got {min_candles_when_empty!r}")

        # Latest closed candle ends at align_floor(now_ms), i.e. the open_time of the
        # currently active candle.
        latest_closed_end = tf.align_floor(now_ms)
        bounds = store.get_bounds(provider, symbol, tf.label, closed_only=True)

        if bounds is None:
            if initial_history_from_ms is not None:
                target_start = tf.align_ceil(initial_history_from_ms)
            else:
                target_start = max(0, latest_closed_end - min_candles_when_empty * tf.interval_ms)
            return GapDetector.find_missing_ranges(
                store,
                provider,
                symbol,
                tf,
                target_start,
                latest_closed_end,
                respect_verified_ranges=respect_verified_ranges,
            )

        min_open, max_open = bounds
        if not fill_internal_gaps:
            tail_start = tf.next_open(max_open)
            return GapDetector.find_missing_ranges(
                store,
                provider,
                symbol,
                tf,
                tail_start,
                latest_closed_end,
                respect_verified_ranges=respect_verified_ranges,
            )

        target_start = min_open
        if initial_history_from_ms is not None:
            aligned_initial = tf.align_ceil(initial_history_from_ms)
            target_start = min(target_start, aligned_initial)

        return GapDetector.find_missing_ranges(
            store,
            provider,
            symbol,
            tf,
            target_start,
            latest_closed_end,
            respect_verified_ranges=respect_verified_ranges,
        )
