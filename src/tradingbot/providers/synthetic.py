"""Deterministic, chunk-invariant synthetic OHLCV candle provider.

Designed for:
- Offline unit and integration tests (zero network dependency).
- Generating learnable synthetic training datasets (multi-scale trends, cycles,
  volatility variation, and volume-range correlation) to validate NeuralNetwork
  convergence before training on live exchange data.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Callable

from tradingbot.config import ProviderConfig
from tradingbot.models import Candle, Timeframe
from tradingbot.providers.base import BaseProvider, ProviderError, default_clock_ms

# Reference UTC timestamp (2024-01-01T00:00:00Z) to keep drift exponents bounded
# for modern epoch timestamps while remaining strictly O(1) per candle.
_REFERENCE_EPOCH_MS = 1_704_067_200_000


def _hash_unit_float(seed: int, key: str, salt: int) -> float:
    """Deterministic uniform float in [0.0, 1.0) derived from BLAKE2b."""
    digest = hashlib.blake2b(
        f"{seed}:{key}:{salt}".encode("utf-8"), digest_size=8
    ).digest()
    val = struct.unpack("<Q", digest)[0]
    return val / float(1 << 64)


def _hash_signed_noise(seed: int, key: str, salt: int) -> float:
    """Deterministic approximate-normal noise in ~[-3, 3] via Irwin-Hall sum."""
    total = sum(_hash_unit_float(seed, key, salt * 16 + i) for i in range(6))
    return (total - 3.0) * math.sqrt(2.0)


class SyntheticProvider(BaseProvider):
    """Generates reproducible OHLCV candles from deterministic multi-cycle + noise dynamics."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        clock_ms: Callable[[], int] = default_clock_ms,
    ) -> None:
        if config.type != "synthetic" or config.synthetic is None:
            raise ProviderError(
                f"SyntheticProvider requires type='synthetic', got {config.type!r}"
            )
        self._config = config
        self._params = config.synthetic
        self._clock_ms = clock_ms

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def max_candles_per_request(self) -> int:
        return self._config.max_candles_per_request

    def _close_price_at_index(self, symbol: str, tf: Timeframe, index: int) -> float:
        series_key = f"{symbol}:{tf.label}"
        seed = self._params.seed
        vol = self._params.volatility
        drift = self._params.drift

        ref_index = (_REFERENCE_EPOCH_MS - tf.anchor_offset_ms) // tf.interval_ms
        rel_k = index - ref_index

        # Deterministic per-series phase offsets so distinct symbols/timeframes differ.
        p1 = _hash_unit_float(seed, series_key, 1) * 2.0 * math.pi
        p2 = _hash_unit_float(seed, series_key, 2) * 2.0 * math.pi
        p3 = _hash_unit_float(seed, series_key, 3) * 2.0 * math.pi

        # Multi-frequency smooth cycles + gentle bounded drift + bar-level noise
        # create learnable momentum/mean-reversion structure while remaining O(1).
        cycle = (
            (3.5 * vol) * math.sin(2.0 * math.pi * rel_k / 24.0 + p1)
            + (7.0 * vol) * math.sin(2.0 * math.pi * rel_k / 96.0 + p2)
            + (12.0 * vol) * math.sin(2.0 * math.pi * rel_k / 360.0 + p3)
        )
        bounded_drift = max(-1.5, min(1.5, drift * rel_k))
        noise = _hash_signed_noise(seed, f"{series_key}:c:{index}", 11) * (0.45 * vol)
        log_ret = max(-3.0, min(3.0, bounded_drift + cycle + noise))
        return max(1e-4, self._params.base_price * math.exp(log_ret))

    def _build_candle(self, symbol: str, tf: Timeframe, open_time_ms: int, now_ms: int) -> Candle:
        idx = (open_time_ms - tf.anchor_offset_ms) // tf.interval_ms
        series_key = f"{symbol}:{tf.label}"
        seed = self._params.seed
        vol = max(self._params.volatility, 1e-4)

        open_price = self._close_price_at_index(symbol, tf, idx - 1)
        close_price = self._close_price_at_index(symbol, tf, idx)

        upper_frac = abs(_hash_signed_noise(seed, f"{series_key}:h:{idx}", 23)) * 0.35 * vol
        lower_frac = abs(_hash_signed_noise(seed, f"{series_key}:l:{idx}", 37)) * 0.35 * vol

        high_price = max(open_price, close_price) * (1.0 + upper_frac)
        low_price = max(1e-6, min(open_price, close_price) * max(0.05, 1.0 - lower_frac))

        body_ratio = abs(close_price - open_price) / (open_price * vol)
        vol_jitter = 0.6 + 0.8 * _hash_unit_float(seed, f"{series_key}:v:{idx}", 53)
        volume = max(0.0, self._params.base_volume * (0.5 + 0.4 * body_ratio) * vol_jitter)

        close_time_ms = open_time_ms + tf.interval_ms - 1
        is_closed = close_time_ms < now_ms

        return Candle(
            provider=self.name,
            symbol=symbol,
            timeframe=tf.label,
            open_time_ms=open_time_ms,
            close_time_ms=close_time_ms,
            open=round(open_price, 6),
            high=round(max(high_price, open_price, close_price), 6),
            low=round(min(low_price, open_price, close_price), 6),
            close=round(close_price, 6),
            volume=round(volume, 4),
            is_closed=is_closed,
        )

    def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe.parse(timeframe)
        now_ms = self._clock_ms()
        effective_limit = (
            min(limit, self.max_candles_per_request)
            if limit is not None
            else self.max_candles_per_request
        )
        if effective_limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        interval = tf.interval_ms
        current_open = tf.align_floor(now_ms)

        if start_ms is None and end_ms is None:
            aligned_end = current_open + interval
            aligned_start = max(0, aligned_end - effective_limit * interval)
        else:
            aligned_end = (
                tf.align_floor(end_ms)
                if end_ms is not None
                else current_open + interval
            )
            aligned_start = (
                tf.align_ceil(start_ms)
                if start_ms is not None
                else max(0, aligned_end - effective_limit * interval)
            )

        # Never generate future candles beyond the currently active bar.
        aligned_end = min(aligned_end, current_open + interval)
        if aligned_start >= aligned_end:
            return []

        candles: list[Candle] = []
        ts = aligned_start
        while ts < aligned_end and len(candles) < effective_limit:
            candles.append(self._build_candle(symbol, tf, ts, now_ms))
            ts += interval
        return candles
