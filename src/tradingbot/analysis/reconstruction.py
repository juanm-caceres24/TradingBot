"""Reconstruct absolute predicted OHLC candles from relative NeuralNetwork outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tradingbot.features import WindowContext
from tradingbot.models import PredictedCandle, Timeframe


def reconstruct_predicted_candles(
    raw_output: Sequence[float],
    window_ctx: WindowContext,
    timeframe: Timeframe | str,
    *,
    base_confidence: float = 0.90,
    horizon_decay: float = 0.08,
    min_confidence: float = 0.20,
) -> tuple[PredictedCandle, ...]:
    """Convert H*4 relative OHLC values into geometrically valid PredictedCandle objects.

    Each future candle k in 1..H is reconstructed using:
        price = max(1e-8, anchor_close + rel_value * scale)
    and projected onto valid OHLC geometry (`high >= max(open, close)` and
    `low <= min(open, close)`). Per-step confidence decays monotonically across
    the forecast horizon so nearer candles render more opaquely than distant ones.
    """
    if len(raw_output) == 0 or len(raw_output) % 4 != 0:
        raise ValueError(
            f"raw_output length must be a positive multiple of 4, got {len(raw_output)}"
        )
    if not all(math.isfinite(float(v)) for v in raw_output):
        raise ValueError("raw_output contains non-finite values")

    tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe.parse(timeframe)
    anchor_close = window_ctx.anchor_close
    scale = window_ctx.scale
    anchor_open_ms = window_ctx.anchor_candle.open_time_ms
    horizon = len(raw_output) // 4

    predicted: list[PredictedCandle] = []
    for idx in range(horizon):
        step = idx + 1
        base = idx * 4
        o_raw = max(1e-8, anchor_close + float(raw_output[base]) * scale)
        h_raw = max(1e-8, anchor_close + float(raw_output[base + 1]) * scale)
        l_raw = max(1e-8, anchor_close + float(raw_output[base + 2]) * scale)
        c_raw = max(1e-8, anchor_close + float(raw_output[base + 3]) * scale)

        high = max(h_raw, l_raw, o_raw, c_raw)
        low = min(h_raw, l_raw, o_raw, c_raw)
        conf = max(
            min_confidence,
            min(1.0, base_confidence * ((1.0 - horizon_decay) ** (step - 1))),
        )

        predicted.append(
            PredictedCandle(
                step=step,
                open_time_ms=anchor_open_ms + step * tf.interval_ms,
                open=round(o_raw, 8),
                high=round(high, 8),
                low=round(low, 8),
                close=round(c_raw, 8),
                confidence=round(conf, 4),
            )
        )

    return tuple(predicted)
