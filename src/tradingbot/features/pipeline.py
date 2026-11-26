"""Window-relative feature extraction and target encoding for NeuralNetwork.

Enforces two critical invariants:
1. Never feeds raw non-stationary prices to the MLP (all price features are
   normalized relative to the anchor candle close and local ATR or % scale).
2. Strictly excludes warm-up candles or any window containing missing/non-finite
   indicator values from both training datasets and live inference vectors.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from tradingbot.config import AssetConfig
from tradingbot.features.schema import (
    SCHEMA_VERSION,
    FeatureSchema,
    compute_feature_hash,
)
from tradingbot.indicators import ATRIndicator, IndicatorEngine
from tradingbot.models import Candle

BASE_BAR_FEATURES = ("open_rel", "high_rel", "low_rel", "close_rel", "vol_rel")
SIGNAL_TARGET_COLUMNS = ("hold", "long", "short")


class InsufficientDataError(ValueError):
    """Raised when not enough complete post-warmup candles exist for a window."""


@dataclass(frozen=True)
class WindowContext:
    """Normalized feature vector plus anchor metadata required for candle reconstruction."""

    anchor_index: int
    anchor_candle: Candle
    anchor_close: float
    scale: float
    vector: list[float]


def _normalize_indicator_value(
    value: float,
    scale_kind: str,
    anchor_close: float,
    scale: float,
) -> float:
    if scale_kind == "price_level":
        return (value - anchor_close) / scale
    if scale_kind == "price_delta":
        return value / scale
    if scale_kind == "rsi_0_100":
        return (value - 50.0) / 50.0
    if scale_kind == "bounded":
        return value
    raise ValueError(f"Unsupported indicator nn_scale_kind: {scale_kind!r}")


class FeaturePipeline:
    """Shared feature & target builder used identically by offline export and live inference."""

    def __init__(self, asset: AssetConfig) -> None:
        self._asset = asset
        self._features_cfg = asset.features
        self._engine = IndicatorEngine(asset.indicators)
        self._nn_indicators = self._engine.nn_indicators()
        self._atr_indicator = (
            ATRIndicator("_norm_atr", {"period": self._features_cfg.atr_period})
            if self._features_cfg.normalization == "atr_relative"
            else None
        )

    @property
    def asset(self) -> AssetConfig:
        return self._asset

    @property
    def indicator_engine(self) -> IndicatorEngine:
        return self._engine

    @property
    def max_warmup(self) -> int:
        ind_warmup = self._engine.max_warmup(nn_only=True)
        norm_warmup = self._atr_indicator.warmup_period() if self._atr_indicator is not None else 0
        return max(ind_warmup, norm_warmup)

    @property
    def min_candles_for_inference(self) -> int:
        return self.max_warmup + self._features_cfg.input_window

    @property
    def min_candles_for_training(self) -> int:
        return self.min_candles_for_inference + self._features_cfg.forecast_horizon

    def per_bar_feature_names(self) -> tuple[str, ...]:
        names = list(BASE_BAR_FEATURES)
        for ind in self._nn_indicators:
            for col in ind.output_columns():
                names.append(f"{ind.indicator_id}.{col}")
        return tuple(names)

    def input_column_names(self) -> tuple[str, ...]:
        per_bar = self.per_bar_feature_names()
        w = self._features_cfg.input_window
        cols: list[str] = []
        for offset in range(w - 1, -1, -1):
            prefix = "t" if offset == 0 else f"t-{offset}"
            for feat in per_bar:
                cols.append(f"{prefix}_{feat}")
        return tuple(cols)

    def target_column_names(self, model_kind: str) -> tuple[str, ...]:
        if model_kind == "candles":
            cols: list[str] = []
            for step in range(1, self._features_cfg.forecast_horizon + 1):
                cols.extend(
                    [
                        f"t+{step}_open_rel",
                        f"t+{step}_high_rel",
                        f"t+{step}_low_rel",
                        f"t+{step}_close_rel",
                    ]
                )
            return tuple(cols)
        if model_kind == "signal":
            return SIGNAL_TARGET_COLUMNS
        raise ValueError(f"Unsupported model_kind {model_kind!r}. Expected 'candles' or 'signal'.")

    def build_schema(self, model_kind: str) -> FeatureSchema:
        per_bar = self.per_bar_feature_names()
        targets = self.target_column_names(model_kind)
        nn_ind_specs = tuple(
            {"id": cfg.id, "type": cfg.type, "params": dict(cfg.params)}
            for cfg in self._asset.indicators
            if cfg.use_in_nn
        )
        tb_dict = (
            {
                "take_profit_pct": self._features_cfg.triple_barrier.take_profit_pct,
                "stop_loss_pct": self._features_cfg.triple_barrier.stop_loss_pct,
            }
            if model_kind == "signal"
            else None
        )
        feature_hash = compute_feature_hash(
            model_kind=model_kind,
            input_window=self._features_cfg.input_window,
            forecast_horizon=self._features_cfg.forecast_horizon,
            normalization=self._features_cfg.normalization,
            atr_period=self._features_cfg.atr_period,
            per_bar_features=per_bar,
            target_columns=targets,
            indicators=nn_ind_specs,
            triple_barrier=tb_dict,
        )
        return FeatureSchema(
            schema_version=SCHEMA_VERSION,
            asset_id=self._asset.id,
            provider=self._asset.provider.name,
            symbol=self._asset.market.symbol,
            timeframe=self._asset.market.timeframe,
            model_kind=model_kind,
            input_window=self._features_cfg.input_window,
            forecast_horizon=self._features_cfg.forecast_horizon,
            normalization=self._features_cfg.normalization,
            atr_period=self._features_cfg.atr_period,
            max_warmup=self.max_warmup,
            input_size=len(per_bar) * self._features_cfg.input_window,
            output_size=len(targets),
            per_bar_features=per_bar,
            target_columns=targets,
            indicators=nn_ind_specs,
            triple_barrier=tb_dict,
            feature_hash=feature_hash,
        )

    def _compute_norm_atr(self, candles: Sequence[Candle]) -> list[float | None] | None:
        if self._atr_indicator is None:
            return None
        return self._atr_indicator.compute(candles)["value"]

    def build_input_window(
        self,
        candles: Sequence[Candle],
        anchor_index: int,
        *,
        precomputed_indicators: dict[str, dict[str, list[float | None]]] | None = None,
        precomputed_atr: list[float | None] | None = None,
    ) -> WindowContext | None:
        """Build the normalized input vector for window [anchor_index - W + 1 .. anchor_index].

        Returns None if the window overlaps indicator warm-up or contains any missing/non-finite value.
        """
        w = self._features_cfg.input_window
        start_idx = anchor_index - w + 1
        if start_idx < self.max_warmup or anchor_index >= len(candles):
            return None

        ind_data = (
            precomputed_indicators
            if precomputed_indicators is not None
            else self._engine.compute_all(candles, nn_only=True)
        )
        atr_series = (
            precomputed_atr
            if precomputed_atr is not None
            else self._compute_norm_atr(candles)
        )

        anchor = candles[anchor_index]
        anchor_close = anchor.close
        if self._features_cfg.normalization == "atr_relative":
            assert atr_series is not None
            raw_atr = atr_series[anchor_index]
            if raw_atr is None or not math.isfinite(raw_atr):
                return None
            scale = max(raw_atr, anchor_close * 1e-6)
        else:
            scale = anchor_close

        window_candles = candles[start_idx : anchor_index + 1]
        if self._asset.market.market_schedule == "24/7":
            interval_ms = self._asset.market.parsed_timeframe.interval_ms
            for prev_c, curr_c in zip(window_candles, window_candles[1:]):
                if curr_c.open_time_ms - prev_c.open_time_ms != interval_ms:
                    return None
        vol_mean = sum(c.volume for c in window_candles) / float(w)
        vol_denom = max(vol_mean, 1e-9)

        vec: list[float] = []
        for idx in range(start_idx, anchor_index + 1):
            c = candles[idx]
            vec.append((c.open - anchor_close) / scale)
            vec.append((c.high - anchor_close) / scale)
            vec.append((c.low - anchor_close) / scale)
            vec.append((c.close - anchor_close) / scale)
            vec.append(math.log1p(c.volume / vol_denom))

            for ind in self._nn_indicators:
                series_map = ind_data[ind.indicator_id]
                for col in ind.output_columns():
                    raw_val = series_map[col][idx]
                    if raw_val is None or not math.isfinite(raw_val):
                        return None
                    norm_val = _normalize_indicator_value(
                        raw_val,
                        ind.nn_scale_kind(col),
                        anchor_close,
                        scale,
                    )
                    if not math.isfinite(norm_val):
                        return None
                    vec.append(norm_val)

        return WindowContext(
            anchor_index=anchor_index,
            anchor_candle=anchor,
            anchor_close=anchor_close,
            scale=scale,
            vector=vec,
        )

    def build_latest_inference_window(self, candles: Sequence[Candle]) -> WindowContext:
        """Build the normalized input vector ending at the latest candle in `candles`."""
        if len(candles) < self.min_candles_for_inference:
            raise InsufficientDataError(
                f"Asset '{self._asset.id}' requires at least {self.min_candles_for_inference} "
                f"candles (warmup={self.max_warmup} + window={self._features_cfg.input_window}), "
                f"got {len(candles)}"
            )
        ctx = self.build_input_window(candles, len(candles) - 1)
        if ctx is None:
            raise InsufficientDataError(
                f"Asset '{self._asset.id}' has incomplete indicator values in the latest window"
            )
        return ctx

    def encode_candle_targets(
        self,
        candles: Sequence[Candle],
        anchor_index: int,
        anchor_close: float,
        scale: float,
    ) -> list[float]:
        """Encode the next H future candles as relative OHLC targets."""
        h = self._features_cfg.forecast_horizon
        targets: list[float] = []
        for offset in range(1, h + 1):
            fc = candles[anchor_index + offset]
            targets.append((fc.open - anchor_close) / scale)
            targets.append((fc.high - anchor_close) / scale)
            targets.append((fc.low - anchor_close) / scale)
            targets.append((fc.close - anchor_close) / scale)
        return targets

    def encode_signal_target(
        self,
        candles: Sequence[Candle],
        anchor_index: int,
        anchor_close: float,
    ) -> list[float]:
        """Triple Barrier Labeling over the next H future candles -> [HOLD, LONG, SHORT].

        Conservative resolution: if both a side's take-profit and stop-loss are
        touched within the exact same candle, that side is invalidated rather than
        assumed profitable.
        """
        h = self._features_cfg.forecast_horizon
        tp_pct = self._features_cfg.triple_barrier.take_profit_pct
        sl_pct = self._features_cfg.triple_barrier.stop_loss_pct

        tp_long = anchor_close * (1.0 + tp_pct)
        sl_long = anchor_close * (1.0 - sl_pct)
        tp_short = anchor_close * (1.0 - tp_pct)
        sl_short = anchor_close * (1.0 + sl_pct)

        long_alive = True
        short_alive = True

        for offset in range(1, h + 1):
            fc = candles[anchor_index + offset]
            long_won = False
            short_won = False

            if long_alive:
                hit_sl = fc.low <= sl_long
                hit_tp = fc.high >= tp_long
                if hit_sl:
                    long_alive = False
                elif hit_tp:
                    long_won = True

            if short_alive:
                hit_sl = fc.high >= sl_short
                hit_tp = fc.low <= tp_short
                if hit_sl:
                    short_alive = False
                elif hit_tp:
                    short_won = True

            if long_won and short_won:
                return [1.0, 0.0, 0.0]
            if long_won:
                return [0.0, 1.0, 0.0]
            if short_won:
                return [0.0, 0.0, 1.0]
            if not long_alive and not short_alive:
                return [1.0, 0.0, 0.0]

        return [1.0, 0.0, 0.0]

    def iter_training_samples(
        self,
        candles: Sequence[Candle],
        model_kind: str,
    ) -> Iterator[tuple[int, list[float], list[float]]]:
        """Yield (anchor_open_time_ms, input_vec, target_vec) across valid sliding windows."""
        if model_kind not in ("candles", "signal"):
            raise ValueError(f"Unsupported model_kind {model_kind!r}")

        n = len(candles)
        if n < self.min_candles_for_training:
            return

        ind_data = self._engine.compute_all(candles, nn_only=True)
        atr_series = self._compute_norm_atr(candles)

        first_anchor = self.max_warmup + self._features_cfg.input_window - 1
        last_anchor_exclusive = n - self._features_cfg.forecast_horizon
        stride = self._features_cfg.stride
        enforce_contiguous = self._asset.market.market_schedule == "24/7"
        interval_ms = self._asset.market.parsed_timeframe.interval_ms
        horizon = self._features_cfg.forecast_horizon

        for anchor_idx in range(first_anchor, last_anchor_exclusive, stride):
            if enforce_contiguous:
                anchor_ts = candles[anchor_idx].open_time_ms
                if any(
                    candles[anchor_idx + step].open_time_ms != anchor_ts + step * interval_ms
                    for step in range(1, horizon + 1)
                ):
                    continue

            ctx = self.build_input_window(
                candles,
                anchor_idx,
                precomputed_indicators=ind_data,
                precomputed_atr=atr_series,
            )
            if ctx is None:
                continue

            if model_kind == "candles":
                target = self.encode_candle_targets(
                    candles, anchor_idx, ctx.anchor_close, ctx.scale
                )
            else:
                target = self.encode_signal_target(candles, anchor_idx, ctx.anchor_close)

            if not all(math.isfinite(v) for v in target):
                continue

            yield (ctx.anchor_candle.open_time_ms, ctx.vector, target)
