"""Built-in transparent technical indicators (zero external dependencies).

Included indicators:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average, SMA-seeded)
- RSI (Wilder's Smoothed Relative Strength Index)
- ATR (Wilder's Average True Range)
- BollingerBands (SMA +/- k * population standard deviation)
- SqueezeMomentum (LazyBear-style Bollinger vs. Keltner squeeze + linear-regression momentum)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tradingbot.indicators.base import (
    BaseIndicator,
    IndicatorError,
    reject_unknown_params,
    require_float_param,
    require_int_param,
)
from tradingbot.models import Candle


def _compute_sma_values(values: Sequence[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def _compute_true_range(candles: Sequence[Candle]) -> list[float]:
    n = len(candles)
    if n == 0:
        return []
    tr = [0.0] * n
    tr[0] = candles[0].high - candles[0].low
    for i in range(1, n):
        prev_c = candles[i - 1].close
        hi = candles[i].high
        lo = candles[i].low
        tr[i] = max(hi - lo, abs(hi - prev_c), abs(lo - prev_c))
    return tr


def _linear_regression_endpoint(window: Sequence[float]) -> float:
    """Evaluate least-squares linear regression fitted over x = 0..m-1 at x = m - 1."""
    m = len(window)
    if m == 1:
        return float(window[0])
    # Closed-form sums for x = 0, 1, ..., m - 1
    sum_x = (m - 1) * m / 2.0
    sum_x2 = (m - 1) * m * (2 * m - 1) / 6.0
    sum_y = 0.0
    sum_xy = 0.0
    for x, y in enumerate(window):
        sum_y += y
        sum_xy += x * y
    denom = m * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return sum_y / m
    slope = (m * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / m
    return intercept + slope * (m - 1)


class SMAIndicator(BaseIndicator):
    indicator_type = "sma"

    def _validate_params(self) -> None:
        reject_unknown_params(self.params, frozenset({"period"}), self.indicator_type)
        self.period = require_int_param(self.params, "period", min_value=1)

    def warmup_period(self) -> int:
        return self.period - 1

    def output_columns(self) -> tuple[str, ...]:
        return ("value",)

    def nn_scale_kind(self, column: str) -> str:
        if column != "value":
            raise IndicatorError(f"Unknown column '{column}' for SMA")
        return "price_level"

    def compute(self, candles: Sequence[Candle]) -> dict[str, list[float | None]]:
        closes = [c.close for c in candles]
        return {"value": _compute_sma_values(closes, self.period)}


class EMAIndicator(BaseIndicator):
    indicator_type = "ema"

    def _validate_params(self) -> None:
        reject_unknown_params(self.params, frozenset({"period"}), self.indicator_type)
        self.period = require_int_param(self.params, "period", min_value=1)

    def warmup_period(self) -> int:
        return self.period - 1

    def output_columns(self) -> tuple[str, ...]:
        return ("value",)

    def nn_scale_kind(self, column: str) -> str:
        if column != "value":
            raise IndicatorError(f"Unknown column '{column}' for EMA")
        return "price_level"

    def compute(self, candles: Sequence[Candle]) -> dict[str, list[float | None]]:
        n = len(candles)
        out: list[float | None] = [None] * n
        if n < self.period:
            return {"value": out}

        alpha = 2.0 / (self.period + 1.0)
        seed = sum(candles[i].close for i in range(self.period)) / self.period
        out[self.period - 1] = seed
        prev = seed
        for i in range(self.period, n):
            prev = alpha * candles[i].close + (1.0 - alpha) * prev
            out[i] = prev
        return {"value": out}


class RSIIndicator(BaseIndicator):
    """Wilder's smoothed Relative Strength Index in [0, 100]."""

    indicator_type = "rsi"

    def _validate_params(self) -> None:
        reject_unknown_params(self.params, frozenset({"period"}), self.indicator_type)
        self.period = require_int_param(self.params, "period", min_value=1)

    def warmup_period(self) -> int:
        return self.period

    def output_columns(self) -> tuple[str, ...]:
        return ("value",)

    def nn_scale_kind(self, column: str) -> str:
        if column != "value":
            raise IndicatorError(f"Unknown column '{column}' for RSI")
        return "rsi_0_100"

    @staticmethod
    def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0.0 and avg_gain == 0.0:
            return 50.0
        if avg_loss == 0.0:
            return 100.0
        if avg_gain == 0.0:
            return 0.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def compute(self, candles: Sequence[Candle]) -> dict[str, list[float | None]]:
        n = len(candles)
        out: list[float | None] = [None] * n
        p = self.period
        if n <= p:
            return {"value": out}

        gain_sum = 0.0
        loss_sum = 0.0
        for i in range(1, p + 1):
            delta = candles[i].close - candles[i - 1].close
            if delta > 0.0:
                gain_sum += delta
            elif delta < 0.0:
                loss_sum -= delta

        avg_gain = gain_sum / p
        avg_loss = loss_sum / p
        out[p] = self._rsi_from_avgs(avg_gain, avg_loss)

        for i in range(p + 1, n):
            delta = candles[i].close - candles[i - 1].close
            gain = delta if delta > 0.0 else 0.0
            loss = -delta if delta < 0.0 else 0.0
            avg_gain = (avg_gain * (p - 1) + gain) / p
            avg_loss = (avg_loss * (p - 1) + loss) / p
            out[i] = self._rsi_from_avgs(avg_gain, avg_loss)

        return {"value": out}


class ATRIndicator(BaseIndicator):
    """Wilder's Average True Range seeded over True Ranges [1..period]."""

    indicator_type = "atr"

    def _validate_params(self) -> None:
        reject_unknown_params(self.params, frozenset({"period"}), self.indicator_type)
        self.period = require_int_param(self.params, "period", min_value=1)

    def warmup_period(self) -> int:
        return self.period

    def output_columns(self) -> tuple[str, ...]:
        return ("value",)

    def nn_scale_kind(self, column: str) -> str:
        if column != "value":
            raise IndicatorError(f"Unknown column '{column}' for ATR")
        return "price_delta"

    def compute(self, candles: Sequence[Candle]) -> dict[str, list[float | None]]:
        n = len(candles)
        out: list[float | None] = [None] * n
        p = self.period
        if n <= p:
            return {"value": out}

        tr = _compute_true_range(candles)
        atr = sum(tr[1 : p + 1]) / p
        out[p] = atr
        for i in range(p + 1, n):
            atr = (atr * (p - 1) + tr[i]) / p
            out[i] = atr
        return {"value": out}


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands (middle SMA, upper, lower using population std dev)."""

    indicator_type = "bollinger"

    def _validate_params(self) -> None:
        reject_unknown_params(self.params, frozenset({"period", "mult"}), self.indicator_type)
        self.period = require_int_param(self.params, "period", min_value=1)
        self.mult = require_float_param(self.params, "mult", exclusive_min=0.0, default=2.0)

    def warmup_period(self) -> int:
        return self.period - 1

    def output_columns(self) -> tuple[str, ...]:
        return ("middle", "upper", "lower")

    def nn_scale_kind(self, column: str) -> str:
        if column not in ("middle", "upper", "lower"):
            raise IndicatorError(f"Unknown column '{column}' for BollingerBands")
        return "price_level"

    def compute(self, candles: Sequence[Candle]) -> dict[str, list[float | None]]:
        n = len(candles)
        middle: list[float | None] = [None] * n
        upper: list[float | None] = [None] * n
        lower: list[float | None] = [None] * n
        p = self.period
        if n < p:
            return {"middle": middle, "upper": upper, "lower": lower}

        closes = [c.close for c in candles]
        for i in range(p - 1, n):
            window = closes[i - p + 1 : i + 1]
            mean = sum(window) / p
            variance = sum((x - mean) ** 2 for x in window) / p
            std = math.sqrt(max(0.0, variance))
            middle[i] = mean
            upper[i] = mean + self.mult * std
            lower[i] = mean - self.mult * std

        return {"middle": middle, "upper": upper, "lower": lower}


class SqueezeMomentumIndicator(BaseIndicator):
    """LazyBear-style Squeeze Momentum Indicator.

    Outputs:
    - `momentum`: Linear regression endpoint (length `mom_length`) of
      `close - avg(avg(highest_high, lowest_low), sma(close))`.
    - `squeeze_state`: +1.0 when Bollinger Bands sit strictly inside Keltner Channels
      (squeeze ON), -1.0 when Bollinger Bands expand outside Keltner Channels
      (squeeze OFF), 0.0 otherwise.
    """

    indicator_type = "squeeze_momentum"

    def _validate_params(self) -> None:
        reject_unknown_params(
            self.params,
            frozenset({"bb_length", "bb_mult", "kc_length", "kc_mult", "mom_length"}),
            self.indicator_type,
        )
        self.bb_length = require_int_param(self.params, "bb_length", min_value=1, default=20)
        self.bb_mult = require_float_param(self.params, "bb_mult", exclusive_min=0.0, default=2.0)
        self.kc_length = require_int_param(self.params, "kc_length", min_value=1, default=20)
        self.kc_mult = require_float_param(self.params, "kc_mult", exclusive_min=0.0, default=1.5)
        self.mom_length = require_int_param(self.params, "mom_length", min_value=1, default=20)

    def warmup_period(self) -> int:
        return max(self.bb_length - 1, self.kc_length - 1, 2 * self.mom_length - 2)

    def output_columns(self) -> tuple[str, ...]:
        return ("momentum", "squeeze_state")

    def nn_scale_kind(self, column: str) -> str:
        if column == "momentum":
            return "price_delta"
        if column == "squeeze_state":
            return "bounded"
        raise IndicatorError(f"Unknown column '{column}' for SqueezeMomentum")

    def compute(self, candles: Sequence[Candle]) -> dict[str, list[float | None]]:
        n = len(candles)
        momentum: list[float | None] = [None] * n
        squeeze_state: list[float | None] = [None] * n
        if n == 0:
            return {"momentum": momentum, "squeeze_state": squeeze_state}

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        tr = _compute_true_range(candles)

        # Bollinger Bands
        bb_upper: list[float | None] = [None] * n
        bb_lower: list[float | None] = [None] * n
        bb_len = self.bb_length
        for i in range(bb_len - 1, n):
            win = closes[i - bb_len + 1 : i + 1]
            mean = sum(win) / bb_len
            std = math.sqrt(max(0.0, sum((x - mean) ** 2 for x in win) / bb_len))
            bb_upper[i] = mean + self.bb_mult * std
            bb_lower[i] = mean - self.bb_mult * std

        # Keltner Channels (LazyBear uses SMA of close and SMA of True Range)
        kc_ma = _compute_sma_values(closes, self.kc_length)
        kc_tr_ma = _compute_sma_values(tr, self.kc_length)

        # Momentum source series: close - avg(avg(highest_high, lowest_low), sma(close))
        mom_len = self.mom_length
        mom_sma = _compute_sma_values(closes, mom_len)
        mom_src: list[float | None] = [None] * n
        for i in range(mom_len - 1, n):
            hh = max(highs[i - mom_len + 1 : i + 1])
            ll = min(lows[i - mom_len + 1 : i + 1])
            donchian_mid = (hh + ll) / 2.0
            sma_val = mom_sma[i]
            assert sma_val is not None
            baseline = (donchian_mid + sma_val) / 2.0
            mom_src[i] = closes[i] - baseline

        warmup = self.warmup_period()
        for i in range(warmup, n):
            u_bb = bb_upper[i]
            l_bb = bb_lower[i]
            m_kc = kc_ma[i]
            r_kc = kc_tr_ma[i]
            assert u_bb is not None and l_bb is not None and m_kc is not None and r_kc is not None

            u_kc = m_kc + self.kc_mult * r_kc
            l_kc = m_kc - self.kc_mult * r_kc

            if l_bb > l_kc and u_bb < u_kc:
                squeeze_state[i] = 1.0
            elif l_bb < l_kc and u_bb > u_kc:
                squeeze_state[i] = -1.0
            else:
                squeeze_state[i] = 0.0

            src_window = mom_src[i - mom_len + 1 : i + 1]
            # Guaranteed all non-None because i >= 2 * mom_len - 2
            clean_window = [float(v) for v in src_window if v is not None]
            momentum[i] = _linear_regression_endpoint(clean_window)

        return {"momentum": momentum, "squeeze_state": squeeze_state}
