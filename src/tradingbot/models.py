"""Core domain models and time utilities for TradingBot."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_TIMEFRAME_PATTERN = re.compile(r"^([1-9][0-9]*)([mhdw])$")

_UNIT_MS: dict[str, int] = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}

_VALID_ACTIONS = frozenset({"HOLD", "LONG", "SHORT"})


def _ensure_strict_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}")
    return value


def _ensure_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return result


def parse_iso_to_ms(value: str) -> int:
    """Parse an ISO-8601 UTC date or datetime string into epoch milliseconds.

    Accepted examples:
    - '2025-01-01'
    - '2025-01-01T00:00:00Z'
    - '2025-01-01T12:30:00+00:00'
    Naive dates/datetimes without timezone offset are interpreted as UTC.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ISO-8601 timestamp must be a non-empty string")

    raw = value.strip()
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    ts_ms = int(round(dt.timestamp() * 1000))
    if ts_ms < 0:
        raise ValueError(f"Timestamp must be >= Unix epoch (1970-01-01 UTC): {value!r}")
    return ts_ms


def format_ms_to_iso(ts_ms: int) -> str:
    """Format UTC epoch milliseconds into an ISO-8601 UTC string."""
    _ensure_strict_int(ts_ms, "ts_ms")
    if ts_ms < 0:
        raise ValueError(f"ts_ms must be >= 0, got {ts_ms}")
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, order=True)
class Timeframe:
    """Canonical candle interval representation (e.g., '1m', '15m', '1h', '1d')."""

    interval_ms: int
    label: str

    @classmethod
    def parse(cls, text: str) -> Timeframe:
        if not isinstance(text, str):
            raise ValueError(f"Timeframe must be a string, got {type(text).__name__}")
        normalized = text.strip().lower()
        match = _TIMEFRAME_PATTERN.match(normalized)
        if not match:
            raise ValueError(
                f"Invalid timeframe {text!r}. Expected '<positive_int><m|h|d|w>' (e.g. '1m', '1h', '1d')."
            )
        amount = int(match.group(1))
        unit = match.group(2)
        return cls(interval_ms=amount * _UNIT_MS[unit], label=f"{amount}{unit}")

    @property
    def interval_seconds(self) -> int:
        return self.interval_ms // 1000

    def is_aligned(self, ts_ms: int) -> bool:
        _ensure_strict_int(ts_ms, "ts_ms")
        return ts_ms >= 0 and (ts_ms % self.interval_ms == 0)

    def align_floor(self, ts_ms: int) -> int:
        """Return the largest aligned candle open_time <= ts_ms."""
        _ensure_strict_int(ts_ms, "ts_ms")
        if ts_ms < 0:
            raise ValueError(f"ts_ms must be >= 0, got {ts_ms}")
        return (ts_ms // self.interval_ms) * self.interval_ms

    def align_ceil(self, ts_ms: int) -> int:
        """Return the smallest aligned candle open_time >= ts_ms."""
        _ensure_strict_int(ts_ms, "ts_ms")
        if ts_ms < 0:
            raise ValueError(f"ts_ms must be >= 0, got {ts_ms}")
        return ((ts_ms + self.interval_ms - 1) // self.interval_ms) * self.interval_ms

    def next_open(self, open_time_ms: int) -> int:
        _ensure_strict_int(open_time_ms, "open_time_ms")
        if open_time_ms < 0:
            raise ValueError(f"open_time_ms must be >= 0, got {open_time_ms}")
        return open_time_ms + self.interval_ms

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True, order=True)
class TimeRange:
    """Half-open UTC millisecond interval [start_ms, end_ms).

    Contains candles whose open_time_ms satisfies start_ms <= open_time_ms < end_ms.
    """

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _ensure_strict_int(self.start_ms, "start_ms")
        _ensure_strict_int(self.end_ms, "end_ms")
        if self.start_ms < 0:
            raise ValueError(f"start_ms must be >= 0, got {self.start_ms}")
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be >= start_ms ({self.start_ms})"
            )

    @property
    def is_empty(self) -> bool:
        return self.start_ms == self.end_ms

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def candle_count(self, timeframe: Timeframe | int) -> int:
        interval_ms = timeframe.interval_ms if isinstance(timeframe, Timeframe) else timeframe
        _ensure_strict_int(interval_ms, "interval_ms")
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be > 0, got {interval_ms}")
        return self.duration_ms // interval_ms

    def contains(self, ts_ms: int) -> bool:
        _ensure_strict_int(ts_ms, "ts_ms")
        return self.start_ms <= ts_ms < self.end_ms

    def overlaps(self, other: TimeRange) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms

    def touches(self, other: TimeRange) -> bool:
        return self.start_ms <= other.end_ms and other.start_ms <= self.end_ms


@dataclass(frozen=True)
class Candle:
    """Canonical OHLCV market candle."""

    provider: str
    symbol: str
    timeframe: str
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        Timeframe.parse(self.timeframe)

        _ensure_strict_int(self.open_time_ms, "open_time_ms")
        _ensure_strict_int(self.close_time_ms, "close_time_ms")
        if self.open_time_ms < 0:
            raise ValueError(f"open_time_ms must be >= 0, got {self.open_time_ms}")
        if self.close_time_ms < self.open_time_ms:
            raise ValueError(
                f"close_time_ms ({self.close_time_ms}) must be >= open_time_ms ({self.open_time_ms})"
            )

        o = _ensure_finite_float(self.open, "open")
        h = _ensure_finite_float(self.high, "high")
        l = _ensure_finite_float(self.low, "low")
        c = _ensure_finite_float(self.close, "close")
        v = _ensure_finite_float(self.volume, "volume")

        if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
            raise ValueError("OHLC prices must be strictly positive")
        if v < 0.0:
            raise ValueError(f"volume must be >= 0, got {v}")
        if h < max(o, c):
            raise ValueError(f"high ({h}) must be >= max(open={o}, close={c})")
        if l > min(o, c):
            raise ValueError(f"low ({l}) must be <= min(open={o}, close={c})")
        if not isinstance(self.is_closed, bool):
            raise ValueError(f"is_closed must be a bool, got {type(self.is_closed).__name__}")

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def full_range(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_closed": self.is_closed,
        }


@dataclass(frozen=True)
class PredictedCandle:
    """Reconstructed future OHLC candle predicted by the Candle Predictor network."""

    step: int
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _ensure_strict_int(self.step, "step")
        _ensure_strict_int(self.open_time_ms, "open_time_ms")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        if self.open_time_ms < 0:
            raise ValueError(f"open_time_ms must be >= 0, got {self.open_time_ms}")

        o = _ensure_finite_float(self.open, "open")
        h = _ensure_finite_float(self.high, "high")
        l = _ensure_finite_float(self.low, "low")
        c = _ensure_finite_float(self.close, "close")
        conf = _ensure_finite_float(self.confidence, "confidence")

        if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
            raise ValueError("Predicted OHLC prices must be strictly positive")
        if h < max(o, c):
            raise ValueError(f"high ({h}) must be >= max(open={o}, close={c})")
        if l > min(o, c):
            raise ValueError(f"low ({l}) must be <= min(open={o}, close={c})")
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {conf}")


@dataclass(frozen=True)
class SignalPrediction:
    """Calibrated multiclass probabilities from the Signal Classifier network."""

    hold_prob: float
    long_prob: float
    short_prob: float
    latency_us: int | None = None

    def __post_init__(self) -> None:
        hp = _ensure_finite_float(self.hold_prob, "hold_prob")
        lp = _ensure_finite_float(self.long_prob, "long_prob")
        sp = _ensure_finite_float(self.short_prob, "short_prob")
        for name, val in (("hold_prob", hp), ("long_prob", lp), ("short_prob", sp)):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        total = hp + lp + sp
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Signal probabilities must sum to ~1.0, got {total:.6f}")
        if self.latency_us is not None:
            _ensure_strict_int(self.latency_us, "latency_us")
            if self.latency_us < 0:
                raise ValueError(f"latency_us must be >= 0, got {self.latency_us}")

    @property
    def best_action(self) -> str:
        probs = {
            "HOLD": self.hold_prob,
            "LONG": self.long_prob,
            "SHORT": self.short_prob,
        }
        return max(probs, key=probs.__getitem__)

    @property
    def best_probability(self) -> float:
        return max(self.hold_prob, self.long_prob, self.short_prob)


@dataclass(frozen=True)
class TradeAnalysis:
    """Actionable trade evaluation derived from candle and/or signal predictions."""

    action: str
    expected_move_pct: float
    estimated_tp: float | None
    estimated_sl: float | None
    risk_reward: float | None
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(_VALID_ACTIONS)}, got {self.action!r}")
        _ensure_finite_float(self.expected_move_pct, "expected_move_pct")
        if self.estimated_tp is not None:
            tp = _ensure_finite_float(self.estimated_tp, "estimated_tp")
            if tp <= 0.0:
                raise ValueError("estimated_tp must be > 0")
        if self.estimated_sl is not None:
            sl = _ensure_finite_float(self.estimated_sl, "estimated_sl")
            if sl <= 0.0:
                raise ValueError("estimated_sl must be > 0")
        if self.risk_reward is not None:
            rr = _ensure_finite_float(self.risk_reward, "risk_reward")
            if rr < 0.0:
                raise ValueError("risk_reward must be >= 0")
        conf = _ensure_finite_float(self.confidence, "confidence")
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {conf}")
        if not isinstance(self.reason, str):
            raise ValueError("reason must be a string")


@dataclass(frozen=True)
class PredictionSnapshot:
    """Complete prediction bundle anchored to a specific candle open time."""

    provider: str
    symbol: str
    timeframe: str
    anchor_open_time_ms: int
    created_at_ms: int
    predicted_candles: tuple[PredictedCandle, ...] = ()
    signal: SignalPrediction | None = None
    analysis: TradeAnalysis | None = None
    candle_latency_us: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        Timeframe.parse(self.timeframe)
        _ensure_strict_int(self.anchor_open_time_ms, "anchor_open_time_ms")
        _ensure_strict_int(self.created_at_ms, "created_at_ms")
        if self.anchor_open_time_ms < 0 or self.created_at_ms < 0:
            raise ValueError("Timestamps must be >= 0")
        if self.candle_latency_us is not None:
            _ensure_strict_int(self.candle_latency_us, "candle_latency_us")
            if self.candle_latency_us < 0:
                raise ValueError("candle_latency_us must be >= 0")
