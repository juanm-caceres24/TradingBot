"""Base abstractions for extensible technical indicators."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from tradingbot.models import Candle


class IndicatorError(ValueError):
    """Raised when an indicator specification or parameter set is invalid."""


# Normalization scale categories declared by indicators so FeaturePipeline
# normalizes newly added indicators automatically without touching core code.
VALID_SCALE_KINDS = frozenset({"price_level", "price_delta", "rsi_0_100", "bounded"})


def require_int_param(
    params: dict[str, Any],
    key: str,
    *,
    min_value: int = 1,
    default: int | None = None,
) -> int:
    if key not in params:
        if default is not None:
            return default
        raise IndicatorError(f"Missing required integer indicator parameter '{key}'")
    val = params[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise IndicatorError(f"Indicator parameter '{key}' must be an integer, got {type(val).__name__}")
    if val < min_value:
        raise IndicatorError(f"Indicator parameter '{key}' must be >= {min_value}, got {val}")
    return val


def require_float_param(
    params: dict[str, Any],
    key: str,
    *,
    exclusive_min: float | None = None,
    min_value: float | None = None,
    default: float | None = None,
) -> float:
    if key not in params:
        if default is not None:
            return default
        raise IndicatorError(f"Missing required numeric indicator parameter '{key}'")
    val = params[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise IndicatorError(f"Indicator parameter '{key}' must be a number, got {type(val).__name__}")
    result = float(val)
    if not math.isfinite(result):
        raise IndicatorError(f"Indicator parameter '{key}' must be finite, got {val!r}")
    if min_value is not None and result < min_value:
        raise IndicatorError(f"Indicator parameter '{key}' must be >= {min_value}, got {result}")
    if exclusive_min is not None and result <= exclusive_min:
        raise IndicatorError(f"Indicator parameter '{key}' must be > {exclusive_min}, got {result}")
    return result


def reject_unknown_params(
    params: dict[str, Any], allowed: frozenset[str], indicator_type: str
) -> None:
    unknown = sorted(set(params.keys()) - allowed)
    if unknown:
        keys_str = ", ".join(repr(k) for k in unknown)
        raise IndicatorError(f"Unknown parameter(s) for indicator '{indicator_type}': {keys_str}")


class BaseIndicator(ABC):
    """Plugin interface implemented by every technical indicator.

    Output convention:
    - `compute(candles)` returns a dict mapping each column name in `output_columns()`
      to a list of length `len(candles)`.
    - Warm-up entries (indices `< warmup_period()`) MUST be `None` so the Web UI can
      render raw candles while omitting missing indicator points, and the feature
      pipeline can strictly exclude incomplete windows from neural-network inputs.
    """

    indicator_type: str = ""

    def __init__(self, indicator_id: str, params: dict[str, Any]) -> None:
        if not isinstance(indicator_id, str) or not indicator_id.strip():
            raise IndicatorError("indicator_id must be a non-empty string")
        self.indicator_id = indicator_id
        self.params = dict(params)
        self._validate_params()

    @abstractmethod
    def _validate_params(self) -> None:
        """Validate and store typed indicator parameters."""

    @abstractmethod
    def warmup_period(self) -> int:
        """Index of the first candle where all output columns become non-None."""

    @abstractmethod
    def output_columns(self) -> tuple[str, ...]:
        """Ordered output series names emitted by this indicator."""

    @abstractmethod
    def nn_scale_kind(self, column: str) -> str:
        """Return normalization category for `column` (one of VALID_SCALE_KINDS)."""

    @abstractmethod
    def compute(self, candles: Sequence[Candle]) -> dict[str, list[float | None]]:
        """Compute indicator output series aligned 1:1 with `candles`."""
