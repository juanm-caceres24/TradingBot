"""Indicator plugin registry and multi-indicator computation engine."""

from __future__ import annotations

from collections.abc import Sequence

from tradingbot.config import IndicatorConfig
from tradingbot.indicators.base import BaseIndicator, IndicatorError
from tradingbot.indicators.builtin import (
    ATRIndicator,
    BollingerBandsIndicator,
    EMAIndicator,
    RSIIndicator,
    SMAIndicator,
    SqueezeMomentumIndicator,
)
from tradingbot.models import Candle

INDICATOR_REGISTRY: dict[str, type[BaseIndicator]] = {
    "sma": SMAIndicator,
    "ema": EMAIndicator,
    "rsi": RSIIndicator,
    "atr": ATRIndicator,
    "bollinger": BollingerBandsIndicator,
    "squeeze_momentum": SqueezeMomentumIndicator,
}


def register_indicator(name: str, cls: type[BaseIndicator]) -> None:
    """Register a custom indicator class under a unique type name."""
    if not isinstance(name, str) or not name.strip():
        raise IndicatorError("Indicator type name must be a non-empty string")
    if not isinstance(cls, type) or not issubclass(cls, BaseIndicator):
        raise IndicatorError("Indicator class must be a subclass of BaseIndicator")
    INDICATOR_REGISTRY[name.strip().lower()] = cls


def create_indicator(config: IndicatorConfig) -> BaseIndicator:
    """Instantiate an indicator plugin from an IndicatorConfig."""
    key = config.type.strip().lower()
    cls = INDICATOR_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(INDICATOR_REGISTRY.keys()))
        raise IndicatorError(
            f"Unknown indicator type {config.type!r}. Available types: {available}"
        )
    return cls(config.id, config.params)


class IndicatorEngine:
    """Computes a configured suite of indicators over raw OHLCV candles."""

    def __init__(self, configs: Sequence[IndicatorConfig]) -> None:
        self._configs = tuple(configs)
        self._indicators = tuple(create_indicator(c) for c in configs)

    @property
    def configs(self) -> tuple[IndicatorConfig, ...]:
        return self._configs

    @property
    def indicators(self) -> tuple[BaseIndicator, ...]:
        return self._indicators

    def nn_indicators(self) -> tuple[BaseIndicator, ...]:
        return tuple(
            ind for cfg, ind in zip(self._configs, self._indicators) if cfg.use_in_nn
        )

    def max_warmup(self, *, nn_only: bool = False) -> int:
        selected = self.nn_indicators() if nn_only else self._indicators
        if not selected:
            return 0
        return max(ind.warmup_period() for ind in selected)

    def compute_all(
        self,
        candles: Sequence[Candle],
        *,
        nn_only: bool = False,
    ) -> dict[str, dict[str, list[float | None]]]:
        """Compute all (or NN-only) indicators aligned 1:1 with `candles`."""
        selected = self.nn_indicators() if nn_only else self._indicators
        results: dict[str, dict[str, list[float | None]]] = {}
        for ind in selected:
            results[ind.indicator_id] = ind.compute(candles)
        return results
