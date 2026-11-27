"""Extensible technical indicator engine and built-in indicator plugins."""

from tradingbot.indicators.base import BaseIndicator, IndicatorError, VALID_SCALE_KINDS
from tradingbot.indicators.builtin import (
    ATRIndicator,
    BollingerBandsIndicator,
    EMAIndicator,
    HTFContextIndicator,
    RSIIndicator,
    SMAIndicator,
    SqueezeMomentumIndicator,
    VolatilityRegimeIndicator,
)
from tradingbot.indicators.registry import (
    INDICATOR_REGISTRY,
    IndicatorEngine,
    create_indicator,
    register_indicator,
)

__all__ = [
    "ATRIndicator",
    "BaseIndicator",
    "BollingerBandsIndicator",
    "EMAIndicator",
    "HTFContextIndicator",
    "INDICATOR_REGISTRY",
    "IndicatorEngine",
    "IndicatorError",
    "RSIIndicator",
    "SMAIndicator",
    "SqueezeMomentumIndicator",
    "VALID_SCALE_KINDS",
    "VolatilityRegimeIndicator",
    "create_indicator",
    "register_indicator",
]
