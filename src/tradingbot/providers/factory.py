"""Provider factory constructing concrete providers from ProviderConfig."""

from __future__ import annotations

import time
from collections.abc import Callable

from tradingbot.config import ProviderConfig
from tradingbot.providers.base import BaseProvider, ProviderError, default_clock_ms
from tradingbot.providers.generic_rest import GenericRestProvider
from tradingbot.providers.synthetic import SyntheticProvider


def create_provider(
    config: ProviderConfig,
    *,
    clock_ms: Callable[[], int] = default_clock_ms,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> BaseProvider:
    """Instantiate the configured market data provider."""
    if config.type == "generic_rest":
        return GenericRestProvider(
            config,
            clock_ms=clock_ms,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )
    if config.type == "synthetic":
        return SyntheticProvider(config, clock_ms=clock_ms)
    raise ProviderError(f"Unsupported provider type: {config.type!r}")
