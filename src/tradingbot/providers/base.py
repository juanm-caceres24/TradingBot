"""Market data provider abstractions and common exceptions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from tradingbot.models import Candle, Timeframe


def default_clock_ms() -> int:
    """Return current UTC wall-clock time in epoch milliseconds."""
    return int(time.time() * 1000)


class ProviderError(RuntimeError):
    """Raised when a market data provider fails to fetch or parse candles."""


class BaseProvider(ABC):
    """Contract implemented by all market data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider identifier persisted in Candle.provider."""

    @property
    @abstractmethod
    def max_candles_per_request(self) -> int:
        """Maximum number of candles returned per single API call."""

    @abstractmethod
    def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        """Fetch candles in ascending open_time_ms order within half-open [start_ms, end_ms).

        If `start_ms` and `end_ms` are omitted, fetches up to `limit` most recent candles.
        """
