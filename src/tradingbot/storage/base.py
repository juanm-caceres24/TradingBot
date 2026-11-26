"""Abstract repository interface for OHLCV candles, sync ranges, and predictions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from tradingbot.models import Candle, TimeRange


class CandleStore(ABC):
    """Storage backend contract for raw OHLCV candles and sync metadata."""

    @abstractmethod
    def close(self) -> None:
        """Close underlying resources cleanly."""

    @abstractmethod
    def upsert_candles(self, candles: Sequence[Candle]) -> int:
        """Insert or update a batch of candles atomically. Returns count processed."""

    @abstractmethod
    def get_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        closed_only: bool = False,
    ) -> list[Candle]:
        """Fetch candles in ascending open_time_ms order within [start_ms, end_ms)."""

    @abstractmethod
    def get_latest_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        closed_only: bool = False,
    ) -> list[Candle]:
        """Fetch up to the latest `limit` candles in ascending chronological order."""

    @abstractmethod
    def get_open_times(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        closed_only: bool = True,
    ) -> list[int]:
        """Fetch sorted open_time_ms timestamps within [start_ms, end_ms)."""

    @abstractmethod
    def get_bounds(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        closed_only: bool = False,
    ) -> tuple[int, int] | None:
        """Return (min_open_time_ms, max_open_time_ms) or None if empty."""

    @abstractmethod
    def count_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        closed_only: bool = False,
    ) -> int:
        """Count stored candles for a specific series."""

    @abstractmethod
    def delete_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> int:
        """Delete candles matching the series and optional [start_ms, end_ms) range."""

    @abstractmethod
    def record_sync_range(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        time_range: TimeRange,
        checked_at_ms: int,
    ) -> None:
        """Record a verified/checked range (e.g., empty weekend/maintenance window)."""

    @abstractmethod
    def get_sync_ranges(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[TimeRange]:
        """Return verified sync ranges overlapping [start_ms, end_ms)."""

    @abstractmethod
    def save_prediction(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        anchor_open_time_ms: int,
        created_at_ms: int,
        model_name: str,
        payload_json: str,
    ) -> int:
        """Persist a prediction audit entry and return its assigned row ID."""

    @abstractmethod
    def get_predictions(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve latest prediction audit entries in descending creation order."""
