"""Declarative JSON-driven HTTP REST market data provider (stdlib only).

Works out-of-the-box with Binance Spot public klines as well as any REST API
described by a provider preset JSON (array_of_arrays or array_of_objects,
nested data_path, ms or s timestamps, asc or desc ordering).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

import tradingbot
from tradingbot.config import ProviderConfig
from tradingbot.models import Candle, Timeframe
from tradingbot.providers.base import BaseProvider, ProviderError, default_clock_ms

_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


def _extract_data_path(payload: Any, data_path: str) -> Any:
    """Traverse dot-separated keys (e.g. 'result.list') inside decoded JSON."""
    if not data_path:
        return payload
    current = payload
    for part in data_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ProviderError(
                f"Response JSON missing expected key {part!r} in data_path {data_path!r}"
            )
        current = current[part]
    return current


def _read_selector(row: Any, selector: int | str, fmt: str, field_name: str) -> Any:
    try:
        if fmt == "array_of_arrays":
            assert isinstance(selector, int)
            return row[selector]
        assert isinstance(selector, str)
        return row[selector]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            f"Failed to read field '{field_name}' using selector {selector!r} from row {row!r}"
        ) from exc


class GenericRestProvider(BaseProvider):
    """Configurable HTTP REST OHLCV provider driven strictly by ProviderConfig."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        max_retries: int = 2,
        retry_backoff_sec: float = 0.5,
        clock_ms: Callable[[], int] = default_clock_ms,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if config.type != "generic_rest":
            raise ProviderError(
                f"GenericRestProvider requires type='generic_rest', got {config.type!r}"
            )
        if config.params_map is None or config.response_mapping is None:
            raise ProviderError(
                f"Provider '{config.name}' is missing params_map or response_mapping"
            )
        self._config = config
        self._params_map = config.params_map
        self._mapping = config.response_mapping
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self._clock_ms = clock_ms
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._rate_lock = threading.Lock()
        self._last_request_mono: float | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def max_candles_per_request(self) -> int:
        return self._config.max_candles_per_request

    def _respect_rate_limit(self) -> None:
        min_gap_sec = self._config.rate_limit_ms / 1000.0
        if min_gap_sec <= 0.0:
            return
        with self._rate_lock:
            now_mono = self._monotonic_fn()
            if self._last_request_mono is not None:
                elapsed = now_mono - self._last_request_mono
                if elapsed < min_gap_sec:
                    self._sleep_fn(min_gap_sec - elapsed)
            self._last_request_mono = self._monotonic_fn()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": f"TradingBot/{tradingbot.__version__}",
        }
        headers.update(self._config.headers)
        if self._config.api_key_env:
            token = os.environ.get(self._config.api_key_env, "").strip()
            if not token:
                raise ProviderError(
                    f"Environment variable '{self._config.api_key_env}' required by "
                    f"provider '{self.name}' is not set"
                )
            headers[self._config.api_key_header] = token
        return headers

    def _to_provider_timestamp(self, ts_ms: int) -> int:
        if self._mapping.timestamp_unit == "s":
            return ts_ms // 1000
        return ts_ms

    def _from_provider_timestamp(self, raw_val: Any, field_name: str) -> int:
        if isinstance(raw_val, bool):
            raise ProviderError(
                f"Invalid boolean timestamp value {raw_val!r} for '{field_name}'"
            )
        try:
            fval = float(raw_val)
            if not math.isfinite(fval):
                raise ValueError("timestamp is not finite")
            num = int(fval)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(
                f"Invalid timestamp value {raw_val!r} for '{field_name}'"
            ) from exc
        if self._mapping.timestamp_unit == "s":
            return num * 1000
        return num

    def _build_url(
        self,
        symbol: str,
        tf: Timeframe,
        *,
        start_ms: int | None,
        end_ms: int | None,
        limit: int,
    ) -> str:
        if tf.label not in self._config.interval_map:
            raise ProviderError(
                f"Timeframe '{tf.label}' is not mapped in provider '{self.name}' interval_map"
            )
        query: dict[str, Any] = {
            self._params_map.symbol: symbol,
            self._params_map.interval: self._config.interval_map[tf.label],
            self._params_map.limit: limit,
        }
        if start_ms is not None:
            query[self._params_map.start_time] = self._to_provider_timestamp(start_ms)
        if end_ms is not None:
            # Convert half-open [start_ms, end_ms) into inclusive API endTime when > start_ms.
            inclusive_end_ms = max(0, end_ms - 1)
            query[self._params_map.end_time] = self._to_provider_timestamp(inclusive_end_ms)

        base = self._config.base_url.rstrip("/")
        endpoint = "/" + self._config.endpoint.lstrip("/")
        return f"{base}{endpoint}?{urllib.parse.urlencode(query)}"

    def _execute_get_json(self, url: str) -> Any:
        headers = self._build_headers()
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            self._respect_rate_limit()
            req = urllib.request.Request(
                url,
                headers=headers,
                method=self._config.http_method,
            )
            try:
                with urllib.request.urlopen(req, timeout=self._config.timeout_sec) as resp:
                    raw_bytes = resp.read()
                try:
                    return json.loads(raw_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProviderError(
                        f"Provider '{self.name}' returned invalid JSON from {url}: {exc}"
                    ) from exc
            except urllib.error.HTTPError as exc:
                status = exc.code
                body_preview = ""
                try:
                    body_preview = exc.read(256).decode("utf-8", errors="replace")
                except Exception:
                    pass
                last_error = ProviderError(
                    f"Provider '{self.name}' HTTP {status} for {url}: {body_preview}"
                )
                if status in _RETRYABLE_HTTP_STATUS and attempt < self._max_retries:
                    self._sleep_fn(self._retry_backoff_sec * (2**attempt))
                    continue
                raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = ProviderError(
                    f"Provider '{self.name}' network error for {url}: {exc}"
                )
                if attempt < self._max_retries:
                    self._sleep_fn(self._retry_backoff_sec * (2**attempt))
                    continue
                raise last_error from exc

        assert last_error is not None
        raise last_error

    def _parse_rows(
        self,
        payload: Any,
        symbol: str,
        tf: Timeframe,
        *,
        start_ms: int | None,
        end_ms: int | None,
        now_ms: int,
    ) -> list[Candle]:
        rows = _extract_data_path(payload, self._mapping.data_path)
        if not isinstance(rows, list):
            raise ProviderError(
                f"Provider '{self.name}' expected a JSON list at data_path "
                f"{self._mapping.data_path!r}, got {type(rows).__name__}"
            )

        if self._mapping.order == "desc":
            rows = list(reversed(rows))

        fmt = self._mapping.format
        fields = self._mapping.fields
        candles: list[Candle] = []

        for row in rows:
            open_time_ms = self._from_provider_timestamp(
                _read_selector(row, fields.open_time, fmt, "open_time"),
                "open_time",
            )
            if start_ms is not None and open_time_ms < start_ms:
                continue
            if end_ms is not None and open_time_ms >= end_ms:
                continue

            if fields.close_time is not None:
                close_time_ms = self._from_provider_timestamp(
                    _read_selector(row, fields.close_time, fmt, "close_time"),
                    "close_time",
                )
            else:
                close_time_ms = open_time_ms + tf.interval_ms - 1

            try:
                open_val = float(_read_selector(row, fields.open, fmt, "open"))
                high_val = float(_read_selector(row, fields.high, fmt, "high"))
                low_val = float(_read_selector(row, fields.low, fmt, "low"))
                close_val = float(_read_selector(row, fields.close, fmt, "close"))
                vol_val = float(_read_selector(row, fields.volume, fmt, "volume"))
            except (TypeError, ValueError) as exc:
                raise ProviderError(
                    f"Provider '{self.name}' returned non-numeric OHLCV values in row {row!r}"
                ) from exc

            # Treat a bar as closed strictly once its close window has elapsed.
            is_closed = (open_time_ms + tf.interval_ms) <= now_ms
            try:
                candles.append(
                    Candle(
                        provider=self.name,
                        symbol=symbol,
                        timeframe=tf.label,
                        open_time_ms=open_time_ms,
                        close_time_ms=close_time_ms,
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=vol_val,
                        is_closed=is_closed,
                    )
                )
            except ValueError as exc:
                raise ProviderError(
                    f"Provider '{self.name}' produced invalid candle at {open_time_ms}: {exc}"
                ) from exc

        candles.sort(key=lambda c: c.open_time_ms)
        return candles

    def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe.parse(timeframe)
        effective_limit = (
            min(limit, self.max_candles_per_request)
            if limit is not None
            else self.max_candles_per_request
        )
        if effective_limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            return []

        url = self._build_url(
            symbol,
            tf,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=effective_limit,
        )
        payload = self._execute_get_json(url)
        now_ms = self._clock_ms()
        return self._parse_rows(
            payload,
            symbol,
            tf,
            start_ms=start_ms,
            end_ms=end_ms,
            now_ms=now_ms,
        )
