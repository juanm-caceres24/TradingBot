"""Unit and loopback HTTP integration tests for market data providers."""

from __future__ import annotations

import copy
import http.server
import json
import os
import threading
import unittest
import urllib.parse
from pathlib import Path
from typing import Any

from tradingbot.config import load_provider_preset, parse_provider_dict
from tradingbot.models import parse_iso_to_ms
from tradingbot.providers import (
    GenericRestProvider,
    ProviderError,
    SyntheticProvider,
    create_provider,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BINANCE_PRESET = ROOT_DIR / "configs" / "providers" / "binance_spot.json"
SYNTHETIC_PRESET = ROOT_DIR / "configs" / "providers" / "synthetic.json"
HOUR_MS = 3_600_000


class _MockExchangeHandler(http.server.BaseHTTPRequestHandler):
    """Configurable local mock HTTP server handler for REST provider tests."""

    routes: dict[str, Any] = {}
    recorded_requests: list[tuple[str, dict[str, str], dict[str, str]]] = []

    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep test output clean

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        headers = {k.lower(): v for k, v in self.headers.items()}
        self.recorded_requests.append((parsed.path, query, headers))

        handler = self.routes.get(parsed.path)
        if handler is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"not found"}')
            return

        status, body = handler(query, headers)
        encoded = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class TestSyntheticProvider(unittest.TestCase):
    def test_determinism_continuity_and_chunk_invariance(self) -> None:
        cfg = load_provider_preset(SYNTHETIC_PRESET)
        start = parse_iso_to_ms("2025-01-01T00:00:00Z")
        now_ms = start + 100 * HOUR_MS + 10_000  # 100 closed candles, 100th active

        provider = create_provider(cfg, clock_ms=lambda: now_ms)
        self.assertIsInstance(provider, SyntheticProvider)

        full = provider.fetch_candles("BTCUSDT", "1h", start_ms=start, end_ms=start + 60 * HOUR_MS)
        self.assertEqual(len(full), 60)

        # Chunk invariance: fetching in two halves matches full fetch bit-for-bit
        part1 = provider.fetch_candles("BTCUSDT", "1h", start_ms=start, end_ms=start + 23 * HOUR_MS)
        part2 = provider.fetch_candles("BTCUSDT", "1h", start_ms=start + 23 * HOUR_MS, end_ms=start + 60 * HOUR_MS)
        self.assertEqual(full, part1 + part2)

        # Continuity: each candle opens at previous candle's close
        for prev, curr in zip(full, full[1:]):
            self.assertAlmostEqual(curr.open, prev.close, places=5)
            self.assertEqual(curr.open_time_ms, prev.open_time_ms + HOUR_MS)
            self.assertTrue(curr.is_closed)

        # Different symbols yield distinct series
        eth = provider.fetch_candles("ETHUSDT", "1h", start_ms=start, end_ms=start + 10 * HOUR_MS)
        self.assertNotAlmostEqual(eth[0].close, full[0].close, places=3)

    def test_latest_fetch_marks_active_candle_unclosed(self) -> None:
        cfg = load_provider_preset(SYNTHETIC_PRESET)
        start = parse_iso_to_ms("2025-01-01T00:00:00Z")
        now_ms = start + 10 * HOUR_MS + 15_000  # bar 10 is currently open

        provider = SyntheticProvider(cfg, clock_ms=lambda: now_ms)
        bars = provider.fetch_candles("BTCUSDT", "1h", limit=5)
        self.assertEqual(len(bars), 5)
        self.assertTrue(all(b.is_closed for b in bars[:-1]))
        self.assertFalse(bars[-1].is_closed)
        self.assertEqual(bars[-1].open_time_ms, start + 10 * HOUR_MS)


class TestGenericRestProvider(unittest.TestCase):
    server: http.server.ThreadingHTTPServer
    thread: threading.Thread
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockExchangeHandler)
        port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)

    def setUp(self) -> None:
        _MockExchangeHandler.routes = {}
        _MockExchangeHandler.recorded_requests = []

    def test_binance_array_of_arrays_mapping_and_filtering(self) -> None:
        start = parse_iso_to_ms("2025-01-01T00:00:00Z")
        now_ms = start + 2 * HOUR_MS + 10_000  # bars 0 and 1 closed; bar 2 still open

        def binance_handler(query: dict[str, str], headers: dict[str, str]) -> tuple[int, Any]:
            # Include one out-of-range bar at index 3 to verify strict [start, end) filtering.
            rows = [
                [start + i * HOUR_MS, "100.0", "105.0", "99.0", str(101.0 + i), "12.5", start + (i + 1) * HOUR_MS - 1]
                for i in range(4)
            ]
            return 200, rows

        _MockExchangeHandler.routes["/api/v3/klines"] = binance_handler

        raw_preset = json.loads(BINANCE_PRESET.read_text(encoding="utf-8"))
        raw_preset["base_url"] = self.base_url
        raw_preset["rate_limit_ms"] = 0
        cfg = parse_provider_dict(raw_preset)

        provider = create_provider(cfg, clock_ms=lambda: now_ms)
        self.assertIsInstance(provider, GenericRestProvider)
        candles = provider.fetch_candles(
            "BTCUSDT", "1h", start_ms=start, end_ms=start + 3 * HOUR_MS, limit=10
        )

        self.assertEqual(len(candles), 3)
        self.assertEqual([c.open_time_ms for c in candles], [start, start + HOUR_MS, start + 2 * HOUR_MS])
        self.assertTrue(candles[0].is_closed)
        self.assertTrue(candles[1].is_closed)
        self.assertFalse(candles[2].is_closed)
        self.assertAlmostEqual(candles[2].close, 103.0)

        # Verify query params matched Binance schema
        path, query, _ = _MockExchangeHandler.recorded_requests[0]
        self.assertEqual(path, "/api/v3/klines")
        self.assertEqual(query["symbol"], "BTCUSDT")
        self.assertEqual(query["interval"], "1h")
        self.assertEqual(query["startTime"], str(start))
        self.assertEqual(query["endTime"], str(start + 3 * HOUR_MS - 1))
        self.assertEqual(query["limit"], "10")

    def test_array_of_objects_nested_path_seconds_desc_and_api_key(self) -> None:
        start_s = 1_735_689_600  # 2025-01-01T00:00:00Z
        now_ms = (start_s + 10 * 3600) * 1000

        def broker_handler(query: dict[str, str], headers: dict[str, str]) -> tuple[int, Any]:
            if headers.get("x-api-key") != "secret-test-key":
                return 401, {"error": "unauthorized"}
            # Descending order, timestamp in seconds, no close_time
            return 200, {
                "result": {
                    "bars": [
                        {"t": start_s + 3600, "o": 201, "h": 205, "l": 199, "c": 204, "v": 80},
                        {"t": start_s, "o": 200, "h": 203, "l": 198, "c": 201, "v": 50},
                    ]
                }
            }

        _MockExchangeHandler.routes["/v2/bars"] = broker_handler

        cfg = parse_provider_dict(
            {
                "name": "mock_broker",
                "type": "generic_rest",
                "base_url": self.base_url,
                "endpoint": "/v2/bars",
                "rate_limit_ms": 0,
                "max_candles_per_request": 500,
                "params_map": {
                    "symbol": "ticker",
                    "interval": "tf",
                    "start_time": "from",
                    "end_time": "to",
                    "limit": "count",
                },
                "interval_map": {"1h": "60"},
                "response_mapping": {
                    "data_path": "result.bars",
                    "format": "array_of_objects",
                    "fields": {
                        "open_time": "t",
                        "open": "o",
                        "high": "h",
                        "low": "l",
                        "close": "c",
                        "volume": "v",
                    },
                    "timestamp_unit": "s",
                    "order": "desc",
                },
                "api_key_env": "TEST_MOCK_BROKER_KEY",
                "api_key_header": "X-API-Key",
            }
        )

        # Missing env var raises clear ProviderError
        os.environ.pop("TEST_MOCK_BROKER_KEY", None)
        provider = GenericRestProvider(cfg, max_retries=0, clock_ms=lambda: now_ms)
        with self.assertRaises(ProviderError):
            provider.fetch_candles("NVDA", "1h", limit=2)

        os.environ["TEST_MOCK_BROKER_KEY"] = "secret-test-key"
        try:
            bars = provider.fetch_candles(
                "NVDA", "1h", start_ms=start_s * 1000, end_ms=(start_s + 7200) * 1000, limit=2
            )
        finally:
            os.environ.pop("TEST_MOCK_BROKER_KEY", None)

        # Reordered to ascending, seconds converted to ms, fallback close_time computed
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].open_time_ms, start_s * 1000)
        self.assertEqual(bars[0].close_time_ms, start_s * 1000 + HOUR_MS - 1)
        self.assertEqual(bars[1].open_time_ms, (start_s + 3600) * 1000)

    def test_retry_on_429_rate_limiting_and_hard_400(self) -> None:
        attempts = {"count": 0}
        sleeps: list[float] = []
        fake_mono = {"t": 100.0}

        def flaky_handler(query: dict[str, str], headers: dict[str, str]) -> tuple[int, Any]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return 429, {"msg": "rate limit"}
            return 200, []

        def bad_handler(query: dict[str, str], headers: dict[str, str]) -> tuple[int, Any]:
            return 400, {"msg": "invalid symbol"}

        _MockExchangeHandler.routes["/flaky"] = flaky_handler
        _MockExchangeHandler.routes["/bad"] = bad_handler

        def fake_sleep(sec: float) -> None:
            sleeps.append(sec)
            fake_mono["t"] += sec

        raw_preset = json.loads(BINANCE_PRESET.read_text(encoding="utf-8"))
        raw_preset["base_url"] = self.base_url
        raw_preset["endpoint"] = "/flaky"
        raw_preset["rate_limit_ms"] = 250
        cfg = parse_provider_dict(raw_preset)

        provider = GenericRestProvider(
            cfg,
            max_retries=2,
            retry_backoff_sec=0.1,
            sleep_fn=fake_sleep,
            monotonic_fn=lambda: fake_mono["t"],
        )
        res = provider.fetch_candles("BTCUSDT", "1h", limit=5)
        self.assertEqual(res, [])
        self.assertEqual(attempts["count"], 2)
        # Backoff (0.1s) + remaining rate limit delay (0.15s) observed
        self.assertGreaterEqual(len(sleeps), 2)

        raw_preset["endpoint"] = "/bad"
        raw_preset["rate_limit_ms"] = 0
        bad_provider = GenericRestProvider(parse_provider_dict(raw_preset), max_retries=2, sleep_fn=fake_sleep)
        with self.assertRaises(ProviderError):
            bad_provider.fetch_candles("INVALID", "1h", limit=5)

    def test_invalid_json_missing_data_path_503_exhaustion_and_bool_timestamp(self) -> None:
        _MockExchangeHandler.routes["/invalid_json"] = lambda _q, _h: (200, "{broken")
        _MockExchangeHandler.routes["/missing_path"] = lambda _q, _h: (200, {"other": []})
        _MockExchangeHandler.routes["/always_503"] = lambda _q, _h: (503, {"error": "down"})
        _MockExchangeHandler.routes["/bool_ts"] = lambda _q, _h: (200, [[True, "100", "101", "99", "100", "1", 10]])

        raw_preset = json.loads(BINANCE_PRESET.read_text(encoding="utf-8"))
        raw_preset["base_url"] = self.base_url
        raw_preset["rate_limit_ms"] = 0

        for ep in ("/invalid_json", "/always_503", "/bool_ts"):
            cfg = copy.deepcopy(raw_preset)
            cfg["endpoint"] = ep
            prov = GenericRestProvider(
                parse_provider_dict(cfg), max_retries=1, retry_backoff_sec=0.0, sleep_fn=lambda _s: None
            )
            with self.subTest(endpoint=ep):
                with self.assertRaises(ProviderError):
                    prov.fetch_candles("BTCUSDT", "1h", limit=5)

        cfg_mp = copy.deepcopy(raw_preset)
        cfg_mp["endpoint"] = "/missing_path"
        cfg_mp["response_mapping"]["data_path"] = "result.list"
        prov_mp = GenericRestProvider(parse_provider_dict(cfg_mp), max_retries=0)
        with self.assertRaises(ProviderError):
            prov_mp.fetch_candles("BTCUSDT", "1h", limit=5)


if __name__ == "__main__":
    unittest.main()
