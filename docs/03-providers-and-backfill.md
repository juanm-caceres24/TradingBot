# Market Data Providers & Historical Backfill (`v0.2.0`)

This document describes the provider layer (`src/tradingbot/providers/`), the declarative JSON REST mapping engine, the deterministic synthetic generator, the multi-asset `SyncService`, and the `apps/backfill_app.py` CLI executable.

---

## 1. Architecture Overview

```text
configs/bot.json ──► AssetConfig (per asset/timeframe)
                           │
                           ▼
                     SyncService
                  ┌────────┼────────┐
                  ▼                 ▼
            GapDetector        BaseProvider (cached per exchange)
         (missing ranges +      ├── GenericRestProvider (JSON-driven: Binance, Bybit, brokers)
          chunk pagination)     └── SyntheticProvider   (deterministic offline generator)
                  │                 │
                  └────────┬────────┘
                           ▼
                   SQLiteCandleStore (candles + sync_ranges)
```

- **Zero external dependencies:** Uses only Python's standard library (`urllib.request`, `json`, `hashlib`, `sqlite3`).
- **Shared rate-limit throttling:** `SyncService` caches provider instances by canonical full `ProviderConfig` JSON, so multiple assets sharing an identical preset (e.g., `btc_15m`, `btc_1h`, `btc_4h`, `btc_1d` on `binance_spot.json`) share one unified rate-limit throttle while distinct per-asset overrides remain isolated.

---

## 2. Declarative REST Provider (`GenericRestProvider`)

Configured completely via JSON presets (e.g. `configs/providers/binance_spot.json`):
- **Supported payload structures:**
  - `array_of_arrays` (Binance Spot `/api/v3/klines`) with integer column indices.
  - `array_of_objects` (brokers / Bybit / Polygon-like APIs) with string field names and optional dot-separated `data_path` (e.g., `"result.bars"`).
- **Timestamp & ordering normalization:**
  - Converts `"s"` or `"ms"` timestamps in both query parameters and responses into canonical UTC epoch milliseconds.
  - Reverses `"desc"` responses into ascending `open_time_ms` order and strictly filters bars to the requested half-open interval `[start_ms, end_ms)`.
  - Computes `close_time_ms = open_time_ms + interval_ms - 1` automatically when the API omits `close_time`.
  - Sets `is_closed = (open_time_ms + interval_ms <= now_ms)`.
- **Resilience & security:**
  - Enforces `rate_limit_ms` between requests and retries transient HTTP `{429, 500, 502, 503, 504}` and network errors with exponential backoff.
  - Optional API keys are loaded strictly from `os.environ[api_key_env]` at request time — never stored in Git.

---

## 3. Deterministic Synthetic Provider (`SyntheticProvider`)

Configured via `configs/providers/synthetic.json`:
- **Chunk-invariant & O(1) per candle:** Fetching `[t0, t100)` yields bit-for-bit identical candles to paginated chunks `[t0, t37)` + `[t37, t100)`.
- **Learnable synthetic dynamics:** Combines multi-scale cyclical waves (24, 96, 360 bars), bounded drift, seeded per-bar noise, continuous `open[k] == close[k-1]` bars, and volatility-correlated volume so downstream `NeuralNetwork` training tests exhibit genuine loss reduction.

---

## 4. Historical Synchronization (`SyncService`) & Empty-Range Tracking

1. Computes missing closed-candle intervals using `GapDetector`.
2. Splits large intervals into pages of at most `provider.max_candles_per_request` bars (`chunk_range`).
3. Upserts returned candles into `SQLiteCandleStore`.
4. **Safe empty-range recording:** For chunks strictly in the past (`chunk.end_ms <= align_floor(now_ms)`):
   - If the API returns 0 bars, records the chunk in `sync_ranges`.
   - If the API returns fewer than `max_candles_per_request` bars (non-truncated page), records any uncovered sub-gaps inside that chunk (e.g., equity weekends/holidays or exchange maintenance windows) so future runs skip them automatically unless `--force-recheck` is passed.

---

## 5. CLI Usage (`apps/backfill_app.py` / `scripts/backfill.sh`)

```bash
# Preview missing ranges without hitting any external API
./scripts/backfill.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-01-10 --dry-run

# Download a bounded historical range (e.g. BTC 1h from Binance Spot)
./scripts/backfill.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-02-01

# Auto-detect and repair head/internal/tail gaps up to now for all enabled assets
./scripts/backfill.sh configs/bot.json --sync-now

# Ensure at least N closed candles exist per asset (useful before first dataset export)
./scripts/backfill.sh configs/bot.json --sync-now --min-candles 5000

# Backfill all configured assets (including any disabled ones) or force-recheck a specific asset
./scripts/backfill.sh configs/bot.json --sync-now --all-assets
./scripts/backfill.sh configs/bot.json --asset btc_1d --start 2020-01-01 --force-recheck
```
