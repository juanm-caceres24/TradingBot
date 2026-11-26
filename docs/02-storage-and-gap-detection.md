# Storage Engine & Gap Detection (`SQLiteCandleStore`, `GapDetector`)

This document describes the local persistence layer in `src/tradingbot/storage/` and the half-open time-interval model used for multi-asset gap detection, prediction auditing, and external-consumer alert persistence.

---

## 1. Design Philosophy

- **Raw OHLCV only in the database:** `candles` stores exclusively raw market bars (`open`, `high`, `low`, `close`, `volume`, timestamps, and `is_closed`). Technical indicators are computed dynamically by the indicator engine so adding or editing indicators never requires database migrations or re-downloading historical data.
- **Multi-asset & multi-timeframe native:** All assets and timeframes (e.g., `BTCUSDT 1h`, `BTCUSDT 1w`, `NVDA 1h`) coexist inside a single local SQLite file in gitignored `data/` (e.g. `data/market.db`).
- **Zero external database servers:** Uses Python's built-in `sqlite3` module with safe additive schema migration support.
- **Abstract contract (`CandleStore`):** All higher-level services depend on the `CandleStore` ABC (`src/tradingbot/storage/base.py`), keeping future storage backends pluggable.

---

## 2. SQLite Schema (`src/tradingbot/storage/sqlite_store.py`)

### `candles` Table (`WITHOUT ROWID`)
```sql
CREATE TABLE IF NOT EXISTS candles (
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time_ms INTEGER NOT NULL,
    close_time_ms INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    is_closed INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (provider, symbol, timeframe, open_time_ms)
) WITHOUT ROWID;
```
- **Automatic deduplication:** Upserts (`ON CONFLICT DO UPDATE`) safely overwrite an active candle as it updates intra-bar and finalize it (`is_closed = 1`) on candle close without creating duplicates.
- **Clustered B-Tree locality:** `WITHOUT ROWID` physically orders candles by `(provider, symbol, timeframe, open_time_ms)`, eliminating a secondary index and making range scans fast across millions of bars.
- **Concurrency:** Configured with `PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout`, and an internal `RLock` so multi-asset background collectors and the Web UI / Alerts API can access the database concurrently without lock errors.

### `sync_ranges` Table
Records half-open intervals `[start_ms, end_ms)` that were already queried from an external API and returned empty (for example, weekends/holidays on equity brokers or exchange maintenance windows). `GapDetector` subtracts verified sync ranges so the collector never retries known-empty periods in an infinite loop.

### `predictions` Table
Logs live and replay neural-network prediction payloads alongside `(asset_id, provider, symbol, timeframe, anchor_open_time_ms, created_at_ms, model_name)` for historical auditability and live accuracy tracking.

### `alerts` Table
Stores triggered trading alerts (`AlertEvent`) with a monotonically increasing `id`, `(asset_id, provider, symbol, timeframe, anchor_open_time_ms, created_at_ms, action, confidence, consecutive_hits, rule_name, payload_json, acknowledged)`:
- Supports **cursor-based polling** via `get_alerts(since_id=N)` (returning `id > N` in ascending `id` order) so external notification consumers (Telegram, WhatsApp, email bots) never miss or duplicate alerts across restarts.
- Supports per-asset filtering, unacknowledged-only queries, O(log N) existence/count checks (`alert_exists(alert_id)`, `count_alerts(...)`), and explicit acknowledgement via `acknowledge_alert(alert_id)`.

---

## 3. Half-Open Time Intervals & Gap Detection (`src/tradingbot/storage/gap_detector.py`)

All ranges use **UTC epoch milliseconds** and **half-open semantics `[start_ms, end_ms)`**. Weekly intervals (`1w`, `Nw`) anchor to **Monday 00:00:00 UTC** (`anchor_offset_ms = 4 * 86_400_000`, matching ISO-8601 and Binance weekly candles rather than Thursday 1970-01-01 Unix epoch):

```text
interval = 1h (3,600,000 ms)
Range [0h, 3h) contains open_time_ms ∈ {0h, 1h, 2h}  →  candle_count = (3h - 0h) / 1h = 3
```

### Key Behaviors
1. **Unclosed-candle safety:** `GapDetector.find_missing_ranges()` queries `closed_only=True`. If the process crashed mid-candle with an unclosed bar persisted (`is_closed = 0`), restarting automatically treats that bar as missing and re-fetches its finalized OHLCV values.
2. **Sync planning (`GapDetector.plan_sync_ranges`):**
   - **Empty DB with `initial_history_from_ms`:** schedules `[align_ceil(initial_history_from_ms), align_floor(now_ms))`.
   - **Empty DB without `initial_history_from_ms`:** schedules the latest `min_candles_when_empty` closed candles.
   - **Populated DB (`fill_internal_gaps=True`):** scans from `min(stored_min, initial_history_from_ms)` to `align_floor(now_ms)`, returning every head, internal, and tail gap.
   - **Populated DB (`fill_internal_gaps=False`):** scans only the tail gap `[stored_max + interval_ms, align_floor(now_ms))` in O(1) relative to historical depth.
3. **API pagination (`chunk_range`):** splits any large missing range into consecutive sub-ranges bounded by `provider.max_candles_per_request` (e.g., 1000 candles per Binance request).
