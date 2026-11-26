# Storage Engine & Gap Detection (`SQLiteCandleStore`, `GapDetector`)

This document describes the local persistence layer in `src/tradingbot/storage/` and the half-open time-interval model used for gap detection and historical synchronization.

---

## 1. Design Philosophy

- **Raw OHLCV only in the database:** `candles` stores exclusively raw market bars (`open`, `high`, `low`, `close`, `volume`, timestamps, and `is_closed`). Technical indicators are computed dynamically by the indicator engine so adding or editing indicators never requires database migrations or re-downloading historical data.
- **Zero external database servers:** Uses Python's built-in `sqlite3` module storing a single local file inside gitignored `data/` (e.g. `data/market.db`).
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
- **Clustered B-Tree locality:** `WITHOUT ROWID` physically orders candles by `(provider, symbol, timeframe, open_time_ms)`, eliminating a secondary index and making range scans fast even across millions of 1-minute bars.
- **Concurrency:** Configured with `PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout`, and an internal `RLock` so the background collector and Web UI can access the database concurrently without lock errors.

### `sync_ranges` Table
Records half-open intervals `[start_ms, end_ms)` that were already queried from an external API and returned empty (for example, weekends/holidays on traditional brokers or exchange maintenance windows). `GapDetector` subtracts verified sync ranges so the collector never retries known-empty periods in an infinite loop.

### `predictions` Table
Logs live and replay neural-network prediction payloads alongside `(provider, symbol, timeframe, anchor_open_time_ms, created_at_ms, model_name)` for historical auditability and live accuracy tracking.

---

## 3. Half-Open Time Intervals & Gap Detection (`src/tradingbot/storage/gap_detector.py`)

All ranges use **UTC epoch milliseconds** and **half-open semantics `[start_ms, end_ms)`**:

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
