# Continuous Multi-Asset Runtime, Historical Replay & Webhooks (`v0.5.0`)

This document describes the live multi-asset orchestrator (`BotOrchestrator`), thread-safe state & event bus (`RuntimeState`, `EventBus`), outbound webhook dispatcher (`WebhookDispatcher`), historical `ReplayEngine`, and CLI apps (`apps/live_app.py`, `apps/replay_app.py`).

---

## 1. Live Orchestrator Architecture (`BotOrchestrator` & `AssetRunner`)

```text
BotOrchestrator (per-asset poll scheduler, graceful SIGINT/SIGTERM stop)
       │
       ├── AssetRunner("btc_1h", poll=5s)  ──► SyncService / GapDetector ──► SQLiteCandleStore
       │                                   ──► IndicatorEngine + PredictionEngine ──► server_app(s)
       │                                   ──► AlertEvaluator ──► SQLite alerts + WebhookDispatcher
       └── AssetRunner("btc_1w", poll=30s) ──► ...
       │
       ├──► RuntimeState (thread-safe RAM snapshots for REST /api/assets/*)
       └──► EventBus     (thread-safe pub/sub events for WebSocket /ws/live & /ws/alerts)
```

### Runtime Modes
- **`sync_live`:** On startup, detects and repairs all missing closed history (head, internal, tail, and previously unclosed bars) up to now, loads the working RAM buffer, computes indicators, and continuously polls for new candles. If a poll pauses longer than one interval, missed closed bars are healed automatically via `SyncService.backfill_asset()`.
- **`live_only`:** Fetches only the minimum bar buffer needed for warm-up + visible candles + NN window without touching older historical gaps (honors `persist_to_db: false` for strictly read-only monitoring).
- **Resilient 24/7 NeuralNetwork startup:** Genuine `SchemaMismatchError` always fails fast to prevent silent garbage predictions; transient connection failures (`NNClientError`) log an error and retry cleanly on subsequent bars unless `--strict-nn` is passed.

---

## 2. Outbound Webhooks (`WebhookDispatcher`)

Whenever a closed candle satisfies an asset's alert rules (`consecutive_candles`, `cooldown_candles`, confidence & R:R thresholds), the persisted `AlertEvent` is dispatched as an HTTP `POST` JSON envelope:

```json
{
  "event": "tradingbot.alert",
  "alert": {
    "id": 1,
    "asset_id": "btc_1h",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "action": "LONG",
    "confidence": 0.82,
    "consecutive_hits": 2,
    "payload": { "risk_reward": 2.4, "estimated_tp": 102000.0, "estimated_sl": 98500.0 }
  }
}
```
- Auth tokens are read at dispatch time from `os.environ[auth_token_env]`.
- Webhook delivery errors are isolated and reported in `WebhookDeliveryResult` without interrupting the collector loop.

---

## 3. Historical Replay Engine (`ReplayEngine`)

Steps chronologically through stored SQLite candles in O(N) time (exploiting strict causality of all built-in indicators), evaluating real `NeuralNetwork` `server_app` predictions, post-hoc directional accuracy, horizon return MAE, and alert triggers without waiting days for live candles to close.

---

## 4. CLI Usage

```bash
# Continuous multi-asset live collector (headless)
./scripts/live.sh configs/bot.json --headless

# Live-only mode for a single asset
./scripts/live.sh configs/bot.json --asset btc_1h --mode live_only --headless

# Historical replay over stored SQLite candles with per-bar verbose output
./scripts/replay.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-02-01 --verbose
```
