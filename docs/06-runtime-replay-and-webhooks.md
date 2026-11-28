# Continuous Multi-Asset Runtime, Historical Replay & Webhooks (`v0.5.0`)

This document describes the live multi-asset orchestrator (`BotOrchestrator`), thread-safe state & event bus (`RuntimeState`, `EventBus`), outbound webhook dispatcher (`WebhookDispatcher`), historical `ReplayEngine`, and CLI apps (`apps/live_app.py`, `apps/replay_app.py`).

---

## 1. Live Orchestrator Architecture (`BotOrchestrator` & `AssetRunner`)

```text
BotOrchestrator (per-asset poll scheduler, thread-safe AssetRunner locks, SIGINT/SIGTERM stop)
       │  (example user-configured assets in local configs/bot.json; 0 assets also supported)
       ├── AssetRunner("btc_15m", poll=5s) ──► SyncService / GapDetector ──► SQLiteCandleStore
       ├── AssetRunner("btc_1h", poll=5s)  ──► SyncService / GapDetector ──► SQLiteCandleStore
       ├── AssetRunner("btc_4h", poll=15s) ──► IndicatorEngine + PredictionEngine ──► server_app(s)
       └── AssetRunner("btc_1d", poll=30s) ──► AlertEvaluator ──► SQLite alerts + WebhookDispatcher
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
    "provider": "binance_spot",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "anchor_open_time_ms": 1735689600000,
    "created_at_ms": 1735693200100,
    "action": "LONG",
    "confidence": 0.82,
    "consecutive_hits": 2,
    "rule_name": "consecutive_2_bars",
    "payload_json": "{\"risk_reward\":2.4,\"estimated_tp\":102000.0,\"estimated_sl\":98500.0}",
    "acknowledged": false,
    "payload": { "risk_reward": 2.4, "estimated_tp": 102000.0, "estimated_sl": 98500.0 }
  }
}
```
- Auth tokens are read at dispatch time from `os.environ[auth_token_env]`.
- Webhook delivery errors are isolated and reported in `WebhookDeliveryResult` without interrupting the collector loop.

---

## 3. Historical Replay Engine (`ReplayEngine`)

Steps chronologically through stored SQLite candles in O(N) time (exploiting strict causality of all built-in indicators), evaluating real `NeuralNetwork` `server_app` predictions and alert triggers without waiting days for live candles to close.

`ReplaySummary` and `apps/replay_app.py` report three complementary post-hoc diagnostic layers (and support `--signal-only` to benchmark `signal_classifier` independently of `candle_predictor`):
- **All-bar candle forecast metrics:** `directional_acc` and `horizon_MAE` across all bars with a valid future horizon `t+H` (when `candle_predictor` is enabled).
- **HOLD rejection funnel (`hold_reasons`):** counts exactly how many `HOLD` bars were filtered by `signal_threshold`, `conflict`, `min_move`, or `min_rr`.
- **Filtered actionable signal metrics (`LONG` / `SHORT` only):** works across dual-model, candle-only, and signal-only modes, reporting `signal_trades_evaluated`, directional accuracy overall and per side (`LONG` / `SHORT`), mean signed horizon return (`signal_avg_return_pct`), and conservative Triple-Barrier outcome counts (`tb_tp_hits`, `tb_sl_hits`, `tb_timeouts`, where same-bar SL+TP collisions resolve conservatively to SL, matching `FeaturePipeline.encode_signal_target`).

---

## 4. CLI Usage

*(The `config` path defaults to `configs/bot.json` and auto-bootstraps minimal defaults if missing. When `web.enabled=true` and `--headless` is omitted, `live.sh` starts the Web Dashboard cleanly even with 0 configured assets).*

```bash
# Start live collector + embedded Web Dashboard (auto-creates configs/bot.json if missing)
./scripts/live.sh --web-host 0.0.0.0 --web-port 8000

# Continuous multi-asset live collector (headless, optional --all-assets, --strict-nn, --max-cycles, --quiet)
./scripts/live.sh configs/bot.json --headless

# Live-only mode for a single configured asset
./scripts/live.sh --asset demo_synthetic_1h --mode live_only --headless

# Historical replay over stored SQLite candles
# (supports --start, --end, --signal-only, --step-delay, --persist-predictions, --persist-alerts, --dispatch-webhooks, --verbose)
./scripts/replay.sh --asset btc_1h --start 2025-01-01 --end 2025-02-01 --verbose
```
