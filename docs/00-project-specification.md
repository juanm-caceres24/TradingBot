# Project Specification — TradingBot

This document is the single source of truth for the requirements, architectural decisions, data formats, module interfaces, `NeuralNetwork` integration contract, external alert API, and milestone roadmap of the **TradingBot** project.

---

## 1. Vision & Objectives

Build a **modular, transparent, 24/7 multi-asset quantitative market data, indicator, neural-network prediction, and alerting platform** in modern **Python (3.11+)** that integrates cleanly with the companion C++17 **`NeuralNetwork` (`libnn` v0.7.0)** project.

Core guiding principles (mirrored directly from `NeuralNetwork`):
- **Full transparency & control:** Every candle, indicator formula, feature normalization transform, gap-fill request, alert evaluation, and neural-network payload is directly implemented and inspectable.
- **Simplicity first (YAGNI):** Keep the entire codebase dependency-free (Python 3.11+ standard library only, zero `pip install`, zero Node.js build tools, zero external CDNs).
- **Zero hardcoded parameters:** Root settings, per-asset profiles, exchange API endpoints, field mappings, database paths, indicator definitions, feature windows, per-asset neural-network servers, alert rules, webhooks, and web bindings live strictly in external JSON files. Unknown JSON keys are rejected at load time.
- **Standardized engineering:** Modular library (`src/tradingbot/`) + thin executable apps (`apps/`), incremental unit testing, living Markdown documentation in `docs/`, and a disciplined Simplified Solo Git Flow.

---

## 2. Target Environment & Use Cases

| Priority | Target Environment | Primary Use Cases |
| :--- | :--- | :--- |
| **Primary** | 24/7 Linux Server (headless or desktop) | Continuous multi-asset OHLCV ingestion into a shared local SQLite database with automatic gap healing; real-time indicator calculation; querying per-asset local `NeuralNetwork` `server_app` processes; evaluating alert rules; serving a lightweight Web UI and external Alerts API reachable over LAN or Tailscale from any browser or companion notification bot (Telegram, WhatsApp, email). |
| **Secondary** | Local Developer Workstation | Running controlled per-asset historical backfills (e.g., 2016–2025), exporting sliding-window CSV datasets and per-asset `nn_config.json` files for offline C++17 `train_app` training, and replaying historical candles to validate models and alert rules. |

---

## 3. Functional Requirements

### 3.1 Multi-Asset & Multi-Timeframe Management
- Track an arbitrary, config-driven list of logical **assets** (identified by unique URL-safe `asset_id`, combining symbol and timeframe — e.g., `btc_1h`, `btc_4h`, `btc_1d`, `sp500_1h`, `nvda_1h`).
- Each asset lives in its own modular JSON file under `configs/assets/` (referenced via `config_path` in `configs/bot.json`, with optional inline overrides such as `"enabled": false`) and independently configures:
  - Its market (`symbol`, `timeframe`, `market_schedule`: `"24/7"` enforces strict `interval_ms` contiguity in sliding windows, replay horizon checks, and consecutive alert streaks; `"sessions"` allows weekend/holiday timestamp gaps cleanly).
  - Its data provider preset (e.g., `binance_spot` for crypto, another REST preset for equities/indices, or `synthetic` for offline tests).
  - Its runtime mode, indicator list, feature window/normalization, Triple Barrier targets, dedicated/shared/disabled neural-network endpoints, analysis thresholds, and alert rules.
- Adding a new asset or timeframe requires **zero code changes**: add a JSON file in `configs/assets/` and register it in `configs/bot.json`.

### 3.2 Configurable Market Data Ingestion
- Fetch historical and real-time OHLCV candles (`open_time`, `close_time`, `open`, `high`, `low`, `close`, `volume`, `is_closed`) from external exchanges, brokers, or web APIs configured entirely via JSON.
- Support declarative REST API presets (starting with **Binance Spot public REST API**, zero API key required) specifying `base_url`, `endpoint`, query parameter mapping, interval mapping, pagination limit, rate-limiting delay, and JSON response mapping (`array_of_arrays` or `array_of_objects`).
- Support an offline **Synthetic Provider** for deterministic unit tests, UI development, and proof-of-concept training datasets without network access.
- Never store API secrets inside JSON files committed to Git; optional private broker or webhook credentials are referenced solely by environment variable name (`api_key_env`, `auth_token_env`).

### 3.3 Local Database & Gap Detection
- Persist raw OHLCV candles for all assets inside the repository's gitignored `data/` folder using a single **SQLite** database (zero external server processes, ACID transactions, WAL mode).
- Store **only raw OHLCV candles** in the `candles` table (indicators are computed dynamically so adding or modifying indicators never requires schema migrations or re-downloading market history).
- Automatically detect and fill per series `(provider, symbol, timeframe)`:
  - **Tail gaps** (e.g., stored history ends on Day 10 and the bot starts on Day 13, or stored history ends in 2025 and the bot starts in October 2026).
  - **Internal gaps** (missing intervals inside an already-downloaded range).
  - **Bounded backfill ranges** (explicit `[start, end)` historical downloads).
- Treat previously persisted **unclosed** candles (`is_closed = 0`) as incomplete on restart so interrupted bars are cleanly re-fetched once closed.
- Track verified empty/checked time windows (`sync_ranges`) so non-24/7 markets (equities, forex) or exchange maintenance windows do not cause infinite retry loops.
- Log live/replay predictions (`predictions` table) and triggered alerts (`alerts` table) for post-hoc auditability and reliable external consumption.

### 3.4 Extensible Technical Indicator Engine
- Registry-based plugin architecture (`BaseIndicator`) exposing `warmup_period()`, `output_columns()`, and `compute(candles)`.
- Built-in initial indicators: `SMA`, `EMA`, `RSI`, `ATR`, `BollingerBands`, and `SqueezeMomentum` (LazyBear-style Bollinger vs. Keltner squeeze + linear-regression momentum).
- Each configured indicator specifies `id`, `type`, `params`, `use_in_nn` (`bool`), and `plot` (`"overlay" | "subchart" | "none"`).
- **Strict warm-up policy:**
  - Early historical candles lacking complete indicator values **are displayed in the Web UI** (with indicator series starting once sufficient candles exist).
  - Incomplete candles/windows with missing (`None`/`NaN`) indicator values are **strictly excluded** from training CSV datasets and live neural-network inference requests.

### 3.5 Feature Normalization, Dataset Export & `NeuralNetwork` v0.7.0 Integration
- **Window-relative normalization:** Never feed raw non-stationary prices to the MLP. Every input window of $W$ candles (e.g., $W = 24$) is normalized relative to the anchor candle close $C_t$ and local volatility (`ATR` or percentage returns), with bounded indicator scaling.
- **Sliding-window dataset generation:** Given $W$ input candles (indices `i .. i + W - 1`, e.g., `0..23` = 24 candles) and $H$ target candles (`i + W .. i + W + H - 1`, e.g., `24..29` = 6 candles for `btc_1h`/`btc_4h`, or `H = 7` for `btc_1d`), slide forward by configurable `stride` starting strictly after `max_warmup`.
- **Per-asset modular multi-network architecture** aligned strictly with `libnn` v0.7.0 constraints (one loss + one output activation per `Network`):
  1. **Candle Predictor (`Huber` or `MSE` + `Linear`):** Input = $W$ normalized candles + indicators; Output = $H \times 4$ relative OHLC values used to reconstruct translucent predicted candles and projected TP/SL zones (`Huber` loss prevents extreme flash-wick outliers from distorting regression scales).
  2. **Signal Classifier (`CCE` + `Softmax`, parallel, enabled by default on `btc_1h`/`btc_4h`/`btc_1d`):** Input = identical normalized feature vector; Output = 3 mutually exclusive probabilities `[HOLD, LONG, SHORT]` trained via Triple Barrier Labeling (`mode: "pct"` or volatility-adaptive `mode: "atr_multiple"`) over the next $H$ candles (leveraging `libnn`'s numerically stable fused `Softmax + CCE` backward path).
  3. **Extensible `extra_models` map:** Supports plugging additional specialized `server_app` models per asset via JSON; `PredictionEngine` and `ReplayEngine` query enabled extra models and expose their raw output vectors in `PredictionSnapshot.extra_outputs` and JSON API responses while `candle_predictor` and `signal_classifier` drive `TradeAnalysis`.
  - Because each logical asset (`btc_1h`, `btc_4h`, `btc_1d`, `nvda_1h`) configures its own endpoints and optional `training` defaults independently, an asset can run dedicated trained models on its own ports, tune its own architecture/regularization, share an ATR-normalized model port for experimentation, or disable NNs entirely (e.g., for charting-only assets).
- **Exact `NeuralNetwork` v0.7.0 interoperability:**
  - Export `libnn`-compatible CSV files per asset (`input_cols + target_cols` numeric columns formatted with `%.9g`).
  - Auto-generate strictly valid `libnn` v0.7.0 `nn_config.json` files per asset/model from optional per-asset `training` defaults and CLI overrides: exact `input_size`/`output_size`, `Huber` (`huber_delta`) or `MSE`/`CCE`, `AdamW` decoupled `weight_decay`, deterministic hidden `dropout_rate`, `Cosine` LR schedule (`min_learning_rate`), global L2 `grad_clip_norm`, **`validation_shuffle: false`** chronological split, **`keep_best_validation_model: true`**, `early_stopping_patience` + `early_stopping_min_delta`, automatic `report_path` (from `export.report_dir`), pre-populated `server.host`/`server.port`, and configurable `export.{output_dir, dataset_path_prefix, model_save_dir, report_dir}` paths in `configs/bot.json` — zero unknown keys.
  - Consume and objectively compare `train_app` v0.7.0 JSON evaluation reports (`apps/analyze_reports_app.py` / `scripts/analyze_reports.sh`) to diagnose baseline beat, class collapse, probability miscalibration, negative $R^2$, and overfitting before live/replay deployment.
  - Save companion `feature_schema.json` files capturing indicator list, feature order, window/horizon sizes, and normalization parameters.
  - Connect to each enabled `server_app` via HTTP/1.1 + JSON (`GET /health`, `GET /info`, `POST /predict`), verifying `input_size` and `output_size` against `feature_schema.json` on startup.

### 3.6 Prediction Reconstruction, Trade Analysis & Alert Engine
- Reconstruct absolute predicted OHLC candles from relative outputs while enforcing valid candle geometry (`high = max(h, l, o, c)`, `low = min(h, l, o, c)`).
- Compute actionable trade analysis metrics (`TradeAnalysis`): recommended action (`HOLD`/`LONG`/`SHORT`), projected `expected_move_pct`, `estimated_tp`, `estimated_sl`, `risk_reward`, confidence (calibrated Softmax probability when Signal Classifier is active, or directional-consistency/R:R heuristic in candle-only mode), and human-readable reasoning.
- **Per-asset Alert Engine (`alerts` config section):**
  - Evaluates closed-candle analysis snapshots against configurable thresholds (`actions`, `min_confidence`, `min_risk_reward`, `min_expected_move_pct`).
  - **Persistence confirmation (`consecutive_candles`):** optionally requires $N$ consecutive closed candles (e.g. 2 or 3) matching the same action above threshold before emitting an alert, filtering single-bar noise.
  - **Anti-spam cooldown (`cooldown_candles`):** suppresses duplicate alerts for the same asset for a configurable number of candles after firing.
  - Persists every triggered `AlertEvent` to SQLite (`alerts` table) with a monotonically increasing `id`.

### 3.7 External Integration API & Outbound Webhooks
Decouples alert generation from notification delivery (Telegram, WhatsApp, email, custom execution bots):
- **Pull API (cursor-based, restart-safe):**
  - `GET /api/assets` — list all configured assets with latest price, signal, health, and unread alert count.
  - `GET /api/assets/{asset_id}/latest` — latest candles, indicators, prediction snapshot, and trade analysis.
  - `GET /api/alerts?since_id=<N>&asset_id=<id>&unacknowledged=true` — ordered oldest-first when `since_id` is supplied so external bots advance their cursor reliably without missing or duplicating alerts across restarts.
  - `POST /api/alerts/{alert_id}/ack` — acknowledge an alert.
- **Real-Time Stream:** `WS /ws/alerts` (dedicated alert stream) and `WS /ws/live` (live candles + predictions + alerts).
- **Optional Outbound Webhooks (`notifications.webhooks`):** HTTP `POST` JSON dispatch on alert emission using Python stdlib, keeping `TradingBot` completely independent of third-party messaging SDKs.

### 3.8 Runtime Modes
All modes are configurable per asset in JSON and overridable via CLI (targeting all enabled assets or a specific `--asset <asset_id>`):
- **`sync_live`:** Detect and backfill missing candles up to now, then run continuously in real time, updating the active candle, running predictions, evaluating alerts, and pushing updates to the Web UI / API.
- **`backfill`:** Download/repair candles strictly within a requested `[start, end)` date range and exit.
- **`live_only`:** Fetch only the minimal warm-up + visible window required for charting/inference without backfilling old historical gaps (optionally with `persist_to_db: false`).
- **`replay`:** Step chronologically through stored historical candles from SQLite via `apps/replay_app.py`, querying the NN server(s), evaluating alerts, printing quantitative accuracy metrics (`directional_acc`, `horizon_MAE`) to console, and optionally persisting predictions/alerts or dispatching webhooks.

### 3.9 Web Dashboard (24/7 Server + LAN/Tailscale Access)
- Embedded zero-dependency Python web server (`ThreadingHTTPServer` + native RFC 6455 WebSocket + REST API), bindable to configurable `host` (e.g. `0.0.0.0` or Tailscale IP) and `port` (e.g. `8000`), with `"enabled": false` / `--headless` support.
- Zero frontend build toolchain and zero external CDNs: clean HTML + CSS + vanilla JS + custom HiDPI HTML5 `<canvas>` financial charting engine (`src/tradingbot/web/static/`) supporting native per-bar translucent predicted candles, TP/SL projection lines, and LazyBear 4-color Squeeze Momentum + squeeze dots.
- Displays:
  - **Multi-asset Watchlist sidebar:** live status, price, latest signal, and active alerts for every configured asset; clicking switches the main chart immediately.
  - Sliding window of real candles (e.g., last 70 candles advancing automatically as new candles arrive).
  - Volume histogram, selected indicator overlays (`SMA`, `EMA`, `Bollinger`), and synchronized subcharts (`RSI`, `SqueezeMomentum`).
  - Predicted future candles rendered in translucent shades on the same price chart, updating with each new candle.
  - Dedicated prediction, trade-analysis, and alerts panel with an extensible toolbar for future control buttons.

---

## 4. Key Architectural & Design Decisions

### 4.1 Canonical Time Representation
- All timestamps are **UTC integer milliseconds since Unix epoch** (`open_time_ms`, `close_time_ms`).
- All query and gap intervals use **half-open ranges `[start_ms, end_ms)`**:
  - A range `[start_ms, end_ms)` with candle interval $\Delta t$ contains candles whose `open_time_ms` satisfies `start_ms <= open_time_ms < end_ms`.
  - Candle count is exact without off-by-one arithmetic: `(end_ms - start_ms) // interval_ms`.
  - Contiguous ranges merge cleanly whenever `a.end_ms == b.start_ms`.

### 4.2 Database Choice: Local SQLite (`sqlite3`)
- Zero external dependencies (Python standard library), single file inside `data/`, WAL concurrency for simultaneous multi-asset collector writes and Web UI/API reads, and clustered `WITHOUT ROWID` primary key `(provider, symbol, timeframe, open_time_ms)` for millisecond range scans and automatic deduplication.
- Encapsulated behind the `CandleStore` abstract interface so alternative storage engines remain pluggable.

### 4.3 Strict Multi-Asset Configuration Policy
- Mirroring `NeuralNetwork/include/nn/config.hpp`, `tradingbot.config` validates every JSON section strictly:
  - Unknown keys raise `ConfigError` immediately (no silent typos).
  - Booleans are never silently accepted where integers or floats are expected.
  - Root `configs/bot.json` configures `storage`, optional `export` (`output_dir`, `dataset_path_prefix`, `model_save_dir`), `web`, `notifications`, and per-asset references (`config_path`), while each asset references provider presets (`preset_path`), both supporting strictly validated inline overrides.

---

## 5. Repository Structure & Module Map

```text
TradingBot/
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/                                # Living Markdown documentation (English)
│   ├── 00-project-specification.md
│   ├── 01-git-workflow.md
│   ├── 02-storage-and-gap-detection.md
│   ├── 03-providers-and-backfill.md
│   ├── 04-indicators-features-and-datasets.md
│   ├── 05-nn-client-analysis-and-alerts.md
│   ├── 06-runtime-replay-and-webhooks.md
│   └── 07-web-dashboard-and-api.md
├── scripts/                             # Helper shell scripts
│   ├── test.sh
│   ├── backfill.sh
│   ├── export_dataset.sh
│   ├── analyze_reports.sh
│   ├── live.sh
│   └── replay.sh
├── configs/                             # External JSON configs (zero hardcoding)
│   ├── bot.json                         # Root multi-asset configuration
│   ├── assets/                          # Per-asset / per-timeframe profiles
│   │   ├── btc_1h.json
│   │   ├── btc_4h.json
│   │   └── btc_1d.json
│   └── providers/                       # Declarative exchange/synthetic presets
│       ├── binance_spot.json
│       └── synthetic.json
├── src/tradingbot/                      # Reusable core library (zero external deps)
│   ├── __init__.py
│   ├── models.py                        # Timeframe (Monday-aligned 1w), Candle, TimeRange, predictions, AlertEvent
│   ├── config.py                        # Strict multi-asset JSON loader & dynamic indicator validator
│   ├── storage/                         # CandleStore ABC, SQLiteCandleStore, GapDetector
│   ├── providers/                       # GenericRestProvider, SyntheticProvider, SyncService (backfill/sync)
│   ├── indicators/                      # Registry + SMA, EMA, RSI, ATR, BB, SqueezeMomentum
│   ├── features/                        # Normalization, window builder, FeatureSchema
│   ├── dataset/                         # libnn v0.7.0 CSV exporter, nn_config.json generator & report analyzer
│   ├── nn_client/                       # Multi-model HTTP client for server_app
│   ├── analysis/                        # Candle reconstruction, TradeAnalyzer, PredictionEngine, AlertEvaluator
│   ├── runtime/                         # Multi-asset BotOrchestrator (sync_live, live_only), ReplayEngine, EventBus, webhooks
│   └── web/                             # Stdlib ThreadingHTTPServer + RFC 6455 WebSockets + Alerts API + HiDPI Canvas UI
├── apps/                                # Thin CLI entrypoints
│   ├── backfill_app.py
│   ├── export_dataset_app.py
│   ├── analyze_reports_app.py
│   ├── live_app.py
│   └── replay_app.py
├── tests/                               # Unit & integration test suite (unittest)
└── data/                                # Gitignored local DBs, CSVs, feature schemas
```

---

## 6. Milestone Roadmap

| Release | Status | Scope | Verification Criteria |
| :--- | :--- | :--- | :--- |
| **`v0.1.0`** | Completed | Repo setup, `docs/00`–`02`, Git Flow, `models.py` (`Timeframe`, `Candle`, `TimeRange`), strict JSON `config.py`, `SQLiteCandleStore` (`WITHOUT ROWID`, WAL, sync ranges, prediction log), `GapDetector`, Binance/Synthetic JSON presets. | Unit tests covering all models, strict config validation & unknown-key rejection, SQLite CRUD/dedup/bounds, and head/tail/internal/verified gap detection (`./scripts/test.sh`). |
| **`v0.1.1`** | Completed | Multi-asset & multi-timeframe configuration architecture (`configs/bot.json` + `configs/assets/*.json`), extensible `extra_models`, per-asset `AlertRulesConfig`, outbound `NotificationsConfig` (webhooks), `AlertEvent` model, SQLite `alerts` table (cursor-based `since_id` + `ack`), and updated documentation. | Unit tests covering multi-asset config loading/overrides/validation, duplicate/invalid `asset_id` rejection, alert rules/webhook validation, and SQLite alert persistence/cursor/ack flows. |
| **`v0.2.0`** | Completed | `BaseProvider`, JSON-driven `GenericRestProvider` (Binance Spot verified), `SyntheticProvider`, paginated & rate-limited multi-asset `SyncService`, `apps/backfill_app.py` (with `--asset` filtering). | Mock HTTP server tests for REST mapping/pagination/rate-limits, synthetic determinism tests, end-to-end multi-asset backfill + gap repair tests. |
| **`v0.3.0`** | Completed | Indicator registry (`SMA`, `EMA`, `RSI`, `ATR`, `BollingerBands`, `SqueezeMomentum`), strict warm-up filtering, window-relative feature pipeline, `feature_schema.json`, per-asset `libnn` CSV exporter (Candle Predictor + Triple-Barrier Signal Classifier), auto `nn_config.json` generator, `apps/export_dataset_app.py`. | Analytical indicator verification tests, NaN/warm-up exclusion tests, CSV format & `nn_config.json` schema parity checks against `NeuralNetwork` v0.6.0. |
| **`v0.4.0`** | Completed | Multi-model `NeuralNetworkClient` (`/health`, `/info`, `/predict`, startup schema check), relative-to-OHLC candle reconstruction, trade analysis engine, and stateful `AlertEvaluator` (thresholds, `consecutive_candles`, `cooldown_candles`). | Mock `server_app` integration tests (`200`/`400`/schema mismatch), candle geometry & trade analysis tests, consecutive-hit & cooldown alert state-machine tests. |
| **`v0.5.0`** | Completed | Multi-asset continuous runtime orchestrator (`sync_live`, `live_only`, `replay`), open vs. closed candle transitions, prediction & alert DB persistence, outbound webhook dispatcher, `apps/live_app.py --headless`, `apps/replay_app.py`. | Simulated clock multi-asset live-loop tests, restart gap-recovery tests, webhook dispatch & replay pipeline tests. |
| **`v0.6.0`** | Completed | Embedded zero-dependency Web Dashboard & External API (`ThreadingHTTPServer` + native RFC 6455 WebSocket `/ws/live` & `/ws/alerts` + REST `/api/assets`, `/api/alerts?since_id=`, `/api/alerts/{id}/ack`, `/api/actions/poll` + HiDPI HTML5 Canvas financial charts): multi-asset watchlist, sliding candle window, volume, overlays, synchronized RSI & Squeeze Momentum subcharts, translucent predicted candles with TP/SL lines, prediction & alerts panel, LAN/Tailscale ready. | API & WebSocket contract tests (including RFC 6455 handshake, Ping/Pong, static asset & path-traversal safety, cursor-based external alert consumer simulation, and `live_app.py` embedded web launch). |
| **`v0.6.1`** | Completed | Additional calibrated multi-timeframe BTC asset profiles (`configs/assets/btc_4h.json`, `configs/assets/btc_1d.json`) registered in `configs/bot.json` with dedicated non-overlapping NN server ports (`8084/8085`, `8086/8087`) and timeframe-scaled Triple Barrier / alert thresholds. | Strict config validation and unique NN port verification across all configured BTC timeframes (`1h`, `4h`, `1d`, `1w`). |
| **`v0.6.2`** | Completed | Full codebase hardening & doc sync: Monday 00:00 UTC alignment for weekly (`1w`) timeframes, dynamic `INDICATOR_REGISTRY` config validation, full provider cache key, thread-safe `AssetRunner` & `BotOrchestrator.poll_asset`, resilient gap-heal error catching, SQL `alert_exists`/`count_alerts`, `market_schedule` contiguity enforcement, `extra_models` integration, HTML escaping & Canvas UI fixes, expanded regression tests, and docs/README parity. | Expanded regression test suite covering weekly alignment, concurrency, custom indicators via JSON, `pct_relative`/`stride`, `SHORT` & edge-case analysis/alerts, HTTP 400/403/404, WS backpressure, and CLI error exits (`./scripts/test.sh` + `node --check`). |
| **`v0.6.3`** | Completed | Follow-up audit polish: double-checked `_ensure_initialized()` in `BotOrchestrator`, `client.py` docstring sync, zero-safe `risk_reward` display in `app.js`, separate unreachable-socket webhook test, and clean `stderr` capture in CLI tests. | Clean `81/81` test execution with zero stray `stderr` output, double-init race verification, and JS syntax check (`./scripts/test.sh` + `node --check`). |
| **`v0.6.4`** | Completed | Removed unused `btc_1w` asset profile from shipped defaults (retaining active `btc_1h`, `btc_4h`, and `btc_1d` profiles and generic Monday-aligned `1w` timeframe support) and synchronized tests and documentation across `docs/` and `README.md`. | Clean `81/81` test suite with zero stale `btc_1w`/port references, `pyproject.toml`/`__init__.py` version parity, and JS syntax check (`./scripts/test.sh` + `node --check`). |
| **`v0.6.5`** | Completed | Added optional root `export` config block (`output_dir`, `dataset_path_prefix`, `model_save_dir`) with CLI overrides (`--output-dir`, `--dataset-path-prefix`, `--model-save-dir`) for direct `NeuralNetwork` relative dataset/model paths, and compacted `btc_4h` (`8082/8083`) and `btc_1d` (`8084/8085`) NN ports. | Strict `export` config validation, relative/absolute `nn_config.json` path assertions, compacted port verification, and full `81/81` test suite (`./scripts/test.sh`). |
| **`v0.7.0`** | Completed | Added volatility-adaptive ATR-multiple Triple Barrier labeling (`mode: "atr_multiple"`) across config, `FeaturePipeline`, `FeatureSchema`, `TradeAnalyzer`, and `ReplayEngine` (with positive TP/SL price floor), fixed RFC 6455 WebSocket magic GUID (`47DA`), hardened `export` path validation and `NeuralNetwork` train/serve hints, and recalibrated `btc_1h`, `btc_4h`, and `btc_1d` profiles (`ema_20`/`ema_55`/`sma_200`, `W=24`, dual models enabled). | Official RFC 6455 Section 1.3 WebSocket handshake vector test, ATR-multiple / extreme-ATR / sub-cent `TradeAnalyzer` tests, strict export path validation, and full `82/82` test suite (`./scripts/test.sh` + `node --check`). |
