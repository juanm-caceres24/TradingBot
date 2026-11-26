# Project Specification — TradingBot

This document is the single source of truth for the requirements, architectural decisions, data formats, module interfaces, `NeuralNetwork` integration contract, and milestone roadmap of the **TradingBot** project.

---

## 1. Vision & Objectives

Build a **modular, transparent, 24/7 quantitative market data, indicator, and neural-network prediction platform** in modern **Python (3.11+)** that integrates cleanly with the companion C++17 **`NeuralNetwork` (`libnn` v0.6.0)** project.

Core guiding principles (mirrored directly from `NeuralNetwork`):
- **Full transparency & control:** Every candle, indicator formula, feature normalization transform, gap-fill request, and neural-network payload is directly implemented and inspectable.
- **Simplicity first (YAGNI):** Keep the core dependency-free (Python standard library) and introduce optional external dependencies only where they deliver clear value (such as the embedded web dashboard server).
- **Zero hardcoded parameters:** Exchange API endpoints, field mappings, database paths, indicator definitions, feature windows, neural-network servers, analysis thresholds, and web server bindings live strictly in external JSON files. Unknown JSON keys are rejected at load time.
- **Standardized engineering:** Modular library (`src/tradingbot/`) + thin executable apps (`apps/`), incremental unit testing, living Markdown documentation in `docs/`, and a disciplined Simplified Solo Git Flow.

---

## 2. Target Environment & Use Cases

| Priority | Target Environment | Primary Use Cases |
| :--- | :--- | :--- |
| **Primary** | 24/7 Linux Server (headless or desktop) | Continuous OHLCV ingestion into a local SQLite database with automatic gap healing; real-time indicator calculation; querying one or more local `NeuralNetwork` `server_app` processes; serving a lightweight Web UI reachable over LAN or Tailscale from any phone/laptop/PC browser. |
| **Secondary** | Local Developer Workstation | Running controlled historical backfills (e.g., 2016–2025), exporting sliding-window CSV datasets and `nn_config.json` files for offline C++17 `train_app` training, and replaying historical candles to inspect model behavior. |

---

## 3. Functional Requirements

### 3.1 Configurable Market Data Ingestion
- Fetch historical and real-time OHLCV candles (`open_time`, `close_time`, `open`, `high`, `low`, `close`, `volume`, `is_closed`) from external exchanges, brokers, or web APIs configured entirely via JSON.
- Support declarative REST API presets (starting with **Binance Spot public REST API**, zero API key required) specifying `base_url`, `endpoint`, query parameter mapping, interval mapping, pagination limit, rate-limiting delay, and JSON response mapping (`array_of_arrays` or `array_of_objects`).
- Support an offline **Synthetic Provider** for deterministic unit tests, UI development, and proof-of-concept training datasets without network access.
- Never store API secrets inside JSON files committed to Git; optional private broker keys are referenced solely by environment variable name (`api_key_env`).

### 3.2 Local Database & Gap Detection
- Persist raw OHLCV candles inside the repository's gitignored `data/` folder using **SQLite** (zero external server processes, ACID transactions, WAL mode).
- Store **only raw OHLCV candles** in the database (indicators are computed dynamically so adding or modifying indicators never requires schema migrations or re-downloading market history).
- Automatically detect and fill:
  - **Tail gaps** (e.g., stored history ends on Day 10 and the bot starts on Day 13, or stored history ends in 2025 and the bot starts in October 2026).
  - **Internal gaps** (missing intervals inside an already-downloaded range).
  - **Bounded backfill ranges** (explicit `[start, end)` historical downloads).
- Track verified empty/checked time windows (`sync_ranges`) so non-24/7 markets (stocks, forex) or exchange maintenance windows do not cause infinite retry loops.
- Log live/replay predictions in a dedicated `predictions` audit table for post-hoc accuracy inspection.

### 3.3 Extensible Technical Indicator Engine
- Registry-based plugin architecture (`BaseIndicator`) exposing `warmup_period()`, `output_columns()`, and `compute(candles)`.
- Built-in initial indicators: `SMA`, `EMA`, `RSI`, `ATR`, `BollingerBands`, and `SqueezeMomentum` (LazyBear-style Bollinger vs. Keltner squeeze + linear-regression momentum).
- Each configured indicator specifies:
  - `id`, `type`, and `params`.
  - `use_in_nn` (`bool`): whether its outputs are included in the neural-network feature vector.
  - `plot` (`"overlay" | "subchart" | "none"`): how the Web UI renders it.
- **Strict warm-up policy:**
  - Early historical candles lacking complete indicator values **are displayed in the Web UI** (with indicator series starting once sufficient candles exist).
  - Incomplete candles/windows with missing (`None`/`NaN`) indicator values are **strictly excluded** from training CSV datasets and live neural-network inference requests.

### 3.4 Feature Normalization, Dataset Export & `NeuralNetwork` v0.6.0 Integration
- **Window-relative normalization:** Never feed raw non-stationary prices to the MLP. Every input window of $W$ candles (e.g., $W = 31$) is normalized relative to the anchor candle close $C_t$ and local volatility (`ATR` or percentage returns), with bounded indicator scaling.
- **Sliding-window dataset generation:** Given $W$ input candles (indices `i .. i + W - 1`, e.g., `0..30` = 31 candles) and $H$ target candles (`i + W .. i + W + H - 1`, e.g., `31..37` = 7 candles), slide forward by configurable `stride` starting strictly after `max_warmup`.
- **Dual-model modular NN architecture** aligned strictly with `libnn` v0.6.0 constraints (one loss + one output activation per `Network`):
  1. **Candle Predictor (`MSE` + `Linear`):** Input = $W$ normalized candles + indicators; Output = $H \times 4$ relative OHLC values used to reconstruct translucent predicted candles and projected TP/SL zones.
  2. **Signal Classifier (`CCE` + `Softmax`, optional/parallel):** Input = identical normalized feature vector; Output = 3 mutually exclusive probabilities `[HOLD, LONG, SHORT]` trained via Triple Barrier Labeling over the next $H$ candles (leveraging `libnn` v0.6.0's numerically stable fused `Softmax + CCE` backward path).
  - Additional specialized models can be added later through configuration without altering the core pipeline.
- **Exact `NeuralNetwork` v0.6.0 interoperability:**
  - Export `libnn`-compatible CSV files (`input_cols + target_cols` numeric columns formatted with `%.9g`).
  - Auto-generate strictly valid `nn_config.json` files (exact `input_size`, `output_size`, `Adam`, `validation_split`, **`validation_shuffle: false`** chronological split, **`keep_best_validation_model: true`** — zero unknown keys).
  - Save a companion `feature_schema.json` capturing indicator list, feature order, window/horizon sizes, and normalization parameters.
  - Connect to one or more `server_app` instances via HTTP/1.1 + JSON (`GET /health`, `GET /info`, `POST /predict`), verifying `input_size` and `output_size` against `feature_schema.json` on startup.

### 3.5 Prediction Reconstruction & Trade Analysis
- Reconstruct absolute predicted OHLC candles from relative outputs while enforcing valid candle geometry (`high >= max(open, close)`, `low <= min(open, close)`).
- Compute actionable trade analysis metrics: projected net move, maximum favorable/adverse excursion (MFE/MAE), estimated Take Profit / Stop Loss levels, Risk-Reward ratio, statistical horizon confidence, and cross-confirmation with the Signal Classifier (`[HOLD, LONG, SHORT]`).

### 3.6 Runtime Modes
All modes are selectable via `config.json` and CLI overrides:
- **`sync_live`:** Detect and backfill missing candles up to now, then stay running in real time, updating the active candle and running predictions on candle close (or optionally intra-candle).
- **`backfill`:** Download/repair candles strictly within a requested `[start, end)` date range and exit.
- **`live_only`:** Fetch only the minimal warm-up + visible window required for charting/inference without backfilling old historical gaps (optionally with `persist_to_db: false`).
- **`replay`:** Step through stored historical candles from SQLite, querying the NN server(s) and streaming results to the UI/console for rapid visual and quantitative evaluation.

### 3.7 Web Dashboard (24/7 Server + LAN/Tailscale Access)
- Embedded Python web server (`FastAPI` + `uvicorn` + WebSocket `/ws/live` + REST API `/api/*`), bindable to configurable `host` (e.g. `0.0.0.0` or Tailscale IP) and `port` (e.g. `8000`), with `"enabled": false` / `--headless` support.
- Zero frontend build toolchain (no Node/npm/React): clean HTML + CSS + vanilla ES modules with **TradingView Lightweight Charts** vendored locally inside `src/tradingbot/web/static/vendor/` (mirroring `cJSON` vendoring in `NeuralNetwork`).
- Displays:
  - Sliding window of real candles (e.g., last 70 candles advancing automatically as new candles arrive).
  - Volume histogram and selected indicator overlays (`SMA`, `EMA`, `Bollinger`) and synchronized subcharts (`RSI`, `SqueezeMomentum`).
  - Predicted future candles rendered in translucent shades on the same price chart, updating with each new candle.
  - Dedicated prediction & trade-analysis panel (expected move, TP/SL, R:R, probabilities, model latency, connection health) with a clean layout ready for future control buttons.

---

## 4. Key Architectural & Design Decisions

### 4.1 Canonical Time Representation
- All timestamps are **UTC integer milliseconds since Unix epoch** (`open_time_ms`, `close_time_ms`).
- All query and gap intervals use **half-open ranges `[start_ms, end_ms)`**:
  - A range `[start_ms, end_ms)` with candle interval $\Delta t$ contains candles whose `open_time_ms` satisfies `start_ms <= open_time_ms < end_ms`.
  - Candle count is exact without off-by-one arithmetic: `(end_ms - start_ms) // interval_ms`.
  - Contiguous ranges merge cleanly whenever `a.end_ms == b.start_ms`.

### 4.2 Database Choice: Local SQLite (`sqlite3`)
- Zero external dependencies (Python standard library), single file inside `data/`, WAL concurrency for simultaneous collector writes and Web UI reads, and clustered `WITHOUT ROWID` primary key `(provider, symbol, timeframe, open_time_ms)` for millisecond range scans and automatic deduplication.
- Encapsulated behind the `CandleStore` abstract interface so alternative storage engines remain pluggable.

### 4.3 Strict Configuration Policy
- Mirroring `NeuralNetwork/include/nn/config.hpp`, `tradingbot.config` validates every JSON section strictly:
  - Unknown keys raise `ConfigError` immediately (no silent typos).
  - Booleans are never silently accepted where integers or floats are expected.
  - Provider configurations support clean external preset files (`preset_path`) with optional inline overrides, all strictly validated.

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
│   └── 02-storage-and-gap-detection.md
├── scripts/                             # Helper shell scripts (test.sh, etc.)
├── configs/                             # External JSON configs (zero hardcoding)
│   ├── btc_1h.json                      # Main bot configuration example
│   └── providers/                       # Declarative exchange/synthetic presets
│       ├── binance_spot.json
│       └── synthetic.json
├── src/tradingbot/                      # Reusable core library
│   ├── __init__.py
│   ├── models.py                        # Timeframe, Candle, TimeRange, prediction models
│   ├── config.py                        # Strict JSON loader & validator (AppConfig)
│   ├── storage/                         # CandleStore ABC, SQLiteCandleStore, GapDetector
│   ├── providers/                       # [v0.2.0] GenericRestProvider, SyntheticProvider
│   ├── indicators/                      # [v0.3.0] Registry + SMA, EMA, RSI, ATR, BB, Squeeze
│   ├── features/                        # [v0.3.0] Normalization, window builder, schema
│   ├── dataset/                         # [v0.3.0] libnn CSV exporter & nn_config generator
│   ├── nn_client/                       # [v0.4.0] HTTP client for NeuralNetwork server_app
│   ├── analysis/                        # [v0.4.0] Candle reconstruction & trade analysis
│   ├── runtime/                         # [v0.5.0] sync_live, backfill, live_only, replay
│   └── web/                             # [v0.6.0] FastAPI + WS + vendored Lightweight Charts
├── apps/                                # [v0.2.0+] Thin CLI entrypoints
│   ├── backfill_app.py
│   ├── export_dataset_app.py
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
| **`v0.2.0`** | Planned | `BaseProvider`, JSON-driven `GenericRestProvider` (Binance Spot verified), `SyntheticProvider`, paginated & rate-limited `SyncService`, `apps/backfill_app.py`. | Mock HTTP server tests for REST mapping/pagination/rate-limits, synthetic determinism tests, end-to-end backfill + gap repair tests. |
| **`v0.3.0`** | Planned | Indicator registry (`SMA`, `EMA`, `RSI`, `ATR`, `BollingerBands`, `SqueezeMomentum`), strict warm-up filtering, window-relative feature pipeline, `feature_schema.json`, `libnn` CSV exporter (both Candle Predictor and Triple-Barrier Signal Classifier), auto `nn_config.json` generator, `apps/export_dataset_app.py`. | Analytical indicator verification tests, NaN/warm-up exclusion tests, CSV format & `nn_config.json` schema parity checks against `NeuralNetwork` v0.6.0. |
| **`v0.4.0`** | Planned | Multi-model `NeuralNetworkClient` (`/health`, `/info`, `/predict`, startup schema check), relative-to-OHLC candle reconstruction, trade analysis & confidence scoring engine. | Mock `server_app` integration tests (`200`/`400`/schema mismatch), candle geometry & trade analysis unit tests. |
| **`v0.5.0`** | Planned | Continuous runtime orchestrator (`sync_live`, `live_only`, `replay`), open vs. closed candle transitions, prediction DB logging, `apps/live_app.py --headless`, `apps/replay_app.py`. | Simulated clock live-loop tests, restart gap-recovery tests, replay pipeline tests. |
| **`v0.6.0`** | Planned | Embedded Web Dashboard (`FastAPI` + WebSocket `/ws/live` + REST `/api/*` + locally vendored TradingView Lightweight Charts): sliding candle window, volume, overlays, synchronized subcharts, translucent predicted candles, prediction panel, LAN/Tailscale ready. | API & WebSocket contract tests, live streaming smoke test with synthetic & Binance feeds, responsive browser verification. |
