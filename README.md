# TradingBot

A modular, transparent, zero-hardcode **Python** multi-asset market data collector, technical indicator engine, dataset generator, neural-network prediction client, alert API, and 24/7 web dashboard designed to integrate seamlessly with the C++17 **`NeuralNetwork` (`libnn` v0.6.0)** training engine and HTTP inference server.

## Features (`v0.6.4`)

- **Zero External Dependencies (Python 3.11+ Standard Library Only):** Runs out of the box on any Linux PC, 24/7 server, or Raspberry Pi with zero `pip install`, zero Node.js build tools, and zero external CDNs.
- **Multi-Asset & Multi-Timeframe Architecture:** Manage independent logical assets (ships ready with `btc_1h`, `btc_4h`, and `btc_1d`, easily extensible to `nvda_1h`, `sp500_1d`, etc.) via strictly validated modular JSON files (`configs/bot.json`, `configs/assets/*.json`, `configs/providers/*.json`).
- **Local SQLite Database & Automatic Gap Healing:** Clustered `WITHOUT ROWID` SQLite storage (`data/market.db`, WAL mode) with Monday-aligned `1w` support, head/internal/tail gap detection, unclosed-candle crash recovery, verified empty `sync_ranges`, prediction audit log, and cursor-based `alerts` table.
- **Declarative REST & Deterministic Synthetic Providers:** JSON-driven `GenericRestProvider` (Binance Spot preset + generic REST/broker mapping with rate-limiting & retry backoff) and O(1) chunk-invariant `SyntheticProvider`.
- **Extensible Indicator Engine & `NeuralNetwork` v0.6.0 Dataset Exporter:** Plugin registry (`SMA`, `EMA`, `RSI`, `ATR`, `BollingerBands`, LazyBear `SqueezeMomentum`), strict warm-up exclusion, window-relative normalization, Triple Barrier labeling, `feature_schema.json` SHA-256 fingerprinting, `%.9g` CSV export, and strictly valid `nn_config.json` generation.
- **Multi-Model `NeuralNetwork` Client, Trade Analysis & Alert Engine:** Consumes `server_app` (`/health`, `/info`, `/predict`), reconstructs translucent predicted OHLC candles, performs dual-model cross-confirmation, evaluates stateful alert rules (`consecutive_candles`, `cooldown_candles`), and dispatches outbound webhooks.
- **Embedded 24/7 Web Dashboard & External Alerts API:** Built-in HTTP/1.1 + native RFC 6455 WebSockets (`/ws/live`, `/ws/alerts`), responsive dark UI for LAN/Tailscale access, HiDPI Canvas financial charts (sliding window, volume, overlays, RSI, 4-color Squeeze Momentum, translucent predicted candles with TP/SL lines), watchlist, and cursor-based `/api/alerts` + `/ack` API for Telegram/WhatsApp/email bots.

## Quick Start

### Requirements
- Python 3.11+ (no external packages required)

### Run Full Test Suite
```bash
./scripts/test.sh
```

### Core CLI Workflows
```bash
# 1) Backfill historical Binance Spot BTC 1h candles into data/market.db
./scripts/backfill.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-02-01

# 2) Export CSV datasets + NeuralNetwork v0.6.0 nn_config.json (SQLite or offline synthetic)
./scripts/export_dataset.sh configs/bot.json --asset btc_1h --kind both --synthetic-candles 2000

# 3) Replay historical candles (requires stored candles from step 1 and running server_app on configured port)
./scripts/replay.sh configs/bot.json --asset btc_1h --verbose

# 4) Run 24/7 continuous live collector + Web Dashboard & Alerts API (LAN / Tailscale)
./scripts/live.sh configs/bot.json --web-host 0.0.0.0 --web-port 8000
```

## Documentation

All project specifications, architecture decisions, and workflows live in [`docs/`](docs/):

- [`docs/00-project-specification.md`](docs/00-project-specification.md) — Full project requirements, multi-asset & dual/multi-model NN architecture, alert engine & external API contract, and milestone roadmap.
- [`docs/01-git-workflow.md`](docs/01-git-workflow.md) — Simplified Solo Git Flow, Conventional Commits, and release conventions.
- [`docs/02-storage-and-gap-detection.md`](docs/02-storage-and-gap-detection.md) — SQLite schema (`candles`, `sync_ranges`, `predictions`, `alerts`), `WITHOUT ROWID` indexing, half-open time ranges, and gap detection algorithm.
- [`docs/03-providers-and-backfill.md`](docs/03-providers-and-backfill.md) — Declarative JSON REST provider, deterministic synthetic generator, `SyncService`, and `backfill_app.py` usage.
- [`docs/04-indicators-features-and-datasets.md`](docs/04-indicators-features-and-datasets.md) — Indicator registry, warm-up rules, window-relative normalization, Triple Barrier labeling, and `NeuralNetwork` v0.6.0 dataset/config export.
- [`docs/05-nn-client-analysis-and-alerts.md`](docs/05-nn-client-analysis-and-alerts.md) — `NeuralNetwork` HTTP client, schema verification, OHLC reconstruction, trade analysis, and stateful alert evaluation.
- [`docs/06-runtime-replay-and-webhooks.md`](docs/06-runtime-replay-and-webhooks.md) — Continuous multi-asset orchestrator, EventBus, outbound webhooks, historical replay, and CLI apps.
- [`docs/07-web-dashboard-and-api.md`](docs/07-web-dashboard-and-api.md) — Embedded 24/7 Web Dashboard, Canvas charting engine, REST/WebSocket API reference, LAN/Tailscale setup, and external alert bot example.
