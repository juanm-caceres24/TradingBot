# TradingBot

A modular, transparent, zero-hardcode **Python** multi-asset market data collector, technical indicator engine, dataset generator, neural-network prediction client, alert API, and 24/7 web dashboard designed to integrate seamlessly with the C++17 **`NeuralNetwork` (`libnn` v0.6.0)** training engine and HTTP inference server.

## Core Principles

- **Simplicity First (YAGNI):** Clean, readable Python with minimal dependencies (zero external dependencies for core storage, ingestion, indicators, feature pipeline, dataset export, NN HTTP client, alert evaluation, live runtime, and webhooks).
- **Zero Hardcoded Parameters:** Multi-asset definitions, exchange/API presets, timeframes, indicators, feature windows, per-asset neural-network servers, alert rules, webhooks, and web bindings live strictly in external JSON files.
- **Multi-Asset & Multi-Timeframe Native:** Track independent assets and timeframes (e.g., `btc_1h`, `btc_1w`, `nvda_1h`, `sp500_1d`) side-by-side, each with its own provider, indicator set, dedicated/shared/disabled neural networks, and alert thresholds.
- **Modular Core + Thin Apps:** All reusable logic lives in `src/tradingbot/`; thin CLI entrypoints live in `apps/`.
- **Living Documentation & Incremental Releases:** Specifications, architecture notes, and workflow conventions live in [`docs/`](docs/) and evolve alongside unit-tested feature branches.

## Current Status (`v0.5.0`)

- Strict multi-asset JSON configuration (`tradingbot.config`), canonical domain models (`tradingbot.models`), local SQLite persistence (`SQLiteCandleStore`), and gap detection (`GapDetector`).
- JSON-driven REST provider (`GenericRestProvider` + Binance Spot preset), deterministic `SyntheticProvider`, multi-asset `SyncService`, and `apps/backfill_app.py`.
- Extensible indicator registry (`SMA`, `EMA`, `RSI`, `ATR`, `BollingerBands`, `SqueezeMomentum`), window-relative `FeaturePipeline`, `feature_schema.json`, `libnn` v0.6.0 CSV exporter, auto `nn_config.json` generator, and `apps/export_dataset_app.py`.
- Multi-model `NeuralNetwork` v0.6.0 HTTP client (`tradingbot.nn_client`) with startup schema verification, predicted OHLC reconstruction, dual-model `TradeAnalyzer`, `PredictionEngine`, and stateful `AlertEvaluator`.
- Continuous multi-asset runtime (`BotOrchestrator`, `RuntimeState`, `EventBus`), outbound `WebhookDispatcher`, historical `ReplayEngine` with post-hoc accuracy metrics, and thin CLI apps (`apps/live_app.py`, `apps/replay_app.py`).

## Quick Start

### Requirements
- Python 3.11+ (standard library only through `v0.5.0`)

### Run Unit Tests
```bash
./scripts/test.sh
```

### Core CLI Workflows
```bash
# 1) Backfill historical Binance Spot BTC 1h candles into data/market.db
./scripts/backfill.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-02-01

# 2) Export CSV datasets + NeuralNetwork v0.6.0 nn_config.json (SQLite or offline synthetic)
./scripts/export_dataset.sh configs/bot.json --asset btc_1h --kind both --synthetic-candles 2000

# 3) Replay historical candles against running NeuralNetwork server_app(s)
./scripts/replay.sh configs/bot.json --asset btc_1h --verbose

# 4) Run 24/7 continuous live collector, predictions, alerts, and webhooks
./scripts/live.sh configs/bot.json --headless
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
