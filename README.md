# TradingBot

A modular, transparent, zero-hardcode **Python** multi-asset market data collector, technical indicator engine, dataset generator, neural-network prediction client, alert API, and 24/7 web dashboard designed to integrate seamlessly with the C++17 **`NeuralNetwork` (`libnn` v0.6.0)** training engine and HTTP inference server.

## Core Principles

- **Simplicity First (YAGNI):** Clean, readable Python with minimal dependencies (zero external dependencies for core storage, ingestion, indicators, feature pipeline, dataset export, NN HTTP client, and alert evaluation).
- **Zero Hardcoded Parameters:** Multi-asset definitions, exchange/API presets, timeframes, indicators, feature windows, per-asset neural-network servers, alert rules, webhooks, and web bindings live strictly in external JSON files.
- **Multi-Asset & Multi-Timeframe Native:** Track independent assets and timeframes (e.g., `btc_1h`, `btc_1w`, `nvda_1h`, `sp500_1d`) side-by-side, each with its own provider, indicator set, dedicated/shared/disabled neural networks, and alert thresholds.
- **Modular Core + Thin Apps:** All reusable logic lives in `src/tradingbot/`; thin CLI entrypoints live in `apps/`.
- **Living Documentation & Incremental Releases:** Specifications, architecture notes, and workflow conventions live in [`docs/`](docs/) and evolve alongside unit-tested feature branches.

## Current Status (`v0.2.0`)

- Strict multi-asset JSON configuration loader (`tradingbot.config`) supporting modular `configs/bot.json`, per-asset files (`configs/assets/*.json`), provider presets (`configs/providers/*.json`), extensible `extra_models`, alert rules, and outbound webhooks.
- Canonical domain models (`tradingbot.models`): `Timeframe`, `Candle`, `TimeRange`, prediction/analysis bundles, and `AlertEvent`.
- Local SQLite persistence (`tradingbot.storage.SQLiteCandleStore`) with WAL mode, composite primary keys (`WITHOUT ROWID`), `sync_ranges`, `predictions` audit logs, and cursor-queryable `alerts`.
- Gap detection & sync planning (`tradingbot.storage.GapDetector`) supporting tail gaps, internal gaps, head backfill ranges, unclosed-candle crash recovery, verified empty ranges, and pagination chunking.
- Market data providers & backfill engine (`tradingbot.providers`): JSON-driven `GenericRestProvider` (Binance Spot preset + generic REST/broker support), deterministic `SyntheticProvider`, multi-asset `SyncService`, and CLI app `apps/backfill_app.py`.

## Quick Start

### Requirements
- Python 3.11+ (standard library only through `v0.2.0`)

### Run Unit Tests
```bash
./scripts/test.sh
```

### Backfill Historical Data
```bash
# Preview missing ranges for BTC 1h without calling the API
./scripts/backfill.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-01-10 --dry-run

# Download BTC 1h candles from Binance Spot into data/market.db
./scripts/backfill.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-01-10

# Auto-sync all enabled assets up to now
./scripts/backfill.sh configs/bot.json --sync-now
```

## Documentation

All project specifications, architecture decisions, and workflows live in [`docs/`](docs/):

- [`docs/00-project-specification.md`](docs/00-project-specification.md) — Full project requirements, multi-asset & dual/multi-model NN architecture, alert engine & external API contract, and milestone roadmap.
- [`docs/01-git-workflow.md`](docs/01-git-workflow.md) — Simplified Solo Git Flow, Conventional Commits, and release conventions.
- [`docs/02-storage-and-gap-detection.md`](docs/02-storage-and-gap-detection.md) — SQLite schema (`candles`, `sync_ranges`, `predictions`, `alerts`), `WITHOUT ROWID` indexing, half-open time ranges, and gap detection algorithm.
- [`docs/03-providers-and-backfill.md`](docs/03-providers-and-backfill.md) — Declarative JSON REST provider, deterministic synthetic generator, `SyncService`, and `backfill_app.py` usage.
