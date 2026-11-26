# TradingBot

A modular, transparent, zero-hardcode **Python** market data collector, technical indicator engine, dataset generator, neural-network prediction client, and 24/7 web dashboard designed to integrate seamlessly with the C++17 **`NeuralNetwork` (`libnn` v0.6.0)** training engine and HTTP inference server.

## Core Principles

- **Simplicity First (YAGNI):** Clean, readable Python with minimal dependencies (zero external dependencies for core storage, ingestion, indicators, feature pipeline, dataset export, and NN HTTP client).
- **Zero Hardcoded Parameters:** All exchange/API endpoints, timeframes, indicators, window sizes, neural-network servers, analysis thresholds, and web server settings live in strictly validated external JSON configuration files.
- **Modular Core + Thin Apps:** All reusable logic lives in `src/tradingbot/`; thin CLI entrypoints live in `apps/`.
- **Living Documentation & Incremental Releases:** Specifications, architecture notes, and workflow conventions live in [`docs/`](docs/) and evolve alongside unit-tested feature branches.

## Current Status (`v0.1.0`)

- Strict JSON configuration loader (`tradingbot.config`) rejecting unknown keys and invalid types across every section.
- Canonical domain models (`tradingbot.models`): `Timeframe`, `Candle`, `TimeRange`, prediction & analysis structures.
- Local SQLite persistence (`tradingbot.storage.SQLiteCandleStore`) with WAL mode, composite primary keys (`WITHOUT ROWID`), sync-range tracking, and prediction audit logs.
- Gap detection & sync planning (`tradingbot.storage.GapDetector`) supporting tail gaps, internal gaps, head backfill ranges, verified empty ranges, and API pagination chunking.
- JSON provider presets for Binance Spot (`configs/providers/binance_spot.json`) and offline synthetic candles (`configs/providers/synthetic.json`).

## Quick Start

### Requirements
- Python 3.11+ (standard library only for `v0.1.0`)

### Run Unit Tests
```bash
./scripts/test.sh
```

## Documentation

All project specifications, architecture decisions, and workflows live in [`docs/`](docs/):

- [`docs/00-project-specification.md`](docs/00-project-specification.md) — Full project requirements, architectural decisions, JSON schemas, `NeuralNetwork` integration protocol, and milestone roadmap.
- [`docs/01-git-workflow.md`](docs/01-git-workflow.md) — Simplified Solo Git Flow, Conventional Commits, and release conventions.
- [`docs/02-storage-and-gap-detection.md`](docs/02-storage-and-gap-detection.md) — SQLite schema, `WITHOUT ROWID` indexing, half-open time ranges, and gap detection algorithm.
