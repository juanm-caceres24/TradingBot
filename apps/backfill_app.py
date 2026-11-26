#!/usr/bin/env python3
"""Thin CLI app for historical candle backfill and gap repair.

Examples:
    PYTHONPATH=src python3 apps/backfill_app.py configs/bot.json --start 2025-01-01 --end 2025-02-01
    PYTHONPATH=src python3 apps/backfill_app.py configs/bot.json --asset btc_1h --sync-now
    PYTHONPATH=src python3 apps/backfill_app.py configs/bot.json --asset btc_1h --start 2024-01-01 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Allow running directly as `python3 apps/backfill_app.py` without editable install.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tradingbot.config import AssetConfig, ConfigError, load_config  # noqa: E402
from tradingbot.models import TimeRange, format_ms_to_iso, parse_iso_to_ms  # noqa: E402
from tradingbot.providers import ProviderError, SyncService, default_clock_ms  # noqa: E402
from tradingbot.storage import SQLiteCandleStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_app",
        description="Download historical OHLCV candles and repair missing gaps into local SQLite.",
    )
    parser.add_argument(
        "config",
        help="Path to root TradingBot JSON configuration (e.g. configs/bot.json).",
    )
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        default=[],
        help="Target specific asset_id (repeatable). Defaults to all enabled assets.",
    )
    parser.add_argument(
        "--all-assets",
        action="store_true",
        help="Include disabled assets from the configuration when --asset is omitted.",
    )
    parser.add_argument(
        "--start",
        default="",
        help="Inclusive UTC ISO-8601 start timestamp (e.g. 2025-01-01 or 2025-01-01T00:00:00Z).",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Exclusive UTC ISO-8601 end timestamp (defaults to current UTC time when --start is given).",
    )
    parser.add_argument(
        "--sync-now",
        action="store_true",
        help="Use GapDetector.plan_sync_ranges to automatically repair missing history up to now.",
    )
    parser.add_argument(
        "--min-candles",
        type=int,
        default=None,
        help="Minimum closed candles to fetch when the database is empty (only with --sync-now).",
    )
    parser.add_argument(
        "--force-recheck",
        action="store_true",
        help="Ignore cached empty sync_ranges and re-query external API for missing periods.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display planned missing ranges without making API requests or modifying the DB.",
    )
    return parser


def _select_assets(
    all_assets: tuple[AssetConfig, ...],
    requested_ids: list[str],
    include_disabled: bool,
) -> list[AssetConfig]:
    if requested_ids:
        by_id = {a.id: a for a in all_assets}
        selected: list[AssetConfig] = []
        for asset_id in requested_ids:
            if asset_id not in by_id:
                available = ", ".join(sorted(by_id.keys()))
                raise ConfigError(
                    f"Unknown asset_id '{asset_id}'. Available assets: {available}"
                )
            selected.append(by_id[asset_id])
        return selected

    if include_disabled:
        return list(all_assets)
    return [a for a in all_assets if a.enabled]


def _print_progress(
    asset_id: str,
    chunk_idx: int,
    total_chunks: int,
    chunk: TimeRange,
    fetched_count: int,
) -> None:
    print(
        f"  [{asset_id}] chunk {chunk_idx}/{total_chunks} "
        f"[{format_ms_to_iso(chunk.start_ms)} -> {format_ms_to_iso(chunk.end_ms)}) "
        f"fetched={fetched_count}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sync_now and (args.start or args.end):
        print("Error: --sync-now cannot be combined with --start or --end.", file=sys.stderr)
        return 2
    if not args.sync_now and not args.start:
        print("Error: specify either --start <ISO> (optional --end <ISO>) or --sync-now.", file=sys.stderr)
        return 2
    if args.min_candles is not None and args.min_candles < 1:
        print("Error: --min-candles must be >= 1.", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
        target_assets = _select_assets(cfg.assets, args.assets, args.all_assets)
        if not target_assets:
            print("No enabled assets selected. Use --asset <id> or --all-assets.")
            return 0

        start_ms: int | None = parse_iso_to_ms(args.start) if args.start else None
        end_ms: int | None = parse_iso_to_ms(args.end) if args.end else None
        if start_ms is not None and end_ms is None:
            end_ms = default_clock_ms()
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            print("Error: --end must be strictly greater than --start.", file=sys.stderr)
            return 2

        respect_verified = not args.force_recheck

        with SQLiteCandleStore(
            cfg.storage.db_path,
            wal_mode=cfg.storage.wal_mode,
            busy_timeout_ms=cfg.storage.busy_timeout_ms,
        ) as store:
            sync_service = SyncService(store)
            for asset in target_assets:
                tf = asset.market.parsed_timeframe
                print(
                    f"=== Asset '{asset.id}' ({asset.provider.name} | "
                    f"{asset.market.symbol} {tf.label}) ==="
                )
                if args.sync_now:
                    report = sync_service.sync_asset_to_now(
                        asset,
                        min_candles_when_empty=args.min_candles,
                        respect_verified_ranges=respect_verified,
                        dry_run=args.dry_run,
                        progress_cb=_print_progress,
                    )
                else:
                    assert start_ms is not None and end_ms is not None
                    report = sync_service.backfill_asset(
                        asset,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        respect_verified_ranges=respect_verified,
                        dry_run=args.dry_run,
                        progress_cb=_print_progress,
                    )

                total_missing_bars = sum(r.candle_count(tf) for r in report.planned_ranges)
                if args.dry_run:
                    print(
                        f"  [DRY RUN] missing_ranges={len(report.planned_ranges)}, "
                        f"missing_candles={total_missing_bars}"
                    )
                    for r in report.planned_ranges:
                        print(
                            f"    - [{format_ms_to_iso(r.start_ms)} -> {format_ms_to_iso(r.end_ms)}) "
                            f"({r.candle_count(tf)} bars)"
                        )
                else:
                    stored_count = store.count_candles(
                        asset.provider.name, asset.market.symbol, tf.label
                    )
                    print(
                        f"  Done in {report.duration_ms} ms | chunks={report.chunks_fetched} "
                        f"| upserted={report.candles_upserted} "
                        f"| empty_ranges_recorded={report.empty_ranges_recorded} "
                        f"| total_stored={stored_count}"
                    )

        return 0
    except (ConfigError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
