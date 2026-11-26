#!/usr/bin/env python3
"""Thin CLI app running continuous multi-asset live ingestion, predictions, and alerts.

Examples:
    ./scripts/live.sh configs/bot.json --headless
    ./scripts/live.sh configs/bot.json --asset btc_1h --mode live_only --headless
    ./scripts/live.sh configs/bot.json --headless --max-cycles 3
"""

from __future__ import annotations

import argparse
import dataclasses
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tradingbot.config import AssetConfig, ConfigError, load_config  # noqa: E402
from tradingbot.models import format_ms_to_iso  # noqa: E402
from tradingbot.nn_client import NNClientError, SchemaMismatchError  # noqa: E402
from tradingbot.runtime import BotOrchestrator, PollOutcome  # noqa: E402
from tradingbot.storage import SQLiteCandleStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_app",
        description="Run continuous multi-asset TradingBot runtime (sync_live / live_only).",
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
        help="Run disabled assets as well when --asset is omitted.",
    )
    parser.add_argument(
        "--mode",
        choices=("sync_live", "live_only"),
        default="",
        help="Optional runtime mode override applied to selected assets.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without starting the embedded Web UI server.",
    )
    parser.add_argument(
        "--strict-nn",
        action="store_true",
        help="Abort immediately on startup if any enabled NeuralNetwork server is unreachable.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional maximum polling cycles before graceful exit (useful for smoke tests).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress routine per-poll status lines (still prints alerts and errors).",
    )
    return parser


def _select_assets(
    all_assets: tuple[AssetConfig, ...],
    requested_ids: list[str],
    include_disabled: bool,
    mode_override: str,
) -> list[AssetConfig]:
    if requested_ids:
        by_id = {a.id: a for a in all_assets}
        selected: list[AssetConfig] = []
        for aid in requested_ids:
            if aid not in by_id:
                raise ConfigError(
                    f"Unknown asset_id '{aid}'. Available: {', '.join(sorted(by_id.keys()))}"
                )
            selected.append(by_id[aid])
    elif include_disabled:
        selected = list(all_assets)
    else:
        selected = [a for a in all_assets if a.enabled]

    if mode_override:
        selected = [
            dataclasses.replace(
                a, runtime=dataclasses.replace(a.runtime, mode=mode_override)
            )
            for a in selected
        ]
    return selected


def _log_outcome(outcome: PollOutcome, *, quiet: bool) -> None:
    if outcome.error:
        print(f"[{outcome.asset_id}] ERROR: {outcome.error}", file=sys.stderr)
    if outcome.alert is not None:
        a = outcome.alert
        print(
            f"[{a.asset_id}] *** ALERT #{a.alert_id} {a.action} "
            f"(conf={a.confidence:.1%}, streak={a.consecutive_hits}, "
            f"bar={format_ms_to_iso(a.anchor_open_time_ms)}) ***"
        )
        for wh in outcome.webhook_results:
            status = f"OK ({wh.status_code})" if wh.success else f"FAILED ({wh.error})"
            print(f"  webhook '{wh.webhook_name}': {status}")
    elif not quiet:
        pred_part = ""
        if outcome.prediction is not None and outcome.prediction.analysis is not None:
            an = outcome.prediction.analysis
            pred_part = f" | action={an.action} conf={an.confidence:.1%} move={an.expected_move_pct:+.2%}"
        tag = "CLOSED" if outcome.new_closed_candle else "tick"
        print(f"[{outcome.asset_id}] {tag} received={outcome.candles_received}{pred_part}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_cycles is not None and args.max_cycles < 0:
        print("Error: --max-cycles must be >= 0.", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
        selected_assets = _select_assets(cfg.assets, args.assets, args.all_assets, args.mode)
        if not selected_assets:
            print("No enabled assets selected. Use --asset <id> or --all-assets.")
            return 0

        if not args.headless and cfg.web.enabled:
            print(
                f"Note: Web Dashboard server is scheduled for v0.6.0; "
                f"running headless collector loop (configured bind {cfg.web.host}:{cfg.web.port})."
            )

        with SQLiteCandleStore(
            cfg.storage.db_path,
            wal_mode=cfg.storage.wal_mode,
            busy_timeout_ms=cfg.storage.busy_timeout_ms,
        ) as store:
            orchestrator = BotOrchestrator(cfg, store, assets=selected_assets)

            def _handle_signal(_signum: int, _frame: Any) -> None:
                print("\nShutdown signal received; stopping gracefully...")
                orchestrator.stop()

            prev_sigint = signal.getsignal(signal.SIGINT)
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            try:
                signal.signal(signal.SIGINT, _handle_signal)
                signal.signal(signal.SIGTERM, _handle_signal)
            except ValueError:
                # Signal registration only works on main thread (e.g. skipped inside threaded tests).
                prev_sigint = None
                prev_sigterm = None

            try:
                asset_list = ", ".join(a.id for a in selected_assets)
                print(f"Starting TradingBot live runtime for assets: {asset_list}")
                cycles = orchestrator.run_loop(
                    strict_nn_startup=args.strict_nn,
                    max_cycles=args.max_cycles,
                    on_outcome=lambda out: _log_outcome(out, quiet=args.quiet),
                )
                print(f"TradingBot stopped cleanly after {cycles} poll cycle(s).")
            finally:
                if prev_sigint is not None:
                    signal.signal(signal.SIGINT, prev_sigint)
                if prev_sigterm is not None:
                    signal.signal(signal.SIGTERM, prev_sigterm)

        return 0
    except (ConfigError, SchemaMismatchError, NNClientError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
