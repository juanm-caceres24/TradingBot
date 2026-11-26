#!/usr/bin/env python3
"""Thin CLI app replaying stored SQLite candles against NeuralNetwork server(s).

Examples:
    ./scripts/replay.sh configs/bot.json --asset btc_1h
    ./scripts/replay.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-02-01 --verbose
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tradingbot.config import AssetConfig, ConfigError, load_config  # noqa: E402
from tradingbot.models import AlertEvent, Candle, PredictionSnapshot, format_ms_to_iso, parse_iso_to_ms  # noqa: E402
from tradingbot.nn_client import NNClientError, SchemaMismatchError  # noqa: E402
from tradingbot.runtime import ReplayEngine, WebhookDispatcher  # noqa: E402
from tradingbot.storage import SQLiteCandleStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_app",
        description="Replay historical candles from SQLite through NeuralNetwork models and alert rules.",
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
        "--start",
        default="",
        help="Optional UTC ISO-8601 start timestamp filter.",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Optional UTC ISO-8601 exclusive end timestamp filter.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between replayed bars (default: 0.0).",
    )
    parser.add_argument(
        "--persist-predictions",
        action="store_true",
        help="Write replay predictions to the SQLite predictions table.",
    )
    parser.add_argument(
        "--persist-alerts",
        action="store_true",
        help="Write triggered replay alerts to the SQLite alerts table.",
    )
    parser.add_argument(
        "--dispatch-webhooks",
        action="store_true",
        help="Send triggered replay alerts to configured outbound webhooks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-bar replay prediction details.",
    )
    return parser


def _select_assets(all_assets: tuple[AssetConfig, ...], requested: list[str]) -> list[AssetConfig]:
    if requested:
        by_id = {a.id: a for a in all_assets}
        selected: list[AssetConfig] = []
        for aid in requested:
            if aid not in by_id:
                raise ConfigError(
                    f"Unknown asset_id '{aid}'. Available: {', '.join(sorted(by_id.keys()))}"
                )
            selected.append(by_id[aid])
        return selected
    return [a for a in all_assets if a.enabled]


def _verbose_step(
    _idx: int,
    candle: Candle,
    snapshot: PredictionSnapshot,
    alert: AlertEvent | None,
) -> None:
    an = snapshot.analysis
    act = an.action if an is not None else "NONE"
    conf = an.confidence if an is not None else 0.0
    move = an.expected_move_pct if an is not None else 0.0
    alert_tag = f" [ALERT {alert.action} streak={alert.consecutive_hits}]" if alert else ""
    print(
        f"  {format_ms_to_iso(candle.open_time_ms)} close={candle.close:.4f} "
        f"action={act} conf={conf:.1%} move={move:+.2%}{alert_tag}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.step_delay < 0.0:
        print("Error: --step-delay must be >= 0.", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
        selected_assets = _select_assets(cfg.assets, args.assets)
        if not selected_assets:
            print("No enabled assets selected. Use --asset <id>.")
            return 0

        start_ms = parse_iso_to_ms(args.start) if args.start else None
        end_ms = parse_iso_to_ms(args.end) if args.end else None
        webhooks = (
            WebhookDispatcher(cfg.notifications.webhooks)
            if args.dispatch_webhooks
            else None
        )

        with SQLiteCandleStore(
            cfg.storage.db_path,
            wal_mode=cfg.storage.wal_mode,
            busy_timeout_ms=cfg.storage.busy_timeout_ms,
        ) as store:
            for asset in selected_assets:
                candles = store.get_candles(
                    asset.provider.name,
                    asset.market.symbol,
                    asset.market.timeframe,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    closed_only=True,
                )
                print(
                    f"=== Replay '{asset.id}' ({asset.market.symbol} {asset.market.timeframe}) | "
                    f"stored_closed_bars={len(candles)} ==="
                )
                engine = ReplayEngine(asset)
                engine.verify_models()
                summary = engine.run(
                    candles,
                    store=store,
                    webhook_dispatcher=webhooks,
                    persist_predictions=args.persist_predictions,
                    persist_alerts=args.persist_alerts,
                    step_delay_sec=args.step_delay,
                    step_cb=_verbose_step if args.verbose else None,
                )
                dir_str = (
                    f"{summary.directional_hits}/{summary.directional_total} "
                    f"({100.0 * summary.directional_hits / summary.directional_total:.1f}%)"
                    if summary.directional_total > 0
                    else "N/A"
                )
                mae_str = (
                    f"{summary.mean_abs_return_error:.4%}"
                    if summary.mean_abs_return_error is not None
                    else "N/A"
                )
                print(
                    f"  evaluated={summary.bars_evaluated} | actions={summary.action_counts} "
                    f"| directional_acc={dir_str} | horizon_MAE={mae_str} "
                    f"| alerts={summary.alerts_triggered}"
                )
                for a in summary.alerts:
                    print(
                        f"    ALERT {a.action} @ {format_ms_to_iso(a.anchor_open_time_ms)} "
                        f"(conf={a.confidence:.1%}, streak={a.consecutive_hits})"
                    )

        return 0
    except (ConfigError, SchemaMismatchError, NNClientError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
