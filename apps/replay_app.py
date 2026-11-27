#!/usr/bin/env python3
"""Thin CLI app replaying stored SQLite candles against NeuralNetwork server(s).

Examples:
    ./scripts/replay.sh configs/bot.json --asset btc_1h
    ./scripts/replay.sh configs/bot.json --asset btc_1h --start 2025-01-01 --end 2025-02-01 --verbose
"""

from __future__ import annotations

import argparse
import dataclasses
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
        "--signal-only",
        action="store_true",
        help="Disable candle_predictor during replay to benchmark signal_classifier alone.",
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
        for aid in dict.fromkeys(requested):
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
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            print("Error: --end must be strictly greater than --start.", file=sys.stderr)
            return 2
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
                run_asset = asset
                mode_tag = ""
                if args.signal_only:
                    if not asset.neural_networks.signal_classifier.enabled:
                        raise ConfigError(
                            f"Asset '{asset.id}' cannot run with --signal-only because "
                            "signal_classifier is disabled."
                        )
                    disabled_cp = dataclasses.replace(
                        asset.neural_networks.candle_predictor, enabled=False
                    )
                    run_asset = dataclasses.replace(
                        asset,
                        neural_networks=dataclasses.replace(
                            asset.neural_networks, candle_predictor=disabled_cp
                        ),
                    )
                    mode_tag = " [signal-only]"

                candles = store.get_candles(
                    run_asset.provider.name,
                    run_asset.market.symbol,
                    run_asset.market.timeframe,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    closed_only=True,
                )
                print(
                    f"=== Replay '{run_asset.id}' ({run_asset.market.symbol} {run_asset.market.timeframe})"
                    f"{mode_tag} | stored_closed_bars={len(candles)} ==="
                )
                engine = ReplayEngine(run_asset)
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
                if summary.hold_reasons:
                    funnel_parts = [
                        f"{k}={summary.hold_reasons[k]}"
                        for k in ("signal_threshold", "conflict", "min_move", "min_rr")
                        if k in summary.hold_reasons
                    ]
                    for extra_k in ("no_models", "other"):
                        if summary.hold_reasons.get(extra_k, 0) > 0:
                            funnel_parts.append(f"{extra_k}={summary.hold_reasons[extra_k]}")
                    if funnel_parts:
                        print("  hold_funnel: " + " | ".join(funnel_parts))
                if summary.signal_trades_evaluated > 0:
                    sig_acc_pct = (
                        100.0 * summary.signal_directional_hits / summary.signal_trades_evaluated
                    )
                    avg_ret_str = (
                        f"{summary.signal_avg_return_pct:+.4%}"
                        if summary.signal_avg_return_pct is not None
                        else "N/A"
                    )
                    tb_resolved = summary.tb_tp_hits + summary.tb_sl_hits
                    tb_wr_str = (
                        f"{100.0 * summary.tb_tp_hits / tb_resolved:.1f}%"
                        if tb_resolved > 0
                        else "N/A"
                    )
                    print(
                        f"  filtered_signals={summary.signal_trades_evaluated} "
                        f"| signal_acc={summary.signal_directional_hits}/{summary.signal_trades_evaluated} "
                        f"({sig_acc_pct:.1f}%) "
                        f"[LONG={summary.signal_long_hits}/{summary.signal_long_total}, "
                        f"SHORT={summary.signal_short_hits}/{summary.signal_short_total}] "
                        f"| avg_ret={avg_ret_str} "
                        f"| TB(TP/SL/TO)={summary.tb_tp_hits}/{summary.tb_sl_hits}/{summary.tb_timeouts} "
                        f"(win_rate={tb_wr_str})"
                    )
                else:
                    print(
                        "  filtered_signals=0 (no actionable LONG/SHORT trades with complete horizon)"
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
