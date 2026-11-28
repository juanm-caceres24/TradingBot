#!/usr/bin/env python3
"""Thin CLI app to export NeuralNetwork (libnn v0.7.0) training CSVs and configs.

Supports exporting from stored SQLite candles OR generating an offline synthetic
test dataset (--synthetic-candles N) to validate NeuralNetwork training immediately.

Examples:
    # Export from local SQLite database for btc_1h (both candle & signal models)
    ./scripts/export_dataset.sh configs/bot.json --asset btc_1h --kind both

    # Generate an offline synthetic test dataset of 2000 candles (no DB/API needed)
    ./scripts/export_dataset.sh configs/bot.json --asset btc_1h --kind both --synthetic-candles 2000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tradingbot.config import (  # noqa: E402
    AssetConfig,
    ConfigError,
    ProviderConfig,
    DEFAULT_CONFIG_PATH,
    SyntheticProviderParams,
    ensure_default_config,
    load_config,
)
from tradingbot.dataset import NNTrainingOptions, export_dataset  # noqa: E402
from tradingbot.models import Candle, parse_iso_to_ms  # noqa: E402
from tradingbot.providers import SyntheticProvider  # noqa: E402
from tradingbot.storage import SQLiteCandleStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_dataset_app",
        description="Export sliding-window CSV datasets, feature_schema.json, and NeuralNetwork v0.7.0 nn_config.json files.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to root TradingBot JSON configuration (default: configs/bot.json; auto-created if missing).",
    )
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        default=[],
        help="Target specific asset_id (repeatable). Defaults to all enabled assets.",
    )
    parser.add_argument(
        "--kind",
        choices=("candles", "signal", "both"),
        default="both",
        help="Which NeuralNetwork model dataset(s) to export (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override export.output_dir from config (default from config: data/exports).",
    )
    parser.add_argument(
        "--dataset-path-prefix",
        default=None,
        help="Override export.dataset_path_prefix for nn_config training.dataset_path (empty string uses absolute CSV path).",
    )
    parser.add_argument(
        "--model-save-dir",
        default=None,
        help="Override export.model_save_dir for nn_config model.save_path (default from config: models).",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Override export.report_dir for nn_config training.report_path (empty string disables automatic JSON report).",
    )
    parser.add_argument(
        "--update-asset-schema",
        action="store_true",
        help="Also write feature_schema.json to the asset's configured neural_networks.*.schema_path.",
    )
    parser.add_argument(
        "--start",
        default="",
        help="Optional UTC ISO-8601 start timestamp filter for SQLite candles.",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Optional UTC ISO-8601 exclusive end timestamp filter for SQLite candles.",
    )
    parser.add_argument(
        "--synthetic-candles",
        type=int,
        default=0,
        help="Generate N offline synthetic candles instead of reading from SQLite (for test training).",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="RNG seed when --synthetic-candles is used (default: 42).",
    )
    parser.add_argument(
        "--hidden",
        default=None,
        help="Override per-asset training.hidden_layers (comma-separated, e.g. 64,32).",
    )
    parser.add_argument(
        "--activation",
        default=None,
        choices=("relu", "tanh", "sigmoid", "leaky_relu"),
        help="Override per-asset training.hidden_activation.",
    )
    parser.add_argument(
        "--candle-loss",
        default=None,
        choices=("mse", "huber"),
        help="Override per-asset training.candle_loss for candle regression models.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=None,
        help="Override per-asset training.huber_delta (> 0, used with huber loss).",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=("adam", "sgd"),
        help="Override per-asset training.optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override per-asset training.epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-asset training.batch_size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override per-asset training.learning_rate.",
    )
    parser.add_argument(
        "--lr-schedule",
        default=None,
        choices=("none", "cosine"),
        help="Override per-asset training.lr_schedule.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Override per-asset training.min_learning_rate for cosine schedule.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override per-asset training.weight_decay (decoupled L2 / AdamW).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override per-asset training.dropout_rate in [0, 1).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Override per-asset training.grad_clip_norm (>= 0; 0 disables).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Override per-asset training.early_stopping_patience (0 disables; requires val-split > 0).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="Override per-asset training.early_stopping_min_delta (>= 0).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override per-asset training.num_threads.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Override per-asset training.validation_split in [0, 1).",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit CSV header row and set dataset_has_header=false in nn_config.json.",
    )
    return parser


def _parse_hidden_layers(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return ()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    layers = tuple(int(p) for p in parts)
    if any(w < 1 for w in layers):
        raise ValueError("All hidden layer widths must be >= 1")
    return layers


def _build_training_options_for_asset(
    asset: AssetConfig,
    args: argparse.Namespace,
    hidden_override: tuple[int, ...] | None,
) -> NNTrainingOptions:
    base = asset.training
    effective_lr = args.lr if args.lr is not None else base.learning_rate
    effective_min_lr = (
        args.min_lr if args.min_lr is not None else min(base.min_learning_rate, effective_lr)
    )
    effective_val_split = (
        args.val_split if args.val_split is not None else base.validation_split
    )
    effective_patience = (
        args.early_stop_patience
        if args.early_stop_patience is not None
        else (0 if effective_val_split <= 0.0 else base.early_stopping_patience)
    )
    return NNTrainingOptions(
        hidden_layers=hidden_override if hidden_override is not None else base.hidden_layers,
        hidden_activation=(
            args.activation if args.activation is not None else base.hidden_activation
        ),
        candle_loss=args.candle_loss if args.candle_loss is not None else base.candle_loss,
        huber_delta=args.huber_delta if args.huber_delta is not None else base.huber_delta,
        epochs=args.epochs if args.epochs is not None else base.epochs,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        learning_rate=effective_lr,
        lr_schedule=args.lr_schedule if args.lr_schedule is not None else base.lr_schedule,
        min_learning_rate=effective_min_lr,
        optimizer=args.optimizer if args.optimizer is not None else base.optimizer,
        adam_beta1=base.adam_beta1,
        adam_beta2=base.adam_beta2,
        adam_epsilon=base.adam_epsilon,
        weight_decay=args.weight_decay if args.weight_decay is not None else base.weight_decay,
        dropout_rate=args.dropout if args.dropout is not None else base.dropout_rate,
        grad_clip_norm=args.grad_clip if args.grad_clip is not None else base.grad_clip_norm,
        validation_split=effective_val_split,
        early_stopping_patience=effective_patience,
        early_stopping_min_delta=(
            args.early_stop_min_delta
            if args.early_stop_min_delta is not None
            else base.early_stopping_min_delta
        ),
        num_threads=args.threads if args.threads is not None else base.num_threads,
        init_seed=base.init_seed,
        seed=base.seed,
        log_every_epochs=base.log_every_epochs,
        server_workers=base.server_workers,
        dataset_has_header=False if args.no_header else base.dataset_has_header,
    )


def _generate_synthetic_candles(
    asset: AssetConfig,
    count: int,
    seed: int,
) -> list[Candle]:
    tf = asset.market.parsed_timeframe
    syn_cfg = ProviderConfig(
        name="synthetic",
        type="synthetic",
        max_candles_per_request=max(1000, count),
        synthetic=SyntheticProviderParams(
            seed=seed,
            base_price=50000.0,
            volatility=0.01,
            drift=0.0001,
            base_volume=120.0,
        ),
    )
    start_ms = parse_iso_to_ms("2024-01-01T00:00:00Z")
    end_ms = start_ms + count * tf.interval_ms
    # Clock placed well after end_ms so every generated bar is closed.
    provider = SyntheticProvider(syn_cfg, clock_ms=lambda: end_ms + tf.interval_ms)
    return provider.fetch_candles(
        asset.market.symbol,
        tf,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=count,
    )


def _select_assets(all_assets: tuple[AssetConfig, ...], requested: list[str]) -> list[AssetConfig]:
    if requested:
        by_id = {a.id: a for a in all_assets}
        selected: list[AssetConfig] = []
        for aid in dict.fromkeys(requested):
            if aid not in by_id:
                available = ", ".join(sorted(by_id.keys())) or "(none — add assets in configs/bot.json)"
                raise ConfigError(
                    f"Unknown asset_id '{aid}'. Available: {available}"
                )
            selected.append(by_id[aid])
        return selected
    return [a for a in all_assets if a.enabled]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.synthetic_candles < 0:
        print("Error: --synthetic-candles must be >= 0.", file=sys.stderr)
        return 2
    if args.output_dir is not None and not args.output_dir.strip():
        print("Error: --output-dir must be a non-empty path.", file=sys.stderr)
        return 2
    if args.model_save_dir is not None and not args.model_save_dir.strip().rstrip("/"):
        print("Error: --model-save-dir must be a valid non-empty directory path.", file=sys.stderr)
        return 2
    if (
        args.dataset_path_prefix is not None
        and args.dataset_path_prefix != ""
        and not args.dataset_path_prefix.strip().rstrip("/")
    ):
        print(
            "Error: --dataset-path-prefix must be empty or a valid non-root path prefix.",
            file=sys.stderr,
        )
        return 2
    if (
        args.report_dir is not None
        and args.report_dir != ""
        and not args.report_dir.strip().rstrip("/")
    ):
        print(
            "Error: --report-dir must be empty or a valid non-root directory path.",
            file=sys.stderr,
        )
        return 2

    try:
        boot = ensure_default_config(args.config)
        if boot is not None:
            print(
                f"Created minimal default configuration at '{boot.config_path}' (0 assets configured). "
                "Copy templates from 'templates/configs/assets/' or run './scripts/init_config.sh --with-demo'."
            )
        cfg = load_config(args.config)
        target_assets = _select_assets(cfg.assets, args.assets)
        if not target_assets:
            if not cfg.assets:
                print("No assets configured in bot.json. Add assets from 'templates/configs/assets/' first.")
            else:
                print("No enabled assets selected. Use --asset <id>.")
            return 0

        hidden_override = (
            _parse_hidden_layers(args.hidden) if args.hidden is not None else None
        )

        kinds = ("candles", "signal") if args.kind == "both" else (args.kind,)
        start_ms = parse_iso_to_ms(args.start) if args.start else None
        end_ms = parse_iso_to_ms(args.end) if args.end else None
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            print("Error: --end must be strictly greater than --start.", file=sys.stderr)
            return 2

        effective_out_dir = args.output_dir if args.output_dir is not None else cfg.export.output_dir
        effective_dataset_prefix = (
            args.dataset_path_prefix
            if args.dataset_path_prefix is not None
            else cfg.export.dataset_path_prefix
        )
        effective_model_save_dir = (
            args.model_save_dir
            if args.model_save_dir is not None
            else cfg.export.model_save_dir
        )
        effective_report_dir = (
            args.report_dir
            if args.report_dir is not None
            else cfg.export.report_dir
        )

        store: SQLiteCandleStore | None = None
        if args.synthetic_candles <= 0:
            store = SQLiteCandleStore(
                cfg.storage.db_path,
                wal_mode=cfg.storage.wal_mode,
                busy_timeout_ms=cfg.storage.busy_timeout_ms,
            )

        try:
            for asset in target_assets:
                options = _build_training_options_for_asset(asset, args, hidden_override)
                if args.synthetic_candles > 0:
                    candles = _generate_synthetic_candles(
                        asset, args.synthetic_candles, args.synthetic_seed
                    )
                    source_label = f"synthetic ({len(candles)} bars, seed={args.synthetic_seed})"
                else:
                    assert store is not None
                    candles = store.get_candles(
                        asset.provider.name,
                        asset.market.symbol,
                        asset.market.timeframe,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        closed_only=True,
                    )
                    source_label = f"sqlite ({len(candles)} closed bars)"

                print(
                    f"=== Asset '{asset.id}' ({asset.market.symbol} {asset.market.timeframe}) | "
                    f"source: {source_label} ==="
                )

                for kind in kinds:
                    endpoint = (
                        asset.neural_networks.candle_predictor
                        if kind == "candles"
                        else asset.neural_networks.signal_classifier
                    )
                    schema_out = (
                        Path(endpoint.schema_path)
                        if args.update_asset_schema and endpoint.schema_path
                        else None
                    )
                    summary = export_dataset(
                        asset,
                        candles,
                        model_kind=kind,
                        output_dir=effective_out_dir,
                        schema_path=schema_out,
                        dataset_path_prefix=effective_dataset_prefix,
                        model_save_dir=effective_model_save_dir,
                        report_dir=effective_report_dir,
                        training_options=options,
                    )
                    print(
                        f"  [{kind}] samples={summary.samples_written} "
                        f"| shape=({summary.input_size} -> {summary.output_size})"
                    )
                    if summary.class_counts is not None and summary.samples_written > 0:
                        counts = summary.class_counts
                        total = summary.samples_written
                        print(
                            f"    classes: HOLD={counts['HOLD']} ({100.0*counts['HOLD']/total:.1f}%), "
                            f"LONG={counts['LONG']} ({100.0*counts['LONG']/total:.1f}%), "
                            f"SHORT={counts['SHORT']} ({100.0*counts['SHORT']/total:.1f}%)"
                        )
                    print(f"    csv:       {summary.csv_path}")
                    print(f"    schema:    {summary.schema_path}")
                    print(f"    nn_config: {summary.nn_config_path}")
                    if summary.report_path:
                        print(f"    report:    {summary.report_path}")
                    if summary.samples_written > 0:
                        compact_flag = " --compact" if kind == "signal" else ""
                        clean_prefix = effective_dataset_prefix.strip().rstrip("/")
                        if clean_prefix:
                            cfg_arg = f"configs/{summary.nn_config_path.name}"
                            print(
                                f"    note:      copy CSV to NeuralNetwork/{clean_prefix}/ "
                                f"and config to NeuralNetwork/configs/"
                            )
                        else:
                            cfg_arg = str(summary.nn_config_path.resolve())
                        print(
                            f"    train cmd: ./build/release/apps/train_app "
                            f"{cfg_arg}{compact_flag}"
                        )
                        print(
                            f"    serve cmd: ./build/release/apps/server_app "
                            f"{cfg_arg}"
                        )
        finally:
            if store is not None:
                store.close()
        return 0
    except (ConfigError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
