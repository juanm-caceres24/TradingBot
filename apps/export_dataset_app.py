#!/usr/bin/env python3
"""Thin CLI app to export NeuralNetwork (libnn v0.6.0) training CSVs and configs.

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
    SyntheticProviderParams,
    load_config,
)
from tradingbot.dataset import NNTrainingOptions, export_dataset  # noqa: E402
from tradingbot.models import Candle, parse_iso_to_ms  # noqa: E402
from tradingbot.providers import SyntheticProvider  # noqa: E402
from tradingbot.storage import SQLiteCandleStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_dataset_app",
        description="Export sliding-window CSV datasets, feature_schema.json, and NeuralNetwork v0.6.0 nn_config.json files.",
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
        default="128,64",
        help="Comma-separated hidden layer sizes for nn_config.json (default: 128,64).",
    )
    parser.add_argument(
        "--activation",
        default="relu",
        choices=("relu", "tanh", "sigmoid", "leaky_relu"),
        help="Hidden layer activation function (default: relu).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs for nn_config.json (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for nn_config.json (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for nn_config.json (default: 0.001).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Training worker threads for nn_config.json (default: 4).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Chronological validation split fraction in [0, 1) (default: 0.2).",
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
                raise ConfigError(
                    f"Unknown asset_id '{aid}'. Available: {', '.join(sorted(by_id.keys()))}"
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

    try:
        cfg = load_config(args.config)
        target_assets = _select_assets(cfg.assets, args.assets)
        if not target_assets:
            print("No enabled assets selected. Use --asset <id>.")
            return 0

        hidden = _parse_hidden_layers(args.hidden)
        options = NNTrainingOptions(
            hidden_layers=hidden,
            hidden_activation=args.activation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_split=args.val_split,
            num_threads=args.threads,
            dataset_has_header=not args.no_header,
        )

        kinds = ("candles", "signal") if args.kind == "both" else (args.kind,)
        start_ms = parse_iso_to_ms(args.start) if args.start else None
        end_ms = parse_iso_to_ms(args.end) if args.end else None
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            print("Error: --end must be strictly greater than --start.", file=sys.stderr)
            return 2

        store: SQLiteCandleStore | None = None
        if args.synthetic_candles <= 0:
            store = SQLiteCandleStore(
                cfg.storage.db_path,
                wal_mode=cfg.storage.wal_mode,
                busy_timeout_ms=cfg.storage.busy_timeout_ms,
            )

        try:
            for asset in target_assets:
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
                    summary = export_dataset(
                        asset,
                        candles,
                        model_kind=kind,
                        output_dir=effective_out_dir,
                        schema_path=schema_out,
                        dataset_path_prefix=effective_dataset_prefix,
                        model_save_dir=effective_model_save_dir,
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
                    if summary.samples_written > 0:
                        compact_flag = " --compact" if kind == "signal" else ""
                        print(
                            f"    train cmd: ./build/release/apps/train_app "
                            f"{summary.nn_config_path.resolve()}{compact_flag}"
                        )
                        print(
                            f"    serve cmd: ./build/release/apps/server_app "
                            f"{summary.nn_config_path.resolve()}"
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
