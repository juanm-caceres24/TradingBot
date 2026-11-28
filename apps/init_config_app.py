#!/usr/bin/env python3
"""Initialize local gitignored TradingBot configuration files from templates.

Examples:
    # Create minimal configs/bot.json (0 assets) + generic provider presets
    ./scripts/init_config.sh

    # Also copy & register the offline synthetic charting demo asset
    ./scripts/init_config.sh --with-demo

    # Also copy the full reference asset template (configs/assets/example_asset.json)
    ./scripts/init_config.sh --copy-example
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
    ConfigError,
    DEFAULT_CONFIG_PATH,
    bootstrap_default_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="init_config_app",
        description="Initialize local configs/ files from versioned templates without overwriting existing configs.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Target root bot configuration path (default: configs/bot.json).",
    )
    parser.add_argument(
        "--with-demo",
        action="store_true",
        help="Copy and register the offline demo_synthetic.json asset for instant charting/export testing.",
    )
    parser.add_argument(
        "--copy-example",
        action="store_true",
        help="Also copy templates/configs/assets/example_asset.json into configs/assets/ as a reference template.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target configuration files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = bootstrap_default_config(
            args.config,
            with_demo=args.with_demo,
            copy_example=args.copy_example,
            force=args.force,
        )
    except (ConfigError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if res.created_files or res.updated_files:
        print(f"Initialized TradingBot configuration at '{res.config_path}':")
        for p in res.created_files:
            print(f"  + created {p}")
        for p in res.updated_files:
            print(f"  ~ updated {p}")
    else:
        print(f"Configuration already exists at '{res.config_path}' (no files overwritten; use --force to reset).")

    if res.skipped_files:
        for p in res.skipped_files:
            print(f"  = kept existing {p}")

    assets_dir = res.config_path.parent / "assets"
    is_default = res.config_path == DEFAULT_CONFIG_PATH
    live_cmd = "./scripts/live.sh" if is_default else f"./scripts/live.sh {res.config_path}"
    export_cmd = (
        "./scripts/export_dataset.sh --asset demo_synthetic_1h --kind both --synthetic-candles 2000"
        if is_default
        else f"./scripts/export_dataset.sh {res.config_path} --asset demo_synthetic_1h --kind both --synthetic-candles 2000"
    )
    if args.with_demo:
        print(
            "\nNext steps:\n"
            f"  - Start dashboard immediately with the offline synthetic demo: {live_cmd}\n"
            f"  - Or export a synthetic training dataset: {export_cmd}\n"
            f"  - To add your own market asset, copy templates/configs/assets/example_asset.json to {assets_dir}/<your_asset>.json and register it in {res.config_path}"
        )
    else:
        print(
            "\nNext steps:\n"
            f"  - Copy templates/configs/assets/example_asset.json to {assets_dir}/<your_asset>.json\n"
            f"  - Register it under \"assets\" in {res.config_path} (or run ./scripts/init_config.sh --with-demo for an offline demo)\n"
            f"  - Start dashboard/runtime: {live_cmd}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
