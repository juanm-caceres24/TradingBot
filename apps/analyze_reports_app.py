#!/usr/bin/env python3
"""Thin CLI app to inspect and compare NeuralNetwork (libnn v0.7.0) JSON training reports.

Examples:
    # Inspect one report with full per-class / per-step diagnostics
    ./scripts/analyze_reports.sh ../NeuralNetwork/reports/btc_1h_signal_report.json

    # Compare all JSON reports inside a directory, ranked by gain over constant baseline
    ./scripts/analyze_reports.sh ../NeuralNetwork/reports/ --task classification

    # Machine-readable JSON summary
    ./scripts/analyze_reports.sh run_a.json run_b.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tradingbot.dataset import (  # noqa: E402
    ReportError,
    TrainingReportSummary,
    compare_reports,
    format_comparison_table,
    format_report_detail,
    load_training_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_reports_app",
        description="Inspect, diagnose, and compare NeuralNetwork v0.7.0 JSON training reports.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more report JSON files or directories containing report JSON files.",
    )
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default=None,
        help="Optional filter to include only classification or regression reports.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-class / per-step breakdown for every report even when comparing multiple files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit ranked summary as JSON.",
    )
    return parser


def _discover_report_files(raw_paths: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        p = Path(raw)
        if p.is_dir():
            for candidate in sorted(p.glob("*.json")):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(candidate)
        elif p.is_file():
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(p)
        else:
            raise ReportError(f"Path not found: '{p}'")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        files = _discover_report_files(args.paths)
        if not files:
            print("Error: No JSON report files found.", file=sys.stderr)
            return 1

        reports: list[TrainingReportSummary] = []
        for f in files:
            reports.append(load_training_report(f))

        if args.task is not None:
            reports = [r for r in reports if r.task == args.task]

        if not reports:
            print(f"No reports matched filter --task {args.task}.")
            return 0

        ranked = compare_reports(reports)
        if args.as_json:
            print(json.dumps([r.to_dict() for r in ranked], indent=2))
            return 0

        if len(ranked) == 1:
            print(format_report_detail(ranked[0]))
        else:
            print(format_comparison_table(ranked))
            if args.detailed:
                print()
                for rep in ranked:
                    print(format_report_detail(rep))
                    print()
        return 0
    except ReportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
