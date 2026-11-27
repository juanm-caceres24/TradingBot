"""Unit and CLI tests for NeuralNetwork v0.7.0 JSON training report analysis."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from tradingbot.dataset import (
    ReportError,
    compare_reports,
    format_comparison_table,
    format_report_detail,
    load_training_report,
    parse_training_report,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ANALYZE_APP_PATH = ROOT_DIR / "apps" / "analyze_reports_app.py"


def _load_analyze_app() -> Any:
    spec = importlib.util.spec_from_file_location("analyze_reports_app", ANALYZE_APP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _classification_report_dict(
    *,
    eval_loss: float = 0.85,
    baseline_loss: float = 1.05,
    macro_f1: float = 0.62,
    collapse_short: bool = False,
    has_validation: bool = True,
) -> dict[str, Any]:
    short_recall = 0.0 if collapse_short else 0.55
    short_prob = 0.01 if collapse_short else 0.24
    return {
        "format_version": 1,
        "task": "classification",
        "eval_split": "validation" if has_validation else "train",
        "model": {
            "input_size": 264,
            "output_size": 3,
            "parameters": 19139,
            "topology": [
                {"size": 264},
                {"size": 64, "activation": "leaky_relu"},
                {"size": 32, "activation": "leaky_relu"},
                {"size": 3, "activation": "softmax"},
            ],
        },
        "training": {
            "loss": "cce",
            "huber_delta": 1.0,
            "optimizer": "adam",
            "epochs_configured": 300,
            "epochs_completed": 72,
            "early_stopped": True,
            "batch_size": 64,
            "learning_rate": 0.001,
            "lr_schedule": "cosine",
            "min_learning_rate": 0.0001,
            "weight_decay": 0.001,
            "dropout_rate": 0.2,
            "grad_clip_norm": 1.0,
            "early_stopping_patience": 25,
            "early_stopping_min_delta": 0.0001,
            "validation_split": 0.2 if has_validation else 0.0,
            "validation_shuffle": False,
            "keep_best_validation_model": has_validation,
            "seed": 100,
            "num_threads": 4,
            "train_samples": 4000,
            "val_samples": 1000 if has_validation else 0,
            "elapsed_ms": 1250.5,
            "initial_train_loss": 1.12,
            "final_train_loss": 0.78,
            "has_validation": has_validation,
            "best_val_epoch": 47 if has_validation else None,
            "best_val_loss": eval_loss if has_validation else None,
            "final_val_loss": eval_loss if has_validation else None,
        },
        "evaluation": {
            "num_classes": 3,
            "loss": eval_loss,
            "baseline_constant_loss": baseline_loss,
            "loss_improvement_vs_baseline": baseline_loss - eval_loss,
            "accuracy": 0.64,
            "majority_baseline_accuracy": 0.50,
            "macro_f1": macro_f1,
            "confusion_matrix": [[380, 60, 60], [80, 150, 20], [90, 20, 140]],
            "per_class": [
                {
                    "class_index": 0,
                    "support": 500,
                    "actual_frequency": 0.50,
                    "mean_predicted_prob": 0.49,
                    "precision": 0.69,
                    "recall": 0.76,
                    "f1": 0.72,
                },
                {
                    "class_index": 1,
                    "support": 250,
                    "actual_frequency": 0.25,
                    "mean_predicted_prob": 0.27,
                    "precision": 0.65,
                    "recall": 0.60,
                    "f1": 0.62,
                },
                {
                    "class_index": 2,
                    "support": 250,
                    "actual_frequency": 0.25,
                    "mean_predicted_prob": short_prob,
                    "precision": 0.0 if collapse_short else 0.64,
                    "recall": short_recall,
                    "f1": 0.0 if collapse_short else 0.59,
                },
            ],
        },
        "history": [
            {"epoch": 1, "train_loss": 1.12, "val_loss": 1.06, "lr": 0.001},
        ],
    }


def _regression_report_dict(*, r2: float = 0.14, eval_loss: float = 0.42, baseline_loss: float = 0.55) -> dict[str, Any]:
    return {
        "format_version": 1,
        "task": "regression",
        "eval_split": "validation",
        "model": {
            "input_size": 264,
            "output_size": 8,
            "parameters": 19304,
            "topology": [
                {"size": 264},
                {"size": 64, "activation": "leaky_relu"},
                {"size": 32, "activation": "leaky_relu"},
                {"size": 8, "activation": "linear"},
            ],
        },
        "training": {
            "loss": "huber",
            "huber_delta": 1.0,
            "optimizer": "adam",
            "epochs_configured": 300,
            "epochs_completed": 88,
            "early_stopped": True,
            "batch_size": 64,
            "learning_rate": 0.001,
            "lr_schedule": "cosine",
            "min_learning_rate": 0.0001,
            "weight_decay": 0.001,
            "dropout_rate": 0.2,
            "grad_clip_norm": 1.0,
            "early_stopping_patience": 25,
            "early_stopping_min_delta": 0.0001,
            "validation_split": 0.2,
            "validation_shuffle": False,
            "keep_best_validation_model": True,
            "seed": 100,
            "num_threads": 4,
            "train_samples": 4000,
            "val_samples": 1000,
            "elapsed_ms": 1410.0,
            "initial_train_loss": 0.68,
            "final_train_loss": 0.39,
            "has_validation": True,
            "best_val_epoch": 63,
            "best_val_loss": eval_loss,
            "final_val_loss": eval_loss,
        },
        "evaluation": {
            "num_outputs": 8,
            "loss": eval_loss,
            "baseline_constant_loss": baseline_loss,
            "loss_improvement_vs_baseline": baseline_loss - eval_loss,
            "mae": 0.58,
            "rmse": 0.81,
            "r2": r2,
            "per_output": [
                {"output_index": i, "mae": 0.5 + 0.02 * i, "rmse": 0.7 + 0.02 * i, "r2": r2}
                for i in range(8)
            ],
        },
        "history": [
            {"epoch": 1, "train_loss": 0.68, "val_loss": 0.59, "lr": 0.001},
        ],
    }


class TestTrainingReportAnalyzer(unittest.TestCase):
    def test_parse_diagnostics_ranking_and_formatters(self) -> None:
        good_cls = parse_training_report(
            _classification_report_dict(eval_loss=0.82, baseline_loss=1.05, macro_f1=0.64),
            source_path="btc_1h_signal_v2.json",
        )
        self.assertTrue(good_cls.beats_baseline)
        self.assertEqual(good_cls.diagnostics, ())
        detail_cls = format_report_detail(good_cls)
        self.assertIn("HOLD", detail_cls)
        self.assertIn("LONG", detail_cls)
        self.assertIn("SHORT", detail_cls)
        self.assertIn("OK (beats constant baseline)", detail_cls)

        collapsed_cls = parse_training_report(
            _classification_report_dict(
                eval_loss=1.10, baseline_loss=1.05, macro_f1=0.31, collapse_short=True, has_validation=False
            ),
            source_path="btc_1h_signal_v1.json",
        )
        self.assertFalse(collapsed_cls.beats_baseline)
        self.assertIn("NO_VALIDATION_SPLIT", collapsed_cls.diagnostics)
        self.assertIn("DOES_NOT_BEAT_BASELINE", collapsed_cls.diagnostics)
        self.assertIn("CLASS_COLLAPSE", collapsed_cls.diagnostics)
        self.assertIn("MISCALIBRATED_PROBS", collapsed_cls.diagnostics)

        ranked = compare_reports([collapsed_cls, good_cls])
        self.assertEqual(ranked[0].name, "btc_1h_signal_v2.json")
        table = format_comparison_table(ranked)
        self.assertIn("btc_1h_signal_v2.json", table)
        self.assertIn("CLASS_COLLAPSE", table)

        # Regression report with OHLC per-step horizon breakdown & negative R2 diagnostic
        good_reg = parse_training_report(_regression_report_dict(r2=0.14), source_path="btc_1h_candles.json")
        self.assertTrue(good_reg.beats_baseline)
        self.assertEqual(good_reg.diagnostics, ())
        detail_reg = format_report_detail(good_reg)
        self.assertIn("Per-step OHLC horizon summary (2 bars)", detail_reg)

        bad_reg = parse_training_report(
            _regression_report_dict(r2=-0.08, eval_loss=0.60, baseline_loss=0.55),
            source_path="btc_1h_candles_bad.json",
        )
        self.assertIn("NEGATIVE_R2", bad_reg.diagnostics)
        self.assertIn("DOES_NOT_BEAT_BASELINE", bad_reg.diagnostics)

        # Nullable non-finite metric values parse cleanly without crashing
        null_reg_raw = _regression_report_dict()
        null_reg_raw["evaluation"]["loss"] = None
        null_reg_raw["evaluation"]["r2"] = None
        null_reg = parse_training_report(null_reg_raw)
        self.assertIn("NON_FINITE_LOSS", null_reg.diagnostics)

        # Unsupported format_version rejected
        bad_ver = _classification_report_dict()
        bad_ver["format_version"] = 2
        with self.assertRaises(ReportError):
            parse_training_report(bad_ver)

    def test_analyze_reports_app_cli(self) -> None:
        app = _load_analyze_app()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            f_cls_good = tmp_path / "btc_1h_signal_v2.json"
            f_cls_bad = tmp_path / "btc_1h_signal_v1.json"
            f_reg = tmp_path / "btc_1h_candles.json"

            f_cls_good.write_text(json.dumps(_classification_report_dict(eval_loss=0.80)), encoding="utf-8")
            f_cls_bad.write_text(
                json.dumps(_classification_report_dict(eval_loss=1.08, collapse_short=True)),
                encoding="utf-8",
            )
            f_reg.write_text(json.dumps(_regression_report_dict()), encoding="utf-8")

            # Single file -> detailed report
            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                rc1 = app.main([str(f_cls_good)])
            self.assertEqual(rc1, 0)
            self.assertIn("btc_1h_signal_v2.json", buf1.getvalue())
            self.assertIn("Confusion matrix", buf1.getvalue())

            # Directory comparison filtered by --task classification
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = app.main([str(tmp_path), "--task", "classification", "--detailed"])
            self.assertEqual(rc2, 0)
            out2 = buf2.getvalue()
            self.assertIn("btc_1h_signal_v2.json", out2)
            self.assertIn("btc_1h_signal_v1.json", out2)
            self.assertNotIn("btc_1h_candles.json", out2)

            # Machine-readable --json output ranked best-first
            buf3 = io.StringIO()
            with redirect_stdout(buf3):
                rc3 = app.main([str(f_cls_bad), str(f_cls_good), "--json"])
            self.assertEqual(rc3, 0)
            parsed_json = json.loads(buf3.getvalue())
            self.assertEqual([r["name"] for r in parsed_json], ["btc_1h_signal_v2.json", "btc_1h_signal_v1.json"])

            # Error path: missing path returns exit code 1
            with redirect_stderr(io.StringIO()):
                self.assertEqual(app.main([str(tmp_path / "missing.json")]), 1)

            # Direct load_training_report error path
            with self.assertRaises(ReportError):
                load_training_report(tmp_path / "missing.json")


if __name__ == "__main__":
    unittest.main()
