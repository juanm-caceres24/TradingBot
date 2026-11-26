"""Unit and end-to-end tests for FeaturePipeline, FeatureSchema, CSV export, and nn_config generation."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from tradingbot.config import load_asset_config, parse_asset_dict
from tradingbot.dataset import (
    NNTrainingOptions,
    export_dataset,
    validate_nn_config_dict,
)
from tradingbot.features import (
    FeaturePipeline,
    InsufficientDataError,
    SchemaError,
    load_feature_schema,
)
from tradingbot.models import Candle, parse_iso_to_ms
from tradingbot.providers import SyntheticProvider, create_provider
from tradingbot.storage import SQLiteCandleStore

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"
BOT_CONFIG_PATH = ROOT_DIR / "configs" / "bot.json"
EXPORT_APP_PATH = ROOT_DIR / "apps" / "export_dataset_app.py"
HOUR_MS = 3_600_000


def _load_export_app_module() -> Any:
    spec = importlib.util.spec_from_file_location("export_dataset_app", EXPORT_APP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_bars(count: int, seed: int = 42) -> list[Candle]:
    raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
    raw["provider"] = {
        "preset_path": "../providers/synthetic.json",
        "max_candles_per_request": max(1000, count),
    }
    asset = parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent)
    start_ms = parse_iso_to_ms("2024-01-01T00:00:00Z")
    end_ms = start_ms + count * HOUR_MS
    provider = create_provider(asset.provider, clock_ms=lambda: end_ms + HOUR_MS)
    assert isinstance(provider, SyntheticProvider)
    return provider.fetch_candles("BTCUSDT", "1h", start_ms=start_ms, end_ms=end_ms, limit=count)


def _bar(idx: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Candle:
    return Candle(
        provider="synthetic",
        symbol="BTCUSDT",
        timeframe="1h",
        open_time_ms=idx * HOUR_MS,
        close_time_ms=(idx + 1) * HOUR_MS - 1,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        is_closed=True,
    )


class TestFeaturePipelineAndSchema(unittest.TestCase):
    def test_btc_1h_feature_dimensions_warmup_and_exclusion(self) -> None:
        asset = load_asset_config(BTC_1H_ASSET_PATH)
        pipeline = FeaturePipeline(asset)

        # btc_1h indicators: sma_20 (19), ema_50 (49), rsi_14 (14), atr_14 (14), squeeze_20 (38)
        self.assertEqual(pipeline.max_warmup, 49)
        # 5 base OHLCV + (sma=1, ema=1, rsi=1, atr=1, squeeze=2) = 11 features/bar * 31 = 341
        schema_c = pipeline.build_schema("candles")
        self.assertEqual(len(schema_c.per_bar_features), 11)
        self.assertEqual(schema_c.input_size, 341)
        self.assertEqual(schema_c.output_size, 28)  # 7 * 4

        schema_s = pipeline.build_schema("signal")
        self.assertEqual(schema_s.input_size, 341)
        self.assertEqual(schema_s.output_size, 3)

        # Strictly reject insufficient bars (max_warmup + input_window = 49 + 31 = 80 required)
        bars_79 = _make_synthetic_bars(79)
        with self.assertRaises(InsufficientDataError):
            pipeline.build_latest_inference_window(bars_79)

        bars_80 = _make_synthetic_bars(80)
        ctx = pipeline.build_latest_inference_window(bars_80)
        self.assertEqual(len(ctx.vector), 341)
        self.assertTrue(all(math.isfinite(v) for v in ctx.vector))
        # Anchor candle itself has close_rel == 0.0 (anchor_close - anchor_close) / scale
        # Last bar's close_rel is at index -11 + 3 = -8
        self.assertAlmostEqual(ctx.vector[-8], 0.0, places=9)

        # Earlier anchor overlapping warmup must return None
        self.assertIsNone(pipeline.build_input_window(bars_80, 78))

    def test_triple_barrier_labeling_long_short_hold_and_ambiguous_bar(self) -> None:
        raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        raw["features"]["forecast_horizon"] = 3
        raw["features"]["triple_barrier"] = {"take_profit_pct": 0.02, "stop_loss_pct": 0.01}
        asset = parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent)
        pipeline = FeaturePipeline(asset)

        # LONG: hits +2% (102.0) without touching -1% (99.0)
        bars_long = [
            _bar(0, 100.0, 100.5, 99.5, 100.0),
            _bar(1, 100.0, 101.0, 99.5, 100.8),
            _bar(2, 100.8, 102.5, 100.2, 102.0),
            _bar(3, 102.0, 102.5, 101.0, 101.5),
        ]
        self.assertEqual(pipeline.encode_signal_target(bars_long, 0, 100.0), [0.0, 1.0, 0.0])

        # SHORT: hits -2% (98.0) without touching +1% (101.0)
        bars_short = [
            _bar(0, 100.0, 100.5, 99.5, 100.0),
            _bar(1, 100.0, 100.5, 99.2, 99.4),
            _bar(2, 99.4, 99.8, 97.5, 98.0),
            _bar(3, 98.0, 99.0, 97.5, 98.5),
        ]
        self.assertEqual(pipeline.encode_signal_target(bars_short, 0, 100.0), [0.0, 0.0, 1.0])

        # Conservative same-bar SL + TP touch invalidates -> HOLD
        bars_whipsaw = [
            _bar(0, 100.0, 100.5, 99.5, 100.0),
            _bar(1, 100.0, 103.0, 97.0, 100.0),
            _bar(2, 100.0, 100.5, 99.5, 100.0),
            _bar(3, 100.0, 100.5, 99.5, 100.0),
        ]
        self.assertEqual(pipeline.encode_signal_target(bars_whipsaw, 0, 100.0), [1.0, 0.0, 0.0])

        # Range-bound -> HOLD
        bars_flat = [_bar(i, 100.0, 100.5, 99.5, 100.0) for i in range(4)]
        self.assertEqual(pipeline.encode_signal_target(bars_flat, 0, 100.0), [1.0, 0.0, 0.0])

    def test_feature_hash_changes_when_indicators_change(self) -> None:
        raw1 = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        raw2 = copy.deepcopy(raw1)
        raw2["indicators"].append(
            {
                "id": "bb_20",
                "type": "bollinger",
                "params": {"period": 20, "mult": 2.0},
                "use_in_nn": True,
                "plot": "overlay",
            }
        )
        s1 = FeaturePipeline(parse_asset_dict(raw1, base_dir=BTC_1H_ASSET_PATH.parent)).build_schema("candles")
        s2 = FeaturePipeline(parse_asset_dict(raw2, base_dir=BTC_1H_ASSET_PATH.parent)).build_schema("candles")
        self.assertNotEqual(s1.feature_hash, s2.feature_hash)
        self.assertEqual(s2.input_size, s1.input_size + 3 * 31)

    def test_pct_relative_stride_24_7_gap_exclusion_and_schema_integrity(self) -> None:
        raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        raw["features"]["normalization"] = "pct_relative"
        raw["features"]["stride"] = 2
        asset = parse_asset_dict(raw, base_dir=BTC_1H_ASSET_PATH.parent)
        pipeline = FeaturePipeline(asset)
        bars = _make_synthetic_bars(120)

        samples = list(pipeline.iter_training_samples(bars, "candles"))
        # first_anchor = 49 + 31 - 1 = 79, last_anchor_excl = 120 - 7 = 113, stride = 2 -> range(79, 113, 2) = 17
        self.assertEqual(len(samples), 17)

        # Introduce a timestamp gap in 24/7 schedule -> windows spanning the gap are excluded!
        gapped = bars[:90] + bars[92:]
        gapped_samples = list(pipeline.iter_training_samples(gapped, "candles"))
        self.assertLess(len(gapped_samples), len(samples))

        # Corrupted schema_version=True or tampered feature_hash rejected with SchemaError
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "bad_schema.json"
            schema_dict = pipeline.build_schema("candles").to_dict()
            bad_bool = copy.deepcopy(schema_dict)
            bad_bool["schema_version"] = True
            sp.write_text(json.dumps(bad_bool), encoding="utf-8")
            with self.assertRaises(SchemaError):
                load_feature_schema(sp)

            tampered = copy.deepcopy(schema_dict)
            tampered["feature_hash"] = "0" * 64
            sp.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(SchemaError):
                load_feature_schema(sp)


class TestDatasetExporterAndNNConfig(unittest.TestCase):
    def test_export_candles_and_signal_libnn_v060_parity(self) -> None:
        asset = load_asset_config(BTC_1H_ASSET_PATH)
        bars = _make_synthetic_bars(150)

        with tempfile.TemporaryDirectory() as tmp:
            # 1) Candle Predictor export
            sum_c = export_dataset(
                asset,
                bars,
                model_kind="candles",
                output_dir=tmp,
                training_options=NNTrainingOptions(hidden_layers=(64, 32), epochs=50),
            )
            # Total=150, first_anchor = 49 + 31 - 1 = 79, last_anchor_excl = 150 - 7 = 143 -> 64 samples
            self.assertEqual(sum_c.samples_written, 64)
            self.assertEqual(sum_c.input_size, 341)
            self.assertEqual(sum_c.output_size, 28)

            lines = sum_c.csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1 + 64)  # header + 64 rows
            header = lines[0].split(",")
            self.assertEqual(len(header), 341 + 28)
            first_row = [float(x) for x in lines[1].split(",")]
            self.assertEqual(len(first_row), 341 + 28)
            self.assertTrue(all(math.isfinite(x) for x in first_row))

            loaded_schema = load_feature_schema(sum_c.schema_path)
            self.assertEqual(loaded_schema.input_size, 341)
            self.assertEqual(loaded_schema.output_size, 28)

            nn_cfg_c = json.loads(sum_c.nn_config_path.read_text(encoding="utf-8"))
            validate_nn_config_dict(nn_cfg_c)
            self.assertEqual(nn_cfg_c["training"]["loss"], "mse")
            self.assertFalse(nn_cfg_c["training"]["validation_shuffle"])
            self.assertTrue(nn_cfg_c["training"]["keep_best_validation_model"])
            self.assertEqual(nn_cfg_c["model"]["layers"][0], {"size": 341})
            self.assertEqual(nn_cfg_c["model"]["layers"][-1], {"size": 28, "activation": "linear"})
            self.assertEqual(nn_cfg_c["server"]["port"], 8080)

            # 2) Signal Classifier export
            sum_s = export_dataset(
                asset,
                bars,
                model_kind="signal",
                output_dir=tmp,
            )
            self.assertEqual(sum_s.samples_written, 64)
            self.assertIsNotNone(sum_s.class_counts)
            assert sum_s.class_counts is not None
            self.assertEqual(sum(sum_s.class_counts.values()), 64)

            nn_cfg_s = json.loads(sum_s.nn_config_path.read_text(encoding="utf-8"))
            validate_nn_config_dict(nn_cfg_s)
            self.assertEqual(nn_cfg_s["training"]["loss"], "cce")
            self.assertEqual(nn_cfg_s["model"]["layers"][-1], {"size": 3, "activation": "softmax"})
            self.assertEqual(nn_cfg_s["server"]["port"], 8081)

    def test_export_dataset_app_cli_synthetic_and_sqlite(self) -> None:
        export_app = _load_export_app_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_syn = tmp_path / "syn_exports"

            # Mode 1: offline synthetic test dataset
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = export_app.main(
                    [
                        str(BOT_CONFIG_PATH),
                        "--asset",
                        "btc_1h",
                        "--kind",
                        "both",
                        "--synthetic-candles",
                        "140",
                        "--output-dir",
                        str(out_syn),
                        "--hidden",
                        "64,32",
                        "--epochs",
                        "25",
                    ]
                )
            self.assertEqual(rc, 0)
            output_text = buf.getvalue()
            self.assertIn("[candles] samples=54", output_text)
            self.assertIn("[signal] samples=54", output_text)
            self.assertTrue((out_syn / "btc_1h_candles_nn_config.json").is_file())
            self.assertTrue((out_syn / "btc_1h_signal_nn_config.json").is_file())

            # Mode 2: export from local SQLite database
            db_path = tmp_path / "market.db"
            out_db = tmp_path / "db_exports"
            asset_raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            asset_raw["provider"] = {"preset_path": str((ROOT_DIR / "configs" / "providers" / "synthetic.json").resolve())}
            asset_raw["neural_networks"]["candle_predictor"]["schema_path"] = str(out_db / "btc_1h_candles_schema.json")
            asset_file = tmp_path / "btc_1h.json"
            asset_file.write_text(json.dumps(asset_raw), encoding="utf-8")

            bot_cfg_path = tmp_path / "bot.json"
            bot_cfg_path.write_text(
                json.dumps(
                    {
                        "storage": {"db_path": str(db_path), "wal_mode": True, "busy_timeout_ms": 5000},
                        "web": {"enabled": False, "host": "127.0.0.1", "port": 8000, "visible_candles": 70},
                        "notifications": {"api_enabled": True, "webhooks": []},
                        "assets": [{"config_path": str(asset_file)}],
                    }
                ),
                encoding="utf-8",
            )

            with SQLiteCandleStore(db_path) as store:
                store.upsert_candles(_make_synthetic_bars(130))

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = export_app.main(
                    [
                        str(bot_cfg_path),
                        "--asset",
                        "btc_1h",
                        "--kind",
                        "candles",
                        "--output-dir",
                        str(out_db),
                    ]
                )
            self.assertEqual(rc2, 0)
            self.assertIn("[candles] samples=44", buf2.getvalue())
            self.assertTrue((out_db / "btc_1h_candles_schema.json").is_file())

            # CLI validation errors
            self.assertEqual(
                export_app.main([str(bot_cfg_path), "--synthetic-candles", "-5"]),
                2,
            )
            self.assertEqual(
                export_app.main([str(bot_cfg_path), "--start", "2025-02-01", "--end", "2025-01-01"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
