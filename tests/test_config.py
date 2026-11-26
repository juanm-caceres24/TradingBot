"""Unit tests for strict JSON configuration loading in tradingbot.config."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tradingbot.config import (
    ConfigError,
    load_config,
    load_provider_preset,
    parse_config_dict,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_CONFIG_PATH = ROOT_DIR / "configs" / "btc_1h.json"
SYNTHETIC_PRESET_PATH = ROOT_DIR / "configs" / "providers" / "synthetic.json"


class TestConfigLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.base_raw = json.loads(BTC_CONFIG_PATH.read_text(encoding="utf-8"))
        self.base_dir = BTC_CONFIG_PATH.parent

    def test_load_default_btc_config_and_binance_preset(self) -> None:
        cfg = load_config(BTC_CONFIG_PATH)
        self.assertEqual(cfg.market.symbol, "BTCUSDT")
        self.assertEqual(cfg.market.timeframe, "1h")
        self.assertEqual(cfg.provider.name, "binance_spot")
        self.assertEqual(cfg.provider.type, "generic_rest")
        self.assertEqual(cfg.provider.max_candles_per_request, 1000)
        self.assertIsNotNone(cfg.provider.response_mapping)
        self.assertEqual(cfg.features.input_window, 31)
        self.assertEqual(cfg.features.forecast_horizon, 7)
        self.assertEqual(len(cfg.indicators), 5)
        self.assertEqual(cfg.web.visible_candles, 70)
        self.assertIsNone(cfg.runtime.initial_history_from_ms())

    def test_load_synthetic_preset_and_override(self) -> None:
        syn = load_provider_preset(SYNTHETIC_PRESET_PATH)
        self.assertEqual(syn.type, "synthetic")
        self.assertIsNotNone(syn.synthetic)
        assert syn.synthetic is not None
        self.assertEqual(syn.synthetic.seed, 42)

        raw = copy.deepcopy(self.base_raw)
        raw["provider"] = {
            "preset_path": "providers/synthetic.json",
            "max_candles_per_request": 500,
        }
        cfg = parse_config_dict(raw, base_dir=self.base_dir)
        self.assertEqual(cfg.provider.type, "synthetic")
        self.assertEqual(cfg.provider.max_candles_per_request, 500)

    def test_reject_unknown_keys_everywhere(self) -> None:
        cases = [
            (["typo_root"], True),
            (["market", "extra"], 1),
            (["storage", "unknown_flag"], False),
            (["runtime", "unknown_mode"], "x"),
            (["features", "extra_window"], 5),
            (["features", "triple_barrier", "extra"], 0.1),
            (["neural_networks", "extra"], 1),
            (["neural_networks", "candle_predictor", "extra"], 1),
            (["analysis", "extra"], 1),
            (["web", "extra"], 1),
        ]
        for path, val in cases:
            with self.subTest(path=path):
                raw = copy.deepcopy(self.base_raw)
                target = raw
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = val
                with self.assertRaises(ConfigError):
                    parse_config_dict(raw, base_dir=self.base_dir)

    def test_reject_unknown_key_in_indicator_and_provider(self) -> None:
        raw = copy.deepcopy(self.base_raw)
        raw["indicators"][0]["unexpected"] = 123
        with self.assertRaises(ConfigError):
            parse_config_dict(raw, base_dir=self.base_dir)

        raw2 = copy.deepcopy(self.base_raw)
        raw2["provider"]["unknown_provider_key"] = "oops"
        with self.assertRaises(ConfigError):
            parse_config_dict(raw2, base_dir=self.base_dir)

    def test_reject_bool_where_int_or_float_expected(self) -> None:
        raw = copy.deepcopy(self.base_raw)
        raw["features"]["input_window"] = True
        with self.assertRaises(ConfigError):
            parse_config_dict(raw, base_dir=self.base_dir)

        raw2 = copy.deepcopy(self.base_raw)
        raw2["runtime"]["poll_interval_sec"] = True
        with self.assertRaises(ConfigError):
            parse_config_dict(raw2, base_dir=self.base_dir)

    def test_reject_unmapped_timeframe_for_rest_provider(self) -> None:
        raw = copy.deepcopy(self.base_raw)
        raw["market"]["timeframe"] = "7h"
        with self.assertRaises(ConfigError):
            parse_config_dict(raw, base_dir=self.base_dir)

    def test_reject_duplicate_indicator_ids(self) -> None:
        raw = copy.deepcopy(self.base_raw)
        raw["indicators"][1]["id"] = raw["indicators"][0]["id"]
        with self.assertRaises(ConfigError):
            parse_config_dict(raw, base_dir=self.base_dir)

    def test_initial_history_from_validation(self) -> None:
        raw = copy.deepcopy(self.base_raw)
        raw["runtime"]["initial_history_from"] = "2024-06-01T00:00:00Z"
        cfg = parse_config_dict(raw, base_dir=self.base_dir)
        self.assertEqual(cfg.runtime.initial_history_from_ms(), 1_717_200_000_000)

        raw_bad = copy.deepcopy(self.base_raw)
        raw_bad["runtime"]["initial_history_from"] = "invalid-date"
        with self.assertRaises(ConfigError):
            parse_config_dict(raw_bad, base_dir=self.base_dir)

    def test_invalid_file_and_invalid_json(self) -> None:
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/path/config.json")

        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "broken.json"
            bad_file.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(bad_file)


if __name__ == "__main__":
    unittest.main()
