"""Unit tests for strict multi-asset JSON configuration loading in tradingbot.config."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tradingbot.config import (
    ConfigError,
    load_asset_config,
    load_config,
    load_provider_preset,
    parse_asset_dict,
    parse_config_dict,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BOT_CONFIG_PATH = ROOT_DIR / "configs" / "bot.json"
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"
SYNTHETIC_PRESET_PATH = ROOT_DIR / "configs" / "providers" / "synthetic.json"


class TestMultiAssetConfigLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.bot_raw = json.loads(BOT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.bot_dir = BOT_CONFIG_PATH.parent
        self.asset_raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        self.asset_dir = BTC_1H_ASSET_PATH.parent

    def test_load_root_bot_config_with_multiple_assets(self) -> None:
        cfg = load_config(BOT_CONFIG_PATH)
        self.assertEqual(cfg.storage.db_path, "data/market.db")
        self.assertEqual(cfg.web.visible_candles, 70)
        self.assertTrue(cfg.notifications.api_enabled)
        self.assertEqual(len(cfg.assets), 2)
        self.assertEqual(len(cfg.enabled_assets), 1)

        btc_1h = cfg.get_asset("btc_1h")
        self.assertTrue(btc_1h.enabled)
        self.assertEqual(btc_1h.market.symbol, "BTCUSDT")
        self.assertEqual(btc_1h.market.timeframe, "1h")
        self.assertEqual(btc_1h.provider.name, "binance_spot")
        self.assertEqual(btc_1h.features.input_window, 31)
        self.assertEqual(btc_1h.features.forecast_horizon, 7)
        self.assertEqual(btc_1h.alerts.consecutive_candles, 2)
        self.assertEqual(btc_1h.alerts.cooldown_candles, 5)
        self.assertEqual(btc_1h.alerts.actions, ("LONG", "SHORT"))

        btc_1w = cfg.get_asset("btc_1w")
        # Inline override in bot.json disabled btc_1w
        self.assertFalse(btc_1w.enabled)
        self.assertEqual(btc_1w.market.timeframe, "1w")
        self.assertFalse(btc_1w.neural_networks.candle_predictor.enabled)

        with self.assertRaises(KeyError):
            cfg.get_asset("nvda_1h")

    def test_load_standalone_asset_and_synthetic_override(self) -> None:
        standalone = load_asset_config(BTC_1H_ASSET_PATH)
        self.assertEqual(standalone.id, "btc_1h")
        self.assertIsNone(standalone.runtime.initial_history_from_ms())

        syn = load_provider_preset(SYNTHETIC_PRESET_PATH)
        self.assertEqual(syn.type, "synthetic")
        assert syn.synthetic is not None
        self.assertEqual(syn.synthetic.seed, 42)

        raw_asset = copy.deepcopy(self.asset_raw)
        raw_asset["provider"] = {
            "preset_path": "../providers/synthetic.json",
            "max_candles_per_request": 500,
        }
        raw_asset["neural_networks"]["extra_models"] = {
            "volatility_model": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8090,
                "schema_path": "data/btc_1h_vol_schema.json",
            }
        }
        parsed = parse_asset_dict(raw_asset, base_dir=self.asset_dir)
        self.assertEqual(parsed.provider.type, "synthetic")
        self.assertEqual(parsed.provider.max_candles_per_request, 500)
        self.assertIn("volatility_model", parsed.neural_networks.extra_models)
        self.assertEqual(parsed.neural_networks.extra_models["volatility_model"].port, 8090)

    def test_notifications_webhooks_validation(self) -> None:
        raw = copy.deepcopy(self.bot_raw)
        raw["notifications"]["webhooks"] = [
            {
                "name": "telegram_bridge",
                "url": "http://127.0.0.1:9000/alert",
                "enabled": True,
                "timeout_sec": 3.0,
                "headers": {"X-Source": "tradingbot"},
                "auth_token_env": "TELEGRAM_BRIDGE_TOKEN",
                "auth_header": "Authorization",
            }
        ]
        cfg = parse_config_dict(raw, base_dir=self.bot_dir)
        self.assertEqual(len(cfg.notifications.webhooks), 1)
        self.assertEqual(cfg.notifications.webhooks[0].name, "telegram_bridge")

        # Reject invalid URL scheme
        raw_bad_url = copy.deepcopy(raw)
        raw_bad_url["notifications"]["webhooks"][0]["url"] = "ftp://example.com"
        with self.assertRaises(ConfigError):
            parse_config_dict(raw_bad_url, base_dir=self.bot_dir)

        # Reject unpaired auth_token_env / auth_header
        raw_bad_auth = copy.deepcopy(raw)
        raw_bad_auth["notifications"]["webhooks"][0]["auth_header"] = ""
        with self.assertRaises(ConfigError):
            parse_config_dict(raw_bad_auth, base_dir=self.bot_dir)

    def test_reject_duplicate_and_empty_assets(self) -> None:
        raw = copy.deepcopy(self.bot_raw)
        raw["assets"] = []
        with self.assertRaises(ConfigError):
            parse_config_dict(raw, base_dir=self.bot_dir)

        raw_dup = copy.deepcopy(self.bot_raw)
        raw_dup["assets"] = [
            {"config_path": "assets/btc_1h.json"},
            {"config_path": "assets/btc_1h.json"},
        ]
        with self.assertRaises(ConfigError):
            parse_config_dict(raw_dup, base_dir=self.bot_dir)

    def test_reject_unknown_keys_root_and_asset(self) -> None:
        for path in (
            ["typo_root"],
            ["storage", "unknown_flag"],
            ["web", "extra"],
            ["notifications", "extra"],
        ):
            with self.subTest(path=path):
                raw = copy.deepcopy(self.bot_raw)
                target = raw
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = 1
                with self.assertRaises(ConfigError):
                    parse_config_dict(raw, base_dir=self.bot_dir)

        for path in (
            ["typo_asset"],
            ["market", "extra"],
            ["runtime", "unknown_mode"],
            ["features", "extra_window"],
            ["features", "triple_barrier", "extra"],
            ["neural_networks", "extra"],
            ["neural_networks", "candle_predictor", "extra"],
            ["analysis", "extra"],
            ["alerts", "extra"],
        ):
            with self.subTest(asset_path=path):
                raw = copy.deepcopy(self.asset_raw)
                target = raw
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = 1
                with self.assertRaises(ConfigError):
                    parse_asset_dict(raw, base_dir=self.asset_dir)

    def test_reject_invalid_asset_fields_and_bool_types(self) -> None:
        # Invalid asset_id
        raw = copy.deepcopy(self.asset_raw)
        raw["id"] = "BTC/1H"
        with self.assertRaises(ConfigError):
            parse_asset_dict(raw, base_dir=self.asset_dir)

        # Bool where int/float expected
        raw_bool = copy.deepcopy(self.asset_raw)
        raw_bool["alerts"]["consecutive_candles"] = True
        with self.assertRaises(ConfigError):
            parse_asset_dict(raw_bool, base_dir=self.asset_dir)

        raw_bool_f = copy.deepcopy(self.asset_raw)
        raw_bool_f["runtime"]["poll_interval_sec"] = False
        with self.assertRaises(ConfigError):
            parse_asset_dict(raw_bool_f, base_dir=self.asset_dir)

        # Unmapped timeframe for REST provider
        raw_tf = copy.deepcopy(self.asset_raw)
        raw_tf["market"]["timeframe"] = "7h"
        with self.assertRaises(ConfigError):
            parse_asset_dict(raw_tf, base_dir=self.asset_dir)

        # Duplicate indicator IDs
        raw_ind = copy.deepcopy(self.asset_raw)
        raw_ind["indicators"][1]["id"] = raw_ind["indicators"][0]["id"]
        with self.assertRaises(ConfigError):
            parse_asset_dict(raw_ind, base_dir=self.asset_dir)

        # Empty alert actions
        raw_act = copy.deepcopy(self.asset_raw)
        raw_act["alerts"]["actions"] = []
        with self.assertRaises(ConfigError):
            parse_asset_dict(raw_act, base_dir=self.asset_dir)

    def test_initial_history_from_and_invalid_files(self) -> None:
        raw = copy.deepcopy(self.asset_raw)
        raw["runtime"]["initial_history_from"] = "2024-06-01T00:00:00Z"
        cfg = parse_asset_dict(raw, base_dir=self.asset_dir)
        self.assertEqual(cfg.runtime.initial_history_from_ms(), 1_717_200_000_000)

        with self.assertRaises(ConfigError):
            load_config("/nonexistent/path/bot.json")

        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "broken.json"
            bad_file.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(bad_file)


if __name__ == "__main__":
    unittest.main()
