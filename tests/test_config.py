"""Unit tests for strict multi-asset JSON configuration loading in tradingbot.config."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tradingbot.config import (
    BootstrapResult,
    ConfigError,
    bootstrap_default_config,
    ensure_default_config,
    load_asset_config,
    load_config,
    load_provider_preset,
    parse_asset_dict,
    parse_config_dict,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BOT_CONFIG_PATH = ROOT_DIR / "templates" / "configs" / "bot.json"
BTC_1H_ASSET_PATH = ROOT_DIR / "templates" / "configs" / "assets" / "example_asset.json"
DEMO_ASSET_PATH = ROOT_DIR / "templates" / "configs" / "assets" / "demo_synthetic.json"
SYNTHETIC_PRESET_PATH = ROOT_DIR / "templates" / "configs" / "providers" / "synthetic.json"
INIT_CONFIG_APP_PATH = ROOT_DIR / "apps" / "init_config_app.py"


class TestMultiAssetConfigLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.bot_raw = json.loads(BOT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.bot_dir = BOT_CONFIG_PATH.parent
        self.asset_raw = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        self.asset_dir = BTC_1H_ASSET_PATH.parent

    def test_load_root_bot_config_with_multiple_assets(self) -> None:
        # Minimal shipped template bot.json boots cleanly with 0 configured assets
        empty_cfg = load_config(BOT_CONFIG_PATH)
        self.assertEqual(empty_cfg.storage.db_path, "data/market.db")
        self.assertEqual(empty_cfg.export.output_dir, "data/exports")
        self.assertEqual(empty_cfg.export.dataset_path_prefix, "data")
        self.assertEqual(empty_cfg.export.model_save_dir, "models")
        self.assertEqual(empty_cfg.export.report_dir, "reports")
        self.assertEqual(empty_cfg.web.visible_candles, 70)
        self.assertTrue(empty_cfg.notifications.api_enabled)
        self.assertEqual(len(empty_cfg.assets), 0)
        self.assertEqual(len(empty_cfg.enabled_assets), 0)

        # Multi-asset root config built from the shipped templates (example_asset + demo_synthetic)
        multi_raw = copy.deepcopy(self.bot_raw)
        multi_raw["assets"] = [
            {"config_path": "assets/example_asset.json"},
            {"config_path": "assets/demo_synthetic.json"},
        ]
        cfg = parse_config_dict(multi_raw, base_dir=self.bot_dir)
        self.assertEqual(len(cfg.assets), 2)
        self.assertEqual(len(cfg.enabled_assets), 2)

        btc_1h = cfg.get_asset("btc_1h")
        self.assertTrue(btc_1h.enabled)
        self.assertEqual(btc_1h.market.symbol, "BTCUSDT")
        self.assertEqual(btc_1h.market.timeframe, "1h")
        self.assertEqual(btc_1h.provider.name, "binance_spot")
        self.assertEqual(btc_1h.features.input_window, 24)
        self.assertEqual(btc_1h.features.forecast_horizon, 6)
        self.assertEqual(btc_1h.features.triple_barrier.mode, "atr_multiple")
        self.assertEqual(btc_1h.features.triple_barrier.take_profit_atr, 1.5)
        self.assertEqual(btc_1h.features.triple_barrier.stop_loss_atr, 1.0)
        self.assertTrue(btc_1h.neural_networks.signal_classifier.enabled)
        self.assertEqual(btc_1h.analysis.signal_decision_mode, "directional_edge")
        self.assertEqual(btc_1h.analysis.tp_sl_source, "triple_barrier")
        self.assertFalse(btc_1h.analysis.require_candle_confirmation)
        self.assertAlmostEqual(btc_1h.analysis.min_directional_prob, 0.40)
        self.assertAlmostEqual(btc_1h.analysis.min_directional_ratio, 1.8)
        self.assertEqual(btc_1h.alerts.consecutive_candles, 2)
        self.assertEqual(btc_1h.alerts.cooldown_candles, 5)
        self.assertEqual(btc_1h.alerts.actions, ("LONG", "SHORT"))
        self.assertEqual(btc_1h.training.hidden_layers, (64, 32))
        self.assertEqual(btc_1h.training.hidden_activation, "leaky_relu")
        self.assertEqual(btc_1h.training.candle_loss, "huber")
        self.assertEqual(btc_1h.training.epochs, 80)
        self.assertAlmostEqual(btc_1h.training.learning_rate, 0.0004)
        self.assertAlmostEqual(btc_1h.training.min_learning_rate, 0.00004)
        self.assertAlmostEqual(btc_1h.training.weight_decay, 0.01)
        self.assertAlmostEqual(btc_1h.training.dropout_rate, 0.3)
        self.assertEqual(btc_1h.training.lr_schedule, "cosine")
        self.assertEqual(btc_1h.training.early_stopping_patience, 20)

        demo = cfg.get_asset("demo_synthetic_1h")
        self.assertTrue(demo.enabled)
        self.assertEqual(demo.provider.type, "synthetic")
        self.assertEqual(demo.market.timeframe, "1h")
        self.assertFalse(demo.neural_networks.candle_predictor.enabled)
        self.assertFalse(demo.neural_networks.signal_classifier.enabled)

        from tradingbot.features import FeaturePipeline

        pipe = FeaturePipeline(btc_1h)
        self.assertEqual(pipe.max_warmup, 199)
        self.assertEqual(pipe.build_schema("candles").input_size, 384)
        self.assertEqual(pipe.build_schema("candles").output_size, 24)
        self.assertEqual(pipe.build_schema("signal").output_size, 3)

        # Reject invalid root-slash export paths
        for bad_export in ({"model_save_dir": "/"}, {"dataset_path_prefix": "/"}, {"report_dir": "/"}):
            raw_exp = copy.deepcopy(self.bot_raw)
            raw_exp["export"] = {**raw_exp["export"], **bad_export}
            with self.assertRaises(ConfigError):
                parse_config_dict(raw_exp, base_dir=self.bot_dir)

        # Verify inline override in bot.json can disable an asset
        raw_override = copy.deepcopy(multi_raw)
        raw_override["assets"][-1]["enabled"] = False
        cfg_override = parse_config_dict(raw_override, base_dir=self.bot_dir)
        self.assertFalse(cfg_override.get_asset("demo_synthetic_1h").enabled)
        self.assertEqual(len(cfg_override.enabled_assets), 1)

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

    def test_allow_empty_assets_and_reject_duplicate_assets(self) -> None:
        raw = copy.deepcopy(self.bot_raw)
        raw["assets"] = []
        parsed_empty = parse_config_dict(raw, base_dir=self.bot_dir)
        self.assertEqual(parsed_empty.assets, ())
        self.assertEqual(parsed_empty.enabled_assets, ())

        raw_bad_type = copy.deepcopy(self.bot_raw)
        raw_bad_type["assets"] = {}
        with self.assertRaises(ConfigError):
            parse_config_dict(raw_bad_type, base_dir=self.bot_dir)

        raw_dup = copy.deepcopy(self.bot_raw)
        raw_dup["assets"] = [
            {"config_path": "assets/example_asset.json"},
            {"config_path": "assets/example_asset.json"},
        ]
        with self.assertRaises(ConfigError):
            parse_config_dict(raw_dup, base_dir=self.bot_dir)

    def test_bootstrap_default_config_ensure_and_init_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "configs" / "bot.json"

            # 1) ensure_default_config ignores arbitrary non-default missing paths
            self.assertIsNone(
                ensure_default_config(tmp_path / "other" / "bot.json", default_path=cfg_path)
            )
            self.assertFalse((tmp_path / "other" / "bot.json").exists())

            # 2) ensure_default_config auto-bootstraps minimal default path (0 assets + providers)
            res = ensure_default_config(cfg_path, default_path=cfg_path)
            self.assertIsInstance(res, BootstrapResult)
            assert res is not None
            self.assertTrue(cfg_path.is_file())
            self.assertTrue((tmp_path / "configs" / "providers" / "binance_spot.json").is_file())
            self.assertTrue((tmp_path / "configs" / "providers" / "synthetic.json").is_file())
            loaded = load_config(cfg_path)
            self.assertEqual(len(loaded.assets), 0)

            # Second call is a no-op once bot.json exists
            self.assertIsNone(ensure_default_config(cfg_path, default_path=cfg_path))

            # 3) Non-overwrite protection preserves user edits; with_demo=True registers demo cleanly without clobbering
            custom_bot = json.loads(cfg_path.read_text(encoding="utf-8"))
            custom_bot["web"]["title"] = "My Custom Bot"
            cfg_path.write_text(json.dumps(custom_bot), encoding="utf-8")

            res_keep = bootstrap_default_config(cfg_path, with_demo=False, force=False)
            self.assertIn(cfg_path, res_keep.skipped_files)
            self.assertEqual(res_keep.updated_files, ())
            self.assertEqual(load_config(cfg_path).web.title, "My Custom Bot")

            res_demo = bootstrap_default_config(cfg_path, with_demo=True, force=False)
            self.assertIn(cfg_path, res_demo.updated_files)
            self.assertTrue((tmp_path / "configs" / "assets" / "demo_synthetic.json").is_file())
            demo_preserved = load_config(cfg_path)
            self.assertEqual(demo_preserved.web.title, "My Custom Bot")
            self.assertEqual([a.id for a in demo_preserved.assets], ["demo_synthetic_1h"])

            # Repeating with_demo=True is strictly idempotent (no duplicate entry)
            res_demo2 = bootstrap_default_config(cfg_path, with_demo=True, force=False)
            self.assertEqual(res_demo2.updated_files, ())
            self.assertIn(cfg_path, res_demo2.skipped_files)
            self.assertEqual([a.id for a in load_config(cfg_path).assets], ["demo_synthetic_1h"])

            # force=True resets bot.json back to template defaults
            res_force = bootstrap_default_config(cfg_path, with_demo=True, copy_example=True, force=True)
            self.assertIn(cfg_path, res_force.created_files)
            self.assertTrue((tmp_path / "configs" / "assets" / "example_asset.json").is_file())
            demo_loaded = load_config(cfg_path)
            self.assertEqual(demo_loaded.web.title, "TradingBot Dashboard")
            self.assertEqual([a.id for a in demo_loaded.assets], ["demo_synthetic_1h"])

            # 4) init_config_app.py CLI smoke test (empty init followed by --with-demo registers demo cleanly)
            spec = importlib.util.spec_from_file_location("init_config_app", INIT_CONFIG_APP_PATH)
            assert spec is not None and spec.loader is not None
            init_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(init_app)

            cli_cfg = tmp_path / "cli_configs" / "bot.json"
            buf_min = io.StringIO()
            with redirect_stdout(buf_min):
                rc_min = init_app.main(["--config", str(cli_cfg)])
            self.assertEqual(rc_min, 0)
            self.assertIn(str(cli_cfg), buf_min.getvalue())
            self.assertEqual(len(load_config(cli_cfg).assets), 0)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = init_app.main(["--config", str(cli_cfg), "--with-demo", "--copy-example"])
            self.assertEqual(rc, 0)
            self.assertIn("~ updated", buf.getvalue())
            self.assertIn(str(cli_cfg), buf.getvalue())
            self.assertEqual([a.id for a in load_config(cli_cfg).assets], ["demo_synthetic_1h"])

            # Re-running CLI without --force keeps existing files untouched
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = init_app.main(["--config", str(cli_cfg), "--with-demo"])
            self.assertEqual(rc2, 0)
            self.assertIn("already exists", buf2.getvalue())

    def test_bootstrap_embedded_fallback_and_missing_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_tpl = tmp_path / "no_templates_here"
            cfg_path = tmp_path / "fb_configs" / "bot.json"

            res = bootstrap_default_config(cfg_path, templates_dir=missing_tpl)
            self.assertIn(cfg_path, res.created_files)
            loaded = load_config(cfg_path)
            self.assertEqual(loaded.assets, ())
            self.assertEqual(
                load_provider_preset(cfg_path.parent / "providers" / "binance_spot.json").type,
                "generic_rest",
            )
            self.assertEqual(
                load_provider_preset(cfg_path.parent / "providers" / "synthetic.json").type,
                "synthetic",
            )

            with self.assertRaises(ConfigError):
                bootstrap_default_config(tmp_path / "d" / "bot.json", templates_dir=missing_tpl, with_demo=True)
            with self.assertRaises(ConfigError):
                bootstrap_default_config(tmp_path / "e" / "bot.json", templates_dir=missing_tpl, copy_example=True)

    def test_reject_unknown_keys_root_and_asset(self) -> None:
        for path in (
            ["typo_root"],
            ["storage", "unknown_flag"],
            ["export", "unknown_flag"],
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
            ["training", "extra"],
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

        # htf_context base_timeframe must match asset.market.timeframe and be a valid non-empty timeframe
        for bad_base_tf in ("4h", "", "invalid_tf", 60):
            raw_htf = copy.deepcopy(self.asset_raw)
            htf_ind = next(ind for ind in raw_htf["indicators"] if ind["type"] == "htf_context")
            htf_ind["params"]["base_timeframe"] = bad_base_tf
            with self.assertRaises(ConfigError):
                parse_asset_dict(raw_htf, base_dir=self.asset_dir)

        # Empty alert actions
        raw_act = copy.deepcopy(self.asset_raw)
        raw_act["alerts"]["actions"] = []
        with self.assertRaises(ConfigError):
            parse_asset_dict(raw_act, base_dir=self.asset_dir)

        # Triple barrier validation (both pct and atr_multiple modes)
        for bad_tb in (
            {"take_profit_pct": 1.0, "stop_loss_pct": 0.02},
            {"take_profit_pct": 0.02, "stop_loss_pct": 1.5},
            {"mode": "atr_multiple", "take_profit_atr": 0.0, "stop_loss_atr": 1.0},
            {"mode": "atr_multiple", "take_profit_atr": 1.5, "stop_loss_pct": 0.01},
            {"mode": "unknown_mode", "take_profit_pct": 0.02, "stop_loss_pct": 0.01},
        ):
            raw_tb = copy.deepcopy(self.asset_raw)
            raw_tb["features"]["triple_barrier"] = bad_tb
            with self.assertRaises(ConfigError):
                parse_asset_dict(raw_tb, base_dir=self.asset_dir)

        # Analysis directional_edge & tp_sl_source validation + backwards-compatible defaults
        for bad_an in (
            {"signal_decision_mode": "voting"},
            {"min_directional_prob": 0.0},
            {"min_directional_prob": 1.2},
            {"min_directional_prob": True},
            {"min_directional_ratio": 0.9},
            {"min_directional_ratio": False},
            {"tp_sl_source": "fixed"},
            {"require_candle_confirmation": "no"},
        ):
            raw_an = copy.deepcopy(self.asset_raw)
            raw_an["analysis"] = {**raw_an["analysis"], **bad_an}
            with self.assertRaises(ConfigError):
                parse_asset_dict(raw_an, base_dir=self.asset_dir)

        raw_legacy_an = copy.deepcopy(self.asset_raw)
        raw_legacy_an["analysis"] = {
            "min_signal_confidence": 0.55,
            "min_risk_reward": 1.5,
            "min_expected_move_pct": 0.005,
        }
        parsed_legacy = parse_asset_dict(raw_legacy_an, base_dir=self.asset_dir)
        self.assertEqual(parsed_legacy.analysis.signal_decision_mode, "argmax")
        self.assertEqual(parsed_legacy.analysis.tp_sl_source, "candles")
        self.assertTrue(parsed_legacy.analysis.require_candle_confirmation)
        self.assertAlmostEqual(parsed_legacy.analysis.min_directional_prob, 0.35)
        self.assertAlmostEqual(parsed_legacy.analysis.min_directional_ratio, 1.5)

        # Per-asset training section validation (libnn v0.7.0 constraints)
        for bad_tr in (
            {"early_stopping_patience": 10, "validation_split": 0.0},
            {"learning_rate": 0.001, "min_learning_rate": 0.005},
            {"dropout_rate": 1.0},
            {"candle_loss": "cce"},
            {"huber_delta": 0.0},
            {"weight_decay": -0.01},
            {"grad_clip_norm": -1.0},
            {"lr_schedule": "linear"},
            {"hidden_layers": [64, 0]},
            {"hidden_activation": "softmax"},
            {"hidden_activation": "linear"},
        ):
            raw_tr = copy.deepcopy(self.asset_raw)
            raw_tr["training"] = bad_tr
            with self.assertRaises(ConfigError):
                parse_asset_dict(raw_tr, base_dir=self.asset_dir)

        # Omitting optional training section falls back cleanly to defaults
        raw_no_tr = copy.deepcopy(self.asset_raw)
        raw_no_tr.pop("training", None)
        parsed_no_tr = parse_asset_dict(raw_no_tr, base_dir=self.asset_dir)
        self.assertEqual(parsed_no_tr.training.candle_loss, "mse")
        self.assertEqual(parsed_no_tr.training.early_stopping_patience, 0)

    def test_custom_registered_indicator_accepted_by_json_config(self) -> None:
        from tradingbot.indicators import BaseIndicator, INDICATOR_REGISTRY, register_indicator

        class CustomOsc(BaseIndicator):
            indicator_type = "custom_osc_test"
            def _validate_params(self) -> None: pass
            def warmup_period(self) -> int: return 0
            def output_columns(self) -> tuple[str, ...]: return ("value",)
            def nn_scale_kind(self, column: str) -> str: return "bounded"
            def compute(self, candles: list) -> dict: return {"value": [0.0 for _ in candles]}

        register_indicator("custom_osc_test", CustomOsc)
        self.addCleanup(INDICATOR_REGISTRY.pop, "custom_osc_test", None)

        raw = copy.deepcopy(self.asset_raw)
        raw["indicators"].append(
            {"id": "my_osc", "type": "custom_osc_test", "params": {}, "use_in_nn": True, "plot": "subchart"}
        )
        parsed = parse_asset_dict(raw, base_dir=self.asset_dir)
        self.assertEqual(parsed.indicators[-1].type, "custom_osc_test")

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
