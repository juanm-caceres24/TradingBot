# Configuration & Templates Reference (`v1.0.0`)

TradingBot separates its **versioned core** from **local user configurations**:
- **`templates/configs/`** (tracked in Git): clean starter templates (`bot.json`, provider presets, `example_asset.json`, and `demo_synthetic.json`).
- **`/configs/`** (gitignored alongside `data/`, `models/`, `/reports/`, and `logs/`): your live instance configs, custom exchanges/brokers, calibrated asset profiles, webhooks, and neural-network ports. Editing `/configs/` never dirties Git or requires new releases.

All configuration files are strictly validated by `src/tradingbot/config.py`: **unknown keys raise `ConfigError` immediately**, and booleans are never coerced into numbers.

---

## 1. Bootstrapping & Initialization

### Automatic Minimal Bootstrap
Every runtime CLI (`./scripts/live.sh`, `./scripts/backfill.sh`, `./scripts/export_dataset.sh`, `./scripts/replay.sh`) defaults its `config` argument to `configs/bot.json`. If `configs/bot.json` does not exist yet on a fresh clone, it is automatically created with `"assets": []` alongside `configs/providers/binance_spot.json` and `configs/providers/synthetic.json`.

### Explicit Initialization (`./scripts/init_config.sh`)
```bash
# Minimal setup: configs/bot.json (0 assets) + generic provider presets
./scripts/init_config.sh

# Instant offline demo: copies & registers assets/demo_synthetic.json (safe even if bot.json already exists)
./scripts/init_config.sh --with-demo

# Also copy the full REST + dual-NN reference asset template (not registered automatically)
./scripts/init_config.sh --with-demo --copy-example

# Reset existing files back to clean template defaults
./scripts/init_config.sh --force
```

---

## 2. Root Configuration (`configs/bot.json`)

Template: `templates/configs/bot.json`

| Section / Key | Type | Default / Allowed | Purpose |
| :--- | :--- | :--- | :--- |
| `storage.db_path` | `str` | Required (e.g. `"data/market.db"`) | Local SQLite database path (parent directories created automatically) |
| `storage.wal_mode` | `bool` | `true` | Enable SQLite Write-Ahead Logging for concurrent reads/writes |
| `storage.busy_timeout_ms` | `int` | `5000` (`>= 0`) | SQLite lock wait timeout in milliseconds |
| `export.output_dir` | `str` | `"data/exports"` | Default output folder for CSVs, schemas, and `nn_config.json` files |
| `export.dataset_path_prefix` | `str` | `""` (template: `"data"`) | Prefix written to `training.dataset_path` inside `nn_config.json` (`""` writes absolute CSV path) |
| `export.model_save_dir` | `str` | `"models"` | Directory written to `model.save_path` inside `nn_config.json` |
| `export.report_dir` | `str` | `""` (template: `"reports"`) | Directory written to `training.report_path` inside `nn_config.json` (`""` disables auto report export) |
| `web.enabled` | `bool` | Required (`true`) | Enable embedded Web Dashboard & Alerts API (`--headless` overrides to off) |
| `web.host` | `str` | Required (`"0.0.0.0"`) | Bind address (`127.0.0.1`, `0.0.0.0`, or Tailscale IP) |
| `web.port` | `int` | Required (`0..65535`) | HTTP/WebSocket listening port (`0` selects an ephemeral OS port) |
| `web.visible_candles` | `int` | Required (`>= 10`) | Default historical candles rendered in the chart viewport |
| `web.title` | `str` | `"TradingBot Dashboard"` | Title displayed in the browser tab and top header |
| `notifications.api_enabled` | `bool` | `true` | Enables `/api/alerts`, `/api/alerts/{id}/ack`, and `WS /ws/alerts` |
| `notifications.webhooks` | `list` | `[]` | Outbound HTTP `POST` alert targets (`name`, `url` http/https, `enabled`, `timeout_sec`, `headers`, `auth_token_env`, `auth_header`) |
| `assets` | `list` | Required (may be `[]`) | List of asset objects or `{"config_path": "assets/<file>.json", ...}` references with optional inline overrides (e.g. `"enabled": false`) |

---

## 3. Provider Presets (`configs/providers/*.json`)

Shipped templates under `templates/configs/providers/`:
- **`binance_spot.json`** — public REST API (`array_of_arrays`, zero API key).
- **`example_rest_objects.json`** — blueprint for brokers/exchanges returning nested JSON objects (`array_of_objects`, `data_path`, Unix seconds, descending order, custom headers, and environment-variable API key).
- **`synthetic.json`** — deterministic offline generator (`type: "synthetic"`).

### `generic_rest` Fields
| Key | Type | Valid Values / Notes |
| :--- | :--- | :--- |
| `name` | `str` | Unique provider label persisted as part of SQLite primary keys |
| `type` | `str` | `"generic_rest"` |
| `base_url` | `str` | `http://` or `https://` origin |
| `endpoint` | `str` | Path starting with `/` (e.g. `"/api/v3/klines"`) |
| `http_method` | `str` | `"GET"` |
| `timeout_sec` | `float` | `> 0` (default `10.0`) |
| `rate_limit_ms` | `int` | `>= 0` delay between requests shared across assets using the identical preset |
| `max_candles_per_request` | `int` | `>= 1` pagination page size |
| `params_map` | `dict` | Maps logical keys (`symbol`, `interval`, `start_time`, `end_time`, `limit`) to exchange query param names |
| `interval_map` | `dict` | Maps TradingBot timeframes (`"15m"`, `"1h"`, `"4h"`, `"1d"`, `"1w"`, etc.) to exchange interval strings |
| `response_mapping.data_path` | `str` | Dot-separated path to candle list (e.g. `""` for root array, `"data.candles"` for nested object) |
| `response_mapping.format` | `str` | `"array_of_arrays"` (integer field indices) or `"array_of_objects"` (string field keys) |
| `response_mapping.fields` | `dict` | Required: `open_time`, `open`, `high`, `low`, `close`, `volume`; optional: `close_time` |
| `response_mapping.timestamp_unit` | `str` | `"ms"` (milliseconds) or `"s"` (Unix seconds) |
| `response_mapping.order` | `str` | `"asc"` or `"desc"` (automatically normalized to ascending) |
| `headers` | `dict` | Optional static HTTP headers |
| `api_key_env` / `api_key_header` | `str` | Optional env-var name and header name (must be both set or both empty; never put secrets in JSON) |

### `synthetic` Fields
| Key | Type | Notes |
| :--- | :--- | :--- |
| `name`, `type` | `str` | `type: "synthetic"` |
| `max_candles_per_request` | `int` | `>= 1` |
| `synthetic.{seed, base_price, volatility, drift, base_volume}` | numbers | Deterministic O(1) multi-scale synthetic OHLCV generator |

---

## 4. Asset Configuration (`configs/assets/*.json`)

Templates: `templates/configs/assets/example_asset.json` (full REST + dual-NN reference) and `templates/configs/assets/demo_synthetic.json` (offline synthetic charting).

### Core Sections
- **`id`**: URL-safe unique identifier (`[A-Za-z0-9_-]+`, e.g. `"btc_1h"`, `"eth_15m"`, `"nvda_1d"`).
- **`enabled`** (`bool`): whether the asset runs by default when `--asset` is omitted.
- **`market`**:
  - `symbol`: exchange symbol (`"BTCUSDT"`, `"NVDA"`, etc.).
  - `timeframe`: `<N><m|h|d|w>` (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`, `1w`, etc.; `1w` is Monday 00:00 UTC aligned).
  - `market_schedule`: `"24/7"` (enforces strict bar contiguity in windows, replay horizons, and alert streaks) or `"sessions"` (allows weekend/holiday gaps).
- **`provider`**: inline provider object or `{"preset_path": "../providers/binance_spot.json", ...}` with optional field overrides.
- **`runtime`**:
  - `mode`: `"sync_live"` | `"live_only"` | `"backfill"` | `"replay"`.
  - `poll_interval_sec` (`> 0`), `initial_history_from` (optional UTC ISO-8601 string or `""`), `fill_internal_gaps` (`bool`), `persist_to_db` (`bool`), `predict_on_open_candle` (`bool`).

### Built-In Technical Indicators (`indicators[]`)
Each entry requires `id` (unique within the asset), `type`, `params`, `use_in_nn` (`bool`), and `plot` (`"overlay" | "subchart" | "none"`):

| `type` | `params` (with defaults) | Output Columns (scale kind) |
| :--- | :--- | :--- |
| `sma` | `period` (`>= 1`) | `value` (`price_level`) |
| `ema` | `period` (`>= 1`) | `value` (`price_level`) |
| `rsi` | `period` (`>= 1`) | `value` (`rsi_0_100`) |
| `atr` | `period` (`>= 1`) | `value` (`price_delta`) |
| `bollinger_bands` | `period` (`>= 2`), `num_std` (`> 0`, default `2.0`) | `upper`, `middle`, `lower` (`price_level`) |
| `squeeze_momentum` | `bb_length` (`20`), `bb_mult` (`2.0`), `kc_length` (`20`), `kc_mult` (`1.5`), `mom_length` (`20`) | `momentum` (`price_delta`), `squeeze_state` (`bounded`) |
| `volatility_regime` | `fast_period` (`14`), `slow_period` (`100`, `> fast_period`), `rank_window` (`100`, `>= 2`) | `vol_ratio` (`bounded`), `vol_rank` (`bounded`) |
| `htf_context` | `base_timeframe` (must match `market.timeframe`), `htf_timeframe` (strict integer multiple), `ema_fast` (`12`), `ema_slow` (`36`, `> ema_fast`), `rsi_period` (`14`) | `trend` (`price_delta`), `close_vs_slow` (`price_delta`), `rsi` (`rsi_0_100`) |

### `features`, `neural_networks`, `analysis`, `alerts`, and `training`
- **`features`**:
  - `input_window` (`>= 1`), `forecast_horizon` (`>= 1`), `stride` (`>= 1`, dataset export step only), `normalization` (`"atr_relative"` | `"pct_relative"`), `atr_period` (`>= 1`).
  - `triple_barrier`:
    - `mode: "pct"` with `take_profit_pct` and `stop_loss_pct` in `(0, 1)`.
    - `mode: "atr_multiple"` with `take_profit_atr` (`> 0`) and `stop_loss_atr` (`> 0`).
- **`neural_networks`**:
  - `timeout_sec` (`> 0`), `verify_schema_on_start` (`bool`).
  - `candle_predictor` & `signal_classifier`: `{enabled, host, port, schema_path}`.
  - `extra_models`: optional map of `{model_name: {enabled, host, port, schema_path}}`.
- **`analysis`**:
  - `signal_decision_mode`: `"argmax"` (default) or `"directional_edge"` (`min_directional_prob` in `(0, 1]`, `min_directional_ratio >= 1.0`).
  - `tp_sl_source`: `"candles"` (default) or `"triple_barrier"`.
  - `require_candle_confirmation`: `true` (dual-model cross-confirmation) or `false` (signal-driven mode while keeping projected candles visible on chart).
  - `min_signal_confidence`, `min_risk_reward`, `min_expected_move_pct`.
- **`alerts`**:
  - `enabled` (`bool`), `actions` (non-empty subset of `["LONG", "SHORT", "HOLD"]`), `min_confidence`, `min_risk_reward`, `min_expected_move_pct`, `consecutive_candles` (`>= 1`), `cooldown_candles` (`>= 0`).
- **`training`** (optional, defaults applied by `export_dataset_app.py` and overridable via CLI flags):
  - `hidden_layers` (e.g. `[64, 32]`), `hidden_activation` (`"relu" | "leaky_relu" | "sigmoid" | "tanh"`), `candle_loss` (`"mse" | "huber"`), `huber_delta` (`> 0`), `optimizer` (`"sgd" | "adam"`), `epochs`, `batch_size`, `learning_rate`, `lr_schedule` (`"none" | "cosine"`), `min_learning_rate` (`<= learning_rate`), `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay`, `dropout_rate` (`[0, 1)`), `grad_clip_norm`, `validation_split` (`[0, 1)`), `early_stopping_patience` (requires `validation_split > 0`), `early_stopping_min_delta`, `num_threads`, `init_seed`, `seed`, `log_every_epochs`, `server_workers`, `dataset_has_header`.

---

## 5. Step-by-Step: Adding a New Asset or Exchange

1. **Add a new exchange (if not Binance Spot):** copy `templates/configs/providers/example_rest_objects.json` (or `binance_spot.json`) to `configs/providers/<exchange>.json`, adjust URL/params/mapping, and export any required API key in your shell (`export MY_EXCHANGE_KEY="..."`).
2. **Add an asset profile:** copy `templates/configs/assets/example_asset.json` to `configs/assets/<asset_id>.json`, set unique `id`, `market.symbol`, `market.timeframe` (update `htf_context.params.base_timeframe` if present), `provider.preset_path`, and unique NN server ports (`candle_predictor.port`, `signal_classifier.port`).
3. **Register it:** add `{"config_path": "assets/<asset_id>.json"}` under `"assets"` in `configs/bot.json`.
4. **Download, export, train, replay, and run live:**
   ```bash
   ./scripts/backfill.sh --asset <asset_id> --start 2024-01-01
   ./scripts/export_dataset.sh --asset <asset_id> --kind both --update-asset-schema
   # Train & serve in NeuralNetwork, then evaluate reports and replay:
   ./scripts/analyze_reports.sh ../NeuralNetwork/reports/
   ./scripts/replay.sh --asset <asset_id> --start 2026-01-01
   ./scripts/live.sh --web-host 0.0.0.0 --web-port 8000
   ```
