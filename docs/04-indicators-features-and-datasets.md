# Indicators, Feature Normalization & `NeuralNetwork` Datasets (`v0.3.0`)

This document describes the extensible indicator registry (`src/tradingbot/indicators/`), window-relative feature extraction (`src/tradingbot/features/`), and `NeuralNetwork` v0.6.0 dataset/config export (`src/tradingbot/dataset/`, `apps/export_dataset_app.py`).

---

## 1. Extensible Indicator Engine (`src/tradingbot/indicators/`)

All indicators inherit from `BaseIndicator` and register in `INDICATOR_REGISTRY`:

| Type | Parameters | Warm-up Index | Output Columns | NN Normalization Scale |
| :--- | :--- | :--- | :--- | :--- |
| `sma` | `period` | `period - 1` | `value` | `price_level` |
| `ema` | `period` | `period - 1` | `value` | `price_level` |
| `rsi` | `period` | `period` | `value` | `rsi_0_100` (`(v - 50) / 50`) |
| `atr` | `period` | `period` | `value` | `price_delta` (`v / scale`) |
| `bollinger` | `period`, `mult` | `period - 1` | `middle`, `upper`, `lower` | `price_level` |
| `squeeze_momentum` | `bb_length`, `bb_mult`, `kc_length`, `kc_mult`, `mom_length` | `max(bb-1, kc-1, 2*mom-2)` | `momentum`, `squeeze_state` | `price_delta`, `bounded` |

### Strict Warm-up Contract
- Indices `< warmup_period()` emit `None` so the Web UI can render all raw historical candles while starting each indicator line only once sufficient bars exist.
- `FeaturePipeline` computes `max_warmup = max(nn_indicator_warmups, atr_warmup)` (including `atr_period` whenever `normalization == "atr_relative"` or `triple_barrier.mode == "atr_multiple"`) and **strictly excludes** any candle or window with missing (`None`) or non-finite values from CSV exports and live NN inference.

### Adding a New Indicator Tomorrow
1. Subclass `BaseIndicator` implementing `_validate_params()`, `warmup_period()`, `output_columns()`, `nn_scale_kind(col)`, and `compute(candles)` and call `register_indicator("my_ind", MyIndicator)`.
2. Add the indicator entry to `configs/assets/<asset>.json` (`"use_in_nn": true/false`, `"plot": "overlay"|"subchart"|"none"`).
3. Zero database migrations or re-downloads required:
   - If `use_in_nn: false`, it computes automatically on all stored history for charting.
   - If `use_in_nn: true`, re-run `./scripts/export_dataset.sh` to recompute historical indicator values, export the updated CSV + `feature_schema.json`, and regenerate `nn_config.json` with the new `input_size`.

---

## 2. Window-Relative Feature Normalization (`FeaturePipeline`)

Raw prices are non-stationary across years and assets and are never fed directly to the MLP. For every input window of $W$ bars ending at anchor bar $t$ with anchor close $C_t$ and local scale $S_t$ (`max(ATR_t, C_t * 1e-6)` for `atr_relative`, or $C_t$ for `pct_relative`):

- **Base per-bar OHLCV features (5):**
  - `(open - C_t) / S_t`, `(high - C_t) / S_t`, `(low - C_t) / S_t`, `(close - C_t) / S_t`
  - `log(1 + volume / max(window_mean_volume, 1e-9))`
- **Contiguity guard:** For `market_schedule: "24/7"`, windows containing missing timestamp gaps (either in the input window or the forecast horizon) are strictly skipped; for `"sessions"` markets, weekend/holiday gaps are permitted.
- **Indicator features (per `use_in_nn: true` indicator):** normalized according to `nn_scale_kind`:
  - `price_level`: `(val - C_t) / S_t`
  - `price_delta`: `val / S_t`
  - `rsi_0_100`: `(val - 50) / 50` in `[-1, +1]`
  - `bounded`: passed through directly (e.g., `squeeze_state ∈ {-1, 0, +1}`)

For `configs/assets/btc_1h.json` (`W = 24`, `H = 6`, `max_warmup = 199` due to `sma_200` → `223` minimum candles for inference, 5 base features + 6 indicator outputs from `ema_20`, `ema_55`, `sma_200`, `rsi_14`, `squeeze_20` = 11 per bar):
`input_size = 24 × 11 = 264` (`btc_4h` uses `24 → 6`; `btc_1d` uses `24 → 7`).

---

## 3. Target Encodings & `NeuralNetwork` v0.6.0 Parity

### Model A — Candle Predictor (`kind = "candles"`)
- **Output (`H × 4`):** relative `(open, high, low, close)` of future candles `t+1 .. t+H` normalized by `(C_t, S_t)`.
- **`libnn` config:** `loss: "mse"`, output activation `"linear"`.

### Model B — Signal Classifier (`kind = "signal"`)
- **Output (`3`):** one-hot `[HOLD, LONG, SHORT]` via conservative **Triple Barrier Labeling** over future bars `t+1 .. t+H` supporting both fixed-percentage (`mode: "pct"` with `take_profit_pct` / `stop_loss_pct`) and volatility-adaptive ATR-multiple barriers (`mode: "atr_multiple"` with `take_profit_atr` / `stop_loss_atr`, keeping class balance stationary across volatility regimes; if a bar touches both SL and TP simultaneously, that side is invalidated).
- **`libnn` config:** `loss: "cce"`, output activation `"softmax"` (activates `libnn` v0.6.0's numerically stable fused `Softmax + CCE` backward path).

### Exact `NeuralNetwork` v0.6.0 Contract
- CSV values are formatted with `%.9g` (exact 32-bit float round-trip matching `libnn::Dataset::save_csv()`).
- Generated `nn_config.json` contains **zero unknown keys**, enforces **`validation_shuffle: false`** (strictly chronological split preventing future data leakage), enables **`keep_best_validation_model: true`**, and pre-populates `server.host` and `server.port` from the asset configuration.
- Root `configs/bot.json` includes an optional `"export"` block (`output_dir`, `dataset_path_prefix`, `model_save_dir`) so exported `nn_config.json` files automatically emit `NeuralNetwork`-ready relative paths (e.g. `"data/<file>.csv"` and `"models/<asset>_<kind>.json"`) without needing CLI flags on every run (set `"dataset_path_prefix": ""` to emit absolute local CSV paths instead).
- `feature_schema.json` stores a SHA-256 `feature_hash` so live inference (`v0.4.0+`) verifies that the loaded model and active indicator configuration match.

---

## 4. CLI Usage (`apps/export_dataset_app.py` / `scripts/export_dataset.sh`)

```bash
# 1) Generate an offline synthetic test dataset + nn_config.json (2000 bars, zero API needed)
./scripts/export_dataset.sh configs/bot.json --asset btc_1h --kind both --synthetic-candles 2000

# 2) Export from real SQLite candles stored by backfill_app using export.* defaults in configs/bot.json
#    (add --update-asset-schema to update the asset's configured schema_path;
#     override paths optionally with --output-dir, --dataset-path-prefix, --model-save-dir)
./scripts/export_dataset.sh configs/bot.json --asset btc_1h --kind both --update-asset-schema

# 3) Train and serve inside your C++17 NeuralNetwork repository:
#    - With configs/bot.json default export.dataset_path_prefix="data", copy CSVs to NeuralNetwork/data/
#      and *_nn_config.json to NeuralNetwork/configs/ (or export with --dataset-path-prefix "" to embed absolute CSV paths)
#    - Use --compact for multiclass signal models
./build/release/apps/train_app configs/btc_1h_candles_nn_config.json
./build/release/apps/train_app configs/btc_1h_signal_nn_config.json --compact
./build/release/apps/server_app configs/btc_1h_candles_nn_config.json
./build/release/apps/server_app configs/btc_1h_signal_nn_config.json
```
