# NeuralNetwork Client, Trade Analysis & Alert Engine (`v0.4.0`)

This document describes the HTTP inference client (`src/tradingbot/nn_client/`), candle reconstruction, trade analysis, and stateful alert rule evaluation (`src/tradingbot/analysis/`).

---

## 1. Companion `NeuralNetwork` v0.6.0 HTTP Client (`src/tradingbot/nn_client/`)

`NeuralNetworkClient` communicates directly with `NeuralNetwork`'s `server_app` over HTTP/1.1 + JSON using only Python's standard library:

- `GET /health` → verifies `{"status": "ok"}`.
- `GET /info` → parses `input_size`, `output_size`, `num_layers`, `parameters`, and `layers`.
- `POST /predict` → sends `{"input": [...]}` (adding `"include_neurons": true` when requested) and parses `{"output": [...], "latency_us": ...}` (validating finite floats and non-negative `latency_us`).

### Startup Schema Integrity Check (`AssetNNClient.verify_all()`)
Before entering live inference, `AssetNNClient` validates every enabled model for an asset (`candle_predictor`, `signal_classifier`, and any `extra_models`):
1. If `schema_path` points to an exported `feature_schema.json` on disk, verifies its integrity and confirms that its SHA-256 `feature_hash` (for `candle_predictor`/`signal_classifier`) or `output_size` (for `extra_models`) matches the active asset configuration. If indicators or window lengths were edited without retraining, startup fails fast with `SchemaMismatchError`. If a configured `schema_path` is missing on disk, a warning is logged and `/info` dimension checks still run.
2. Queries `GET /info` on the running `server_app` and confirms exact `input_size` (and `output_size` for primary models) parity.

---

## 2. Predicted Candle Reconstruction (`reconstruct_predicted_candles`)

Transforms $H \times 4$ relative OHLC outputs back into absolute prices using the anchor candle's `(anchor_close, scale)`:
- `price = max(1e-8, anchor_close + rel * scale)`
- Enforces valid OHLC geometry (`high = max(h, l, o, c)`, `low = min(h, l, o, c)`).
- Assigns monotonically decaying per-step horizon confidence (`base_confidence * (1 - decay)^(step - 1)`) so nearer predicted bars render more opaquely than distant bars on the Web UI chart.

---

## 3. Trade Analysis (`TradeAnalyzer`)

Supports all four per-asset model combinations cleanly (`extra_models` outputs are forwarded verbatim in `PredictionSnapshot.extra_outputs`):
1. **Dual-Model Cross-Confirmation (`candle_predictor` + `signal_classifier`):** Requires `signal.best_probability >= min_signal_confidence`, rejects active directional contradictions and moves with `|expected_move_pct| < min_expected_move_pct`, estimates TP/SL from the projected OHLC path (with Triple-Barrier stop floor if predicted drawdown is zero), and enforces `risk_reward >= min_risk_reward`.
2. **Candle Predictor Only:** Evaluates `expected_move_pct` against `min_expected_move_pct`, computes directional consistency across all $H$ bars and R:R, and produces an explicit heuristic confidence score.
3. **Signal Classifier Only:** Evaluates calibrated Softmax probability against `min_signal_confidence` and derives TP/SL and R:R from the asset's `triple_barrier` parameters (`0 < pct < 1`).
4. **No Primary Models Enabled:** Emits neutral `HOLD` (for charting-only assets).

---

## 4. Stateful Alert Engine (`AlertEvaluator`)

Evaluates closed-candle `PredictionSnapshot`s per asset:
- **Thresholds:** action in `alerts.actions`, `confidence >= min_confidence`, `|expected_move_pct| >= min_expected_move_pct`, and `risk_reward >= min_risk_reward` (when `min_risk_reward > 0`, missing `risk_reward` does not qualify).
- **Persistence (`consecutive_candles`):** tracks consecutive matching closed candles; any `HOLD`, below-threshold bar, or multi-bar gap (in `"24/7"` schedule) resets the streak, while a direction flip restarts the streak at 1.
- **Anti-spam spacing (`cooldown_candles`):** enforces a minimum separation of `cooldown_candles` closed bars between emitted alerts (i.e., `cooldown_candles = 3` suppresses the next 2 bars and permits re-firing on the 3rd bar; `0` or `1` imposes no extra spacing).
- **Idempotency & restart safety:** ignores re-polls of the same or older `anchor_open_time_ms`; on startup, `AssetRunner` seeds `_last_anchor_ms` from the latest persisted SQLite alert for that asset.
