"""Per-asset multi-model NeuralNetwork client with startup schema verification."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from tradingbot.config import AssetConfig, ModelEndpointConfig
from tradingbot.features import FeaturePipeline, FeatureSchema, SchemaError, load_feature_schema
from tradingbot.models import SignalPrediction
from tradingbot.nn_client.client import (
    ModelInfo,
    NNClientError,
    NeuralNetworkClient,
    RawPrediction,
    SchemaMismatchError,
)

_LOGGER = logging.getLogger(__name__)


def verify_endpoint_schema(
    client: NeuralNetworkClient,
    expected_schema: FeatureSchema,
    *,
    model_label: str,
    schema_file_path: str = "",
) -> ModelInfo:
    """Verify server_app /info topology and optional persisted feature_schema.json."""
    if schema_file_path:
        path = Path(schema_file_path)
        if path.is_file():
            try:
                saved = load_feature_schema(path)
            except SchemaError as exc:
                raise SchemaMismatchError(
                    f"Model '{model_label}' schema file '{path}' is invalid: {exc}"
                ) from exc
            if saved.feature_hash != expected_schema.feature_hash:
                raise SchemaMismatchError(
                    f"Model '{model_label}' feature_hash mismatch against '{path}': "
                    f"trained model expects hash {saved.feature_hash[:12]} "
                    f"(input={saved.input_size}, output={saved.output_size}), "
                    f"current config produces {expected_schema.feature_hash[:12]} "
                    f"(input={expected_schema.input_size}, output={expected_schema.output_size}). "
                    "Re-export dataset and retrain the model."
                )
        else:
            _LOGGER.warning(
                "Schema file '%s' configured for model '%s' does not exist on disk; "
                "falling back to /info dimension checks only.",
                path,
                model_label,
            )

    info = client.info()
    if info.input_size != expected_schema.input_size:
        raise SchemaMismatchError(
            f"Model '{model_label}' at {client.base_url} expects input_size={info.input_size}, "
            f"but current asset feature pipeline produces input_size={expected_schema.input_size}."
        )
    if info.output_size != expected_schema.output_size:
        raise SchemaMismatchError(
            f"Model '{model_label}' at {client.base_url} produces output_size={info.output_size}, "
            f"but expected output_size={expected_schema.output_size}."
        )
    return info


class AssetNNClient:
    """Coordinates all configured NeuralNetwork endpoints for a single AssetConfig."""

    def __init__(self, asset: AssetConfig) -> None:
        self._asset = asset
        self._pipeline = FeaturePipeline(asset)
        nn_cfg = asset.neural_networks
        timeout = nn_cfg.timeout_sec

        self._candle_cfg: ModelEndpointConfig = nn_cfg.candle_predictor
        self._signal_cfg: ModelEndpointConfig = nn_cfg.signal_classifier

        self._candle_client: NeuralNetworkClient | None = (
            NeuralNetworkClient(self._candle_cfg.host, self._candle_cfg.port, timeout_sec=timeout)
            if self._candle_cfg.enabled
            else None
        )
        self._signal_client: NeuralNetworkClient | None = (
            NeuralNetworkClient(self._signal_cfg.host, self._signal_cfg.port, timeout_sec=timeout)
            if self._signal_cfg.enabled
            else None
        )
        self._extra_clients: dict[str, tuple[ModelEndpointConfig, NeuralNetworkClient]] = {}
        for name, ep_cfg in nn_cfg.extra_models.items():
            if ep_cfg.enabled:
                self._extra_clients[name] = (
                    ep_cfg,
                    NeuralNetworkClient(ep_cfg.host, ep_cfg.port, timeout_sec=timeout),
                )

    @property
    def has_any_model(self) -> bool:
        return (
            self._candle_client is not None
            or self._signal_client is not None
            or bool(self._extra_clients)
        )

    @property
    def candle_enabled(self) -> bool:
        return self._candle_client is not None

    @property
    def signal_enabled(self) -> bool:
        return self._signal_client is not None

    def health_summary(self) -> dict[str, bool]:
        summary: dict[str, bool] = {}
        if self._candle_client is not None:
            summary["candle_predictor"] = self._candle_client.health()
        if self._signal_client is not None:
            summary["signal_classifier"] = self._signal_client.health()
        for name, (_cfg, client) in self._extra_clients.items():
            summary[name] = client.health()
        return summary

    def verify_all(self) -> dict[str, ModelInfo]:
        """Verify health and input/output dimensions of all enabled models for this asset."""
        infos: dict[str, ModelInfo] = {}
        if self._candle_client is not None:
            schema_c = self._pipeline.build_schema("candles")
            infos["candle_predictor"] = verify_endpoint_schema(
                self._candle_client,
                schema_c,
                model_label=f"{self._asset.id}:candle_predictor",
                schema_file_path=self._candle_cfg.schema_path,
            )
        if self._signal_client is not None:
            schema_s = self._pipeline.build_schema("signal")
            infos["signal_classifier"] = verify_endpoint_schema(
                self._signal_client,
                schema_s,
                model_label=f"{self._asset.id}:signal_classifier",
                schema_file_path=self._signal_cfg.schema_path,
            )
        expected_input = self._pipeline.build_schema("candles").input_size
        for name, (ep_cfg, client) in self._extra_clients.items():
            info = client.info()
            if info.input_size != expected_input:
                raise SchemaMismatchError(
                    f"Extra model '{self._asset.id}:{name}' at {client.base_url} expects "
                    f"input_size={info.input_size}, got {expected_input}"
                )
            if ep_cfg.schema_path:
                ep_path = Path(ep_cfg.schema_path)
                if ep_path.is_file():
                    try:
                        saved = load_feature_schema(ep_path)
                    except SchemaError as exc:
                        raise SchemaMismatchError(
                            f"Extra model '{self._asset.id}:{name}' schema file '{ep_path}' is invalid: {exc}"
                        ) from exc
                    if info.output_size != saved.output_size:
                        raise SchemaMismatchError(
                            f"Extra model '{self._asset.id}:{name}' at {client.base_url} produces "
                            f"output_size={info.output_size}, schema expects {saved.output_size}"
                        )
            infos[name] = info
        return infos

    def predict_candles(
        self,
        input_vector: Sequence[float],
        *,
        include_neurons: bool = False,
    ) -> RawPrediction | None:
        if self._candle_client is None:
            return None
        pred = self._candle_client.predict(input_vector, include_neurons=include_neurons)
        expected_out = self._asset.features.forecast_horizon * 4
        if len(pred.output) != expected_out:
            raise NNClientError(
                f"Candle predictor returned {len(pred.output)} values, expected {expected_out}"
            )
        return pred

    def predict_signal(
        self,
        input_vector: Sequence[float],
        *,
        include_neurons: bool = False,
    ) -> SignalPrediction | None:
        if self._signal_client is None:
            return None
        pred = self._signal_client.predict(input_vector, include_neurons=include_neurons)
        if len(pred.output) != 3:
            raise NNClientError(
                f"Signal classifier returned {len(pred.output)} values, expected 3 ([HOLD, LONG, SHORT])"
            )
        hold_p, long_p, short_p = pred.output
        if hold_p < -1e-6 or long_p < -1e-6 or short_p < -1e-6:
            raise NNClientError(
                f"Signal classifier returned negative probability: {pred.output!r}"
            )
        # Clamp tiny floating-point rounding drift and renormalize if sum is close to 1.
        h = max(0.0, hold_p)
        l = max(0.0, long_p)
        s = max(0.0, short_p)
        total = h + l + s
        if abs(total - 1.0) > 1e-2 or total <= 0.0:
            raise NNClientError(
                f"Signal classifier probabilities do not sum to ~1.0: {pred.output!r}"
            )
        return SignalPrediction(
            hold_prob=h / total,
            long_prob=l / total,
            short_prob=s / total,
            latency_us=pred.latency_us,
        )

    def predict_extra(
        self,
        input_vector: Sequence[float],
    ) -> dict[str, RawPrediction]:
        results: dict[str, RawPrediction] = {}
        for name, (_cfg, client) in self._extra_clients.items():
            results[name] = client.predict(input_vector)
        return results
