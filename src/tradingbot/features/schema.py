"""Feature schema serialization and integrity fingerprinting.

Guarantees that live inference (v0.4.0+) produces the exact same feature vector
layout, indicator parameters, and normalization rules used during offline
NeuralNetwork training (v0.3.0).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a feature_schema.json file is missing, malformed, or incompatible."""


@dataclass(frozen=True)
class FeatureSchema:
    """Immutable contract describing the exact inputs and targets of a trained NN model."""

    schema_version: int
    asset_id: str
    provider: str
    symbol: str
    timeframe: str
    model_kind: str
    input_window: int
    forecast_horizon: int
    normalization: str
    atr_period: int
    max_warmup: int
    input_size: int
    output_size: int
    per_bar_features: tuple[str, ...]
    target_columns: tuple[str, ...]
    indicators: tuple[dict[str, Any], ...]
    triple_barrier: dict[str, float] | None
    feature_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "provider": self.provider,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_kind": self.model_kind,
            "input_window": self.input_window,
            "forecast_horizon": self.forecast_horizon,
            "normalization": self.normalization,
            "atr_period": self.atr_period,
            "max_warmup": self.max_warmup,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "per_bar_features": list(self.per_bar_features),
            "target_columns": list(self.target_columns),
            "indicators": list(self.indicators),
            "triple_barrier": self.triple_barrier,
            "feature_hash": self.feature_hash,
        }


def compute_feature_hash(
    *,
    model_kind: str,
    input_window: int,
    forecast_horizon: int,
    normalization: str,
    atr_period: int,
    per_bar_features: tuple[str, ...],
    target_columns: tuple[str, ...],
    indicators: tuple[dict[str, Any], ...],
    triple_barrier: dict[str, float] | None,
) -> str:
    """Compute a deterministic SHA-256 fingerprint of the feature & target definition."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": model_kind,
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "normalization": normalization,
        "atr_period": atr_period,
        "per_bar_features": list(per_bar_features),
        "target_columns": list(target_columns),
        "indicators": list(indicators),
        "triple_barrier": triple_barrier,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_feature_schema(schema: FeatureSchema, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(schema.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return file_path


def load_feature_schema(path: str | Path) -> FeatureSchema:
    file_path = Path(path)
    if not file_path.is_file():
        raise SchemaError(f"Feature schema file not found: {file_path}")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Invalid JSON in feature schema '{file_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise SchemaError("Feature schema root must be a JSON object")

    required = {
        "schema_version",
        "asset_id",
        "provider",
        "symbol",
        "timeframe",
        "model_kind",
        "input_window",
        "forecast_horizon",
        "normalization",
        "atr_period",
        "max_warmup",
        "input_size",
        "output_size",
        "per_bar_features",
        "target_columns",
        "indicators",
        "triple_barrier",
        "feature_hash",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise SchemaError(f"Feature schema missing required keys: {', '.join(missing)}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"Unsupported feature schema_version {raw['schema_version']!r} (expected {SCHEMA_VERSION})"
        )
    if raw["model_kind"] not in ("candles", "signal"):
        raise SchemaError(f"Invalid model_kind {raw['model_kind']!r}")

    return FeatureSchema(
        schema_version=int(raw["schema_version"]),
        asset_id=str(raw["asset_id"]),
        provider=str(raw["provider"]),
        symbol=str(raw["symbol"]),
        timeframe=str(raw["timeframe"]),
        model_kind=str(raw["model_kind"]),
        input_window=int(raw["input_window"]),
        forecast_horizon=int(raw["forecast_horizon"]),
        normalization=str(raw["normalization"]),
        atr_period=int(raw["atr_period"]),
        max_warmup=int(raw["max_warmup"]),
        input_size=int(raw["input_size"]),
        output_size=int(raw["output_size"]),
        per_bar_features=tuple(str(x) for x in raw["per_bar_features"]),
        target_columns=tuple(str(x) for x in raw["target_columns"]),
        indicators=tuple(dict(ind) for ind in raw["indicators"]),
        triple_barrier=dict(raw["triple_barrier"]) if raw["triple_barrier"] is not None else None,
        feature_hash=str(raw["feature_hash"]),
    )
