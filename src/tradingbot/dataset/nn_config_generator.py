"""Strict NeuralNetwork (libnn v0.6.0) config.json generator and verifier.

Every key produced matches the strict schema in NeuralNetwork/docs/05-config-and-serialization.md
with zero unknown keys and strictly chronological time-series validation splits
(`validation_shuffle: false`, `keep_best_validation_model: true`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradingbot.config import AssetConfig
from tradingbot.features.schema import FeatureSchema

_LIBNN_ROOT_KEYS = frozenset({"model", "training", "server"})
_LIBNN_MODEL_KEYS = frozenset(
    {"layers", "init_seed", "load_path", "save_path", "save_on_finish"}
)
_LIBNN_TRAINING_KEYS = frozenset(
    {
        "dataset_path",
        "dataset_has_header",
        "dataset_generator",
        "loss",
        "optimizer",
        "epochs",
        "batch_size",
        "learning_rate",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "shuffle",
        "num_threads",
        "seed",
        "validation_split",
        "validation_shuffle",
        "keep_best_validation_model",
        "log_every_epochs",
    }
)
_LIBNN_SERVER_KEYS = frozenset({"host", "port", "num_workers"})
_LIBNN_ACTIVATIONS = frozenset({"linear", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"})


@dataclass(frozen=True)
class NNTrainingOptions:
    """Hyperparameters used when auto-generating a NeuralNetwork v0.6.0 config.json."""

    hidden_layers: tuple[int, ...] = (128, 64)
    hidden_activation: str = "relu"
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    validation_split: float = 0.2
    num_threads: int = 4
    init_seed: int = 42
    seed: int = 100
    log_every_epochs: int = 10
    server_workers: int = 4
    dataset_has_header: bool = True

    def __post_init__(self) -> None:
        if any(h < 1 for h in self.hidden_layers):
            raise ValueError("All hidden_layers sizes must be >= 1")
        if self.hidden_activation not in _LIBNN_ACTIVATIONS:
            raise ValueError(f"Unsupported hidden_activation: {self.hidden_activation!r}")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError(f"Unsupported optimizer: {self.optimizer!r}")
        if self.epochs < 1 or self.batch_size < 1 or self.num_threads < 1 or self.server_workers < 1:
            raise ValueError("epochs, batch_size, num_threads, and server_workers must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if not (0.0 <= self.validation_split < 1.0):
            raise ValueError("validation_split must be in [0, 1)")


def build_nn_config_dict(
    asset: AssetConfig,
    schema: FeatureSchema,
    *,
    dataset_path: str,
    model_save_path: str,
    options: NNTrainingOptions | None = None,
) -> dict[str, Any]:
    """Generate a strictly valid NeuralNetwork v0.6.0 config.json dictionary."""
    opts = options if options is not None else NNTrainingOptions()

    if schema.model_kind == "candles":
        loss = "mse"
        output_activation = "linear"
        endpoint = asset.neural_networks.candle_predictor
    elif schema.model_kind == "signal":
        loss = "cce"
        output_activation = "softmax"
        endpoint = asset.neural_networks.signal_classifier
    else:
        raise ValueError(f"Unsupported schema.model_kind: {schema.model_kind!r}")

    layers: list[dict[str, Any]] = [{"size": schema.input_size}]
    for width in opts.hidden_layers:
        layers.append({"size": width, "activation": opts.hidden_activation})
    layers.append({"size": schema.output_size, "activation": output_activation})

    cfg: dict[str, Any] = {
        "model": {
            "layers": layers,
            "init_seed": opts.init_seed,
            "load_path": "",
            "save_path": model_save_path,
            "save_on_finish": True,
        },
        "training": {
            "dataset_path": dataset_path,
            "dataset_has_header": opts.dataset_has_header,
            "dataset_generator": "",
            "loss": loss,
            "optimizer": opts.optimizer,
            "epochs": opts.epochs,
            "batch_size": opts.batch_size,
            "learning_rate": opts.learning_rate,
            "adam_beta1": opts.adam_beta1,
            "adam_beta2": opts.adam_beta2,
            "adam_epsilon": opts.adam_epsilon,
            "shuffle": True,
            "num_threads": opts.num_threads,
            "seed": opts.seed,
            "validation_split": opts.validation_split,
            "validation_shuffle": False,
            "keep_best_validation_model": opts.validation_split > 0.0,
            "log_every_epochs": opts.log_every_epochs,
        },
        "server": {
            "host": endpoint.host,
            "port": endpoint.port,
            "num_workers": opts.server_workers,
        },
    }
    validate_nn_config_dict(cfg)
    return cfg


def validate_nn_config_dict(cfg: dict[str, Any]) -> None:
    """Verify strict schema parity against NeuralNetwork v0.6.0 config.hpp."""
    unknown_root = set(cfg.keys()) - _LIBNN_ROOT_KEYS
    if unknown_root:
        raise ValueError(f"Unknown root keys for libnn config: {sorted(unknown_root)}")

    unknown_model = set(cfg["model"].keys()) - _LIBNN_MODEL_KEYS
    if unknown_model:
        raise ValueError(f"Unknown model keys for libnn config: {sorted(unknown_model)}")

    unknown_train = set(cfg["training"].keys()) - _LIBNN_TRAINING_KEYS
    if unknown_train:
        raise ValueError(f"Unknown training keys for libnn config: {sorted(unknown_train)}")

    unknown_server = set(cfg["server"].keys()) - _LIBNN_SERVER_KEYS
    if unknown_server:
        raise ValueError(f"Unknown server keys for libnn config: {sorted(unknown_server)}")

    layers = cfg["model"]["layers"]
    if len(layers) < 2:
        raise ValueError("libnn model must have at least an input and output layer")
    if set(layers[0].keys()) != {"size"}:
        raise ValueError("libnn input layer must contain only 'size'")
    for idx, layer in enumerate(layers[1:], start=1):
        if set(layer.keys()) != {"size", "activation"}:
            raise ValueError(f"libnn layer[{idx}] must contain only 'size' and 'activation'")
        if layer["activation"] not in _LIBNN_ACTIVATIONS:
            raise ValueError(f"Unsupported libnn activation: {layer['activation']!r}")


def save_nn_config(cfg: dict[str, Any], path: str | Path) -> Path:
    validate_nn_config_dict(cfg)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return file_path
