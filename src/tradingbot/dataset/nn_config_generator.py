"""Strict NeuralNetwork (libnn v0.7.0) config.json generator and verifier.

Every key produced matches the strict schema in NeuralNetwork/docs/05-config-and-serialization.md
and docs/08-regularization-and-metrics.md with zero unknown keys and strictly chronological
time-series validation splits (`validation_shuffle: false`, `keep_best_validation_model: true`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradingbot.config import AssetConfig, NNTrainingConfig
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
        "huber_delta",
        "optimizer",
        "epochs",
        "batch_size",
        "learning_rate",
        "lr_schedule",
        "min_learning_rate",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "weight_decay",
        "dropout_rate",
        "grad_clip_norm",
        "shuffle",
        "num_threads",
        "seed",
        "validation_split",
        "validation_shuffle",
        "keep_best_validation_model",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "report_path",
        "log_every_epochs",
    }
)
_LIBNN_SERVER_KEYS = frozenset({"host", "port", "num_workers"})
_LIBNN_HIDDEN_ACTIVATIONS = frozenset({"relu", "tanh", "sigmoid", "leaky_relu"})
_LIBNN_ACTIVATIONS = frozenset({"linear", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"})
_LIBNN_LOSSES = frozenset(
    {
        "mse",
        "huber",
        "smooth_l1",
        "bce",
        "binary_cross_entropy",
        "cce",
        "categorical_cross_entropy",
        "cross_entropy",
    }
)
_LIBNN_CANDLE_LOSSES = frozenset({"mse", "huber"})
_LIBNN_OPTIMIZERS = frozenset({"sgd", "adam"})
_LIBNN_LR_SCHEDULES = frozenset({"none", "cosine"})


def _is_real_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


@dataclass(frozen=True)
class NNTrainingOptions:
    """Hyperparameters used when auto-generating a NeuralNetwork v0.7.0 config.json."""

    hidden_layers: tuple[int, ...] = (128, 64)
    hidden_activation: str = "relu"
    candle_loss: str = "mse"
    huber_delta: float = 1.0
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    lr_schedule: str = "none"
    min_learning_rate: float = 0.0
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    dropout_rate: float = 0.0
    grad_clip_norm: float = 0.0
    validation_split: float = 0.2
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    num_threads: int = 4
    init_seed: int = 42
    seed: int = 100
    log_every_epochs: int = 10
    server_workers: int = 4
    dataset_has_header: bool = True

    def __post_init__(self) -> None:
        if any(not _is_real_int(h) or h < 1 for h in self.hidden_layers):
            raise ValueError("All hidden_layers sizes must be integers >= 1")
        if self.hidden_activation not in _LIBNN_HIDDEN_ACTIVATIONS:
            raise ValueError(
                f"Unsupported hidden_activation: {self.hidden_activation!r}. "
                f"Expected one of {sorted(_LIBNN_HIDDEN_ACTIVATIONS)}."
            )
        if self.candle_loss not in _LIBNN_CANDLE_LOSSES:
            raise ValueError(
                f"Unsupported candle_loss: {self.candle_loss!r}. Expected 'mse' or 'huber'."
            )
        if not _is_finite_number(self.huber_delta) or self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be a finite number > 0")
        if self.optimizer not in _LIBNN_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {self.optimizer!r}")
        if (
            not _is_real_int(self.epochs)
            or self.epochs < 1
            or not _is_real_int(self.batch_size)
            or self.batch_size < 1
            or not _is_real_int(self.num_threads)
            or self.num_threads < 1
            or not _is_real_int(self.server_workers)
            or self.server_workers < 1
        ):
            raise ValueError("epochs, batch_size, num_threads, and server_workers must be >= 1")
        if (
            not _is_real_int(self.init_seed)
            or self.init_seed < 0
            or not _is_real_int(self.seed)
            or self.seed < 0
            or not _is_real_int(self.log_every_epochs)
            or self.log_every_epochs < 0
        ):
            raise ValueError("init_seed, seed, and log_every_epochs must be integers >= 0")
        if not _is_finite_number(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a finite number > 0")
        if self.lr_schedule not in _LIBNN_LR_SCHEDULES:
            raise ValueError(f"Unsupported lr_schedule: {self.lr_schedule!r}")
        if not _is_finite_number(self.min_learning_rate) or not (
            0.0 <= self.min_learning_rate <= self.learning_rate
        ):
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        if not _is_finite_number(self.adam_beta1) or not (0.0 <= self.adam_beta1 < 1.0):
            raise ValueError("adam_beta1 must be in [0, 1)")
        if not _is_finite_number(self.adam_beta2) or not (0.0 <= self.adam_beta2 < 1.0):
            raise ValueError("adam_beta2 must be in [0, 1)")
        if not _is_finite_number(self.adam_epsilon) or self.adam_epsilon <= 0.0:
            raise ValueError("adam_epsilon must be > 0")
        if not _is_finite_number(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if not _is_finite_number(self.dropout_rate) or not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("dropout_rate must be in [0, 1)")
        if not _is_finite_number(self.grad_clip_norm) or self.grad_clip_norm < 0.0:
            raise ValueError("grad_clip_norm must be >= 0")
        if not _is_finite_number(self.validation_split) or not (0.0 <= self.validation_split < 1.0):
            raise ValueError("validation_split must be in [0, 1)")
        if not _is_real_int(self.early_stopping_patience) or self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be an integer >= 0")
        if self.early_stopping_patience > 0 and self.validation_split <= 0.0:
            raise ValueError("early_stopping_patience > 0 requires validation_split > 0")
        if (
            not _is_finite_number(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0.0
        ):
            raise ValueError("early_stopping_min_delta must be >= 0")
        if not isinstance(self.dataset_has_header, bool):
            raise ValueError("dataset_has_header must be a boolean")

    @classmethod
    def from_asset_config(cls, cfg: NNTrainingConfig) -> NNTrainingOptions:
        """Convert an AssetConfig.training block into NNTrainingOptions."""
        return cls(
            hidden_layers=cfg.hidden_layers,
            hidden_activation=cfg.hidden_activation,
            candle_loss=cfg.candle_loss,
            huber_delta=cfg.huber_delta,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            lr_schedule=cfg.lr_schedule,
            min_learning_rate=cfg.min_learning_rate,
            optimizer=cfg.optimizer,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            adam_epsilon=cfg.adam_epsilon,
            weight_decay=cfg.weight_decay,
            dropout_rate=cfg.dropout_rate,
            grad_clip_norm=cfg.grad_clip_norm,
            validation_split=cfg.validation_split,
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_min_delta=cfg.early_stopping_min_delta,
            num_threads=cfg.num_threads,
            init_seed=cfg.init_seed,
            seed=cfg.seed,
            log_every_epochs=cfg.log_every_epochs,
            server_workers=cfg.server_workers,
            dataset_has_header=cfg.dataset_has_header,
        )


def build_nn_config_dict(
    asset: AssetConfig,
    schema: FeatureSchema,
    *,
    dataset_path: str,
    model_save_path: str,
    report_path: str = "",
    options: NNTrainingOptions | None = None,
) -> dict[str, Any]:
    """Generate a strictly valid NeuralNetwork v0.7.0 config.json dictionary."""
    opts = (
        options
        if options is not None
        else NNTrainingOptions.from_asset_config(asset.training)
    )

    if schema.model_kind == "candles":
        loss = opts.candle_loss
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
            "huber_delta": opts.huber_delta,
            "optimizer": opts.optimizer,
            "epochs": opts.epochs,
            "batch_size": opts.batch_size,
            "learning_rate": opts.learning_rate,
            "lr_schedule": opts.lr_schedule,
            "min_learning_rate": opts.min_learning_rate,
            "adam_beta1": opts.adam_beta1,
            "adam_beta2": opts.adam_beta2,
            "adam_epsilon": opts.adam_epsilon,
            "weight_decay": opts.weight_decay,
            "dropout_rate": opts.dropout_rate,
            "grad_clip_norm": opts.grad_clip_norm,
            "shuffle": True,
            "num_threads": opts.num_threads,
            "seed": opts.seed,
            "validation_split": opts.validation_split,
            "validation_shuffle": False,
            "keep_best_validation_model": opts.validation_split > 0.0,
            "early_stopping_patience": opts.early_stopping_patience,
            "early_stopping_min_delta": opts.early_stopping_min_delta,
            "report_path": report_path,
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
    """Verify strict schema and value parity against NeuralNetwork v0.7.0 config.hpp."""
    if not isinstance(cfg, dict):
        raise ValueError("libnn config must be a JSON object")
    missing_root = _LIBNN_ROOT_KEYS - set(cfg.keys())
    if missing_root:
        raise ValueError(f"Missing required root keys for libnn config: {sorted(missing_root)}")
    unknown_root = set(cfg.keys()) - _LIBNN_ROOT_KEYS
    if unknown_root:
        raise ValueError(f"Unknown root keys for libnn config: {sorted(unknown_root)}")

    for sec in ("model", "training", "server"):
        if not isinstance(cfg[sec], dict):
            raise ValueError(f"libnn config '{sec}' must be a JSON object")

    unknown_model = set(cfg["model"].keys()) - _LIBNN_MODEL_KEYS
    if unknown_model:
        raise ValueError(f"Unknown model keys for libnn config: {sorted(unknown_model)}")

    t_cfg = cfg["training"]
    unknown_train = set(t_cfg.keys()) - _LIBNN_TRAINING_KEYS
    if unknown_train:
        raise ValueError(f"Unknown training keys for libnn config: {sorted(unknown_train)}")

    unknown_server = set(cfg["server"].keys()) - _LIBNN_SERVER_KEYS
    if unknown_server:
        raise ValueError(f"Unknown server keys for libnn config: {sorted(unknown_server)}")

    layers = cfg["model"].get("layers")
    if not isinstance(layers, list) or len(layers) < 2:
        raise ValueError("libnn model must have at least an input and output layer")
    if not isinstance(layers[0], dict) or set(layers[0].keys()) != {"size"}:
        raise ValueError("libnn input layer must contain only 'size'")
    last_idx = len(layers) - 1
    for idx, layer in enumerate(layers[1:], start=1):
        if not isinstance(layer, dict) or set(layer.keys()) != {"size", "activation"}:
            raise ValueError(f"libnn layer[{idx}] must contain only 'size' and 'activation'")
        act = layer["activation"]
        if act not in _LIBNN_ACTIVATIONS:
            raise ValueError(f"Unsupported libnn activation: {act!r}")
        if idx < last_idx and act == "softmax":
            raise ValueError(f"Hidden layer[{idx}] must not use 'softmax' activation")

    loss = t_cfg.get("loss", "mse")
    if loss not in _LIBNN_LOSSES:
        raise ValueError(f"Unsupported libnn loss: {loss!r}")
    huber_delta = t_cfg.get("huber_delta", 1.0)
    if not _is_finite_number(huber_delta) or huber_delta <= 0.0:
        raise ValueError(f"Invalid libnn huber_delta: {huber_delta!r}")
    optimizer = t_cfg.get("optimizer", "sgd")
    if optimizer not in _LIBNN_OPTIMIZERS:
        raise ValueError(f"Unsupported libnn optimizer: {optimizer!r}")
    lr = t_cfg.get("learning_rate", 0.01)
    if not _is_finite_number(lr) or lr <= 0.0:
        raise ValueError(f"Invalid libnn learning_rate: {lr!r}")
    lr_sched = t_cfg.get("lr_schedule", "none")
    if lr_sched not in _LIBNN_LR_SCHEDULES:
        raise ValueError(f"Unsupported libnn lr_schedule: {lr_sched!r}")
    min_lr = t_cfg.get("min_learning_rate", 0.0)
    if not _is_finite_number(min_lr) or not (0.0 <= min_lr <= lr):
        raise ValueError(f"Invalid libnn min_learning_rate: {min_lr!r} (lr={lr!r})")
    wd = t_cfg.get("weight_decay", 0.0)
    if not _is_finite_number(wd) or wd < 0.0:
        raise ValueError(f"Invalid libnn weight_decay: {wd!r}")
    dropout = t_cfg.get("dropout_rate", 0.0)
    if not _is_finite_number(dropout) or not (0.0 <= dropout < 1.0):
        raise ValueError(f"Invalid libnn dropout_rate: {dropout!r}")
    clip = t_cfg.get("grad_clip_norm", 0.0)
    if not _is_finite_number(clip) or clip < 0.0:
        raise ValueError(f"Invalid libnn grad_clip_norm: {clip!r}")
    val_split = t_cfg.get("validation_split", 0.0)
    if not _is_finite_number(val_split) or not (0.0 <= val_split < 1.0):
        raise ValueError(f"Invalid libnn validation_split: {val_split!r}")
    patience = t_cfg.get("early_stopping_patience", 0)
    if not _is_real_int(patience) or patience < 0:
        raise ValueError(f"Invalid libnn early_stopping_patience: {patience!r}")
    if patience > 0 and val_split <= 0.0:
        raise ValueError("libnn early_stopping_patience > 0 requires validation_split > 0")
    min_delta = t_cfg.get("early_stopping_min_delta", 0.0)
    if not _is_finite_number(min_delta) or min_delta < 0.0:
        raise ValueError(f"Invalid libnn early_stopping_min_delta: {min_delta!r}")
    report_path = t_cfg.get("report_path", "")
    if not isinstance(report_path, str):
        raise ValueError("libnn report_path must be a string")


def save_nn_config(cfg: dict[str, Any], path: str | Path) -> Path:
    validate_nn_config_dict(cfg)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return file_path
