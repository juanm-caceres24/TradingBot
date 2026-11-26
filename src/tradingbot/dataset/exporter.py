"""Dataset CSV exporter producing NeuralNetwork (libnn v0.6.0) training artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tradingbot.config import AssetConfig
from tradingbot.dataset.nn_config_generator import (
    NNTrainingOptions,
    build_nn_config_dict,
    save_nn_config,
)
from tradingbot.features import FeaturePipeline, save_feature_schema
from tradingbot.models import Candle


@dataclass(frozen=True)
class ExportSummary:
    """Summary of exported CSV dataset, feature schema, and NeuralNetwork config."""

    asset_id: str
    model_kind: str
    input_size: int
    output_size: int
    total_candles: int
    samples_written: int
    csv_path: Path
    schema_path: Path
    nn_config_path: Path
    class_counts: dict[str, int] | None = None


def export_dataset(
    asset: AssetConfig,
    candles: Sequence[Candle],
    *,
    model_kind: str,
    output_dir: str | Path,
    csv_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    nn_config_path: str | Path | None = None,
    model_save_path: str | None = None,
    dataset_path_prefix: str = "",
    model_save_dir: str = "models",
    training_options: NNTrainingOptions | None = None,
) -> ExportSummary:
    """Export sliding-window CSV, feature_schema.json, and libnn nn_config.json."""
    if model_kind not in ("candles", "signal"):
        raise ValueError(f"Unsupported model_kind {model_kind!r}. Expected 'candles' or 'signal'.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FeaturePipeline(asset)
    schema = pipeline.build_schema(model_kind)
    opts = training_options if training_options is not None else NNTrainingOptions()

    w = asset.features.input_window
    h = asset.features.forecast_horizon
    stem = f"{asset.id}_{model_kind}_{w}to{h}"

    resolved_csv = Path(csv_path) if csv_path is not None else (out_dir / f"{stem}.csv")
    resolved_schema = (
        Path(schema_path)
        if schema_path is not None
        else (out_dir / f"{asset.id}_{model_kind}_schema.json")
    )
    resolved_nn_cfg = (
        Path(nn_config_path)
        if nn_config_path is not None
        else (out_dir / f"{asset.id}_{model_kind}_nn_config.json")
    )
    clean_model_dir = model_save_dir.strip().rstrip("/")
    resolved_model_save = (
        model_save_path
        if model_save_path is not None
        else (
            f"{clean_model_dir}/{asset.id}_{model_kind}.json"
            if clean_model_dir
            else f"{asset.id}_{model_kind}.json"
        )
    )

    resolved_csv.parent.mkdir(parents=True, exist_ok=True)
    header_cols = list(pipeline.input_column_names()) + list(
        pipeline.target_column_names(model_kind)
    )

    samples_written = 0
    class_counts: dict[str, int] | None = (
        {"HOLD": 0, "LONG": 0, "SHORT": 0} if model_kind == "signal" else None
    )

    with resolved_csv.open("w", encoding="utf-8", newline="\n") as fh:
        if opts.dataset_has_header:
            fh.write(",".join(header_cols) + "\n")
        for _open_time_ms, input_vec, target_vec in pipeline.iter_training_samples(
            candles, model_kind
        ):
            # Format floats with %.9g matching libnn::Dataset::save_csv() for exact float32 round-trip.
            row_str = ",".join(f"{v:.9g}" for v in (*input_vec, *target_vec))
            fh.write(row_str + "\n")
            samples_written += 1
            if class_counts is not None:
                if target_vec[1] > 0.5:
                    class_counts["LONG"] += 1
                elif target_vec[2] > 0.5:
                    class_counts["SHORT"] += 1
                else:
                    class_counts["HOLD"] += 1

    save_feature_schema(schema, resolved_schema)
    clean_prefix = dataset_path_prefix.strip().rstrip("/")
    nn_dataset_path = (
        f"{clean_prefix}/{resolved_csv.name}"
        if clean_prefix
        else str(resolved_csv.resolve())
    )
    nn_cfg = build_nn_config_dict(
        asset,
        schema,
        dataset_path=nn_dataset_path,
        model_save_path=resolved_model_save,
        options=opts,
    )
    save_nn_config(nn_cfg, resolved_nn_cfg)

    return ExportSummary(
        asset_id=asset.id,
        model_kind=model_kind,
        input_size=schema.input_size,
        output_size=schema.output_size,
        total_candles=len(candles),
        samples_written=samples_written,
        csv_path=resolved_csv,
        schema_path=resolved_schema,
        nn_config_path=resolved_nn_cfg,
        class_counts=class_counts,
    )
