"""Parser, diagnostic evaluator, and comparator for NeuralNetwork (libnn v0.7.0) JSON reports.

Consumes the domain-agnostic format_version=1 JSON reports emitted by
`train_app --report <path>` (or `training.report_path`) as specified in
NeuralNetwork/docs/08-regularization-and-metrics.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRADING_3CLASS_NAMES = ("HOLD", "LONG", "SHORT")
_OHLC_FIELD_NAMES = ("open", "high", "low", "close")


class ReportError(ValueError):
    """Raised when a NeuralNetwork v0.7.0 training report JSON is invalid or unsupported."""


def _as_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportError(f"Expected object at '{path}', got {type(value).__name__}")
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReportError(f"Expected array at '{path}', got {type(value).__name__}")
    return value


def _as_int(value: Any, path: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"Expected integer at '{path}', got {type(value).__name__}")
    return value


def _as_float(value: Any, path: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"Expected number at '{path}', got {type(value).__name__}")
    fval = float(value)
    if not math.isfinite(fval):
        return None if allow_none else float("nan")
    return fval


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ReportError(f"Expected boolean at '{path}', got {type(value).__name__}")
    return value


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ReportError(f"Expected string at '{path}', got {type(value).__name__}")
    return value


def _fmt_opt(value: float | None, fmt: str = ".4f", default: str = "n/a") -> str:
    if value is None or not math.isfinite(value):
        return default
    return format(value, fmt)


@dataclass(frozen=True)
class ClassMetricRow:
    class_index: int
    support: int
    actual_frequency: float
    mean_predicted_prob: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class OutputMetricRow:
    output_index: int
    mae: float | None
    rmse: float | None
    r2: float | None


@dataclass(frozen=True)
class TrainingReportSummary:
    """Structured summary and diagnostics for one NeuralNetwork v0.7.0 training report."""

    source_path: str
    task: str  # "classification" or "regression"
    eval_split: str  # "validation" or "train"
    input_size: int
    output_size: int
    parameters: int
    topology: tuple[int, ...]
    hidden_activation: str
    loss_name: str
    optimizer: str
    epochs_configured: int
    epochs_completed: int
    early_stopped: bool
    batch_size: int
    learning_rate: float
    lr_schedule: str
    weight_decay: float
    dropout_rate: float
    grad_clip_norm: float
    early_stopping_patience: int
    train_samples: int
    val_samples: int
    elapsed_ms: float
    has_validation: bool
    initial_train_loss: float | None
    final_train_loss: float | None
    best_val_epoch: int | None
    best_val_loss: float | None
    final_val_loss: float | None
    eval_loss: float | None
    baseline_constant_loss: float | None
    loss_improvement_vs_baseline: float | None
    # Classification fields
    num_classes: int | None = None
    accuracy: float | None = None
    majority_baseline_accuracy: float | None = None
    macro_f1: float | None = None
    confusion_matrix: tuple[tuple[int, ...], ...] = ()
    per_class: tuple[ClassMetricRow, ...] = ()
    # Regression fields
    num_outputs: int | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    per_output: tuple[OutputMetricRow, ...] = ()
    # Computed diagnostics
    diagnostics: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return Path(self.source_path).name if self.source_path else "<memory>"

    @property
    def topology_str(self) -> str:
        return "->".join(str(x) for x in self.topology)

    @property
    def beats_baseline(self) -> bool:
        return (
            self.loss_improvement_vs_baseline is not None
            and self.loss_improvement_vs_baseline > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_path": self.source_path,
            "name": self.name,
            "task": self.task,
            "eval_split": self.eval_split,
            "topology": self.topology_str,
            "parameters": self.parameters,
            "loss": self.loss_name,
            "optimizer": self.optimizer,
            "epochs_completed": self.epochs_completed,
            "epochs_configured": self.epochs_configured,
            "early_stopped": self.early_stopped,
            "best_val_epoch": self.best_val_epoch,
            "final_train_loss": self.final_train_loss,
            "best_val_loss": self.best_val_loss,
            "eval_loss": self.eval_loss,
            "baseline_constant_loss": self.baseline_constant_loss,
            "loss_improvement_vs_baseline": self.loss_improvement_vs_baseline,
            "beats_baseline": self.beats_baseline,
            "regularization": {
                "weight_decay": self.weight_decay,
                "dropout_rate": self.dropout_rate,
                "lr_schedule": self.lr_schedule,
                "grad_clip_norm": self.grad_clip_norm,
                "early_stopping_patience": self.early_stopping_patience,
            },
            "diagnostics": list(self.diagnostics),
        }
        if self.task == "classification":
            out["accuracy"] = self.accuracy
            out["majority_baseline_accuracy"] = self.majority_baseline_accuracy
            out["macro_f1"] = self.macro_f1
        else:
            out["mae"] = self.mae
            out["rmse"] = self.rmse
            out["r2"] = self.r2
        return out


def _compute_diagnostics(
    *,
    task: str,
    has_validation: bool,
    epochs_completed: int,
    final_train_loss: float | None,
    best_val_loss: float | None,
    final_val_loss: float | None,
    eval_loss: float | None,
    loss_improvement: float | None,
    per_class: Sequence[ClassMetricRow],
    r2: float | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not has_validation:
        warnings.append("NO_VALIDATION_SPLIT")
    if eval_loss is None:
        warnings.append("NON_FINITE_LOSS")
    elif loss_improvement is not None and loss_improvement <= 0.0:
        warnings.append("DOES_NOT_BEAT_BASELINE")

    if task == "classification":
        collapsed = any(
            (row.support > 0 and row.recall == 0.0)
            or (row.actual_frequency >= 0.10 and row.mean_predicted_prob < 0.02)
            for row in per_class
        )
        if collapsed:
            warnings.append("CLASS_COLLAPSE")
        miscalibrated = any(
            abs(row.mean_predicted_prob - row.actual_frequency) > 0.15
            for row in per_class
        )
        if miscalibrated:
            warnings.append("MISCALIBRATED_PROBS")
    else:
        if r2 is not None and r2 < 0.0:
            warnings.append("NEGATIVE_R2")

    if (
        has_validation
        and epochs_completed >= 10
        and final_train_loss is not None
        and final_train_loss > 0.0
    ):
        if best_val_loss is not None and best_val_loss > final_train_loss * 1.35:
            warnings.append("OVERFITTING")
        elif (
            best_val_loss is not None
            and final_val_loss is not None
            and best_val_loss > 0.0
            and final_val_loss > best_val_loss * 1.15
        ):
            warnings.append("OVERFITTING")

    return tuple(warnings)


def parse_training_report(
    raw: dict[str, Any],
    *,
    source_path: str = "",
) -> TrainingReportSummary:
    """Parse and validate a NeuralNetwork v0.7.0 training report JSON dictionary."""
    root = _as_dict(raw, "root")
    fmt_ver = _as_int(root.get("format_version"), "format_version")
    if fmt_ver != 1:
        raise ReportError(f"Unsupported training report format_version: {fmt_ver!r} (expected 1)")

    task = _as_str(root.get("task"), "task")
    if task not in ("classification", "regression"):
        raise ReportError(f"Unsupported report task: {task!r}")
    eval_split = _as_str(root.get("eval_split"), "eval_split")

    model = _as_dict(root.get("model"), "model")
    topology_raw = _as_list(model.get("topology"), "model.topology")
    if len(topology_raw) < 2:
        raise ReportError("model.topology must contain at least 2 layers")
    topology_sizes: list[int] = []
    hidden_act = "none"
    for idx, layer in enumerate(topology_raw):
        layer_d = _as_dict(layer, f"model.topology[{idx}]")
        size = _as_int(layer_d.get("size"), f"model.topology[{idx}].size")
        assert size is not None
        topology_sizes.append(size)
        if 0 < idx < len(topology_raw) - 1 and hidden_act == "none":
            hidden_act = _as_str(layer_d.get("activation", "unknown"), f"model.topology[{idx}].activation")

    tr = _as_dict(root.get("training"), "training")
    ev = _as_dict(root.get("evaluation"), "evaluation")

    input_size = _as_int(model.get("input_size"), "model.input_size")
    output_size = _as_int(model.get("output_size"), "model.output_size")
    parameters = _as_int(model.get("parameters"), "model.parameters")
    assert input_size is not None and output_size is not None and parameters is not None

    epochs_configured = _as_int(tr.get("epochs_configured"), "training.epochs_configured")
    epochs_completed = _as_int(tr.get("epochs_completed"), "training.epochs_completed")
    batch_size = _as_int(tr.get("batch_size"), "training.batch_size")
    patience = _as_int(tr.get("early_stopping_patience", 0), "training.early_stopping_patience")
    train_samples = _as_int(tr.get("train_samples"), "training.train_samples")
    val_samples = _as_int(tr.get("val_samples", 0), "training.val_samples")
    assert (
        epochs_configured is not None
        and epochs_completed is not None
        and batch_size is not None
        and patience is not None
        and train_samples is not None
        and val_samples is not None
    )

    lr = _as_float(tr.get("learning_rate"), "training.learning_rate")
    wd = _as_float(tr.get("weight_decay", 0.0), "training.weight_decay")
    dropout = _as_float(tr.get("dropout_rate", 0.0), "training.dropout_rate")
    grad_clip = _as_float(tr.get("grad_clip_norm", 0.0), "training.grad_clip_norm")
    elapsed_ms = _as_float(tr.get("elapsed_ms", 0.0), "training.elapsed_ms")
    assert (
        lr is not None
        and wd is not None
        and dropout is not None
        and grad_clip is not None
        and elapsed_ms is not None
    )

    has_validation = _as_bool(tr.get("has_validation", False), "training.has_validation")
    early_stopped = _as_bool(tr.get("early_stopped", False), "training.early_stopped")
    initial_train_loss = _as_float(tr.get("initial_train_loss"), "training.initial_train_loss", allow_none=True)
    final_train_loss = _as_float(tr.get("final_train_loss"), "training.final_train_loss", allow_none=True)
    best_val_epoch = _as_int(tr.get("best_val_epoch"), "training.best_val_epoch", allow_none=True)
    best_val_loss = _as_float(tr.get("best_val_loss"), "training.best_val_loss", allow_none=True)
    final_val_loss = _as_float(tr.get("final_val_loss"), "training.final_val_loss", allow_none=True)

    eval_loss = _as_float(ev.get("loss"), "evaluation.loss", allow_none=True)
    baseline_loss = _as_float(
        ev.get("baseline_constant_loss"), "evaluation.baseline_constant_loss", allow_none=True
    )
    loss_improvement = _as_float(
        ev.get("loss_improvement_vs_baseline"),
        "evaluation.loss_improvement_vs_baseline",
        allow_none=True,
    )

    num_classes: int | None = None
    accuracy: float | None = None
    maj_acc: float | None = None
    macro_f1: float | None = None
    confusion: list[tuple[int, ...]] = []
    per_class: list[ClassMetricRow] = []

    num_outputs: int | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    per_output: list[OutputMetricRow] = []

    if task == "classification":
        num_classes = _as_int(ev.get("num_classes"), "evaluation.num_classes")
        accuracy = _as_float(ev.get("accuracy"), "evaluation.accuracy", allow_none=True)
        maj_acc = _as_float(
            ev.get("majority_baseline_accuracy"),
            "evaluation.majority_baseline_accuracy",
            allow_none=True,
        )
        macro_f1 = _as_float(ev.get("macro_f1"), "evaluation.macro_f1", allow_none=True)
        cm_raw = _as_list(ev.get("confusion_matrix", []), "evaluation.confusion_matrix")
        for r_idx, row in enumerate(cm_raw):
            row_l = _as_list(row, f"evaluation.confusion_matrix[{r_idx}]")
            vals: list[int] = []
            for c_idx, cell in enumerate(row_l):
                cval = _as_int(cell, f"evaluation.confusion_matrix[{r_idx}][{c_idx}]")
                assert cval is not None
                vals.append(cval)
            confusion.append(tuple(vals))
        pc_raw = _as_list(ev.get("per_class", []), "evaluation.per_class")
        for i, item in enumerate(pc_raw):
            d = _as_dict(item, f"evaluation.per_class[{i}]")
            ci = _as_int(d.get("class_index"), f"evaluation.per_class[{i}].class_index")
            sup = _as_int(d.get("support"), f"evaluation.per_class[{i}].support")
            freq = _as_float(d.get("actual_frequency", 0.0), f"evaluation.per_class[{i}].actual_frequency")
            mprob = _as_float(
                d.get("mean_predicted_prob", 0.0), f"evaluation.per_class[{i}].mean_predicted_prob"
            )
            prec = _as_float(d.get("precision", 0.0), f"evaluation.per_class[{i}].precision")
            rec = _as_float(d.get("recall", 0.0), f"evaluation.per_class[{i}].recall")
            f1 = _as_float(d.get("f1", 0.0), f"evaluation.per_class[{i}].f1")
            assert (
                ci is not None
                and sup is not None
                and freq is not None
                and mprob is not None
                and prec is not None
                and rec is not None
                and f1 is not None
            )
            per_class.append(
                ClassMetricRow(
                    class_index=ci,
                    support=sup,
                    actual_frequency=freq,
                    mean_predicted_prob=mprob,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                )
            )
    else:
        num_outputs = _as_int(ev.get("num_outputs"), "evaluation.num_outputs")
        mae = _as_float(ev.get("mae"), "evaluation.mae", allow_none=True)
        rmse = _as_float(ev.get("rmse"), "evaluation.rmse", allow_none=True)
        r2 = _as_float(ev.get("r2"), "evaluation.r2", allow_none=True)
        po_raw = _as_list(ev.get("per_output", []), "evaluation.per_output")
        for i, item in enumerate(po_raw):
            d = _as_dict(item, f"evaluation.per_output[{i}]")
            oi = _as_int(d.get("output_index"), f"evaluation.per_output[{i}].output_index")
            assert oi is not None
            per_output.append(
                OutputMetricRow(
                    output_index=oi,
                    mae=_as_float(d.get("mae"), f"evaluation.per_output[{i}].mae", allow_none=True),
                    rmse=_as_float(d.get("rmse"), f"evaluation.per_output[{i}].rmse", allow_none=True),
                    r2=_as_float(d.get("r2"), f"evaluation.per_output[{i}].r2", allow_none=True),
                )
            )

    diagnostics = _compute_diagnostics(
        task=task,
        has_validation=has_validation,
        epochs_completed=epochs_completed,
        final_train_loss=final_train_loss,
        best_val_loss=best_val_loss,
        final_val_loss=final_val_loss,
        eval_loss=eval_loss,
        loss_improvement=loss_improvement,
        per_class=per_class,
        r2=r2,
    )

    return TrainingReportSummary(
        source_path=source_path,
        task=task,
        eval_split=eval_split,
        input_size=input_size,
        output_size=output_size,
        parameters=parameters,
        topology=tuple(topology_sizes),
        hidden_activation=hidden_act,
        loss_name=_as_str(tr.get("loss"), "training.loss"),
        optimizer=_as_str(tr.get("optimizer"), "training.optimizer"),
        epochs_configured=epochs_configured,
        epochs_completed=epochs_completed,
        early_stopped=early_stopped,
        batch_size=batch_size,
        learning_rate=lr,
        lr_schedule=_as_str(tr.get("lr_schedule", "none"), "training.lr_schedule"),
        weight_decay=wd,
        dropout_rate=dropout,
        grad_clip_norm=grad_clip,
        early_stopping_patience=patience,
        train_samples=train_samples,
        val_samples=val_samples,
        elapsed_ms=elapsed_ms,
        has_validation=has_validation,
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        best_val_epoch=best_val_epoch,
        best_val_loss=best_val_loss,
        final_val_loss=final_val_loss,
        eval_loss=eval_loss,
        baseline_constant_loss=baseline_loss,
        loss_improvement_vs_baseline=loss_improvement,
        num_classes=num_classes,
        accuracy=accuracy,
        majority_baseline_accuracy=maj_acc,
        macro_f1=macro_f1,
        confusion_matrix=tuple(confusion),
        per_class=tuple(per_class),
        num_outputs=num_outputs,
        mae=mae,
        rmse=rmse,
        r2=r2,
        per_output=tuple(per_output),
        diagnostics=diagnostics,
    )


def load_training_report(path: str | Path) -> TrainingReportSummary:
    """Load and parse a NeuralNetwork v0.7.0 JSON report file from disk."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ReportError(f"Training report file not found: '{file_path}'")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Failed to read training report '{file_path}': {exc}") from exc
    return parse_training_report(raw, source_path=str(file_path))


def compare_reports(
    reports: Sequence[TrainingReportSummary],
) -> list[TrainingReportSummary]:
    """Rank training reports objectively from best to worst generalization vs constant baseline."""

    def _sort_key(rep: TrainingReportSummary) -> tuple[float, float, float]:
        imp = (
            rep.loss_improvement_vs_baseline
            if rep.loss_improvement_vs_baseline is not None
            else float("-inf")
        )
        if rep.task == "classification":
            secondary = rep.macro_f1 if rep.macro_f1 is not None else float("-inf")
        else:
            secondary = rep.r2 if rep.r2 is not None else float("-inf")
        neg_loss = -rep.eval_loss if rep.eval_loss is not None else float("-inf")
        return (imp, secondary, neg_loss)

    return sorted(reports, key=_sort_key, reverse=True)


def _class_label(index: int, total_classes: int | None) -> str:
    if total_classes == 3 and 0 <= index < 3:
        return _TRADING_3CLASS_NAMES[index]
    return f"class_{index}"


def format_report_detail(report: TrainingReportSummary) -> str:
    """Format a complete human-readable summary and diagnostics for one training report."""
    lines: list[str] = []
    lines.append(f"=== {report.name} ({report.task} | split={report.eval_split}) ===")
    lines.append(
        f"  Model:   {report.topology_str} ({report.hidden_activation}) | "
        f"params={report.parameters:,} | samples train={report.train_samples} val={report.val_samples}"
    )
    stop_tag = " [early-stopped]" if report.early_stopped else ""
    lines.append(
        f"  Train:   loss={report.loss_name} opt={report.optimizer} lr={report.learning_rate:g} "
        f"sched={report.lr_schedule} bs={report.batch_size} "
        f"epochs={report.epochs_completed}/{report.epochs_configured}{stop_tag}"
    )
    lines.append(
        f"  Reg:     weight_decay={report.weight_decay:g} dropout={report.dropout_rate:g} "
        f"grad_clip={report.grad_clip_norm:g} patience={report.early_stopping_patience}"
    )
    best_ep = str(report.best_val_epoch) if report.best_val_epoch is not None else "n/a"
    lines.append(
        f"  Losses:  train={_fmt_opt(report.final_train_loss, '.5f')} | "
        f"best_val={_fmt_opt(report.best_val_loss, '.5f')} (ep {best_ep}) | "
        f"eval={_fmt_opt(report.eval_loss, '.5f')} vs baseline={_fmt_opt(report.baseline_constant_loss, '.5f')} "
        f"(delta={_fmt_opt(report.loss_improvement_vs_baseline, '+.5f')})"
    )

    if report.task == "classification":
        lines.append(
            f"  Metrics: acc={_fmt_opt(report.accuracy, '.2%')} "
            f"(majority={_fmt_opt(report.majority_baseline_accuracy, '.2%')}) | "
            f"macro_f1={_fmt_opt(report.macro_f1, '.4f')}"
        )
        if report.per_class:
            lines.append("  Per-class breakdown:")
            lines.append("    class   support   actual%   pred_prob%   prec     recall   f1")
            for row in report.per_class:
                lbl = _class_label(row.class_index, report.num_classes)
                lines.append(
                    f"    {lbl:<7} {row.support:>7}   {row.actual_frequency*100:>6.1f}%   "
                    f"{row.mean_predicted_prob*100:>9.1f}%   "
                    f"{row.precision:>6.3f}   {row.recall:>6.3f}   {row.f1:>6.3f}"
                )
        if report.confusion_matrix:
            lines.append("  Confusion matrix (rows=true, cols=pred):")
            for r_idx, row in enumerate(report.confusion_matrix):
                lbl = _class_label(r_idx, report.num_classes)
                cells = " ".join(f"{c:>6}" for c in row)
                lines.append(f"    {lbl:<7} [{cells}]")
    else:
        lines.append(
            f"  Metrics: MAE={_fmt_opt(report.mae, '.5f')} | "
            f"RMSE={_fmt_opt(report.rmse, '.5f')} | R2={_fmt_opt(report.r2, '+.4f')}"
        )
        if report.per_output and len(report.per_output) >= 4 and len(report.per_output) % 4 == 0:
            horizon = len(report.per_output) // 4
            lines.append(f"  Per-step OHLC horizon summary ({horizon} bars):")
            for step in range(horizon):
                chunk = report.per_output[step * 4 : (step + 1) * 4]
                maes = [o.mae for o in chunk if o.mae is not None]
                r2s = [o.r2 for o in chunk if o.r2 is not None]
                mean_mae = sum(maes) / len(maes) if maes else None
                mean_r2 = sum(r2s) / len(r2s) if r2s else None
                close_r2 = chunk[3].r2
                lines.append(
                    f"    t+{step + 1:<2}: mean_MAE={_fmt_opt(mean_mae, '.5f')} | "
                    f"mean_R2={_fmt_opt(mean_r2, '+.4f')} | close_R2={_fmt_opt(close_r2, '+.4f')}"
                )

    status = "OK (beats constant baseline)" if not report.diagnostics else ", ".join(report.diagnostics)
    lines.append(f"  Diag:    {status}")
    return "\n".join(lines)


def format_comparison_table(reports: Sequence[TrainingReportSummary]) -> str:
    """Format a ranked side-by-side comparison table across multiple training reports."""
    if not reports:
        return "No training reports found."
    ranked = compare_reports(reports)
    header = (
        f"{'#':>2}  {'Report':<34} {'Task':<6} {'Loss':<6} {'EvalLoss':>9} "
        f"{'vsBase':>9} {'KeyMetric':>14} {'Ep':>7}  {'Diagnostics'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for rank, rep in enumerate(ranked, start=1):
        task_short = "class" if rep.task == "classification" else "regr"
        if rep.task == "classification":
            key_metric = f"F1={_fmt_opt(rep.macro_f1, '.3f')}"
        else:
            key_metric = f"R2={_fmt_opt(rep.r2, '+.3f')}"
        ep_str = f"{rep.epochs_completed}/{rep.epochs_configured}"
        diag = "OK" if not rep.diagnostics else ",".join(rep.diagnostics)
        name = rep.name if len(rep.name) <= 34 else ("..." + rep.name[-31:])
        lines.append(
            f"{rank:>2}  {name:<34} {task_short:<6} {rep.loss_name:<6} "
            f"{_fmt_opt(rep.eval_loss, '.5f'):>9} "
            f"{_fmt_opt(rep.loss_improvement_vs_baseline, '+.5f'):>9} "
            f"{key_metric:>14} {ep_str:>7}  {diag}"
        )
    return "\n".join(lines)
