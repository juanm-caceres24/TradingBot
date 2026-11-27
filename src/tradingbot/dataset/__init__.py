"""NeuralNetwork (libnn v0.7.0) CSV dataset export, config generation, and report analysis."""

from tradingbot.dataset.exporter import ExportSummary, export_dataset
from tradingbot.dataset.nn_config_generator import (
    NNTrainingOptions,
    build_nn_config_dict,
    save_nn_config,
    validate_nn_config_dict,
)
from tradingbot.dataset.report_analyzer import (
    ClassMetricRow,
    OutputMetricRow,
    ReportError,
    TrainingReportSummary,
    compare_reports,
    format_comparison_table,
    format_report_detail,
    load_training_report,
    parse_training_report,
)

__all__ = [
    "ClassMetricRow",
    "ExportSummary",
    "NNTrainingOptions",
    "OutputMetricRow",
    "ReportError",
    "TrainingReportSummary",
    "build_nn_config_dict",
    "compare_reports",
    "export_dataset",
    "format_comparison_table",
    "format_report_detail",
    "load_training_report",
    "parse_training_report",
    "save_nn_config",
    "validate_nn_config_dict",
]
