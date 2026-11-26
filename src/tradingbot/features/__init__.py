"""Feature normalization, sliding-window extraction, and schema persistence."""

from tradingbot.features.pipeline import (
    BASE_BAR_FEATURES,
    SIGNAL_TARGET_COLUMNS,
    FeaturePipeline,
    InsufficientDataError,
    WindowContext,
)
from tradingbot.features.schema import (
    SCHEMA_VERSION,
    FeatureSchema,
    SchemaError,
    compute_feature_hash,
    load_feature_schema,
    save_feature_schema,
)

__all__ = [
    "BASE_BAR_FEATURES",
    "FeaturePipeline",
    "FeatureSchema",
    "InsufficientDataError",
    "SCHEMA_VERSION",
    "SIGNAL_TARGET_COLUMNS",
    "SchemaError",
    "WindowContext",
    "compute_feature_hash",
    "load_feature_schema",
    "save_feature_schema",
]
