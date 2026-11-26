"""Strict external JSON configuration loader and validator for TradingBot.

Mirrors the strict schema discipline of NeuralNetwork (v0.6.0):
- Zero hardcoded runtime parameters.
- Unknown keys in any JSON section raise ConfigError immediately.
- Strict type validation (booleans are never accepted as integers or floats).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tradingbot.models import Timeframe, parse_iso_to_ms


class ConfigError(ValueError):
    """Raised when a configuration JSON file violates schema or value constraints."""


_VALID_SCHEDULES = frozenset({"24/7", "sessions"})
_VALID_PROVIDER_TYPES = frozenset({"generic_rest", "synthetic"})
_VALID_HTTP_METHODS = frozenset({"GET"})
_VALID_RESPONSE_FORMATS = frozenset({"array_of_arrays", "array_of_objects"})
_VALID_TIMESTAMP_UNITS = frozenset({"ms", "s"})
_VALID_ORDERS = frozenset({"asc", "desc"})
_VALID_RUNTIME_MODES = frozenset({"sync_live", "backfill", "live_only", "replay"})
_VALID_INDICATOR_TYPES = frozenset(
    {"sma", "ema", "rsi", "atr", "bollinger", "squeeze_momentum"}
)
_VALID_PLOT_MODES = frozenset({"overlay", "subchart", "none"})
_VALID_NORMALIZATIONS = frozenset({"atr_relative", "pct_relative"})


def _reject_unknown_keys(
    obj: dict[str, Any], allowed: frozenset[str], section: str
) -> None:
    unknown = sorted(set(obj.keys()) - allowed)
    if unknown:
        keys_str = ", ".join(repr(k) for k in unknown)
        raise ConfigError(f"Unknown key(s) in '{section}': {keys_str}")


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{path}' must be a JSON object, got {type(value).__name__}")
    return value


def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"'{path}' must be a JSON array, got {type(value).__name__}")
    return value


def _expect_str(value: Any, path: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"'{path}' must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ConfigError(f"'{path}' must be a non-empty string")
    return value


def _expect_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{path}' must be a boolean, got {type(value).__name__}")
    return value


def _expect_int(
    value: Any,
    path: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{path}' must be an integer, got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ConfigError(f"'{path}' must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"'{path}' must be <= {max_value}, got {value}")
    return value


def _expect_float(
    value: Any,
    path: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{path}' must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"'{path}' must be finite, got {value!r}")
    if min_value is not None and result < min_value:
        raise ConfigError(f"'{path}' must be >= {min_value}, got {result}")
    if exclusive_min is not None and result <= exclusive_min:
        raise ConfigError(f"'{path}' must be > {exclusive_min}, got {result}")
    if max_value is not None and result > max_value:
        raise ConfigError(f"'{path}' must be <= {max_value}, got {result}")
    return result


def _expect_enum(value: Any, path: str, allowed: frozenset[str]) -> str:
    text = _expect_str(value, path, allow_empty=False)
    if text not in allowed:
        raise ConfigError(
            f"'{path}' must be one of {sorted(allowed)}, got {text!r}"
        )
    return text


def _require_key(obj: dict[str, Any], key: str, section: str) -> Any:
    if key not in obj:
        raise ConfigError(f"Missing required key '{key}' in '{section}'")
    return obj[key]


@dataclass(frozen=True)
class MarketConfig:
    symbol: str
    timeframe: str
    market_schedule: str = "24/7"

    @property
    def parsed_timeframe(self) -> Timeframe:
        return Timeframe.parse(self.timeframe)


@dataclass(frozen=True)
class RestParamsMap:
    symbol: str
    interval: str
    start_time: str
    end_time: str
    limit: str


@dataclass(frozen=True)
class RestFieldsMap:
    open_time: int | str
    open: int | str
    high: int | str
    low: int | str
    close: int | str
    volume: int | str
    close_time: int | str | None = None


@dataclass(frozen=True)
class RestResponseMapping:
    data_path: str
    format: str
    fields: RestFieldsMap
    timestamp_unit: str
    order: str


@dataclass(frozen=True)
class SyntheticProviderParams:
    seed: int
    base_price: float
    volatility: float
    drift: float
    base_volume: float


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    type: str
    max_candles_per_request: int
    base_url: str = ""
    endpoint: str = ""
    http_method: str = "GET"
    timeout_sec: float = 10.0
    rate_limit_ms: int = 200
    params_map: RestParamsMap | None = None
    interval_map: dict[str, str] = field(default_factory=dict)
    response_mapping: RestResponseMapping | None = None
    headers: dict[str, str] = field(default_factory=dict)
    api_key_env: str = ""
    api_key_header: str = ""
    synthetic: SyntheticProviderParams | None = None


@dataclass(frozen=True)
class StorageConfig:
    db_path: str
    wal_mode: bool = True
    busy_timeout_ms: int = 5000


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    poll_interval_sec: float
    initial_history_from: str = ""
    fill_internal_gaps: bool = True
    persist_to_db: bool = True
    predict_on_open_candle: bool = False

    def initial_history_from_ms(self) -> int | None:
        if not self.initial_history_from:
            return None
        return parse_iso_to_ms(self.initial_history_from)


@dataclass(frozen=True)
class IndicatorConfig:
    id: str
    type: str
    params: dict[str, Any]
    use_in_nn: bool
    plot: str


@dataclass(frozen=True)
class TripleBarrierConfig:
    take_profit_pct: float
    stop_loss_pct: float


@dataclass(frozen=True)
class FeaturesConfig:
    input_window: int
    forecast_horizon: int
    stride: int
    normalization: str
    atr_period: int
    triple_barrier: TripleBarrierConfig


@dataclass(frozen=True)
class ModelEndpointConfig:
    enabled: bool
    host: str
    port: int
    schema_path: str = ""


@dataclass(frozen=True)
class NeuralNetworksConfig:
    timeout_sec: float
    verify_schema_on_start: bool
    candle_predictor: ModelEndpointConfig
    signal_classifier: ModelEndpointConfig


@dataclass(frozen=True)
class AnalysisConfig:
    min_signal_confidence: float
    min_risk_reward: float
    min_expected_move_pct: float


@dataclass(frozen=True)
class WebConfig:
    enabled: bool
    host: str
    port: int
    visible_candles: int
    title: str = "TradingBot Dashboard"


@dataclass(frozen=True)
class AppConfig:
    market: MarketConfig
    provider: ProviderConfig
    storage: StorageConfig
    runtime: RuntimeConfig
    indicators: tuple[IndicatorConfig, ...]
    features: FeaturesConfig
    neural_networks: NeuralNetworksConfig
    analysis: AnalysisConfig
    web: WebConfig


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{label} file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {label} '{path}': {exc}") from exc
    return _expect_dict(raw, str(path))


def _parse_field_selector(value: Any, path: str, fmt: str) -> int | str:
    if fmt == "array_of_arrays":
        return _expect_int(value, path, min_value=0)
    return _expect_str(value, path, allow_empty=False)


def parse_provider_dict(
    raw: dict[str, Any], *, base_dir: Path | None = None, section: str = "provider"
) -> ProviderConfig:
    """Strictly parse and validate a provider JSON object (preset or inline)."""
    obj = _expect_dict(raw, section)

    # Resolve optional preset_path and layer any inline overrides on top.
    if "preset_path" in obj:
        preset_rel = _expect_str(obj["preset_path"], f"{section}.preset_path", allow_empty=False)
        preset_file = Path(preset_rel)
        if not preset_file.is_absolute():
            candidate = (base_dir / preset_file) if base_dir is not None else preset_file
            preset_file = candidate if candidate.is_file() else preset_file
        preset_data = _load_json_file(preset_file, "Provider preset")
        if "preset_path" in preset_data:
            raise ConfigError(
                f"Nested 'preset_path' is not allowed inside preset '{preset_file}'"
            )
        merged = dict(preset_data)
        for key, value in obj.items():
            if key != "preset_path":
                merged[key] = value
        obj = merged

    provider_type = _expect_enum(
        _require_key(obj, "type", section),
        f"{section}.type",
        _VALID_PROVIDER_TYPES,
    )
    name = _expect_str(_require_key(obj, "name", section), f"{section}.name", allow_empty=False)
    max_candles = _expect_int(
        _require_key(obj, "max_candles_per_request", section),
        f"{section}.max_candles_per_request",
        min_value=1,
    )

    if provider_type == "generic_rest":
        allowed = frozenset(
            {
                "name",
                "type",
                "base_url",
                "endpoint",
                "http_method",
                "timeout_sec",
                "rate_limit_ms",
                "max_candles_per_request",
                "params_map",
                "interval_map",
                "response_mapping",
                "headers",
                "api_key_env",
                "api_key_header",
            }
        )
        _reject_unknown_keys(obj, allowed, section)

        base_url = _expect_str(
            _require_key(obj, "base_url", section),
            f"{section}.base_url",
            allow_empty=False,
        )
        endpoint = _expect_str(
            _require_key(obj, "endpoint", section),
            f"{section}.endpoint",
            allow_empty=False,
        )
        http_method = _expect_enum(
            obj.get("http_method", "GET"),
            f"{section}.http_method",
            _VALID_HTTP_METHODS,
        )
        timeout_sec = _expect_float(
            obj.get("timeout_sec", 10.0),
            f"{section}.timeout_sec",
            exclusive_min=0.0,
        )
        rate_limit_ms = _expect_int(
            obj.get("rate_limit_ms", 200),
            f"{section}.rate_limit_ms",
            min_value=0,
        )

        # params_map
        pm_raw = _expect_dict(
            _require_key(obj, "params_map", section), f"{section}.params_map"
        )
        _reject_unknown_keys(
            pm_raw,
            frozenset({"symbol", "interval", "start_time", "end_time", "limit"}),
            f"{section}.params_map",
        )
        params_map = RestParamsMap(
            symbol=_expect_str(_require_key(pm_raw, "symbol", f"{section}.params_map"), f"{section}.params_map.symbol", allow_empty=False),
            interval=_expect_str(_require_key(pm_raw, "interval", f"{section}.params_map"), f"{section}.params_map.interval", allow_empty=False),
            start_time=_expect_str(_require_key(pm_raw, "start_time", f"{section}.params_map"), f"{section}.params_map.start_time", allow_empty=False),
            end_time=_expect_str(_require_key(pm_raw, "end_time", f"{section}.params_map"), f"{section}.params_map.end_time", allow_empty=False),
            limit=_expect_str(_require_key(pm_raw, "limit", f"{section}.params_map"), f"{section}.params_map.limit", allow_empty=False),
        )

        # interval_map
        im_raw = _expect_dict(
            _require_key(obj, "interval_map", section), f"{section}.interval_map"
        )
        if not im_raw:
            raise ConfigError(f"'{section}.interval_map' must not be empty")
        interval_map: dict[str, str] = {}
        for k, v in im_raw.items():
            try:
                canonical = Timeframe.parse(k).label
            except ValueError as exc:
                raise ConfigError(f"Invalid timeframe key in '{section}.interval_map': {exc}") from exc
            interval_map[canonical] = _expect_str(
                v, f"{section}.interval_map.{k}", allow_empty=False
            )

        # response_mapping
        rm_raw = _expect_dict(
            _require_key(obj, "response_mapping", section),
            f"{section}.response_mapping",
        )
        _reject_unknown_keys(
            rm_raw,
            frozenset({"data_path", "format", "fields", "timestamp_unit", "order"}),
            f"{section}.response_mapping",
        )
        fmt = _expect_enum(
            _require_key(rm_raw, "format", f"{section}.response_mapping"),
            f"{section}.response_mapping.format",
            _VALID_RESPONSE_FORMATS,
        )
        data_path = _expect_str(
            rm_raw.get("data_path", ""),
            f"{section}.response_mapping.data_path",
            allow_empty=True,
        )
        timestamp_unit = _expect_enum(
            _require_key(rm_raw, "timestamp_unit", f"{section}.response_mapping"),
            f"{section}.response_mapping.timestamp_unit",
            _VALID_TIMESTAMP_UNITS,
        )
        order = _expect_enum(
            rm_raw.get("order", "asc"),
            f"{section}.response_mapping.order",
            _VALID_ORDERS,
        )

        fields_raw = _expect_dict(
            _require_key(rm_raw, "fields", f"{section}.response_mapping"),
            f"{section}.response_mapping.fields",
        )
        _reject_unknown_keys(
            fields_raw,
            frozenset({"open_time", "open", "high", "low", "close", "volume", "close_time"}),
            f"{section}.response_mapping.fields",
        )
        close_time_sel: int | str | None = None
        if "close_time" in fields_raw and fields_raw["close_time"] is not None:
            close_time_sel = _parse_field_selector(
                fields_raw["close_time"],
                f"{section}.response_mapping.fields.close_time",
                fmt,
            )

        fields_map = RestFieldsMap(
            open_time=_parse_field_selector(_require_key(fields_raw, "open_time", f"{section}.response_mapping.fields"), f"{section}.response_mapping.fields.open_time", fmt),
            open=_parse_field_selector(_require_key(fields_raw, "open", f"{section}.response_mapping.fields"), f"{section}.response_mapping.fields.open", fmt),
            high=_parse_field_selector(_require_key(fields_raw, "high", f"{section}.response_mapping.fields"), f"{section}.response_mapping.fields.high", fmt),
            low=_parse_field_selector(_require_key(fields_raw, "low", f"{section}.response_mapping.fields"), f"{section}.response_mapping.fields.low", fmt),
            close=_parse_field_selector(_require_key(fields_raw, "close", f"{section}.response_mapping.fields"), f"{section}.response_mapping.fields.close", fmt),
            volume=_parse_field_selector(_require_key(fields_raw, "volume", f"{section}.response_mapping.fields"), f"{section}.response_mapping.fields.volume", fmt),
            close_time=close_time_sel,
        )

        headers_raw = _expect_dict(obj.get("headers", {}), f"{section}.headers")
        headers = {
            _expect_str(k, f"{section}.headers key", allow_empty=False): _expect_str(
                v, f"{section}.headers.{k}", allow_empty=True
            )
            for k, v in headers_raw.items()
        }
        api_key_env = _expect_str(obj.get("api_key_env", ""), f"{section}.api_key_env")
        api_key_header = _expect_str(obj.get("api_key_header", ""), f"{section}.api_key_header")
        if bool(api_key_env) != bool(api_key_header):
            raise ConfigError(
                f"'{section}.api_key_env' and '{section}.api_key_header' must both be set or both empty"
            )

        return ProviderConfig(
            name=name,
            type=provider_type,
            max_candles_per_request=max_candles,
            base_url=base_url,
            endpoint=endpoint,
            http_method=http_method,
            timeout_sec=timeout_sec,
            rate_limit_ms=rate_limit_ms,
            params_map=params_map,
            interval_map=interval_map,
            response_mapping=RestResponseMapping(
                data_path=data_path,
                format=fmt,
                fields=fields_map,
                timestamp_unit=timestamp_unit,
                order=order,
            ),
            headers=headers,
            api_key_env=api_key_env,
            api_key_header=api_key_header,
        )

    # provider_type == "synthetic"
    allowed_syn = frozenset({"name", "type", "max_candles_per_request", "synthetic"})
    _reject_unknown_keys(obj, allowed_syn, section)
    syn_raw = _expect_dict(_require_key(obj, "synthetic", section), f"{section}.synthetic")
    _reject_unknown_keys(
        syn_raw,
        frozenset({"seed", "base_price", "volatility", "drift", "base_volume"}),
        f"{section}.synthetic",
    )
    syn_params = SyntheticProviderParams(
        seed=_expect_int(_require_key(syn_raw, "seed", f"{section}.synthetic"), f"{section}.synthetic.seed", min_value=0),
        base_price=_expect_float(_require_key(syn_raw, "base_price", f"{section}.synthetic"), f"{section}.synthetic.base_price", exclusive_min=0.0),
        volatility=_expect_float(_require_key(syn_raw, "volatility", f"{section}.synthetic"), f"{section}.synthetic.volatility", min_value=0.0),
        drift=_expect_float(_require_key(syn_raw, "drift", f"{section}.synthetic"), f"{section}.synthetic.drift"),
        base_volume=_expect_float(_require_key(syn_raw, "base_volume", f"{section}.synthetic"), f"{section}.synthetic.base_volume", exclusive_min=0.0),
    )
    return ProviderConfig(
        name=name,
        type=provider_type,
        max_candles_per_request=max_candles,
        synthetic=syn_params,
    )


def load_provider_preset(path: str | Path) -> ProviderConfig:
    """Load and validate a standalone provider preset JSON file."""
    file_path = Path(path)
    raw = _load_json_file(file_path, "Provider preset")
    return parse_provider_dict(raw, base_dir=file_path.parent)


def _parse_model_endpoint(raw: Any, path: str) -> ModelEndpointConfig:
    obj = _expect_dict(raw, path)
    _reject_unknown_keys(obj, frozenset({"enabled", "host", "port", "schema_path"}), path)
    return ModelEndpointConfig(
        enabled=_expect_bool(_require_key(obj, "enabled", path), f"{path}.enabled"),
        host=_expect_str(_require_key(obj, "host", path), f"{path}.host", allow_empty=False),
        port=_expect_int(_require_key(obj, "port", path), f"{path}.port", min_value=1, max_value=65535),
        schema_path=_expect_str(obj.get("schema_path", ""), f"{path}.schema_path"),
    )


def parse_config_dict(
    raw: dict[str, Any], *, base_dir: Path | None = None
) -> AppConfig:
    """Strictly parse and validate a full TradingBot configuration dictionary."""
    root = _expect_dict(raw, "root")
    _reject_unknown_keys(
        root,
        frozenset(
            {
                "market",
                "provider",
                "storage",
                "runtime",
                "indicators",
                "features",
                "neural_networks",
                "analysis",
                "web",
            }
        ),
        "root",
    )

    # market
    m_raw = _expect_dict(_require_key(root, "market", "root"), "market")
    _reject_unknown_keys(m_raw, frozenset({"symbol", "timeframe", "market_schedule"}), "market")
    symbol = _expect_str(_require_key(m_raw, "symbol", "market"), "market.symbol", allow_empty=False)
    tf_raw = _expect_str(_require_key(m_raw, "timeframe", "market"), "market.timeframe", allow_empty=False)
    try:
        timeframe = Timeframe.parse(tf_raw).label
    except ValueError as exc:
        raise ConfigError(f"Invalid 'market.timeframe': {exc}") from exc
    market_schedule = _expect_enum(
        m_raw.get("market_schedule", "24/7"),
        "market.market_schedule",
        _VALID_SCHEDULES,
    )
    market = MarketConfig(symbol=symbol, timeframe=timeframe, market_schedule=market_schedule)

    # provider
    provider = parse_provider_dict(
        _require_key(root, "provider", "root"),
        base_dir=base_dir,
        section="provider",
    )
    if provider.type == "generic_rest" and timeframe not in provider.interval_map:
        raise ConfigError(
            f"Timeframe '{timeframe}' is not mapped in provider '{provider.name}' interval_map"
        )

    # storage
    s_raw = _expect_dict(_require_key(root, "storage", "root"), "storage")
    _reject_unknown_keys(s_raw, frozenset({"db_path", "wal_mode", "busy_timeout_ms"}), "storage")
    storage = StorageConfig(
        db_path=_expect_str(_require_key(s_raw, "db_path", "storage"), "storage.db_path", allow_empty=False),
        wal_mode=_expect_bool(s_raw.get("wal_mode", True), "storage.wal_mode"),
        busy_timeout_ms=_expect_int(s_raw.get("busy_timeout_ms", 5000), "storage.busy_timeout_ms", min_value=0),
    )

    # runtime
    r_raw = _expect_dict(_require_key(root, "runtime", "root"), "runtime")
    _reject_unknown_keys(
        r_raw,
        frozenset(
            {
                "mode",
                "poll_interval_sec",
                "initial_history_from",
                "fill_internal_gaps",
                "persist_to_db",
                "predict_on_open_candle",
            }
        ),
        "runtime",
    )
    init_from = _expect_str(r_raw.get("initial_history_from", ""), "runtime.initial_history_from")
    if init_from:
        try:
            parse_iso_to_ms(init_from)
        except ValueError as exc:
            raise ConfigError(f"Invalid 'runtime.initial_history_from': {exc}") from exc

    runtime = RuntimeConfig(
        mode=_expect_enum(_require_key(r_raw, "mode", "runtime"), "runtime.mode", _VALID_RUNTIME_MODES),
        poll_interval_sec=_expect_float(
            _require_key(r_raw, "poll_interval_sec", "runtime"),
            "runtime.poll_interval_sec",
            exclusive_min=0.0,
        ),
        initial_history_from=init_from,
        fill_internal_gaps=_expect_bool(r_raw.get("fill_internal_gaps", True), "runtime.fill_internal_gaps"),
        persist_to_db=_expect_bool(r_raw.get("persist_to_db", True), "runtime.persist_to_db"),
        predict_on_open_candle=_expect_bool(
            r_raw.get("predict_on_open_candle", False), "runtime.predict_on_open_candle"
        ),
    )

    # indicators
    ind_list_raw = _expect_list(_require_key(root, "indicators", "root"), "indicators")
    indicators: list[IndicatorConfig] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(ind_list_raw):
        sec = f"indicators[{idx}]"
        item_obj = _expect_dict(item, sec)
        _reject_unknown_keys(item_obj, frozenset({"id", "type", "params", "use_in_nn", "plot"}), sec)
        ind_id = _expect_str(_require_key(item_obj, "id", sec), f"{sec}.id", allow_empty=False)
        if ind_id in seen_ids:
            raise ConfigError(f"Duplicate indicator id '{ind_id}' in 'indicators'")
        seen_ids.add(ind_id)
        ind_type = _expect_enum(
            _require_key(item_obj, "type", sec), f"{sec}.type", _VALID_INDICATOR_TYPES
        )
        ind_params = _expect_dict(item_obj.get("params", {}), f"{sec}.params")
        indicators.append(
            IndicatorConfig(
                id=ind_id,
                type=ind_type,
                params=dict(ind_params),
                use_in_nn=_expect_bool(_require_key(item_obj, "use_in_nn", sec), f"{sec}.use_in_nn"),
                plot=_expect_enum(_require_key(item_obj, "plot", sec), f"{sec}.plot", _VALID_PLOT_MODES),
            )
        )

    # features
    f_raw = _expect_dict(_require_key(root, "features", "root"), "features")
    _reject_unknown_keys(
        f_raw,
        frozenset(
            {
                "input_window",
                "forecast_horizon",
                "stride",
                "normalization",
                "atr_period",
                "triple_barrier",
            }
        ),
        "features",
    )
    tb_raw = _expect_dict(_require_key(f_raw, "triple_barrier", "features"), "features.triple_barrier")
    _reject_unknown_keys(tb_raw, frozenset({"take_profit_pct", "stop_loss_pct"}), "features.triple_barrier")
    triple_barrier = TripleBarrierConfig(
        take_profit_pct=_expect_float(
            _require_key(tb_raw, "take_profit_pct", "features.triple_barrier"),
            "features.triple_barrier.take_profit_pct",
            exclusive_min=0.0,
        ),
        stop_loss_pct=_expect_float(
            _require_key(tb_raw, "stop_loss_pct", "features.triple_barrier"),
            "features.triple_barrier.stop_loss_pct",
            exclusive_min=0.0,
        ),
    )
    features = FeaturesConfig(
        input_window=_expect_int(_require_key(f_raw, "input_window", "features"), "features.input_window", min_value=1),
        forecast_horizon=_expect_int(_require_key(f_raw, "forecast_horizon", "features"), "features.forecast_horizon", min_value=1),
        stride=_expect_int(f_raw.get("stride", 1), "features.stride", min_value=1),
        normalization=_expect_enum(_require_key(f_raw, "normalization", "features"), "features.normalization", _VALID_NORMALIZATIONS),
        atr_period=_expect_int(f_raw.get("atr_period", 14), "features.atr_period", min_value=1),
        triple_barrier=triple_barrier,
    )

    # neural_networks
    nn_raw = _expect_dict(_require_key(root, "neural_networks", "root"), "neural_networks")
    _reject_unknown_keys(
        nn_raw,
        frozenset(
            {
                "timeout_sec",
                "verify_schema_on_start",
                "candle_predictor",
                "signal_classifier",
            }
        ),
        "neural_networks",
    )
    neural_networks = NeuralNetworksConfig(
        timeout_sec=_expect_float(nn_raw.get("timeout_sec", 5.0), "neural_networks.timeout_sec", exclusive_min=0.0),
        verify_schema_on_start=_expect_bool(
            nn_raw.get("verify_schema_on_start", True),
            "neural_networks.verify_schema_on_start",
        ),
        candle_predictor=_parse_model_endpoint(
            _require_key(nn_raw, "candle_predictor", "neural_networks"),
            "neural_networks.candle_predictor",
        ),
        signal_classifier=_parse_model_endpoint(
            _require_key(nn_raw, "signal_classifier", "neural_networks"),
            "neural_networks.signal_classifier",
        ),
    )

    # analysis
    a_raw = _expect_dict(_require_key(root, "analysis", "root"), "analysis")
    _reject_unknown_keys(
        a_raw,
        frozenset({"min_signal_confidence", "min_risk_reward", "min_expected_move_pct"}),
        "analysis",
    )
    analysis = AnalysisConfig(
        min_signal_confidence=_expect_float(
            _require_key(a_raw, "min_signal_confidence", "analysis"),
            "analysis.min_signal_confidence",
            exclusive_min=0.0,
            max_value=1.0,
        ),
        min_risk_reward=_expect_float(
            _require_key(a_raw, "min_risk_reward", "analysis"),
            "analysis.min_risk_reward",
            min_value=0.0,
        ),
        min_expected_move_pct=_expect_float(
            _require_key(a_raw, "min_expected_move_pct", "analysis"),
            "analysis.min_expected_move_pct",
            min_value=0.0,
        ),
    )

    # web
    w_raw = _expect_dict(_require_key(root, "web", "root"), "web")
    _reject_unknown_keys(
        w_raw,
        frozenset({"enabled", "host", "port", "visible_candles", "title"}),
        "web",
    )
    web = WebConfig(
        enabled=_expect_bool(_require_key(w_raw, "enabled", "web"), "web.enabled"),
        host=_expect_str(_require_key(w_raw, "host", "web"), "web.host", allow_empty=False),
        port=_expect_int(_require_key(w_raw, "port", "web"), "web.port", min_value=0, max_value=65535),
        visible_candles=_expect_int(
            _require_key(w_raw, "visible_candles", "web"),
            "web.visible_candles",
            min_value=10,
        ),
        title=_expect_str(w_raw.get("title", "TradingBot Dashboard"), "web.title", allow_empty=False),
    )

    return AppConfig(
        market=market,
        provider=provider,
        storage=storage,
        runtime=runtime,
        indicators=tuple(indicators),
        features=features,
        neural_networks=neural_networks,
        analysis=analysis,
        web=web,
    )


def load_config(path: str | Path) -> AppConfig:
    """Load and strictly validate a TradingBot JSON configuration file."""
    file_path = Path(path)
    raw = _load_json_file(file_path, "Configuration")
    return parse_config_dict(raw, base_dir=file_path.parent)
