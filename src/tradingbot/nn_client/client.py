"""HTTP/1.1 + JSON client matching NeuralNetwork (libnn v0.7.0) server_app API.

Endpoints consumed (docs/06-inference-server-api.md):
- GET  /health  -> {"status": "ok"}
- GET  /info    -> {"input_size", "output_size", "num_layers", "parameters", "layers"}
- POST /predict -> {"input": [...]} (plus "include_neurons": true when requested)
                   -> {"output": [...], "latency_us": ..., "layers": ...?}
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class NNClientError(RuntimeError):
    """Raised on network, HTTP, or payload errors communicating with server_app."""


class SchemaMismatchError(NNClientError):
    """Raised when a NeuralNetwork server's topology or feature hash does not match config."""


@dataclass(frozen=True)
class ModelInfo:
    """Parsed metadata returned by NeuralNetwork server_app GET /info."""

    input_size: int
    output_size: int
    num_layers: int
    parameters: int
    layers: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RawPrediction:
    """Raw output vector and server latency returned by POST /predict."""

    output: tuple[float, ...]
    latency_us: int
    layers: tuple[dict[str, Any], ...] | None = None


class NeuralNetworkClient:
    """Stateless stdlib HTTP client for one NeuralNetwork v0.7.0 server_app instance."""

    def __init__(self, host: str, port: int, *, timeout_sec: float = 5.0) -> None:
        if not host or not isinstance(host, str):
            raise ValueError("host must be a non-empty string")
        if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError(f"port must be in 1..65535, got {port!r}")
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(float(timeout_sec))
            or float(timeout_sec) <= 0.0
        ):
            raise ValueError(f"timeout_sec must be a finite number > 0, got {timeout_sec!r}")

        self._host = host
        self._port = port
        self._timeout_sec = float(timeout_sec)
        self._base_url = f"http://{host}:{port}"

    @property
    def base_url(self) -> str:
        return self._base_url

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        body = (
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(512).decode("utf-8", errors="replace")
            except Exception:
                pass
            raise NNClientError(
                f"NeuralNetwork server {url} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise NNClientError(
                f"Failed to connect to NeuralNetwork server {url}: {exc}"
            ) from exc

        try:
            decoded = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NNClientError(
                f"NeuralNetwork server {url} returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(decoded, dict):
            raise NNClientError(
                f"NeuralNetwork server {url} expected JSON object, got {type(decoded).__name__}"
            )
        return decoded

    def health(self) -> bool:
        """Query GET /health. Returns True iff server responds with status == 'ok'."""
        try:
            data = self._request_json("GET", "/health")
        except NNClientError:
            return False
        return data.get("status") == "ok"

    def info(self) -> ModelInfo:
        """Query GET /info and parse loaded model topology."""
        data = self._request_json("GET", "/info")
        for key in ("input_size", "output_size", "num_layers", "parameters", "layers"):
            if key not in data:
                raise NNClientError(f"GET /info missing required key '{key}'")
        layers_raw = data["layers"]
        if not isinstance(layers_raw, list):
            raise NNClientError("GET /info 'layers' must be a list")

        try:
            return ModelInfo(
                input_size=int(data["input_size"]),
                output_size=int(data["output_size"]),
                num_layers=int(data["num_layers"]),
                parameters=int(data["parameters"]),
                layers=tuple(dict(layer) for layer in layers_raw),
            )
        except (TypeError, ValueError) as exc:
            raise NNClientError(f"GET /info returned malformed numeric/layer fields: {exc}") from exc

    def predict(
        self,
        input_vector: Sequence[float],
        *,
        include_neurons: bool = False,
    ) -> RawPrediction:
        """Send POST /predict with a finite numeric feature vector."""
        clean_input = [float(v) for v in input_vector]
        if not clean_input:
            raise ValueError("input_vector must not be empty")
        if not all(math.isfinite(v) for v in clean_input):
            raise ValueError("input_vector contains non-finite values (NaN/inf)")

        payload: dict[str, Any] = {"input": clean_input}
        if include_neurons:
            payload["include_neurons"] = True

        data = self._request_json("POST", "/predict", payload)
        if "output" not in data or not isinstance(data["output"], list):
            raise NNClientError("POST /predict response missing list field 'output'")
        if "latency_us" not in data:
            raise NNClientError("POST /predict response missing field 'latency_us'")

        try:
            output = tuple(float(v) for v in data["output"])
            latency_us = int(data["latency_us"])
        except (TypeError, ValueError) as exc:
            raise NNClientError(f"POST /predict returned malformed numeric values: {exc}") from exc

        if latency_us < 0:
            raise NNClientError(f"POST /predict returned negative latency_us: {latency_us}")
        if not all(math.isfinite(v) for v in output):
            raise NNClientError("POST /predict returned non-finite output values")

        layers: tuple[dict[str, Any], ...] | None = None
        if "layers" in data and isinstance(data["layers"], list):
            layers = tuple(dict(layer) for layer in data["layers"])

        return RawPrediction(output=output, latency_us=latency_us, layers=layers)
