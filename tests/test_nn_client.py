"""Loopback HTTP integration tests for NeuralNetworkClient and AssetNNClient."""

from __future__ import annotations

import copy
import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from tradingbot.config import parse_asset_dict
from tradingbot.features import FeaturePipeline, save_feature_schema
from tradingbot.nn_client import (
    AssetNNClient,
    NNClientError,
    NeuralNetworkClient,
    SchemaMismatchError,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BTC_1H_ASSET_PATH = ROOT_DIR / "configs" / "assets" / "btc_1h.json"


def _make_mock_nn_server(
    input_size: int,
    output_vector: list[float],
    *,
    latency_us: int = 7,
) -> tuple[http.server.ThreadingHTTPServer, threading.Thread, int]:
    """Spin up a lightweight mock matching NeuralNetwork v0.7.0 server_app HTTP/JSON contract."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/info":
                self._send_json(
                    200,
                    {
                        "input_size": input_size,
                        "output_size": len(output_vector),
                        "num_layers": 2,
                        "parameters": input_size * 32 + 32 + 32 * len(output_vector) + len(output_vector),
                        "layers": [
                            {"in_size": input_size, "out_size": 32, "activation": "relu"},
                            {"in_size": 32, "out_size": len(output_vector), "activation": "linear"},
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
            vec = req.get("input")
            if not isinstance(vec, list) or len(vec) != input_size:
                self._send_json(400, {"error": "input length mismatch"})
                return
            resp: dict[str, Any] = {
                "output": list(output_vector),
                "latency_us": latency_us,
            }
            if req.get("include_neurons"):
                resp["layers"] = [{"z": [0.1], "a": [0.1]}]
            self._send_json(200, resp)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t, port


class TestNeuralNetworkClient(unittest.TestCase):
    def test_health_info_predict_and_error_handling(self) -> None:
        srv, thread, port = _make_mock_nn_server(4, [0.1, 0.2], latency_us=11)
        try:
            client = NeuralNetworkClient("127.0.0.1", port, timeout_sec=2.0)
            self.assertTrue(client.health())

            info = client.info()
            self.assertEqual(info.input_size, 4)
            self.assertEqual(info.output_size, 2)
            self.assertEqual(info.num_layers, 2)

            pred = client.predict([1.0, 2.0, 3.0, 4.0], include_neurons=True)
            self.assertEqual(pred.output, (0.1, 0.2))
            self.assertEqual(pred.latency_us, 11)
            self.assertIsNotNone(pred.layers)

            # Wrong length triggers HTTP 400 -> NNClientError
            with self.assertRaises(NNClientError):
                client.predict([1.0, 2.0])
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=2.0)

        # Unreachable server returns False for health() and raises on info()
        self.assertFalse(client.health())
        with self.assertRaises(NNClientError):
            client.info()

    def test_asset_nn_client_dual_models_and_schema_mismatch(self) -> None:
        raw_asset = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
        # btc_1h expects input_size=384, candle output=24, signal output=3
        srv_c, th_c, port_c = _make_mock_nn_server(384, [0.05] * 24, latency_us=14)
        srv_s, th_s, port_s = _make_mock_nn_server(384, [0.15, 0.75, 0.10], latency_us=9)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                schema_c_path = Path(tmp) / "candles_schema.json"
                schema_s_path = Path(tmp) / "signal_schema.json"

                raw_asset["neural_networks"]["candle_predictor"] = {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": port_c,
                    "schema_path": str(schema_c_path),
                }
                raw_asset["neural_networks"]["signal_classifier"] = {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": port_s,
                    "schema_path": str(schema_s_path),
                }
                asset = parse_asset_dict(raw_asset, base_dir=BTC_1H_ASSET_PATH.parent)
                pipeline = FeaturePipeline(asset)
                save_feature_schema(pipeline.build_schema("candles"), schema_c_path)
                save_feature_schema(pipeline.build_schema("signal"), schema_s_path)

                client = AssetNNClient(asset)
                self.assertEqual(
                    client.health_summary(),
                    {"candle_predictor": True, "signal_classifier": True},
                )
                infos = client.verify_all()
                self.assertEqual(infos["candle_predictor"].output_size, 24)
                self.assertEqual(infos["signal_classifier"].output_size, 3)

                vec = [0.0] * 384
                raw_c = client.predict_candles(vec)
                assert raw_c is not None
                self.assertEqual(len(raw_c.output), 24)

                sig = client.predict_signal(vec)
                assert sig is not None
                self.assertEqual(sig.best_action, "LONG")
                self.assertAlmostEqual(sig.long_prob, 0.75)

                # Mutating an indicator invalidates feature_hash -> SchemaMismatchError!
                raw_modified = copy.deepcopy(raw_asset)
                raw_modified["indicators"][0]["params"]["period"] = 25
                modified_asset = parse_asset_dict(raw_modified, base_dir=BTC_1H_ASSET_PATH.parent)
                with self.assertRaises(SchemaMismatchError):
                    AssetNNClient(modified_asset).verify_all()
        finally:
            for srv, th in ((srv_c, th_c), (srv_s, th_s)):
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)

    def test_timeout_validation_and_signal_candle_output_guards(self) -> None:
        with self.assertRaises(ValueError):
            NeuralNetworkClient("127.0.0.1", 8080, timeout_sec=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            NeuralNetworkClient("127.0.0.1", 8080, timeout_sec=float("nan"))

        # Test invalid output shapes and probability bounds from mock servers
        srv_bad_c, th_bad_c, port_bad_c = _make_mock_nn_server(384, [0.1] * 4)  # wrong length (expects 24)
        srv_neg_s, th_neg_s, port_neg_s = _make_mock_nn_server(384, [-0.1, 0.9, 0.2])  # negative prob
        srv_sum_s, th_sum_s, port_sum_s = _make_mock_nn_server(384, [0.5, 0.5, 0.5])  # sum=1.5
        srv_len_s, th_len_s, port_len_s = _make_mock_nn_server(384, [0.5, 0.5])  # wrong length (expects 3)
        srv_extra, th_extra, port_extra = _make_mock_nn_server(384, [0.42, 0.58])  # extra model

        servers = (
            (srv_bad_c, th_bad_c),
            (srv_neg_s, th_neg_s),
            (srv_sum_s, th_sum_s),
            (srv_len_s, th_len_s),
            (srv_extra, th_extra),
        )
        try:
            vec = [0.0] * 384
            raw_base = json.loads(BTC_1H_ASSET_PATH.read_text(encoding="utf-8"))
            raw_base["neural_networks"]["candle_predictor"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_bad_c,
                "schema_path": "",
            }
            raw_base["neural_networks"]["signal_classifier"] = {
                "enabled": True,
                "host": "127.0.0.1",
                "port": port_neg_s,
                "schema_path": "",
            }
            raw_base["neural_networks"]["extra_models"] = {
                "volatility_head": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": port_extra,
                    "schema_path": "",
                }
            }
            client1 = AssetNNClient(parse_asset_dict(raw_base, base_dir=BTC_1H_ASSET_PATH.parent))
            with self.assertRaises(NNClientError):
                client1.predict_candles(vec)
            with self.assertRaises(NNClientError):
                client1.predict_signal(vec)
            extra_res = client1.predict_extra(vec)
            self.assertIn("volatility_head", extra_res)
            self.assertEqual(extra_res["volatility_head"].output, (0.42, 0.58))

            for bad_port in (port_sum_s, port_len_s):
                raw_s = copy.deepcopy(raw_base)
                raw_s["neural_networks"]["candle_predictor"]["enabled"] = False
                raw_s["neural_networks"]["signal_classifier"]["port"] = bad_port
                raw_s["neural_networks"]["extra_models"] = {}
                c_s = AssetNNClient(parse_asset_dict(raw_s, base_dir=BTC_1H_ASSET_PATH.parent))
                with self.assertRaises(NNClientError):
                    c_s.predict_signal(vec)

            # Extra model verify_all succeeds when input_size matches
            raw_ex = copy.deepcopy(raw_base)
            raw_ex["neural_networks"]["candle_predictor"]["enabled"] = False
            raw_ex["neural_networks"]["signal_classifier"]["enabled"] = False
            infos = AssetNNClient(parse_asset_dict(raw_ex, base_dir=BTC_1H_ASSET_PATH.parent)).verify_all()
            self.assertEqual(infos["volatility_head"].output_size, 2)
        finally:
            for srv, th in servers:
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
