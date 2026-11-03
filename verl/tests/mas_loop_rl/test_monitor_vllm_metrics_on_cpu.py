from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import psutil

from mas_loop_rl.monitor_vllm_metrics import _address, _metrics, _pid_addresses


def test_address_uses_latest_server_manager_line(tmp_path: Path):
    log = tmp_path / "launcher.log"
    log.write_text(
        "LLMServerManager: ['127.0.0.1:1111']\n"
        "other line\n"
        'LLMServerManager: ["127.0.0.1:2222"]\n'
    )
    assert _address(log) == "127.0.0.1:2222"


def test_address_falls_back_to_nested_full_train_log(tmp_path: Path):
    launcher = tmp_path / "trial_launcher.log"
    launcher.write_text("outer launcher only\n")
    nested = tmp_path / "train" / "full_train.log"
    nested.parent.mkdir()
    nested.write_text("Ray prefix LLMServerManager: ['127.0.0.1:3333']\n")
    assert _address(launcher) == "127.0.0.1:3333"


def test_pid_addresses_uses_only_vllm_http_server_listeners(tmp_path: Path, monkeypatch):
    launcher = tmp_path / "trial_launcher.log"
    launcher.write_text("outer launcher only\n")
    nested = tmp_path / "train" / "full_train.log"
    nested.parent.mkdir()
    nested.write_text("(vLLMHttpServer pid=12345) engine starting\n")

    listeners = [
        SimpleNamespace(
            status=psutil.CONN_LISTEN,
            laddr=SimpleNamespace(ip="0.0.0.0", port=4567),
        ),
        SimpleNamespace(
            status="ESTABLISHED",
            laddr=SimpleNamespace(ip="127.0.0.1", port=9999),
        ),
    ]

    class FakeProcess:
        def __init__(self, pid):
            assert pid == 12345

        def net_connections(self, kind):
            assert kind == "tcp"
            return listeners

    monkeypatch.setattr("mas_loop_rl.monitor_vllm_metrics.psutil.Process", FakeProcess)
    assert _pid_addresses(launcher) == ("127.0.0.1:4567",)


def test_metrics_parses_prometheus_scheduler_gauges():
    payload = b"""# HELP vllm:kv_cache_usage_perc KV usage
vllm:kv_cache_usage_perc{model_name=\"test\"} 0.73
vllm:num_requests_running{model_name=\"a\"} 10
vllm:num_requests_running{model_name=\"b\"} 5
vllm:num_requests_waiting{model_name=\"a\"} 3
vllm:num_requests_waiting{model_name=\"b\"} 2
"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/metrics"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        values = _metrics(f"127.0.0.1:{server.server_port}", timeout=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert values["vllm:kv_cache_usage_perc"] == 0.73
    assert values["vllm:num_requests_running"] == 15
    assert values["vllm:num_requests_waiting"] == 5
