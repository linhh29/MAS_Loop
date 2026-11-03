"""Poll the colocated Orchestra vLLM Prometheus endpoint during a trial.

Verl's colocated V1 AsyncLLM exposes scheduler gauges on its internal HTTP
server, but unlike the standalone OpenAI Editor/Sub-agent services it does not
reliably emit the periodic human-readable scheduler line.  This monitor waits
for ``LLMServerManager`` to print the server address, polls ``/metrics``, and
writes the same compact fields consumed by ``analyze_vllm_concurrency_run``.
"""
from __future__ import annotations

import argparse
import os
import re
import time
import urllib.request

import psutil
from pathlib import Path

ADDRESS_RE = re.compile(r"LLMServerManager:\s*\[['\"]([^'\"]+)['\"]")
HTTP_SERVER_PID_RE = re.compile(r"vLLMHttpServer pid=(\d+)")
METRIC_RE = re.compile(r"^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc|num_requests_running|num_requests_waiting))(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _candidate_logs(path: Path) -> tuple[Path, ...]:
    """Return logs that may contain Ray's colocated vLLM address.

    The outer concurrency launcher writes a small ``trial_launcher.log`` while
    the nested MAS-Loop script tees Ray/Verl output to ``train/full_train.log``.
    Production launchers may pass the latter directly, so preserve the caller's
    path first and add the nested log only when applicable.
    """

    candidates = [path]
    nested = path.parent / "train" / "full_train.log"
    if nested != path:
        candidates.append(nested)
    return tuple(candidates)


def _address(path: Path) -> str | None:
    matches: list[str] = []
    for candidate in _candidate_logs(path):
        if candidate.is_file():
            matches.extend(ADDRESS_RE.findall(candidate.read_text(errors="ignore")))
    return matches[-1] if matches else None


def _pid_addresses(path: Path) -> tuple[str, ...]:
    """Discover Ray vLLM HTTP listeners when manager stdout is delayed.

    Ray can buffer ``LLMServerManager`` even when Python flushes.  The
    ``vLLMHttpServer pid=...`` prefix appears earlier, so inspect only those
    processes' LISTEN sockets and let the caller validate ``/metrics``.
    """

    pids: list[int] = []
    for candidate in _candidate_logs(path):
        if candidate.is_file():
            pids.extend(int(value) for value in HTTP_SERVER_PID_RE.findall(candidate.read_text(errors="ignore")))
    addresses: list[str] = []
    for pid in dict.fromkeys(pids):
        try:
            connections = psutil.Process(pid).net_connections(kind="tcp")
        except (psutil.Error, OSError):
            continue
        for connection in connections:
            if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                continue
            host, port = connection.laddr.ip, connection.laddr.port
            if host in {"0.0.0.0", "::"}:
                host = "127.0.0.1"
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            addresses.append(f"{host}:{port}")
    return tuple(dict.fromkeys(addresses))


def _metrics(address: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(f"http://{address}/metrics", timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    values: dict[str, list[float]] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line.strip())
        if match:
            values.setdefault(match.group(1), []).append(float(match.group(2)))
    result: dict[str, float] = {}
    for key, entries in values.items():
        # One TP=8 replica exposes one engine. Summation also remains correct if
        # future configurations expose multiple DP engines from one endpoint.
        if key.endswith(("num_requests_running", "num_requests_waiting")):
            result[key] = sum(entries)
        else:
            result[key] = max(entries)
    return result


def monitor(launcher_log: Path, output: Path, pid: int, interval: float, request_timeout: float) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    address = None
    last_error = None
    with output.open("a", buffering=1) as stream:
        while _alive(pid):
            if address is None:
                candidates = tuple(filter(None, (_address(launcher_log),))) + _pid_addresses(launcher_log)
                for candidate in dict.fromkeys(candidates):
                    try:
                        _metrics(candidate, request_timeout)
                    except Exception:
                        continue
                    address = candidate
                    stream.write(f"# discovered={address} time={time.time():.6f}\n")
                    break
                if address is None:
                    time.sleep(interval)
                    continue
            try:
                metrics = _metrics(address, request_timeout)
                kv = metrics.get("vllm:kv_cache_usage_perc")
                if kv is None:
                    kv = metrics.get("vllm:gpu_cache_usage_perc")
                running = metrics.get("vllm:num_requests_running")
                waiting = metrics.get("vllm:num_requests_waiting")
                if kv is not None and running is not None and waiting is not None:
                    stream.write(
                        f"time={time.time():.6f} Running: {int(running)} reqs, "
                        f"Waiting: {int(waiting)} reqs, GPU KV cache usage: {kv * 100.0:.4f}%\n"
                    )
                last_error = None
            except Exception as exc:  # endpoint may be sleeping or shutting down
                message = f"{type(exc).__name__}: {exc}"
                if message != last_error:
                    stream.write(f"# poll_error={message} time={time.time():.6f}\n")
                    last_error = message
            time.sleep(interval)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=2.0)
    args = parser.parse_args()
    raise SystemExit(monitor(args.launcher_log, args.output, args.pid, args.interval, args.request_timeout))


if __name__ == "__main__":
    main()
