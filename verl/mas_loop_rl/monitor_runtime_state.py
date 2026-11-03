"""High-frequency runtime telemetry for MAS-Loop throughput trials.

The five-minute health monitor is intentionally human-facing, but vLLM phases
can finish in less than five minutes.  This sampler records the evidence needed
for concurrency tuning every few seconds: GPUs 8-15, the live phase coordinator
socket, and standalone Editor/Sub-agent Prometheus scheduler counters.  Errors
are stored as data and never terminate training.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from mas_loop_rl.vllm_arbiter import read_state

_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")
_WANTED = {
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _prometheus(url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    grouped: dict[str, list[float]] = {}
    for raw in text.splitlines():
        match = _METRIC_RE.match(raw.strip())
        if not match or match.group(1) not in _WANTED:
            continue
        grouped.setdefault(match.group(1), []).append(float(match.group(2)))
    result: dict[str, float] = {}
    for key, values in grouped.items():
        if key.endswith(("num_requests_running", "num_requests_waiting", "_total")):
            result[key] = sum(values)
        else:
            result[key] = max(values)
    return result


def _gpu_snapshot(indices: tuple[int, ...], timeout: float) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=True)
    wanted = set(indices)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 5:
            continue
        index = int(fields[0])
        if index not in wanted:
            continue
        rows.append(
            {
                "index": index,
                "utilization_gpu_pct": float(fields[1]),
                "memory_used_mib": float(fields[2]),
                "memory_total_mib": float(fields[3]),
                "power_draw_w": None if fields[4] in {"N/A", "[N/A]"} else float(fields[4]),
            }
        )
    return rows




def _dispatch_snapshot(directory: Path, stale_after_sec: float) -> dict[str, Any]:
    """Aggregate per-AgentLoop-process dispatch snapshots without touching OSSFS.

    Each worker overwrites one small JSON file under the run-specific /tmp
    control directory.  Summing the counters distinguishes three very
    different bottlenecks that GPU memory alone cannot reveal: executor queue
    saturation, HTTP/service latency, and phase-coordinator admission waiting.
    """
    now = time.time()
    result: dict[str, Any] = {
        "directory": str(directory),
        "processes": 0,
        "stale_processes": 0,
        "workers_total": 0,
        "phases": {},
    }
    if not directory.is_dir():
        result["missing"] = True
        return result
    for path in directory.glob("dispatch-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            age = max(0.0, now - float(payload.get("time", 0.0)))
            if age > stale_after_sec:
                result["stale_processes"] += 1
                continue
            result["processes"] += 1
            result["workers_total"] += int(payload.get("workers_per_process", 0) or 0)
            for phase, stats in payload.get("phases", {}).items():
                dst = result["phases"].setdefault(
                    phase,
                    {
                        "submitted": 0, "started": 0, "completed": 0, "errors": 0,
                        "active": 0, "active_peak_sum": 0, "queued": 0,
                        "queued_peak_sum": 0, "queue_wait_max_sec": 0.0,
                        "service_max_sec": 0.0,
                    },
                )
                for key in ("submitted", "started", "completed", "errors", "active", "queued"):
                    dst[key] += int(stats.get(key, 0) or 0)
                dst["active_peak_sum"] += int(stats.get("active_peak", 0) or 0)
                dst["queued_peak_sum"] += int(stats.get("queued_peak", 0) or 0)
                dst["queue_wait_max_sec"] = max(
                    dst["queue_wait_max_sec"], float(stats.get("queue_wait_sec", {}).get("max", 0.0) or 0.0)
                )
                dst["service_max_sec"] = max(
                    dst["service_max_sec"], float(stats.get("service_sec", {}).get("max", 0.0) or 0.0)
                )
        except Exception:
            continue
    return result


def _ray_lb_snapshot(directory: Path, stale_after_sec: float) -> dict[str, Any]:
    """Read process-local snapshots of the single global Ray admission gate.

    The counters are not additive because every client observes the same actor.
    We therefore expose the newest complete status and maxima across fresh
    reporters, plus the number of reporting AgentLoop processes.
    """
    now = time.time()
    fresh: list[dict[str, Any]] = []
    stale = 0
    if directory.is_dir():
        for path in directory.glob("ray-lb-*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if now - float(payload.get("time", 0.0)) > stale_after_sec:
                    stale += 1
                else:
                    fresh.append(payload)
            except Exception:
                continue
    result: dict[str, Any] = {
        "directory": str(directory),
        "reporters": len(fresh),
        "stale_reporters": stale,
    }
    if not fresh:
        result["missing"] = not directory.is_dir()
        return result
    newest = max(fresh, key=lambda value: float(value.get("time", 0.0)))
    for key in (
        "time", "pid", "servers", "total_inflight", "active_servers",
        "max_outstanding", "admission_waiters", "admission_peak",
        "acquire_wait_sec",
    ):
        if key in newest:
            result[key] = newest[key]
    result["observed_max_total_inflight"] = max(int(v.get("total_inflight", 0) or 0) for v in fresh)
    result["observed_max_waiters"] = max(int(v.get("admission_waiters", 0) or 0) for v in fresh)
    result["observed_max_acquire_wait_sec"] = max(float(v.get("acquire_wait_sec", 0.0) or 0.0) for v in fresh)
    return result


def sample(
    *, endpoints: dict[str, str], gpu_indices: tuple[int, ...], request_timeout: float,
    dispatch_dir: Path | None = None, ray_lb_dir: Path | None = None,
    dispatch_stale_after_sec: float = 90.0,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "time": time.time(), "gpus": [], "coordinator": {}, "vllm": {},
        "dispatch": {}, "ray_load_balancer": {}, "errors": {},
    }
    try:
        record["gpus"] = _gpu_snapshot(gpu_indices, timeout=max(2.0, request_timeout))
    except Exception as exc:
        record["errors"]["gpus"] = f"{type(exc).__name__}: {exc}"
    try:
        state = read_state()
        record["coordinator"] = {
            "owner": state.get("owner"),
            "status": state.get("status"),
            "active": state.get("active", state.get("count")),
            "phase_id": state.get("phase_id"),
            "phase_requests": state.get("phase_requests"),
            "waiters": state.get("waiters", {}),
            "metrics": state.get("metrics", {}),
        }
    except Exception as exc:
        record["errors"]["coordinator"] = f"{type(exc).__name__}: {exc}"
    if dispatch_dir is not None:
        record["dispatch"] = _dispatch_snapshot(dispatch_dir, dispatch_stale_after_sec)
    if ray_lb_dir is not None:
        record["ray_load_balancer"] = _ray_lb_snapshot(ray_lb_dir, dispatch_stale_after_sec)
    for role, endpoint in endpoints.items():
        try:
            record["vllm"][role] = _prometheus(endpoint, request_timeout)
        except Exception as exc:
            record["errors"][role] = f"{type(exc).__name__}: {exc}"
    return record


def monitor(
    *, output: Path, pid: int, interval: float, endpoints: dict[str, str],
    gpu_indices: tuple[int, ...], request_timeout: float, dispatch_dir: Path | None,
    ray_lb_dir: Path | None, dispatch_stale_after_sec: float,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", buffering=1) as stream:
        while _alive(pid):
            started = time.monotonic()
            stream.write(json.dumps(sample(endpoints=endpoints, gpu_indices=gpu_indices, request_timeout=request_timeout, dispatch_dir=dispatch_dir, ray_lb_dir=ray_lb_dir, dispatch_stale_after_sec=dispatch_stale_after_sec), ensure_ascii=False) + "\n")
            time.sleep(max(0.0, interval - (time.monotonic() - started)))
        stream.write(json.dumps({"time": time.time(), "event": "parent_exited", "pid": pid}) + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=2.0)
    parser.add_argument("--gpus", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--editor-url", default="http://127.0.0.1:8111")
    parser.add_argument("--subagent-url", default="http://127.0.0.1:8112")
    parser.add_argument("--dispatch-dir", type=Path)
    parser.add_argument("--ray-lb-dir", type=Path)
    parser.add_argument("--dispatch-stale-after-sec", type=float, default=90.0)
    args = parser.parse_args()
    gpu_indices = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    endpoints = {"editor": args.editor_url, "sub_agent": args.subagent_url}
    raise SystemExit(
        monitor(
            output=args.output, pid=args.pid, interval=max(1.0, args.interval),
            endpoints=endpoints, gpu_indices=gpu_indices, request_timeout=max(0.2, args.request_timeout),
            dispatch_dir=args.dispatch_dir,
            ray_lb_dir=args.ray_lb_dir,
            dispatch_stale_after_sec=max(5.0, args.dispatch_stale_after_sec),
        )
    )


if __name__ == "__main__":
    main()
