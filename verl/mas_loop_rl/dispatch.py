"""Per-AgentLoop-worker bounded dispatch pool for blocking local model calls.

The request path only updates process-local counters.  A daemon periodically
writes one atomic JSON snapshot under /tmp, so profiling cannot turn OSSFS into
the rollout bottleneck.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

_LOCK = threading.Lock()
_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_REPORTER_STARTED = False
_METRICS_LOCK = threading.Lock()
_PHASES: dict[str, dict[str, Any]] = {}
_SAMPLE_LIMIT = max(128, int(os.getenv("MAS_LOOP_DISPATCH_METRICS_SAMPLE_LIMIT", "4096")))
_PROCESS_STARTED = time.time()


def workers_per_process() -> int:
    explicit = os.getenv("MAS_LOOP_DISPATCH_WORKERS_PER_PROCESS", "").strip()
    if explicit:
        return max(1, int(explicit))
    total = max(1, int(os.getenv("MAS_LOOP_MAX_CONCURRENCY", "1024")))
    processes = max(1, int(os.getenv("MAS_LOOP_AGENT_LOOP_WORKER_PROCESSES", "8")))
    return max(4, int(math.ceil(total / processes)))


def _new_phase() -> dict[str, Any]:
    return {
        "submitted": 0,
        "started": 0,
        "completed": 0,
        "errors": 0,
        "active": 0,
        "active_peak": 0,
        "queued": 0,
        "queued_peak": 0,
        "queue_wait_sec": deque(maxlen=_SAMPLE_LIMIT),
        "service_sec": deque(maxlen=_SAMPLE_LIMIT),
    }


def _phase(name: str) -> dict[str, Any]:
    key = name or "unspecified"
    value = _PHASES.get(key)
    if value is None:
        value = _new_phase()
        _PHASES[key] = value
    return value


def _percentiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    values.sort()
    n = len(values)
    def at(q: float) -> float:
        return float(values[min(n - 1, max(0, math.ceil(q * n) - 1))])
    return {
        "samples": n,
        "mean": float(sum(values) / n),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": float(values[-1]),
    }


def metrics_snapshot() -> dict[str, Any]:
    with _METRICS_LOCK:
        phases: dict[str, Any] = {}
        for name, value in _PHASES.items():
            phases[name] = {
                key: int(value[key])
                for key in ("submitted", "started", "completed", "errors", "active", "active_peak", "queued", "queued_peak")
            }
            phases[name]["queue_wait_sec"] = _percentiles(list(value["queue_wait_sec"]))
            phases[name]["service_sec"] = _percentiles(list(value["service_sec"]))
    return {
        "time": time.time(),
        "pid": os.getpid(),
        "process_started": _PROCESS_STARTED,
        "workers_per_process": workers_per_process(),
        "phases": phases,
    }


def _write_snapshot() -> None:
    directory = os.getenv("MAS_LOOP_DISPATCH_METRICS_DIR", "").strip()
    if not directory:
        return
    path = Path(directory) / f"dispatch-{os.getpid()}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(metrics_snapshot(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # Metrics are diagnostic and must never fail a rollout.
        pass


def _reporter() -> None:
    interval = max(1.0, float(os.getenv("MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC", "30")))
    while True:
        time.sleep(interval)
        _write_snapshot()


def _start_reporter() -> None:
    global _REPORTER_STARTED
    if _REPORTER_STARTED or not os.getenv("MAS_LOOP_DISPATCH_METRICS_DIR", "").strip():
        return
    _REPORTER_STARTED = True
    threading.Thread(target=_reporter, name="mas-loop-dispatch-metrics", daemon=True).start()


def shared_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            count = workers_per_process()
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=count, thread_name_prefix="mas-loop-dispatch"
            )
            _start_reporter()
        return _EXECUTOR


async def run_blocking(fn: Callable[..., Any], *args: Any, phase: str = "", **kwargs: Any) -> Any:
    """Dispatch a blocking model call and record queue/service concurrency."""
    submitted_at = time.monotonic()
    name = phase or "unspecified"
    with _METRICS_LOCK:
        stats = _phase(name)
        stats["submitted"] += 1
        stats["queued"] += 1
        stats["queued_peak"] = max(stats["queued_peak"], stats["queued"])
    loop = asyncio.get_running_loop()

    def invoke() -> Any:
        started_at = time.monotonic()
        with _METRICS_LOCK:
            stats = _phase(name)
            stats["started"] += 1
            stats["queued"] = max(0, stats["queued"] - 1)
            stats["active"] += 1
            stats["active_peak"] = max(stats["active_peak"], stats["active"])
            stats["queue_wait_sec"].append(started_at - submitted_at)
        failed = False
        try:
            return fn(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            finished_at = time.monotonic()
            with _METRICS_LOCK:
                stats = _phase(name)
                stats["active"] = max(0, stats["active"] - 1)
                stats["completed"] += 1
                stats["errors"] += int(failed)
                stats["service_sec"].append(finished_at - started_at)

    return await loop.run_in_executor(shared_executor(), invoke)
