"""Summarize one MAS-Loop vLLM concurrency benchmark.

Combines scheduler logs with the high-frequency ``runtime_metrics.jsonl``
telemetry.  Tuning decisions should use request/KV pressure and application
queueing, not allocated GPU memory alone.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r"GPU KV cache usage:\s*([0-9.]+)%")
KV_MEMORY_RE = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB")
KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
MAX_CONCURRENCY_RE = re.compile(r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x")
LOAD_RE = re.compile(r"Running:\s*(\d+) reqs,\s*Waiting:\s*(\d+) reqs")
ENGINE_RE = re.compile(r"Engine\s+(\d+):")
LOG_TIME_RE = re.compile(r"(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
METRIC_RE = re.compile(r"(?:^|\s-\s)(timing_s/(?:gen|step|update_actor)|perf/throughput):([0-9.eE+-]+)")


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    index = (len(xs) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - index) + xs[hi] * (index - lo)


def _stats(values: list[float], *, total: bool = False) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "p50": None, "p95": None, "max": None}
    result: dict[str, float | int | None] = {
        "samples": len(values),
        "mean": round(mean(values), 4),
        "p50": round(_percentile(values, 0.50) or 0.0, 4),
        "p95": round(_percentile(values, 0.95) or 0.0, 4),
        "max": round(max(values), 4),
    }
    if total:
        result["last"] = round(values[-1], 4)
        result["delta"] = round(max(0.0, values[-1] - values[0]), 4)
    return result


def parse_scheduler_log(path: Path, extra_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "samples": 0, "missing": True}
    texts = [path.read_text(errors="ignore")]
    texts.extend(p.read_text(errors="ignore") for p in extra_paths if p.is_file())
    text = ANSI_RE.sub("", "\n".join(texts))
    kv = [float(x) for x in KV_RE.findall(text)]
    kv_memory = [float(x) for x in KV_MEMORY_RE.findall(text)]
    kv_tokens = [int(x.replace(",", "")) for x in KV_TOKENS_RE.findall(text)]
    maximum_concurrency = [
        {"tokens_per_request": int(tokens.replace(",", "")), "concurrency_x": float(value)}
        for tokens, value in MAX_CONCURRENCY_RE.findall(text)
    ]
    loads = [(int(a), int(b)) for a, b in LOAD_RE.findall(text)]
    # vLLM DP logs one scheduler line per Engine/rank.  A raw max over all
    # lines reports the per-rank cap (128 here), not the global load (up to
    # 1024 for DP8).  Aggregate rows sharing the same log timestamp so the
    # report exposes both views without changing the legacy fields.
    rank_loads: dict[int, list[tuple[int, int]]] = defaultdict(list)
    timestamp_loads: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
    for line in text.splitlines():
        match = LOAD_RE.search(line)
        if not match:
            continue
        load = (int(match.group(1)), int(match.group(2)))
        rank_match = ENGINE_RE.search(line)
        if not rank_match:
            continue
        rank = int(rank_match.group(1))
        rank_loads[rank].append(load)
        timestamp = LOG_TIME_RE.search(line)
        if timestamp:
            timestamp_loads[timestamp.group(1)][rank] = load
    running_per_rank_peak = {str(rank): max((x for x, _ in rows), default=0) for rank, rows in sorted(rank_loads.items())}
    waiting_per_rank_peak = {str(rank): max((x for _, x in rows), default=0) for rank, rows in sorted(rank_loads.items())}
    global_rows = [
        (sum(load[0] for load in rows.values()), sum(load[1] for load in rows.values()))
        for rows in timestamp_loads.values() if rows
    ]
    # If a log has rank labels but no parseable timestamps, retain a safe
    # fallback rather than manufacturing a cross-time sum.
    global_running_peak = max((x for x, _ in global_rows), default=None)
    global_waiting_peak = max((x for _, x in global_rows), default=None)
    rank_count = len(rank_loads)
    pct_stats = _stats(kv)
    pool_gib = mean(kv_memory) if kv_memory else None
    active_gib = {
        key: round(pool_gib * float(value) / 100.0, 4)
        if pool_gib is not None and value is not None and key in {"mean", "p50", "p95", "max"}
        else None
        for key, value in pct_stats.items()
        if key in {"mean", "p50", "p95", "max"}
    }
    return {
        "path": str(path),
        "samples": len(kv),
        "kv_cache_pct": pct_stats,
        "kv_pool": {
            "rank_samples": len(kv_memory),
            "gib_per_gpu_mean": round(pool_gib, 4) if pool_gib is not None else None,
            "gib_per_gpu_min": round(min(kv_memory), 4) if kv_memory else None,
            "gib_per_gpu_max": round(max(kv_memory), 4) if kv_memory else None,
            "tokens_per_gpu_mean": round(mean(kv_tokens), 1) if kv_tokens else None,
            "target_70pct_gib_per_gpu": round(pool_gib * 0.70, 4) if pool_gib is not None else None,
        },
        "active_kv_gib_per_gpu": active_gib,
        "maximum_concurrency": maximum_concurrency[-1] if maximum_concurrency else None,
        # For DP logs these are global peaks; for legacy single-scheduler logs
        # they remain identical to the old values.
        "running_max": global_running_peak if rank_count else max((x for x, _ in loads), default=None),
        "waiting_max": global_waiting_peak if rank_count else max((x for _, x in loads), default=None),
        "rank_count": rank_count or 1 if loads else 0,
        "running_per_rank_peak": running_per_rank_peak,
        "waiting_per_rank_peak": waiting_per_rank_peak,
        "running_global_sum_peak": global_running_peak if rank_count else max((x for x, _ in loads), default=None),
        "waiting_global_sum_peak": global_waiting_peak if rank_count else max((x for _, x in loads), default=None),
    }


def parse_step_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = ANSI_RE.sub("", path.read_text(errors="ignore"))
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, value in METRIC_RE.findall(text):
        grouped[key].append(float(value))
    return {key: {"values": values, **_stats(values)} for key, values in grouped.items()}


def _read_runtime(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line in path.open(errors="ignore"):
        try:
            item = json.loads(line)
            if isinstance(item, dict) and "event" not in item:
                records.append(item)
        except (ValueError, TypeError):
            continue
    return records


def parse_runtime_metrics(path: Path) -> dict[str, Any]:
    records = _read_runtime(path)
    if not records:
        return {"path": str(path), "samples": 0, "missing": not path.is_file()}

    gpu: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    roles: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    dispatch: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    coordinator: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    timestamps: list[float] = []

    for item in records:
        timestamps.append(float(item.get("time", 0.0) or 0.0))
        for row in item.get("gpus", []):
            try:
                idx = int(row["index"])
            except (KeyError, TypeError, ValueError):
                continue
            for key in ("utilization_gpu_pct", "memory_used_mib", "power_draw_w"):
                value = row.get(key)
                if value is not None:
                    gpu[idx][key].append(float(value))
        for role, values in item.get("vllm", {}).items():
            for key, value in values.items():
                roles[role][key].append(float(value))
        for phase, values in item.get("dispatch", {}).get("phases", {}).items():
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    dispatch[phase][key].append(float(value))
        state = item.get("coordinator", {})
        owner = str(state.get("owner") or "idle")
        for key in ("active", "phase_requests"):
            value = state.get(key)
            if isinstance(value, (int, float)):
                coordinator[owner][key].append(float(value))
        waiters = state.get("waiters", {})
        if isinstance(waiters, dict):
            coordinator[owner]["waiters_total"].append(float(sum(float(v or 0) for v in waiters.values())))

    duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
    gpu_summary: dict[str, Any] = {}
    pooled: dict[str, list[float]] = defaultdict(list)
    for idx, values in sorted(gpu.items()):
        gpu_summary[str(idx)] = {key: _stats(vals) for key, vals in values.items()}
        for key, vals in values.items():
            pooled[key].extend(vals)
    gpu_summary["all"] = {key: _stats(vals) for key, vals in pooled.items()}

    role_summary: dict[str, Any] = {}
    for role, values in roles.items():
        role_summary[role] = {}
        for key, vals in values.items():
            is_counter = key.endswith("_total")
            role_summary[role][key] = _stats(vals, total=is_counter)
        # Prometheus exposes these gauges as fractions in [0, 1] despite the
        # historical ``_perc`` suffix. Publish an explicit percent field so
        # tuning reports cannot mistake 0.95 for 0.95% instead of 95%.
        kv_fraction = values.get("vllm:kv_cache_usage_perc") or values.get("vllm:gpu_cache_usage_perc", [])
        role_summary[role]["kv_cache_usage_pct"] = _stats([100.0 * x for x in kv_fraction])
        prompt = values.get("vllm:prompt_tokens_total", [])
        generation = values.get("vllm:generation_tokens_total", [])
        role_summary[role]["token_rate_per_sec"] = {
            "prompt": round(max(0.0, prompt[-1] - prompt[0]) / duration, 4) if prompt and duration > 0 else None,
            "generation": round(max(0.0, generation[-1] - generation[0]) / duration, 4) if generation and duration > 0 else None,
        }

    return {
        "path": str(path),
        "samples": len(records),
        "duration_sec": round(duration, 3),
        "gpus": gpu_summary,
        "vllm": role_summary,
        "dispatch": {phase: {key: _stats(vals, total=key in {"submitted", "started", "completed", "errors"}) for key, vals in values.items()} for phase, values in dispatch.items()},
        "coordinator": {owner: {key: _stats(vals) for key, vals in values.items()} for owner, values in coordinator.items()},
    }


def summarize(run_root: Path) -> dict[str, Any]:
    train = run_root / "train"
    candidates = (train / "full_train.log", run_root / "full_launch.log")
    full = next((path for path in candidates if path.is_file()), candidates[0])
    return {
        "run_root": str(run_root),
        "roles": {
            "orchestra": parse_scheduler_log(train / "logs" / "orchestra_vllm_metrics.log", extra_paths=(full, run_root / "trial_launcher.log")),
            "editor": parse_scheduler_log(train / "logs" / "editor-peer_vllm.log"),
            "sub_agent": parse_scheduler_log(train / "logs" / "subagent_vllm.log"),
        },
        "runtime": parse_runtime_metrics(run_root / "runtime_metrics.jsonl"),
        "step_metrics": parse_step_metrics(full),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_root)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
