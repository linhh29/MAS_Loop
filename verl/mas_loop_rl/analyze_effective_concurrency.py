"""Summarize configured versus effective MAS-Loop rollout concurrency.

This deliberately reads only diagnostic files produced by one MAS-Loop run. It
separates four different quantities that are easy to confuse while tuning:
coordinator admission, process-local blocking dispatch, vLLM scheduler load,
and HTTP/model failures recorded in canonical pair shards.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

_VLLM_RE = re.compile(
    r"Avg prompt throughput:\s*([0-9.]+) tokens/s,\s*"
    r"Avg generation throughput:\s*([0-9.]+) tokens/s,\s*"
    r"Running:\s*(\d+) reqs,\s*Waiting:\s*(\d+) reqs,\s*"
    r"GPU KV cache usage:\s*([0-9.]+)%"
)
_ACTOR_RE = re.compile(
    r"Running:\s*(\d+) reqs,\s*Waiting:\s*(\d+) reqs,\s*GPU KV cache usage:\s*([0-9.]+)%"
)
_ENGINE_RE = re.compile(r"Engine\s+(\d+):")
_LOG_TIME_RE = re.compile(r"(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))])


def coordinator(control_dir: Path) -> dict[str, Any]:
    phases = []
    events = list(_jsonl(control_dir / "events.jsonl"))
    acquired: dict[int, dict[str, Any]] = {}
    for row in events:
        phase_id = int(row.get("phase_id") or 0)
        event = row.get("event")
        if event == "phase_acquired":
            acquired[phase_id] = row
        elif event == "phase_complete":
            phases.append({
                "phase_id": phase_id,
                "owner": row.get("owner", ""),
                "requests": int(row.get("requests") or 0),
                "active_peak": int(row.get("active_peak") or 0),
                "duration_sec": float(row.get("duration_sec") or 0.0),
                "sleep_sec": float(row.get("sleep_duration_sec") or 0.0),
                "generation_plus_wake_sec": max(
                    0.0,
                    float(row.get("duration_sec") or 0.0) - float(row.get("sleep_duration_sec") or 0.0),
                ),
            })
    state = _json(control_dir / "state.json")
    return {"completed_phases": phases, "last_state": state}


def dispatch(control_dir: Path) -> dict[str, Any]:
    by_phase: dict[str, dict[str, Any]] = {}
    process_count = 0
    workers = 0
    for path in sorted((control_dir / "dispatch_metrics").glob("dispatch-*.json")):
        row = _json(path)
        if not row:
            continue
        process_count += 1
        workers += int(row.get("workers_per_process") or 0)
        for name, phase in (row.get("phases") or {}).items():
            out = by_phase.setdefault(name, {
                "submitted": 0, "started": 0, "completed": 0, "errors": 0,
                "active_peak_sum": 0, "queued_peak_sum": 0,
                "queue_wait_p95_max_sec": 0.0, "service_p95_max_sec": 0.0,
            })
            for key in ("submitted", "started", "completed", "errors"):
                out[key] += int(phase.get(key) or 0)
            out["active_peak_sum"] += int(phase.get("active_peak") or 0)
            out["queued_peak_sum"] += int(phase.get("queued_peak") or 0)
            out["queue_wait_p95_max_sec"] = max(
                out["queue_wait_p95_max_sec"], float((phase.get("queue_wait_sec") or {}).get("p95") or 0.0)
            )
            out["service_p95_max_sec"] = max(
                out["service_p95_max_sec"], float((phase.get("service_sec") or {}).get("p95") or 0.0)
            )
    return {"process_count": process_count, "worker_capacity_sum": workers, "phases": by_phase}


def vllm_log(path: Path) -> dict[str, Any]:
    prompt: list[float] = []
    generation: list[float] = []
    running: list[int] = []
    waiting: list[int] = []
    kv: list[float] = []
    errors: dict[str, int] = {}
    per_rank: dict[int, dict[str, list[float]]] = {}
    timestamp_rows: dict[str, dict[int, tuple[float, float, int, int]]] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _VLLM_RE.search(line)
                if match:
                    p, g, r, w, k = float(match.group(1)), float(match.group(2)), int(match.group(3)), int(match.group(4)), float(match.group(5))
                    prompt.append(p); generation.append(g); running.append(r); waiting.append(w); kv.append(k)
                    rank_match = _ENGINE_RE.search(line)
                    if rank_match:
                        rank = int(rank_match.group(1))
                        row = per_rank.setdefault(rank, {"running": [], "waiting": [], "kv": []})
                        row["running"].append(r); row["waiting"].append(w); row["kv"].append(k)
                        time_match = _LOG_TIME_RE.search(line)
                        if time_match:
                            timestamp_rows.setdefault(time_match.group(1), {})[rank] = (p, g, r, w)
                else:
                    actor = _ACTOR_RE.search(line)
                    if actor:
                        running.append(int(actor.group(1))); waiting.append(int(actor.group(2))); kv.append(float(actor.group(3)))
                lower = line.lower()
                for name, needle in (
                    ("connect_timeout", "connection timed out"), ("timeout", "timeout"),
                    ("http_4xx", "http/1.1 4"), ("http_5xx", "http/1.1 5"),
                    ("engine_error", "engine core proc"),
                ):
                    if needle in lower:
                        errors[name] = errors.get(name, 0) + 1
    except OSError:
        pass
    global_rows = [
        (
            sum(row[0] for row in rows.values()),
            sum(row[1] for row in rows.values()),
            sum(row[2] for row in rows.values()),
            sum(row[3] for row in rows.values()),
        )
        for rows in timestamp_rows.values() if rows
    ]
    global_running = max((r for _, _, r, _ in global_rows), default=None)
    global_waiting = max((w for _, _, _, w in global_rows), default=None)
    rank_count = len(per_rank)
    running_series = [r for _, _, r, _ in global_rows] if rank_count else running
    peak = max(running_series, default=0)
    peak_index = running_series.index(peak) if running_series else -1
    tail_samples = 0
    if peak > 0 and peak_index >= 0:
        threshold = max(1, math.ceil(peak * 0.10))
        for idx in range(peak_index + 1, len(running_series)):
            if running_series[idx] < threshold and max(running_series[idx:], default=0) < threshold:
                tail_samples = len(running_series) - idx
                break
    return {
        "samples": len(running),
        "running_peak": peak,
        "running_mean": mean(running) if running else 0.0,
        "waiting_peak": global_waiting if rank_count else max(waiting, default=0),
        "kv_peak_pct": max(kv, default=0.0), "kv_mean_pct": mean(kv) if kv else 0.0,
        "prompt_tps_peak": max(prompt, default=0.0), "prompt_tps_mean": mean(prompt) if prompt else 0.0,
        "generation_tps_peak": max(generation, default=0.0), "generation_tps_mean": mean(generation) if generation else 0.0,
        "prompt_tps_global_sum_peak": max((p for p, _, _, _ in global_rows), default=max(prompt, default=0.0)),
        "prompt_tps_global_sum_mean": mean([p for p, _, _, _ in global_rows]) if global_rows else (mean(prompt) if prompt else 0.0),
        "generation_tps_global_sum_peak": max((g for _, g, _, _ in global_rows), default=max(generation, default=0.0)),
        "generation_tps_global_sum_mean": mean([g for _, g, _, _ in global_rows]) if global_rows else (mean(generation) if generation else 0.0),
        "rank_count": rank_count or 1 if running else 0,
        "running_per_rank_peak": {str(rank): max(row["running"], default=0) for rank, row in sorted(per_rank.items())},
        "waiting_per_rank_peak": {str(rank): max(row["waiting"], default=0) for rank, row in sorted(per_rank.items())},
        "running_global_sum_peak": global_running if rank_count else max(running, default=0),
        "waiting_global_sum_peak": global_waiting if rank_count else max(waiting, default=0),
        "post_peak_below_10pct_samples": tail_samples,
        "approx_tail_sec_at_10s_log_interval": tail_samples * 10,
        "error_lines": errors,
    }


def canonical_errors(log_root: Path) -> dict[str, Any]:
    counts = {
        "rows": 0, "rows_with_errors": 0, "editor_error": 0,
        "source_agent_error": 0, "updated_agent_error": 0, "connection_timeout": 0,
    }
    canonical_dirs = list(log_root.glob("**/canonical_rollouts"))
    for directory in canonical_dirs:
        for path in directory.glob("*.jsonl"):
            for row in _jsonl(path):
                counts["rows"] += 1
                text = json.dumps({
                    "error": row.get("error"),
                    "violations": row.get("violations"),
                    "source": row.get("src_trajectory", {}).get("error_message") if isinstance(row.get("src_trajectory"), dict) else "",
                    "updated": row.get("updated_trajectory", {}).get("error_message") if isinstance(row.get("updated_trajectory"), dict) else "",
                }, ensure_ascii=False).lower()
                if text not in ('{"error": "", "violations": [], "source": null, "updated": null}',):
                    if any(x in text for x in ("error", "timeout", "failed")):
                        counts["rows_with_errors"] += 1
                counts["editor_error"] += int("peer_edit_failed" in text)
                counts["source_agent_error"] += int("source_subagent" in text and "error" in text)
                counts["updated_agent_error"] += int("updated_subagent" in text and "error" in text)
                counts["connection_timeout"] += int("connection timed out" in text or "connecttimeout" in text)
    return counts


def analyze(run_root: Path, control_dir: Path | None = None) -> dict[str, Any]:
    if control_dir is None:
        candidates = sorted(Path("/tmp").glob("mas_loop_*_control_*"), key=lambda p: p.stat().st_mtime)
        control_dir = candidates[-1] if candidates else Path("/nonexistent")
    log_dirs = [run_root / "train" / "logs", run_root / "logs"]
    logs: dict[str, Any] = {}
    for directory in log_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.log")):
            if "vllm" in path.name:
                logs[path.name] = vllm_log(path)
    return {
        "run_root": str(run_root),
        "control_dir": str(control_dir),
        "coordinator": coordinator(control_dir),
        "dispatch": dispatch(control_dir),
        "vllm": logs,
        "canonical": canonical_errors(run_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--control-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = analyze(Path(args.run_root), Path(args.control_dir) if args.control_dir else None)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
