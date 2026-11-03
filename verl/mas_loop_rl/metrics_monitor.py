#!/usr/bin/env python3
"""Extract VERL console metrics and continuously render MAS-Loop training curves.

This is deliberately a sidecar: it never imports the trainer or touches model/GPU
state. It can therefore be attached to an already-running experiment safely.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PHASE_RE = re.compile(r"trainer\.experiment_name=(?:'|\")?([^\s'\"]+)")
PHASE_CONFIG_RE = re.compile(r"['\"]experiment_name['\"]\s*:\s*['\"]([^'\"]+)")
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
STOP = False


def _stop(*_: Any) -> None:
    global STOP
    STOP = True


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase = "unknown"
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI_RE.sub("", raw).strip()
            match = PHASE_RE.search(line) or PHASE_CONFIG_RE.search(line)
            if match:
                phase = match.group(1).rstrip(",")
            step_match = STEP_RE.search(line)
            if not step_match or "training/global_step:" not in line:
                continue
            metric_text = line[step_match.end():]
            row: dict[str, Any] = {"phase": phase, "step": int(step_match.group(1))}
            for item in metric_text.split(" - "):
                if ":" not in item:
                    continue
                key, value = item.rsplit(":", 1)
                key, value = key.strip(), value.strip()
                if not key or not NUMBER_RE.match(value):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                row[key] = number if math.isfinite(number) else None
            # The explicit metric is authoritative if present.
            if row.get("training/global_step") is not None:
                row["step"] = int(row["training/global_step"])
            rows.append(row)
    # Keep the last occurrence if a partially duplicated log was resumed.
    dedup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        dedup[(str(row["phase"]), int(row["step"]))] = row
    return list(dedup.values())


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_structured(rows: list[dict[str, Any]], out_dir: Path) -> None:
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_text(out_dir / "training_metrics.jsonl", jsonl)
    keys = ["phase", "step"] + sorted({k for row in rows for k in row if k not in {"phase", "step"}})
    tmp = out_dir / "training_metrics.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out_dir / "training_metrics.csv")
    latest = {}
    for row in rows:
        latest[str(row["phase"])] = row
    atomic_text(out_dir / "latest_metrics.json", json.dumps(latest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def plot_group(rows: list[dict[str, Any]], out: Path, title: str, metrics: list[str], logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    plotted = 0
    phases = list(dict.fromkeys(str(r["phase"]) for r in rows))
    for phase in phases:
        phase_rows = sorted((r for r in rows if str(r["phase"]) == phase), key=lambda r: int(r["step"]))
        for metric in metrics:
            points = [(int(r["step"]), r.get(metric)) for r in phase_rows if r.get(metric) is not None]
            if not points:
                continue
            xs, ys = zip(*points)
            label = f"{phase}: {metric}" if len(phases) > 1 else metric
            ax.plot(xs, ys, marker="o" if len(xs) < 20 else None, linewidth=1.6, markersize=3.5, label=label)
            plotted += 1
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.grid(True, alpha=0.25)
    if logy:
        ax.set_yscale("symlog", linthresh=1e-6)
    if plotted:
        ax.legend(fontsize=7.5, ncol=2)
    else:
        ax.text(0.5, 0.5, "No completed-step metrics yet", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    tmp = out.with_name(out.stem + ".tmp.png")
    fig.savefig(tmp)
    plt.close(fig)
    os.replace(tmp, out)


def render(rows: list[dict[str, Any]], out_dir: Path) -> None:
    plot_group(rows, out_dir / "01_actor_losses.png", "Actor / GRPO optimization", [
        "actor/loss", "actor/pg_loss", "actor/kl_loss", "actor/entropy", "actor/grad_norm"
    ])
    plot_group(rows, out_dir / "02_kl_diagnostics.png", "KL diagnostics", [
        "actor/ppo_kl", "rollout_corr/kl", "rollout_corr/k3_kl", "actor/kl_coef"
    ], logy=True)
    plot_group(rows, out_dir / "03_reward_score.png", "Reward, score and advantage", [
        "critic/score/mean", "critic/rewards/mean", "critic/advantages/mean",
        "critic/score/max", "critic/score/min"
    ])
    plot_group(rows, out_dir / "04_timing.png", "Step timing (seconds)", [
        "timing_s/step", "timing_s/gen", "timing_s/update_actor", "timing_s/old_log_prob", "timing_s/ref"
    ])
    plot_group(rows, out_dir / "05_throughput_lengths.png", "Throughput and sequence lengths", [
        "perf/throughput", "response_length/mean", "prompt_length/mean"
    ])
    plot_group(rows, out_dir / "06_memory.png", "Actor memory (GiB)", [
        "actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb", "actor/perf/cpu_memory_used_gb"
    ])
    plot_group(rows, out_dir / "07_policy_stability.png", "Policy stability", [
        "actor/pg_clipfrac", "training/rollout_probs_diff_mean", "training/rollout_probs_diff_max",
        "training/rollout_actor_probs_pearson_corr"
    ])


def write_readme(out_dir: Path, log_path: Path, rows: list[dict[str, Any]]) -> None:
    phases: dict[str, int] = {}
    for row in rows:
        phases[str(row["phase"])] = max(phases.get(str(row["phase"]), 0), int(row["step"]))
    text = f"""# MAS-Loop training metrics\n\nSource log: `{log_path}`\n\nCompleted steps by phase: `{json.dumps(phases, ensure_ascii=False)}`\n\n- `training_metrics.csv`: one structured row per completed optimizer step.\n- `training_metrics.jsonl`: lossless JSONL representation.\n- `latest_metrics.json`: latest metrics for each Orchestra/Editor phase.\n- `01_actor_losses.png`: actor total/policy/KL loss, entropy and gradient norm.\n- `02_kl_diagnostics.png`: PPO KL and rollout/training-distribution KL.\n- `03_reward_score.png`: GRPO reward/score/advantage.\n- `04_timing.png`: rollout generation and update timing.\n- `05_throughput_lengths.png`: throughput and token lengths.\n- `06_memory.png`: actor CPU/GPU memory.\n- `07_policy_stability.png`: clipping and rollout-policy consistency.\n\nThese files are refreshed by the sidecar after each polling interval. A metric row appears only after a full optimizer step finishes.\n"""
    atomic_text(out_dir / "README.md", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    previous_signature = None
    consecutive_io_errors = 0
    while not STOP:
        try:
            rows = parse_log(args.log_file)
            signature = tuple((r["phase"], r["step"], len(r)) for r in rows)
            if signature != previous_signature:
                write_structured(rows, args.output_dir)
                render(rows, args.output_dir)
                write_readme(args.output_dir, args.log_file, rows)
                atomic_text(args.output_dir / "monitor.status", f"updated_at={time.strftime('%F %T %Z')}\nrows={len(rows)}\nsource={args.log_file}\n")
                print(f"[METRICS] {time.strftime('%F %T')} refreshed {len(rows)} completed steps -> {args.output_dir}", flush=True)
                previous_signature = signature
            consecutive_io_errors = 0
        except OSError as exc:
            # Shared/FUSE checkpoint storage can transiently return EIO while a
            # large training log is being appended. This sidecar must never die
            # because the next poll can recover and the trainer is unaffected.
            consecutive_io_errors += 1
            if consecutive_io_errors == 1 or consecutive_io_errors % 10 == 0:
                print(
                    f"[METRICS][WARN] {time.strftime('%F %T')} transient I/O error "
                    f"(consecutive={consecutive_io_errors}): {exc}",
                    flush=True,
                )
        if args.once:
            break
        time.sleep(max(5, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
