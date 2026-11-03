"""Wait until GPU 8-15 are isolated for a MAS_Loop trial.

The guard checks both physical GPU processes and foreign project command lines/cwds.
This matters when a foreign vLLM engine is in sleep mode: gpustat can show 2 MiB,
while the still-live process may wake later and collide with MAS_Loop.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


def _ancestor_pids(pid: int) -> set[int]:
    result: set[int] = set()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            pid = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
    return result


GPU_VISIBILITY_KEYS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def _parse_visible_gpus(environ: bytes) -> tuple[int, ...] | None:
    """Return explicit numeric GPU visibility, or None when it is unspecified/opaque.

    An explicitly empty value means the process cannot see a GPU. UUID-style
    visibility remains opaque and therefore fail-closed in the isolation guard.
    """
    entries = environ.split(b"\0")
    for key in GPU_VISIBILITY_KEYS:
        prefix = f"{key}=".encode()
        for entry in entries:
            if not entry.startswith(prefix):
                continue
            value = entry[len(prefix):].decode(errors="replace").strip()
            if not value or value == "-1":
                return ()
            try:
                return tuple(int(item.strip()) for item in value.split(",") if item.strip())
            except ValueError:
                return None
    return None


def _foreign_processes(
    forbidden_roots: tuple[str, ...],
    forbidden_text: tuple[str, ...],
    protected_gpus: tuple[int, ...] = (),
) -> list[dict[str, object]]:
    ignored = _ancestor_pids(os.getpid())
    protected = set(protected_gpus)
    blockers: list[dict[str, object]] = []
    # Do not block harmless shells, editors, filesystem daemons, or monitoring
    # processes merely because their cwd is another project.  We only need to
    # guard processes that can own/wake a training or inference runtime.
    runtime_markers = (
        "trainer.main_ppo",
        "vllm.entrypoints",
        "ray::",
    )
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid in ignored:
            continue
        try:
            cmdline = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            try:
                cwd = os.readlink(proc / "cwd")
            except OSError:
                cwd = ""
            visible_gpus = _parse_visible_gpus(proc.joinpath("environ").read_bytes())
        except OSError:
            continue
        root_matches = [root for root in forbidden_roots if root in cmdline or cwd.startswith(root)]
        text_matches = [text for text in forbidden_text if text in cmdline]
        is_runtime = any(marker in cmdline for marker in runtime_markers)
        # A live foreign runtime with an explicit, numeric CUDA visibility that
        # is disjoint from the protected physical GPU set cannot wake onto the
        # protected GPUs. This permits another project on GPUs 0-7 and this trial on 8-15,
        # while unspecified/UUID visibility remains fail-closed.
        explicitly_disjoint = visible_gpus is not None and protected.isdisjoint(visible_gpus)
        if not explicitly_disjoint and (text_matches or (root_matches and is_runtime)):
            blockers.append(
                {
                    "pid": pid,
                    "cwd": cwd,
                    "cmd": cmdline[:500],
                    "matched": [*root_matches, *text_matches],
                    "visible_gpus": visible_gpus,
                }
            )
    return sorted(blockers, key=lambda row: int(row["pid"]))


def _gpu_processes(gpu_indices: tuple[int, ...]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw = subprocess.check_output(["gpustat", "--json"], text=True)
    payload = json.loads(raw)
    blockers: list[dict[str, object]] = []
    snapshot: list[dict[str, object]] = []
    wanted = set(gpu_indices)
    for gpu in payload.get("gpus", []):
        index = int(gpu["index"])
        if index not in wanted:
            continue
        processes = gpu.get("processes") or []
        row = {
            "index": index,
            "util": gpu.get("utilization.gpu"),
            "memory_used_mb": gpu.get("memory.used"),
            "process_count": len(processes),
        }
        snapshot.append(row)
        for proc in processes:
            blockers.append(
                {
                    "gpu": index,
                    "pid": proc.get("pid"),
                    "command": proc.get("command"),
                    "gpu_memory_mb": proc.get("gpu_memory_usage"),
                }
            )
    return blockers, snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--forbid-root", action="append", default=[], help="Optional project roots to exclude from the protected GPUs")
    parser.add_argument("--forbid-text", action="append", default=[], help="Optional process command markers to exclude")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=0.0, help="0 means wait indefinitely")
    args = parser.parse_args()

    gpu_indices = tuple(int(x) for x in args.gpus.split(",") if x.strip())
    forbidden_roots = tuple(dict.fromkeys(args.forbid_root))
    forbidden_text = tuple(dict.fromkeys(args.forbid_text))
    started = time.monotonic()
    stable = 0
    while True:
        now = dt.datetime.now().astimezone().strftime("%F %T %Z")
        foreign = _foreign_processes(forbidden_roots, forbidden_text, gpu_indices)
        try:
            gpu_blockers, snapshot = _gpu_processes(gpu_indices)
            gpu_error = None
        except Exception as exc:  # fail closed when gpustat is unavailable
            gpu_blockers, snapshot = [], []
            gpu_error = f"{type(exc).__name__}: {exc}"

        clear = not foreign and not gpu_blockers and gpu_error is None
        stable = stable + 1 if clear else 0
        report = {
            "time": now,
            "clear": clear,
            "stable_checks": stable,
            "required_stable_checks": args.stable_checks,
            "foreign_process_count": len(foreign),
            "foreign_processes": foreign[:12],
            "gpu_process_count": len(gpu_blockers),
            "gpu_processes": gpu_blockers[:24],
            "gpu_snapshot": snapshot,
            "gpu_error": gpu_error,
        }
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if stable >= max(1, args.stable_checks):
            print(f"[gpu-isolation] READY after {time.monotonic() - started:.1f}s", flush=True)
            return 0
        if args.timeout > 0 and time.monotonic() - started >= args.timeout:
            print("[gpu-isolation] TIMEOUT", flush=True)
            return 3
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
