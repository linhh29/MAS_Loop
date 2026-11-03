#!/usr/bin/env python3
"""Wait for the formal candidate banks, release the dedicated vLLM, then run local selectors."""
from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED = {
    ("hotpotqa", "cot"): 200 * 16,
    ("hotpotqa", "mas"): 200 * 16,
    ("bcp", "cot"): 168 * 16,
    ("bcp", "mas"): 168 * 16,
}
RM_METHODS = [
    "skywork_v2_qwen3_8b",
    "orch_rm",
    "mas_judgerm_orchestration",
    "mas_judgerm_trajectory",
]
API_METHODS = ["deepseek_v4_flash", "kimi_k3", "qwen3_8_2_4t_a95b"]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_banks(run_dir: Path) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {"checked_at": now(), "banks": {}}
    complete = True
    for (dataset, kind), expected in EXPECTED.items():
        path = run_dir / "candidates" / dataset / f"{kind}.jsonl"
        try:
            rows = read_rows(path)
            pairs = collections.Counter((str(r["sample_id"]), int(r["candidate_idx"])) for r in rows)
            duplicates = sum(v - 1 for v in pairs.values() if v > 1)
            sample_count = len({str(r["sample_id"]) for r in rows})
            ok = len(rows) == expected and len(pairs) == expected and duplicates == 0
            report["banks"][f"{dataset}/{kind}"] = {
                "rows": len(rows), "expected": expected, "unique_pairs": len(pairs),
                "duplicates": duplicates, "samples": sample_count, "complete": ok,
            }
            complete = complete and ok
        except Exception as exc:
            report["banks"][f"{dataset}/{kind}"] = {"complete": False, "error": f"{type(exc).__name__}: {exc}"}
            complete = False
    report["complete"] = complete
    return complete, report


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return ""


def stop_dedicated_vllm(pid: int, log) -> None:
    if not pid_alive(pid):
        print(f"[{now()}] vLLM pid {pid} already exited", file=log, flush=True)
        return
    command = cmdline(pid)
    required = ("vllm", "Qwen3-30B-A3B", "--port 8800", "qwen3-30b-a3b")
    if not all(part in command for part in required):
        raise RuntimeError(f"refusing to stop pid {pid}; unexpected cmdline: {command!r}")
    print(f"[{now()}] stopping dedicated vLLM pid={pid}: {command}", file=log, flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 300
    while pid_alive(pid) and time.time() < deadline:
        time.sleep(2)
    if pid_alive(pid):
        raise RuntimeError(f"vLLM pid {pid} did not exit within 300 seconds; refusing SIGKILL")
    print(f"[{now()}] dedicated vLLM stopped", file=log, flush=True)


def run_selector(*, workspace: Path, config: Path, run_id: str, method: str, log_dir: Path, log) -> None:
    command = [
        sys.executable, "testing_time_scaling/run.py", "select",
        "--config", str(config), "--run-id", run_id,
        "--datasets", "hotpotqa", "bcp", "--methods", method,
    ]
    method_log = log_dir / f"select.{method}.log"
    print(f"[{now()}] starting selector {method}: {' '.join(command)}", file=log, flush=True)
    with method_log.open("a", encoding="utf-8") as out:
        out.write(f"\n[{now()}] COMMAND: {' '.join(command)}\n")
        out.flush()
        proc = subprocess.run(command, cwd=workspace, stdout=out, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"selector {method} failed with rc={proc.returncode}; see {method_log}")
    print(f"[{now()}] selector {method} completed", file=log, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-pid", type=int, action="append", required=True)
    parser.add_argument("--vllm-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--run-api-judges", action="store_true", help="Send orchestration prompts to configured external APIs")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    run_dir = workspace / "testing_time_scaling" / "results" / args.run_id
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    controller_log = log_dir / "post_generation_controller.log"
    status_path = run_dir / "post_generation_status.json"

    with controller_log.open("a", encoding="utf-8") as log:
        print(f"[{now()}] controller started", file=log, flush=True)
        while True:
            complete, report = audit_banks(run_dir)
            generation_alive = {str(pid): pid_alive(pid) for pid in args.generation_pid}
            report.update({"phase": "waiting_for_candidates", "generation_pids": args.generation_pid, "generation_alive": generation_alive})
            atomic_json(status_path, report)
            compact = ", ".join(f"{k}={v.get('rows', 0)}/{v.get('expected', '?')}" for k, v in report["banks"].items())
            print(f"[{now()}] {compact}; generation_alive={report['generation_alive']}", file=log, flush=True)
            if complete:
                break
            # Never wait forever on a generator that exited before completing its
            # resumable JSONL banks.  A non-zero exit leaves the exact checkpoint
            # counts in post_generation_status.json for the next invocation.
            if not any(generation_alive.values()):
                report["phase"] = "generation_exited_incomplete"
                report["error"] = "all candidate generators exited before all banks completed"
                atomic_json(status_path, report)
                raise RuntimeError(f"all generators exited before candidate banks completed: {compact}")
            time.sleep(args.poll_seconds)

        # The final row may be visible milliseconds before the generator closes its HTTP client.
        deadline = time.time() + 600
        while any(pid_alive(pid) for pid in args.generation_pid) and time.time() < deadline:
            alive = [pid for pid in args.generation_pid if pid_alive(pid)]
            print(f"[{now()}] banks complete; waiting for generator pids {alive} to exit", file=log, flush=True)
            time.sleep(5)
        alive = [pid for pid in args.generation_pid if pid_alive(pid)]
        if alive:
            raise RuntimeError(f"candidate banks are complete but generation processes {alive} did not exit within 600 seconds")

        complete, report = audit_banks(run_dir)
        if not complete:
            raise RuntimeError("candidate-bank audit regressed after generator exit")
        report["phase"] = "candidate_audit_passed"
        atomic_json(status_path, report)
        stop_dedicated_vllm(args.vllm_pid, log)

        # API judges are independent and rate-limited per endpoint. Run all three
        # concurrently while deterministic/local-RM selectors use CPU/GPU.
        api_processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
        for method in (API_METHODS if args.run_api_judges else []):
            command = [
                sys.executable, "testing_time_scaling/run.py", "select",
                "--config", str(args.config), "--run-id", args.run_id,
                "--datasets", "hotpotqa", "bcp", "--methods", method,
            ]
            method_log = (log_dir / f"select.{method}.log").open("a", encoding="utf-8")
            method_log.write(f"\n[{now()}] COMMAND: {' '.join(command)}\n")
            method_log.flush()
            proc = subprocess.Popen(command, cwd=workspace, stdout=method_log, stderr=subprocess.STDOUT)
            api_processes.append((method, proc, method_log))
            print(f"[{now()}] started API selector {method} pid={proc.pid}", file=log, flush=True)

        # Rebuild deterministic results/table first; this validates both CoT banks.
        run_selector(workspace=workspace, config=args.config, run_id=args.run_id, method="cot_lb", log_dir=log_dir, log=log)
        run_selector(workspace=workspace, config=args.config, run_id=args.run_id, method="majority_voting", log_dir=log_dir, log=log)
        run_selector(workspace=workspace, config=args.config, run_id=args.run_id, method="oracle_ub", log_dir=log_dir, log=log)
        for method in RM_METHODS:
            run_selector(workspace=workspace, config=args.config, run_id=args.run_id, method=method, log_dir=log_dir, log=log)

        api_failures = []
        for method, proc, method_log in api_processes:
            rc = proc.wait()
            method_log.close()
            if rc != 0:
                api_failures.append(f"{method}: rc={rc}")
            else:
                print(f"[{now()}] API selector {method} completed", file=log, flush=True)
        if api_failures:
            raise RuntimeError("API selectors failed: " + ", ".join(api_failures))

        # Concurrent selector processes each write a table snapshot. Rebuild once
        # after all methods finish so final/paper tables include every artifact.
        run_selector(workspace=workspace, config=args.config, run_id=args.run_id, method="cot_lb", log_dir=log_dir, log=log)

        audit_output = run_dir / "audit.full.json"
        audit_command = [
            sys.executable, "testing_time_scaling/audit_run.py",
            "--run-dir", str(run_dir), *(["--require-api"] if args.run_api_judges else []), "--output", str(audit_output),
        ]
        audit_log = log_dir / "audit.local.log"
        print(f"[{now()}] running strict local completion audit", file=log, flush=True)
        with audit_log.open("a", encoding="utf-8") as out:
            proc = subprocess.run(audit_command, cwd=workspace, stdout=out, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"strict local completion audit failed; see {audit_output} and {audit_log}")

        _, final_report = audit_banks(run_dir)
        final_report.update({
            "phase": "all_selectors_completed" if args.run_api_judges else "local_selectors_completed",
            "local_methods": RM_METHODS, "api_methods": API_METHODS if args.run_api_judges else [],
            "api_judges_pending_approval": not args.run_api_judges,
            "strict_audit": str(audit_output), "completed_at": now(),
        })
        atomic_json(status_path, final_report)
        completed_scope = "local and API selectors" if args.run_api_judges else "local selectors; API judges pending explicit data-export approval"
        print(f"[{now()}] {completed_scope} plus strict audit completed", file=log, flush=True)


if __name__ == "__main__":
    main()
