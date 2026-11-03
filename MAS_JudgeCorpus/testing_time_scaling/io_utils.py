from __future__ import annotations

import json
import fcntl
import os
import uuid
from pathlib import Path
from typing import Any, Iterable


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp path: shard workers may publish same manifest/table concurrently.
    # Fixed ``.tmp`` causes NFS collisions and Errno 95 during resume.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one complete JSONL record, serialized across resume/shard workers.

    MAS records can be several MiB, so relying on O_APPEND alone is unsafe when
    multiple shard processes share the canonical checkpoint file: a text write
    may be split into multiple syscalls.  A sidecar flock protects the whole
    record while preserving the existing resumable file layout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    encoded = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("ab", buffering=0) as f:
                f.write(encoded)
                f.flush()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completed_keys(path: Path, fields: Iterable[str]) -> set[tuple[Any, ...]]:
    names = tuple(fields)
    return {tuple(row.get(k) for k in names) for row in read_jsonl(path) if all(k in row for k in names)}
