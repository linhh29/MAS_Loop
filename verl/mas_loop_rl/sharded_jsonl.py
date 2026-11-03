from __future__ import annotations

import atexit
import glob
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator


class _ProcessShardWriter:
    """One append-only JSONL descriptor per process and logical stream.

    Each process writes its own shard, so no inter-process ``flock`` is needed.
    ``os.write`` hands every encoded record directly to the kernel rather than
    retaining Python buffered data that could disappear when a Ray worker is
    terminated.  A small thread lock only protects threads in the *same*
    process (notably MASExecutor's dispatch pool).
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self.lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        payload = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        view = memoryview(payload)
        with self.lock:
            while view:
                written = os.write(self.fd, view)
                if written <= 0:
                    raise OSError(f"short JSONL write to {self.path}: {written}")
                view = view[written:]

    def close(self) -> None:
        with self.lock:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1


_registry_lock = threading.Lock()
_registry_pid = os.getpid()
_registry: dict[str, _ProcessShardWriter] = {}


def _close_all() -> None:
    global _registry
    with _registry_lock:
        writers, _registry = list(_registry.values()), {}
    for writer in writers:
        try:
            writer.close()
        except OSError:
            pass


atexit.register(_close_all)


def _writer(stream_dir: Path, shard_prefix: str = "shard") -> _ProcessShardWriter:
    global _registry_pid, _registry
    pid = os.getpid()
    with _registry_lock:
        # Ray/worker code can fork after importing this module. Never inherit a
        # parent's descriptor or shard identity into the child.
        if _registry_pid != pid:
            for old in _registry.values():
                try:
                    old.close()
                except OSError:
                    pass
            _registry = {}
            _registry_pid = pid
        key = f"{stream_dir.resolve()}::{shard_prefix}"
        writer = _registry.get(key)
        if writer is None:
            writer = _ProcessShardWriter(stream_dir / f"{shard_prefix}-{pid}.jsonl")
            _registry[key] = writer
        return writer


def append_jsonl_shard(stream_dir: str | Path, record: dict[str, Any], *, shard_prefix: str = "shard") -> Path:
    writer = _writer(Path(stream_dir), shard_prefix=shard_prefix)
    writer.append(record)
    return writer.path


def close_jsonl_writers(
    stream_dir: str | Path | None = None,
    *,
    shard_prefix: str | None = None,
) -> int:
    """Close matching process-local shard writers and return their count.

    Long-lived rollout workers normally keep one append descriptor open until
    process exit.  Bounded post-processing jobs, however, may need to read back
    a just-written shard immediately.  Explicitly closing that descriptor
    avoids read-after-write stalls on shared FUSE filesystems while preserving
    the append-only worker fast path.
    """
    if shard_prefix is not None and stream_dir is None:
        raise ValueError("shard_prefix requires stream_dir")

    global _registry_pid, _registry
    pid = os.getpid()
    resolved_dir = str(Path(stream_dir).resolve()) if stream_dir is not None else None
    selected: list[_ProcessShardWriter] = []
    inherited: list[_ProcessShardWriter] = []
    with _registry_lock:
        if _registry_pid != pid:
            inherited, _registry = list(_registry.values()), {}
            _registry_pid = pid
        else:
            for key, writer in list(_registry.items()):
                writer_dir, writer_prefix = key.rsplit("::", 1)
                if resolved_dir is not None and writer_dir != resolved_dir:
                    continue
                if shard_prefix is not None and writer_prefix != shard_prefix:
                    continue
                selected.append(_registry.pop(key))

    for writer in [*inherited, *selected]:
        writer.close()
    return len(selected)


def resolve_jsonl_inputs(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files, shard directories, and glob expressions deterministically."""

    resolved: set[Path] = set()
    for raw in inputs:
        text = str(raw)
        path = Path(text)
        if path.is_file():
            resolved.add(path)
            continue
        if path.is_dir():
            resolved.update(p for p in path.rglob("*.jsonl") if p.is_file())
            continue
        for match in glob.glob(text, recursive=True):
            candidate = Path(match)
            if candidate.is_file():
                resolved.add(candidate)
            elif candidate.is_dir():
                resolved.update(p for p in candidate.rglob("*.jsonl") if p.is_file())
    return sorted(resolved, key=lambda p: str(p))


def iter_jsonl(inputs: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for path in resolve_jsonl_inputs(inputs):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
