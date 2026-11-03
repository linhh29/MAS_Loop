"""Cross-process exclusive ownership for colocated MAS-Loop vLLM engines.

The optimized backend uses a Unix-domain local control plane and keeps phase
refcounts in RAM.  The original flock/JSON implementation remains as a fallback
when ``MAS_LOOP_VLLM_CONTROL_SOCKET`` is unset.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


logger = logging.getLogger(__name__)


@dataclass
class CoordinatorToken:
    token: str
    owner: str
    kind: str


def control_socket_path() -> str:
    return os.getenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "").strip()


def using_coordinator() -> bool:
    return bool(control_socket_path()) and os.getenv("MAS_LOOP_PHASE_LEASE", "1") not in {"0", "false", "False"}


def wait_for_external_vllm_idle_enabled() -> bool:
    return (
        os.getenv("MAS_LOOP_EXCLUSIVE_VLLM", "0").strip().lower() in {"1", "true", "yes", "on"}
        and using_coordinator()
    )


def _rpc(op: str, *, timeout_sec: float | None = None, **payload: Any) -> dict[str, Any]:
    path = control_socket_path()
    if not path:
        raise RuntimeError("MAS_LOOP_VLLM_CONTROL_SOCKET is empty")
    timeout = (
        float(timeout_sec)
        if timeout_sec is not None
        else float(os.getenv("MAS_LOOP_VLLM_CONTROL_TIMEOUT_SEC", "3600"))
    )
    if timeout <= 0:
        raise ValueError(f"coordinator RPC timeout must be positive, got {timeout}")
    request = json.dumps({"op": op, **payload}, ensure_ascii=False).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(path)
        client.sendall(request)
        chunks = bytearray()
        while not chunks.endswith(b"\n"):
            part = client.recv(65536)
            if not part:
                break
            chunks.extend(part)
    response = json.loads(bytes(chunks).decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError(response.get("error") or f"coordinator {op} failed")
    result = response.get("result")
    return result if isinstance(result, dict) else {}


def wait_until_idle(*, timeout_sec: float | None = None, poll_sec: float | None = None) -> dict[str, Any]:
    """Wait until the coordinator has acknowledged that all external vLLMs slept.

    The strict state predicate intentionally requires ``status=idle`` in
    addition to zero active requests.  During the grace and sleeping states,
    active is already zero but model memory may still be resident.
    """

    if not using_coordinator():
        raise RuntimeError("external vLLM idle barrier requires the phase coordinator")
    timeout = (
        float(timeout_sec)
        if timeout_sec is not None
        else float(os.getenv("MAS_LOOP_VLLM_IDLE_TIMEOUT_SEC", "300"))
    )
    poll = (
        float(poll_sec)
        if poll_sec is not None
        else float(os.getenv("MAS_LOOP_VLLM_IDLE_POLL_SEC", "0.05"))
    )
    if timeout <= 0:
        raise ValueError(f"idle barrier timeout must be positive, got {timeout}")
    if poll <= 0:
        raise ValueError(f"idle barrier poll interval must be positive, got {poll}")

    started = time.monotonic()
    deadline = started + timeout
    last_state: dict[str, Any] = {}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"external vLLM coordinator did not become idle within {timeout:.3f}s; "
                f"last_state={last_state!r}"
            )
        # Security: fail closed on coordinator RPC failures; continuing into the
        # optimizer with an unknown residency state can cause a GPU OOM.
        last_state = _rpc("state", timeout_sec=min(remaining, 5.0))
        owner = str(last_state.get("owner") or "")
        active = int(last_state.get("active", last_state.get("count", 0)) or 0)
        status = str(last_state.get("status") or "")
        if not owner and active == 0 and status == "idle":
            logger.info(
                "external vLLM idle barrier completed in %.3fs phase_id=%s",
                time.monotonic() - started,
                last_state.get("phase_id"),
            )
            return last_state
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))


async def _async_rpc(op: str, **payload: Any) -> dict[str, Any]:
    """Async coordinator RPC used from Ray asyncio actors.

    Do not implement this with ``asyncio.to_thread(_rpc, ...)``.  When the
    phase reaches ``max_active``, acquire calls intentionally wait.  Enough
    waiting acquires can occupy every worker in asyncio's default executor and
    starve the release calls that would free admission slots, producing a
    classic thread-pool deadlock.  Native asyncio Unix sockets let any number
    of acquires wait without consuming threads, while releases remain runnable.
    """

    path = control_socket_path()
    if not path:
        raise RuntimeError("MAS_LOOP_VLLM_CONTROL_SOCKET is empty")
    control_timeout = float(os.getenv("MAS_LOOP_VLLM_CONTROL_TIMEOUT_SEC", "3600"))
    # Acquires can legitimately wait across several long model phases.  Keep
    # their deadline separate from short control RPCs; the coordinator also
    # cancels the server-side waiter when this client disconnects, so a timeout
    # or task cancellation cannot later materialize as an orphaned lease.
    timeout = (
        float(os.getenv("MAS_LOOP_VLLM_ACQUIRE_TIMEOUT_SEC", "14400"))
        if op == "acquire"
        else control_timeout
    )
    if timeout <= 0:
        raise ValueError(f"coordinator {op} timeout must be positive, got {timeout}")
    request = json.dumps({"op": op, **payload}, ensure_ascii=False).encode("utf-8") + b"\n"
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=timeout)
        writer.write(request)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not raw:
            raise RuntimeError(f"coordinator {op} closed without a response")
        response = json.loads(raw.decode("utf-8"))
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or f"coordinator {op} failed")
        result = response.get("result")
        return result if isinstance(result, dict) else {}
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass


def lock_path() -> str:
    return os.getenv("MAS_LOOP_VLLM_SLEEP_LOCK", "/tmp/mas_loop_vllm_sleep.lock")


def state_path() -> str:
    return os.getenv("MAS_LOOP_VLLM_SLEEP_STATE", lock_path() + ".state.json")


def events_path() -> str:
    return os.getenv("MAS_LOOP_VLLM_SLEEP_EVENTS", state_path() + ".events.jsonl")


def read_state() -> dict[str, Any]:
    if using_coordinator():
        try:
            return _rpc("state")
        except Exception:
            pass
    try:
        with open(state_path(), "r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {"owner": "", "count": 0, "updated": 0.0}


def write_state(value: dict[str, Any]) -> None:
    path = state_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_event(event: str, owner: str, **fields: Any) -> None:
    # Coordinator owns critical-path phase events.  Optional caller-side events
    # stay on the configured local path and never fsync.
    path = events_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.time(), "pid": os.getpid(), "event": event, "owner": owner, **fields}
    try:
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass


def acquire_transition(owner: str, *, max_active: int = 1024) -> tuple[IO[str] | CoordinatorToken | None, bool]:
    if using_coordinator():
        result = _rpc("acquire", owner=owner, max_active=max_active)
        first = bool(result.get("first"))
        token = CoordinatorToken(str(result.get("token") or ""), owner, "wake") if first else None
        return token, first

    path = lock_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wait = float(os.getenv("MAS_LOOP_VLLM_OWNER_WAIT_SEC", "0.1"))
    last_report = 0.0
    while True:
        lock = open(path, "a+", encoding="utf-8")
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_state()
        current = str(state.get("owner") or "")
        count = int(state.get("count") or 0)
        if not current or count <= 0:
            log_event("transition_acquired", owner, previous_owner=current)
            return lock, True
        if current == owner and count < max_active:
            write_state({"owner": owner, "count": count + 1, "updated": time.time()})
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            log_event("shared_acquire", owner, count=count + 1)
            return None, False
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        now = time.time()
        if now - last_report >= 30:
            log_event("waiting", owner, blocking_owner=current, blocking_count=count)
            last_report = now
        time.sleep(wait)


def commit_first_acquire(lock: IO[str] | CoordinatorToken, owner: str) -> None:
    if isinstance(lock, CoordinatorToken):
        _rpc("commit", owner=owner, token=lock.token)
        return
    write_state({"owner": owner, "count": 1, "updated": time.time()})
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    log_event("wake_complete", owner, count=1)


def abort_first_acquire(lock: IO[str] | CoordinatorToken, owner: str, error: BaseException) -> None:
    if isinstance(lock, CoordinatorToken):
        _rpc("abort", owner=owner, token=lock.token, error=f"{type(error).__name__}:{error}")
        return
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    log_event("wake_failed", owner, error=f"{type(error).__name__}:{error}")


def acquire_release_transition(owner: str) -> tuple[IO[str] | CoordinatorToken | None, int, bool]:
    if using_coordinator():
        result = _rpc("release", owner=owner)
        should_sleep = bool(result.get("should_sleep"))
        token = CoordinatorToken(str(result.get("token") or ""), owner, "sleep") if should_sleep else None
        return token, int(result.get("count") or 0), should_sleep

    path = lock_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lock = open(path, "a+", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = read_state()
    current = str(state.get("owner") or "")
    count = int(state.get("count") or 0)
    if current != owner or count <= 0:
        return lock, 0, False
    new_count = count - 1
    if new_count > 0:
        write_state({"owner": owner, "count": new_count, "updated": time.time()})
        return lock, new_count, False
    write_state({"owner": owner, "count": 1, "updated": time.time(), "transition": "sleeping"})
    return lock, 0, True


def finish_release(lock: IO[str] | CoordinatorToken | None, owner: str, *, slept: bool, error: BaseException | None = None) -> None:
    if isinstance(lock, CoordinatorToken):
        _rpc("finish", owner=owner, token=lock.token, slept=slept, error=str(error or ""))
        return
    if lock is None:
        return
    if slept:
        write_state({"owner": "", "count": 0, "updated": time.time()})
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    log_event("sleep_complete" if error is None else "sleep_failed", owner, slept=slept, error=str(error or ""))


async def async_acquire_transition(
    owner: str, *, max_active: int = 1024
) -> tuple[IO[str] | CoordinatorToken | None, bool]:
    """Non-blocking phase admission for asyncio AgentLoop workers."""

    if not using_coordinator():
        return await asyncio.to_thread(acquire_transition, owner, max_active=max_active)
    result = await _async_rpc("acquire", owner=owner, max_active=max_active)
    first = bool(result.get("first"))
    token = CoordinatorToken(str(result.get("token") or ""), owner, "wake") if first else None
    return token, first


async def async_commit_first_acquire(lock: IO[str] | CoordinatorToken, owner: str) -> None:
    if not isinstance(lock, CoordinatorToken):
        await asyncio.to_thread(commit_first_acquire, lock, owner)
        return
    await _async_rpc("commit", owner=owner, token=lock.token)


async def async_abort_first_acquire(lock: IO[str] | CoordinatorToken, owner: str, error: BaseException) -> None:
    if not isinstance(lock, CoordinatorToken):
        await asyncio.to_thread(abort_first_acquire, lock, owner, error)
        return
    await _async_rpc("abort", owner=owner, token=lock.token, error=f"{type(error).__name__}:{error}")


async def async_acquire_release_transition(
    owner: str,
) -> tuple[IO[str] | CoordinatorToken | None, int, bool]:
    """Non-blocking release; it must never queue behind waiting acquires."""

    if not using_coordinator():
        return await asyncio.to_thread(acquire_release_transition, owner)
    result = await _async_rpc("release", owner=owner)
    should_sleep = bool(result.get("should_sleep"))
    token = CoordinatorToken(str(result.get("token") or ""), owner, "sleep") if should_sleep else None
    return token, int(result.get("count") or 0), should_sleep


async def async_finish_release(
    lock: IO[str] | CoordinatorToken | None,
    owner: str,
    *,
    slept: bool,
    error: BaseException | None = None,
) -> None:
    if not isinstance(lock, CoordinatorToken):
        await asyncio.to_thread(finish_release, lock, owner, slept=slept, error=error)
        return
    await _async_rpc("finish", owner=owner, token=lock.token, slept=slept, error=str(error or ""))
