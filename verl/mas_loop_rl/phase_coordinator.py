"""Local in-memory control plane for exclusive MAS-Loop vLLM phases.

All colocated engines span GPUs 8-15.  Only one model may be resident at once,
but hundreds of requests for that same model should enter as one phase.  The
coordinator keeps request refcounts in RAM behind a Unix socket, coalesces short
zero-active gaps into one lease, and serializes wake/sleep acknowledgements.
No request-path state is written to OSSFS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any


class PhaseCoordinator:
    def __init__(
        self,
        *,
        socket_path: str,
        state_path: str,
        events_path: str,
        grace_sec: float = 0.75,
        backlog: int = 16384,
        state_interval_sec: float = 1.0,
    ):
        self.socket_path = socket_path
        self.state_path = state_path
        self.events_path = events_path
        self.grace_sec = max(0.0, float(grace_sec))
        self.backlog = max(128, int(backlog))
        self.state_interval_sec = max(0.05, float(state_interval_sec))
        self.condition = asyncio.Condition()
        self.owner = ""
        self.active = 0
        self.status = "idle"
        self.phase_id = 0
        self.transition_token = ""
        self.phase_started = 0.0
        self.phase_requests = 0
        self.total_requests = 0
        self.waiters: dict[str, int] = {}
        self.active_peak = 0
        self.global_active_peak = 0
        self.owner_active_peaks: dict[str, int] = {}
        self.waiters_peak = 0
        self.owner_waiter_peaks: dict[str, int] = {}
        self.acquire_wait_count = 0
        self.acquire_wait_sum_sec = 0.0
        self.acquire_wait_max_sec = 0.0
        self.owner_acquire_wait: dict[str, dict[str, float | int]] = {}
        self.wake_started_mono = 0.0
        self.sleep_started_mono = 0.0
        self.last_wake_duration_sec = 0.0
        self.last_sleep_duration_sec = 0.0
        self.last_phase_duration_sec = 0.0
        self.completed_phases = 0
        self.server: asyncio.AbstractServer | None = None
        self._event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._event_task: asyncio.Task | None = None
        self._state_task: asyncio.Task | None = None
        self._client_tasks: set[asyncio.Task[Any]] = set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "count": self.active,
            "active": self.active,
            "status": self.status,
            "phase_id": self.phase_id,
            "phase_requests": self.phase_requests,
            "phase_started": self.phase_started,
            "waiters": dict(self.waiters),
            "metrics": {
                "active_peak_current_phase": self.active_peak,
                "active_peak_global": self.global_active_peak,
                "active_peak_by_owner": dict(self.owner_active_peaks),
                "waiters_total": sum(self.waiters.values()),
                "waiters_peak_global": self.waiters_peak,
                "waiters_peak_by_owner": dict(self.owner_waiter_peaks),
                "acquire_wait": {
                    "count": self.acquire_wait_count,
                    "mean_sec": self.acquire_wait_sum_sec / self.acquire_wait_count if self.acquire_wait_count else 0.0,
                    "max_sec": self.acquire_wait_max_sec,
                    "by_owner": dict(self.owner_acquire_wait),
                },
                "last_wake_duration_sec": self.last_wake_duration_sec,
                "last_sleep_duration_sec": self.last_sleep_duration_sec,
                "last_phase_duration_sec": self.last_phase_duration_sec,
                "completed_phases": self.completed_phases,
                "total_requests": self.total_requests,
                "backlog": self.backlog,
            },
            "updated": time.time(),
            "pid": os.getpid(),
            "backend": "unix_phase_coordinator_v1",
        }

    def event(self, event: str, owner: str = "", **fields: Any) -> None:
        self._event_queue.put_nowait({
            "time": time.time(), "pid": os.getpid(), "event": event,
            "owner": owner, "phase_id": self.phase_id, **fields,
        })

    def _note_waiters(self, owner: str) -> None:
        total = sum(self.waiters.values())
        self.waiters_peak = max(self.waiters_peak, total)
        self.owner_waiter_peaks[owner] = max(self.owner_waiter_peaks.get(owner, 0), self.waiters.get(owner, 0))

    def _note_active(self, owner: str) -> None:
        self.active_peak = max(self.active_peak, self.active)
        self.global_active_peak = max(self.global_active_peak, self.active)
        self.owner_active_peaks[owner] = max(self.owner_active_peaks.get(owner, 0), self.active)

    def _note_acquire_wait(self, owner: str, wait_sec: float) -> None:
        self.acquire_wait_count += 1
        self.acquire_wait_sum_sec += wait_sec
        self.acquire_wait_max_sec = max(self.acquire_wait_max_sec, wait_sec)
        stats = self.owner_acquire_wait.setdefault(owner, {"count": 0, "sum_sec": 0.0, "max_sec": 0.0})
        stats["count"] = int(stats["count"]) + 1
        stats["sum_sec"] = float(stats["sum_sec"]) + wait_sec
        stats["max_sec"] = max(float(stats["max_sec"]), wait_sec)
        stats["mean_sec"] = float(stats["sum_sec"]) / int(stats["count"])

    def _write_state_snapshot(self) -> None:
        """Publish a current snapshot without touching the admission critical path."""
        state_path = Path(self.state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name(
            f".{state_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            tmp.write_text(json.dumps(self.snapshot(), ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, state_path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    async def _state_writer(self) -> None:
        # Shared acquire/release can happen tens of thousands of times per phase.
        # A periodic publisher keeps monitoring fresh without adding filesystem IO
        # to those request admission operations.
        while True:
            try:
                self._write_state_snapshot()
            except Exception:
                pass
            await asyncio.sleep(self.state_interval_sec)

    async def _event_writer(self) -> None:
        # This path is expected to live under /tmp.  It is intentionally detached
        # from the admission condition, so profiling can never stall dispatch.
        event_path = Path(self.events_path)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            record = await self._event_queue.get()
            if record is None:
                return
            try:
                with event_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._write_state_snapshot()
            except Exception:
                pass

    async def acquire(self, owner: str, max_active: int) -> dict[str, Any]:
        started = time.monotonic()
        max_active = max(1, int(max_active))
        async with self.condition:
            self.waiters[owner] = self.waiters.get(owner, 0) + 1
            self._note_waiters(owner)
            try:
                while True:
                    if not self.owner:
                        self.owner = owner
                        self.active = 1
                        self.status = "waking"
                        self.phase_id += 1
                        self.phase_started = time.time()
                        self.wake_started_mono = time.monotonic()
                        self.phase_requests = 1
                        self.active_peak = 1
                        self.total_requests += 1
                        self.transition_token = uuid.uuid4().hex
                        wait_sec = time.monotonic() - started
                        self._note_active(owner)
                        self._note_acquire_wait(owner, wait_sec)
                        self.event("phase_acquired", owner, first=True, wait_sec=wait_sec)
                        return {"first": True, "token": self.transition_token, "count": 1, "phase_id": self.phase_id}
                    if self.owner == owner and self.status in {"ready", "grace"} and self.active < max_active:
                        if self.status == "grace":
                            self.status = "ready"
                            self.event("phase_grace_reused", owner)
                        self.active += 1
                        self.phase_requests += 1
                        self.total_requests += 1
                        wait_sec = time.monotonic() - started
                        self._note_active(owner)
                        self._note_acquire_wait(owner, wait_sec)
                        self.condition.notify_all()
                        return {"first": False, "token": "", "count": self.active, "phase_id": self.phase_id}
                    await self.condition.wait()
            finally:
                self.waiters[owner] = max(0, self.waiters.get(owner, 1) - 1)
                if not self.waiters[owner]:
                    self.waiters.pop(owner, None)

    async def commit(self, owner: str, token: str) -> dict[str, Any]:
        async with self.condition:
            if self.owner != owner or self.status != "waking" or token != self.transition_token:
                raise RuntimeError(f"invalid wake commit owner={owner} current={self.owner} status={self.status}")
            self.status = "ready"
            self.transition_token = ""
            self.last_wake_duration_sec = max(0.0, time.monotonic() - self.wake_started_mono)
            self.event("wake_complete", owner, count=self.active, duration_sec=self.last_wake_duration_sec)
            self.condition.notify_all()
            return {"ok": True}

    async def abort(self, owner: str, token: str, error: str) -> dict[str, Any]:
        async with self.condition:
            if self.owner == owner and self.status == "waking" and token == self.transition_token:
                self.event("wake_failed", owner, error=error)
                self.owner, self.active, self.status, self.transition_token = "", 0, "idle", ""
                self.condition.notify_all()
            return {"ok": True}

    async def release(self, owner: str) -> dict[str, Any]:
        async with self.condition:
            if self.owner != owner or self.active <= 0:
                return {"count": 0, "should_sleep": False, "token": "", "phase_id": self.phase_id}
            self.active -= 1
            if self.active > 0:
                # Wake admission waiters that were blocked only because active
                # had reached max_active.  Without this notification, a batch
                # larger than the active cap can deadlock after the first wave.
                self.condition.notify_all()
                return {"count": self.active, "should_sleep": False, "token": "", "phase_id": self.phase_id}

            # Last request does not immediately evict the model.  A short lease
            # absorbs dispatch jitter, while a different owner remains blocked.
            self.status = "grace"
            grace_phase = self.phase_id
            deadline = asyncio.get_running_loop().time() + self.grace_sec
            self.event("phase_grace_started", owner, grace_sec=self.grace_sec, requests=self.phase_requests)
            while self.owner == owner and self.phase_id == grace_phase and self.active == 0:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
            if self.active > 0:
                self.status = "ready"
                return {"count": self.active, "should_sleep": False, "token": "", "phase_id": self.phase_id}
            token = uuid.uuid4().hex
            self.transition_token = token
            self.status = "sleeping"
            self.sleep_started_mono = time.monotonic()
            self.event(
                "phase_sleep_requested", owner, requests=self.phase_requests,
                duration_sec=time.time() - self.phase_started,
            )
            return {"count": 0, "should_sleep": True, "token": token, "phase_id": self.phase_id}

    async def finish(self, owner: str, token: str, slept: bool, error: str) -> dict[str, Any]:
        async with self.condition:
            if self.owner != owner or self.status != "sleeping" or token != self.transition_token:
                raise RuntimeError(f"invalid sleep finish owner={owner} current={self.owner} status={self.status}")
            if slept:
                self.last_sleep_duration_sec = max(0.0, time.monotonic() - self.sleep_started_mono)
                self.last_phase_duration_sec = max(0.0, time.time() - self.phase_started)
                self.completed_phases += 1
                self.event(
                    "phase_complete", owner, requests=self.phase_requests, error=error,
                    active_peak=self.active_peak, sleep_duration_sec=self.last_sleep_duration_sec,
                    duration_sec=self.last_phase_duration_sec,
                )
                self.owner, self.active, self.status, self.transition_token = "", 0, "idle", ""
            else:
                # Never wake a competing model after an unacknowledged sleep.
                self.status = "sleep_failed"
                self.event("sleep_failed", owner, error=error)
            self.condition.notify_all()
            return {"ok": True}

    async def _acquire_while_connected(
        self,
        reader: asyncio.StreamReader,
        owner: str,
        max_active: int,
    ) -> dict[str, Any] | None:
        """Cancel a queued acquire as soon as its client disconnects.

        An acquire may wait across multiple model phases.  Previously, when the
        client timed out or its AgentLoop task was cancelled, ``handle`` kept
        waiting inside ``acquire``.  That abandoned waiter could later be
        admitted, increment ``active``, and never issue a matching release.
        Racing admission against EOF keeps the coordinator refcount tied to a
        live request connection.
        """

        acquire_task = asyncio.create_task(self.acquire(owner, max_active))
        disconnect_task = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait(
                {acquire_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if acquire_task in done:
                # Admission won the race.  Stop the EOF watcher before replying.
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)
                return acquire_task.result()

            # EOF (or unexpected extra protocol bytes) arrived before admission.
            # Cancelling acquire executes its waiter-accounting finally block and
            # prevents a future phantom active lease.
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
            return None
        finally:
            for task in (acquire_task, disconnect_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(acquire_task, disconnect_task, return_exceptions=True)

    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        response: dict[str, Any] | None = None
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=3600)
            req = json.loads(raw.decode("utf-8"))
            op = str(req.get("op") or "")
            if op == "acquire":
                result = await self._acquire_while_connected(
                    reader,
                    str(req["owner"]),
                    int(req.get("max_active", 1024)),
                )
                if result is None:
                    return
            elif op == "commit":
                result = await self.commit(str(req["owner"]), str(req["token"]))
            elif op == "abort":
                result = await self.abort(str(req["owner"]), str(req["token"]), str(req.get("error", "")))
            elif op == "release":
                result = await self.release(str(req["owner"]))
            elif op == "finish":
                result = await self.finish(
                    str(req["owner"]), str(req["token"]), bool(req.get("slept")), str(req.get("error", ""))
                )
            elif op == "state":
                result = self.snapshot()
            elif op == "ping":
                result = {"ok": True, "pid": os.getpid()}
            else:
                raise ValueError(f"unknown op: {op}")
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
        try:
            if response is not None:
                writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The disconnect-aware acquire path prevents queued requests from
            # becoming leases.  A peer can still disappear while a small reply
            # is being written; closing the transport is sufficient here.
            pass
        finally:
            writer.close()
            try:
                # A peer that vanished while queued must not keep coordinator
                # shutdown waiting forever in transport cleanup.
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
                pass

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        try:
            await self._handle_request(reader, writer)
        finally:
            if task is not None:
                self._client_tasks.discard(task)

    async def run(self) -> None:
        socket = Path(self.socket_path)
        socket.parent.mkdir(parents=True, exist_ok=True)
        if socket.exists():
            socket.unlink()
        self._event_task = asyncio.create_task(self._event_writer())
        self._state_task = asyncio.create_task(self._state_writer())
        self.server = await asyncio.start_unix_server(self.handle, path=self.socket_path, backlog=self.backlog)
        os.chmod(self.socket_path, 0o600)
        self.event("coordinator_started", socket=self.socket_path, grace_sec=self.grace_sec, backlog=self.backlog)
        # The listener is already serving after start_unix_server returns.
        # Wait on a cancellable sentinel instead of Server.serve_forever(): the
        # latter owns its own close-on-cancel lifecycle and can wait on a
        # disconnected long-poll client before our explicit handler cleanup runs.
        await asyncio.Future()

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
        # asyncio.Server.close() stops accepting connections but its
        # wait_closed() may wait for existing handler transports.  Cancel and
        # drain those handlers first, then wait for the listener to close.
        current = asyncio.current_task()
        clients = [task for task in self._client_tasks if task is not current and not task.done()]
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
        if self.server is not None:
            try:
                await asyncio.wait_for(self.server.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                # The listener has already stopped accepting and all tracked
                # handlers are drained.  Do not let an asyncio transport-level
                # close notification prevent coordinator process shutdown.
                pass
        self.event("coordinator_stopped")
        await self._event_queue.put(None)
        if self._event_task is not None:
            await self._event_task
        if self._state_task is not None:
            self._state_task.cancel()
            try:
                await self._state_task
            except asyncio.CancelledError:
                pass
        try:
            self._write_state_snapshot()
        except Exception:
            pass
        try:
            Path(self.socket_path).unlink()
        except FileNotFoundError:
            pass


async def _main(args: argparse.Namespace) -> None:
    coordinator = PhaseCoordinator(
        socket_path=args.socket, state_path=args.state, events_path=args.events,
        grace_sec=args.grace_sec, backlog=args.backlog, state_interval_sec=args.state_interval_sec,
    )
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    task = asyncio.create_task(coordinator.run())
    waiter = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    if waiter in done:
        # Close the listener and cancel live client handlers before awaiting the
        # serve_forever task.  A disconnected long-poll acquire otherwise leaves
        # an open handler that can delay cancellation of the server context.
        task.cancel()
        # ``run`` waits on a plain cancellable sentinel, so let it unwind before
        # waiting for the listener's close notification.  Calling wait_closed
        # while the run task is still registered can otherwise form a shutdown
        # dependency cycle.
        await asyncio.gather(task, return_exceptions=True)
        await coordinator.close()
        return

    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    try:
        await task
    finally:
        await coordinator.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--grace-sec", type=float, default=0.75)
    parser.add_argument("--backlog", type=int, default=int(os.getenv("MAS_LOOP_PHASE_SOCKET_BACKLOG", "16384")))
    parser.add_argument(
        "--state-interval-sec", type=float,
        default=float(os.getenv("MAS_LOOP_PHASE_STATE_INTERVAL_SEC", "1.0")),
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
