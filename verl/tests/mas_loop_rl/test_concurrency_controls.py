from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time


def test_phase_coordinator_same_owner_parallel_and_cross_owner_exclusive(tmp_path: Path) -> None:
    from mas_loop_rl.phase_coordinator import PhaseCoordinator

    async def scenario() -> None:
        c = PhaseCoordinator(
            socket_path=str(tmp_path / "c.sock"),
            state_path=str(tmp_path / "state.json"),
            events_path=str(tmp_path / "events.jsonl"),
            grace_sec=0.001,
            backlog=8192,
        )
        first = await c.acquire("actor", 3)
        assert first["first"]
        await c.commit("actor", first["token"])

        actor_waiters = [asyncio.create_task(c.acquire("actor", 3)) for _ in range(4)]
        editor_waiter = asyncio.create_task(c.acquire("editor", 3))
        await asyncio.sleep(0.01)
        assert c.owner == "actor"
        assert c.active == 3
        assert sum(t.done() for t in actor_waiters) == 2
        assert not editor_waiter.done()

        # Two releases let the remaining actor requests enter without allowing
        # the competing owner to break phase exclusivity.
        await c.release("actor")
        await asyncio.sleep(0)
        await c.release("actor")
        await asyncio.gather(*actor_waiters)
        assert c.owner == "actor"
        assert c.active == 3
        assert not editor_waiter.done()

        await c.release("actor")
        await c.release("actor")
        sleep_transition = await c.release("actor")
        assert sleep_transition["should_sleep"]
        await c.finish("actor", sleep_transition["token"], True, "")

        editor_first = await asyncio.wait_for(editor_waiter, timeout=1)
        assert editor_first["first"]
        await c.commit("editor", editor_first["token"])
        editor_sleep = await c.release("editor")
        assert editor_sleep["should_sleep"]
        await c.finish("editor", editor_sleep["token"], True, "")

        metrics = c.snapshot()["metrics"]
        assert metrics["active_peak_global"] == 3
        assert metrics["active_peak_by_owner"]["actor"] == 3
        assert metrics["waiters_peak_by_owner"]["actor"] >= 2
        assert metrics["waiters_peak_by_owner"]["editor"] >= 1
        assert metrics["completed_phases"] == 2
        assert metrics["backlog"] == 8192

    asyncio.run(scenario())


def test_dispatch_metrics_report_real_queue_and_active(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAS_LOOP_DISPATCH_WORKERS_PER_PROCESS", "4")
    monkeypatch.setenv("MAS_LOOP_DISPATCH_METRICS_DIR", str(tmp_path))
    monkeypatch.setenv("MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC", "3600")

    import mas_loop_rl.dispatch as dispatch

    def blocking() -> int:
        time.sleep(0.02)
        return 1

    async def scenario() -> None:
        values = await asyncio.gather(*(dispatch.run_blocking(blocking, phase="subagent") for _ in range(20)))
        assert sum(values) == 20

    asyncio.run(scenario())
    dispatch._write_snapshot()
    metric_file = tmp_path / f"dispatch-{os.getpid()}.json"
    payload = json.loads(metric_file.read_text())
    stats = payload["phases"]["subagent"]
    assert stats["submitted"] == 20
    assert stats["completed"] == 20
    assert stats["active_peak"] == 4
    assert stats["queued_peak"] >= 16
    assert stats["queue_wait_sec"]["p95"] > 0
    assert stats["service_sec"]["p50"] >= 0.015


def test_managed_blocking_waits_for_phase_before_dispatch():
    from contextlib import asynccontextmanager
    from mas_loop_rl.loop_agent import _MASLoopBase

    async def scenario() -> None:
        gate = asyncio.Event()
        dispatch_calls = []

        class Lease:
            @asynccontextmanager
            async def async_awake_session(self):
                await gate.wait()
                yield

        class FakeLoop:
            _managed_blocking = _MASLoopBase._managed_blocking

            async def _blocking(self, fn, *args, phase="", **kwargs):
                dispatch_calls.append(phase)
                return fn(*args, **kwargs)

        task = asyncio.create_task(
            FakeLoop()._managed_blocking(Lease(), lambda: 7, phase="editor")
        )
        await asyncio.sleep(0.02)
        assert dispatch_calls == []
        gate.set()
        assert await asyncio.wait_for(task, timeout=1) == 7
        assert dispatch_calls == ["editor"]

    asyncio.run(scenario())
