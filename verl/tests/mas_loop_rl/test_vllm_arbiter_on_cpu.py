from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import pytest

from mas_loop_rl import vllm_arbiter
from mas_loop_rl.local_client import LocalChatClient


@pytest.fixture
def arbiter_env(tmp_path, monkeypatch):
    lock = tmp_path / "exclusive.lock"
    state = tmp_path / "state.json"
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("MAS_LOOP_VLLM_SLEEP_LOCK", str(lock))
    monkeypatch.setenv("MAS_LOOP_VLLM_SLEEP_STATE", str(state))
    monkeypatch.setenv("MAS_LOOP_VLLM_SLEEP_EVENTS", str(events))
    monkeypatch.setenv("MAS_LOOP_VLLM_OWNER_WAIT_SEC", "0.01")
    monkeypatch.delenv("MAS_LOOP_VLLM_CONTROL_SOCKET", raising=False)
    return state


def test_reference_count_and_last_release_sleeps(arbiter_env):
    lock, first = vllm_arbiter.acquire_transition("actor")
    assert first
    vllm_arbiter.commit_first_acquire(lock, "actor")

    lock2, first2 = vllm_arbiter.acquire_transition("actor")
    assert lock2 is None and not first2
    assert vllm_arbiter.read_state()["count"] == 2

    release, count, should_sleep = vllm_arbiter.acquire_release_transition("actor")
    assert count == 1 and not should_sleep
    vllm_arbiter.finish_release(release, "actor", slept=False)

    release, count, should_sleep = vllm_arbiter.acquire_release_transition("actor")
    assert count == 0 and should_sleep
    vllm_arbiter.finish_release(release, "actor", slept=True)
    assert vllm_arbiter.read_state()["owner"] == ""


def test_different_owner_waits_for_acknowledged_sleep(arbiter_env):
    lock, _ = vllm_arbiter.acquire_transition("actor")
    vllm_arbiter.commit_first_acquire(lock, "actor")
    entered = threading.Event()

    def waiter():
        other_lock, first = vllm_arbiter.acquire_transition("subagent")
        assert first
        vllm_arbiter.commit_first_acquire(other_lock, "subagent")
        entered.set()

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    time.sleep(0.05)
    assert not entered.is_set()

    release, _, should_sleep = vllm_arbiter.acquire_release_transition("actor")
    assert should_sleep
    # While transition lock is held, the next owner still cannot enter.
    time.sleep(0.05)
    assert not entered.is_set()
    vllm_arbiter.finish_release(release, "actor", slept=True)
    assert entered.wait(1)

    release, _, should_sleep = vllm_arbiter.acquire_release_transition("subagent")
    assert should_sleep
    vllm_arbiter.finish_release(release, "subagent", slept=True)
    thread.join(1)


def test_local_client_wakes_once_and_sleeps_after_last_request(arbiter_env, monkeypatch):
    client = LocalChatClient(base_url="http://127.0.0.1:9999/v1", model="peer", manage_sleep=True)
    sleeping = {"value": True}
    calls = []

    monkeypatch.setattr(client, "is_sleeping", lambda: sleeping["value"])

    def wake():
        calls.append("wake")
        sleeping["value"] = False

    def sleep(level=None):
        calls.append(f"sleep:{level}")
        sleeping["value"] = True

    monkeypatch.setattr(client, "wake_up", wake)
    monkeypatch.setattr(client, "sleep", sleep)

    with client.awake_session():
        with client.awake_session():
            assert calls == ["wake"]
        assert calls == ["wake"]
    assert calls == ["wake", "sleep:1"]
    state = json.loads(arbiter_env.read_text())
    assert state["owner"] == "" and state["count"] == 0


def test_last_release_waits_for_dp_quiescence_before_sleep(arbiter_env, monkeypatch):
    monkeypatch.setenv("MAS_LOOP_VLLM_DP_QUIESCE_SEC", "2.5")
    client = LocalChatClient(base_url="http://127.0.0.1:9999/v1", model="judge", manage_sleep=True)
    sleeping = {"value": True}
    calls = []
    monkeypatch.setattr(client, "is_sleeping", lambda: sleeping["value"])
    monkeypatch.setattr(client, "wake_up", lambda: (calls.append("wake"), sleeping.__setitem__("value", False)))

    def sleep(level=None):
        calls.append(f"sleep:{level}")
        sleeping["value"] = True

    monkeypatch.setattr(client, "sleep", sleep)
    monkeypatch.setattr("mas_loop_rl.local_client.time.sleep", lambda seconds: calls.append(f"dwell:{seconds}"))

    with client.awake_session():
        pass

    assert calls == ["wake", "dwell:2.5", "sleep:1"]


def test_async_local_client_wakes_once_and_last_release_sleeps(arbiter_env, monkeypatch):
    client = LocalChatClient(base_url="http://127.0.0.1:9999/v1", model="peer", manage_sleep=True)
    sleeping = {"value": True}
    calls = []
    monkeypatch.setattr(client, "is_sleeping", lambda: sleeping["value"])

    def wake():
        calls.append("wake")
        sleeping["value"] = False

    def sleep(level=None):
        calls.append(f"sleep:{level}")
        sleeping["value"] = True

    monkeypatch.setattr(client, "wake_up", wake)
    monkeypatch.setattr(client, "sleep", sleep)

    async def scenario():
        entered = 0
        all_entered = asyncio.Event()
        release = asyncio.Event()

        async def request():
            nonlocal entered
            async with client.async_awake_session():
                entered += 1
                if entered == 4:
                    all_entered.set()
                await release.wait()

        tasks = [asyncio.create_task(request()) for _ in range(4)]
        await asyncio.wait_for(all_entered.wait(), timeout=2)
        assert calls == ["wake"]
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())
    assert calls == ["wake", "sleep:1"]
    state = json.loads(arbiter_env.read_text())
    assert state["owner"] == "" and state["count"] == 0


def test_async_local_client_cancellation_releases_lease(arbiter_env, monkeypatch):
    client = LocalChatClient(base_url="http://127.0.0.1:9999/v1", model="peer", manage_sleep=True)
    sleeping = {"value": True}
    monkeypatch.setattr(client, "is_sleeping", lambda: sleeping["value"])
    monkeypatch.setattr(client, "wake_up", lambda: sleeping.__setitem__("value", False))
    monkeypatch.setattr(client, "sleep", lambda level=None: sleeping.__setitem__("value", True))

    async def scenario():
        entered = asyncio.Event()

        async def request():
            async with client.async_awake_session():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(request())
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    state = json.loads(arbiter_env.read_text())
    assert state["owner"] == "" and state["count"] == 0


def test_wait_until_idle_returns_immediately_for_acknowledged_idle(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "/tmp/test-control.sock")
    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "1")
    calls = []

    def fake_rpc(op, *, timeout_sec=None, **payload):
        calls.append((op, timeout_sec, payload))
        return {"owner": "", "active": 0, "status": "idle", "phase_id": 18}

    monkeypatch.setattr(vllm_arbiter, "_rpc", fake_rpc)
    state = vllm_arbiter.wait_until_idle(timeout_sec=1, poll_sec=0.01)
    assert state["phase_id"] == 18
    assert len(calls) == 1


def test_wait_until_idle_waits_for_active_phase(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "/tmp/test-control.sock")
    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "1")
    states = iter([
        {"owner": "judge", "active": 2, "status": "ready"},
        {"owner": "judge", "active": 0, "status": "grace"},
        {"owner": "", "active": 0, "status": "idle"},
    ])
    calls = 0

    def fake_rpc(op, *, timeout_sec=None, **payload):
        nonlocal calls
        calls += 1
        return next(states)

    monkeypatch.setattr(vllm_arbiter, "_rpc", fake_rpc)
    vllm_arbiter.wait_until_idle(timeout_sec=1, poll_sec=0.001)
    assert calls == 3


def test_wait_until_idle_does_not_accept_sleeping_with_zero_active(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "/tmp/test-control.sock")
    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "1")
    states = iter([
        {"owner": "judge", "active": 0, "status": "sleeping"},
        {"owner": "", "active": 0, "status": "idle"},
    ])
    calls = 0

    def fake_rpc(op, *, timeout_sec=None, **payload):
        nonlocal calls
        calls += 1
        return next(states)

    monkeypatch.setattr(vllm_arbiter, "_rpc", fake_rpc)
    vllm_arbiter.wait_until_idle(timeout_sec=1, poll_sec=0.001)
    assert calls == 2


def test_wait_until_idle_times_out_fail_closed(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "/tmp/test-control.sock")
    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "1")
    monkeypatch.setattr(
        vllm_arbiter,
        "_rpc",
        lambda op, *, timeout_sec=None, **payload: {
            "owner": "judge",
            "active": 0,
            "status": "sleeping",
        },
    )
    with pytest.raises(TimeoutError, match="did not become idle"):
        vllm_arbiter.wait_until_idle(timeout_sec=0.02, poll_sec=0.002)


def test_wait_until_idle_propagates_coordinator_rpc_failure(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "/tmp/test-control.sock")
    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "1")

    def fail_rpc(op, *, timeout_sec=None, **payload):
        raise RuntimeError("coordinator unavailable")

    monkeypatch.setattr(vllm_arbiter, "_rpc", fail_rpc)
    with pytest.raises(RuntimeError, match="coordinator unavailable"):
        vllm_arbiter.wait_until_idle(timeout_sec=1, poll_sec=0.01)


def test_external_idle_barrier_enablement_is_mas_only(monkeypatch):
    monkeypatch.delenv("MAS_LOOP_EXCLUSIVE_VLLM", raising=False)
    monkeypatch.delenv("MAS_LOOP_VLLM_CONTROL_SOCKET", raising=False)
    assert not vllm_arbiter.wait_for_external_vllm_idle_enabled()

    monkeypatch.setenv("MAS_LOOP_EXCLUSIVE_VLLM", "1")
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "/tmp/test-control.sock")
    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "1")
    assert vllm_arbiter.wait_for_external_vllm_idle_enabled()

    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "0")
    assert not vllm_arbiter.wait_for_external_vllm_idle_enabled()


def test_async_acquire_uses_dedicated_long_timeout(monkeypatch):
    captured = {}

    async def fake_open(path):
        class Reader:
            async def readline(self):
                return b'{"ok":true,"result":{"first":false}}\n'
        class Writer:
            def write(self, data):
                captured["request"] = data
            async def drain(self):
                return None
            def close(self):
                return None
            async def wait_closed(self):
                return None
        return Reader(), Writer()

    real_wait_for = asyncio.wait_for

    async def recording_wait_for(awaitable, timeout):
        captured.setdefault("timeouts", []).append(timeout)
        return await real_wait_for(awaitable, timeout=1)

    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", "/tmp/test-control.sock")
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_TIMEOUT_SEC", "7")
    monkeypatch.setenv("MAS_LOOP_VLLM_ACQUIRE_TIMEOUT_SEC", "19")
    monkeypatch.setattr(asyncio, "open_unix_connection", fake_open)
    monkeypatch.setattr(asyncio, "wait_for", recording_wait_for)

    result = asyncio.run(vllm_arbiter._async_rpc("acquire", owner="judge", max_active=2))
    assert result == {"first": False}
    assert captured["timeouts"] == [19.0, 19.0, 19.0]
