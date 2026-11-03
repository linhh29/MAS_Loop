from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from mas_loop_rl import vllm_arbiter


def _start(tmp_path: Path, monkeypatch, grace: float = 0.12):
    socket = tmp_path / "control.sock"
    state = tmp_path / "state.json"
    events = tmp_path / "events.jsonl"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parents[2]), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(
        [sys.executable, "-m", "mas_loop_rl.phase_coordinator", "--socket", str(socket),
         "--state", str(state), "--events", str(events), "--grace-sec", str(grace),
         "--state-interval-sec", "0.05"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.time() + 5
    while time.time() < deadline and not socket.exists():
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read())
        time.sleep(0.01)
    assert socket.exists()
    monkeypatch.setenv("MAS_LOOP_VLLM_CONTROL_SOCKET", str(socket))
    monkeypatch.setenv("MAS_LOOP_PHASE_LEASE", "1")
    return proc, events


def _stop(proc: subprocess.Popen):
    proc.terminate()
    proc.wait(timeout=5)


def test_local_coordinator_coalesces_same_owner_zero_active_gap(tmp_path, monkeypatch):
    proc, events = _start(tmp_path, monkeypatch)
    try:
        first_token, first = vllm_arbiter.acquire_transition("subagent")
        assert first
        vllm_arbiter.commit_first_acquire(first_token, "subagent")

        release_result = {}
        def last_release():
            token, count, should_sleep = vllm_arbiter.acquire_release_transition("subagent")
            release_result.update(token=token, count=count, should_sleep=should_sleep)

        thread = threading.Thread(target=last_release)
        thread.start()
        time.sleep(0.03)  # inside the phase grace window
        token2, first2 = vllm_arbiter.acquire_transition("subagent")
        assert token2 is None and not first2
        thread.join(timeout=2)
        assert release_result["should_sleep"] is False

        token3, count3, sleep3 = vllm_arbiter.acquire_release_transition("subagent")
        assert count3 == 0 and sleep3
        vllm_arbiter.finish_release(token3, "subagent", slept=True)
        assert vllm_arbiter.read_state()["owner"] == ""
    finally:
        _stop(proc)
    assert "phase_grace_reused" in events.read_text()


def test_local_coordinator_blocks_next_owner_until_sleep_ack(tmp_path, monkeypatch):
    proc, _ = _start(tmp_path, monkeypatch, grace=0.02)
    try:
        token, first = vllm_arbiter.acquire_transition("actor")
        assert first
        vllm_arbiter.commit_first_acquire(token, "actor")
        entered = threading.Event()

        def waiter():
            other, other_first = vllm_arbiter.acquire_transition("editor")
            assert other_first
            vllm_arbiter.commit_first_acquire(other, "editor")
            entered.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        release, _, should_sleep = vllm_arbiter.acquire_release_transition("actor")
        assert should_sleep and not entered.is_set()
        time.sleep(0.03)
        assert not entered.is_set()
        vllm_arbiter.finish_release(release, "actor", slept=True)
        assert entered.wait(2)
        release2, _, sleep2 = vllm_arbiter.acquire_release_transition("editor")
        assert sleep2
        vllm_arbiter.finish_release(release2, "editor", slept=True)
        thread.join(timeout=2)
    finally:
        _stop(proc)


def test_async_admission_over_capacity_can_release_without_threadpool_deadlock(tmp_path, monkeypatch):
    """A waiting acquire must not prevent active requests from releasing.

    This is the small CPU analogue of batch>128 with rollout_n=8: requests
    above MAS_LOOP_VLLM_MAX_ACTIVE wait, while completions must still release
    admission slots.  The former to_thread implementation could starve those
    releases when the default executor filled with waiting acquires.
    """

    proc, _ = _start(tmp_path, monkeypatch, grace=0.01)

    async def scenario():
        first_token, first = await vllm_arbiter.async_acquire_transition("actor", max_active=2)
        assert first
        await vllm_arbiter.async_commit_first_acquire(first_token, "actor")
        second_token, second_first = await vllm_arbiter.async_acquire_transition("actor", max_active=2)
        assert second_token is None and not second_first

        waiting = asyncio.create_task(vllm_arbiter.async_acquire_transition("actor", max_active=2))
        await asyncio.sleep(0.03)
        assert not waiting.done()

        release, count, should_sleep = await vllm_arbiter.async_acquire_release_transition("actor")
        assert release is None and count == 1 and not should_sleep
        third_token, third_first = await asyncio.wait_for(waiting, timeout=1)
        assert third_token is None and not third_first

        release2, count2, sleep2 = await vllm_arbiter.async_acquire_release_transition("actor")
        assert release2 is None and count2 == 1 and not sleep2
        release3, count3, sleep3 = await vllm_arbiter.async_acquire_release_transition("actor")
        assert count3 == 0 and sleep3
        await vllm_arbiter.async_finish_release(release3, "actor", slept=True)

    try:
        asyncio.run(scenario())
    finally:
        _stop(proc)


def test_state_file_refreshes_during_shared_phase_without_new_events(tmp_path, monkeypatch):
    proc, _ = _start(tmp_path, monkeypatch, grace=0.02)
    state_path = tmp_path / "state.json"
    try:
        token, first = vllm_arbiter.acquire_transition("subagent", max_active=4)
        assert first
        vllm_arbiter.commit_first_acquire(token, "subagent")
        token2, first2 = vllm_arbiter.acquire_transition("subagent", max_active=4)
        assert token2 is None and not first2

        # The second shared acquire emits no event.  The periodic publisher must
        # still expose active=2 rather than leaving the wake_complete snapshot.
        deadline = time.time() + 2
        payload = {}
        while time.time() < deadline:
            if state_path.exists():
                payload = __import__("json").loads(state_path.read_text())
                if payload.get("active") == 2:
                    break
            time.sleep(0.02)
        assert payload["active"] == 2
        assert time.time() - payload["updated"] < 1

        release, count, should_sleep = vllm_arbiter.acquire_release_transition("subagent")
        assert release is None and count == 1 and not should_sleep
        release2, count2, should_sleep2 = vllm_arbiter.acquire_release_transition("subagent")
        assert count2 == 0 and should_sleep2
        vllm_arbiter.finish_release(release2, "subagent", slept=True)
    finally:
        _stop(proc)


def test_disconnected_queued_acquire_never_becomes_active(tmp_path, monkeypatch):
    """A timed-out/cancelled waiter must not become an orphaned lease later."""

    import json
    import socket

    proc, _ = _start(tmp_path, monkeypatch, grace=0.01)
    socket_path = str(tmp_path / "control.sock")
    try:
        token, first = vllm_arbiter.acquire_transition("subagent", max_active=1)
        assert first
        vllm_arbiter.commit_first_acquire(token, "subagent")

        abandoned = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        abandoned.settimeout(1)
        abandoned.connect(socket_path)
        abandoned.sendall(
            json.dumps({"op": "acquire", "owner": "judge", "max_active": 1}).encode("utf-8") + b"\n"
        )
        time.sleep(0.05)
        abandoned.close()

        # Give the server-side EOF watcher time to cancel the queued acquire.
        deadline = time.time() + 2
        while time.time() < deadline:
            state = vllm_arbiter.read_state()
            if not (state.get("waiters") or {}).get("judge"):
                break
            time.sleep(0.02)
        assert not (vllm_arbiter.read_state().get("waiters") or {}).get("judge")

        release, _, should_sleep = vllm_arbiter.acquire_release_transition("subagent")
        assert should_sleep
        vllm_arbiter.finish_release(release, "subagent", slept=True)
        time.sleep(0.05)

        state = vllm_arbiter.read_state()
        assert state["owner"] == ""
        assert state["active"] == 0
        assert state["phase_requests"] == 1
    finally:
        _stop(proc)
