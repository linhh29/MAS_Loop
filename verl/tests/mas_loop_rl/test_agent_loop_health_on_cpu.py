import pytest

from verl.trainer.ppo.v1 import agent_loop_tq


class _FakeHealthMethod:
    def __init__(self, worker):
        self._worker = worker

    def remote(self):
        self._worker.calls += 1
        return f"probe-{self._worker.name}-{self._worker.calls}"


class _FakeWorker:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.health = _FakeHealthMethod(self)


def _manager(*workers):
    manager = object.__new__(agent_loop_tq.AgentLoopManagerTQ)
    manager.agent_loop_workers = list(workers)
    return manager


def test_health_timeout_is_busy_and_reuses_pending_probe(monkeypatch):
    worker = _FakeWorker("a")
    manager = _manager(worker)
    waits = []

    def fake_wait(refs, *, num_returns, timeout):
        waits.append((list(refs), num_returns, timeout))
        return [], list(refs)

    monkeypatch.setattr(agent_loop_tq.ray, "wait", fake_wait)
    monkeypatch.setattr(
        agent_loop_tq.ray,
        "get",
        lambda _refs: pytest.fail("ray.get must not be called for pending probes"),
    )
    monkeypatch.setenv("MAS_LOOP_AGENT_LOOP_HEALTH_TIMEOUT_SEC", "0.01")

    manager.assert_healthy()
    manager.assert_healthy()

    assert worker.calls == 1
    assert waits == [(["probe-a-1"], 1, 0.01), (["probe-a-1"], 1, 0.01)]


def test_successful_probe_is_replaced_on_next_check(monkeypatch):
    worker = _FakeWorker("a")
    manager = _manager(worker)
    received = []

    monkeypatch.setattr(
        agent_loop_tq.ray,
        "wait",
        lambda refs, *, num_returns, timeout: (list(refs), []),
    )
    monkeypatch.setattr(agent_loop_tq.ray, "get", lambda refs: received.append(list(refs)))

    manager.assert_healthy()
    manager.assert_healthy()

    assert worker.calls == 2
    assert received == [["probe-a-1"], ["probe-a-2"]]


def test_completed_probe_failure_aborts_optimizer_step(monkeypatch):
    worker = _FakeWorker("a")
    manager = _manager(worker)

    monkeypatch.setattr(
        agent_loop_tq.ray,
        "wait",
        lambda refs, *, num_returns, timeout: (list(refs), []),
    )

    def fake_get(_refs):
        raise RuntimeError("simulated completed actor failure")

    monkeypatch.setattr(agent_loop_tq.ray, "get", fake_get)

    with pytest.raises(RuntimeError, match="health check failed") as exc_info:
        manager.assert_healthy()

    assert "simulated completed actor failure" in str(exc_info.value.__cause__)


def test_health_timeout_must_be_positive(monkeypatch):
    manager = _manager(_FakeWorker("a"))
    monkeypatch.setenv("MAS_LOOP_AGENT_LOOP_HEALTH_TIMEOUT_SEC", "0")

    with pytest.raises(ValueError, match="must be positive"):
        manager.assert_healthy()
