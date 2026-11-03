from __future__ import annotations

from mas_loop_rl import vllm_arbiter
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class _CheckpointManager:
    def __init__(self, calls):
        self.calls = calls

    def sleep_replicas(self):
        self.calls.append("integrated_sleep")


def _trainer(calls):
    trainer = object.__new__(PPOTrainerSync)
    trainer.checkpoint_manager = _CheckpointManager(calls)
    return trainer


def test_on_sample_end_skips_external_barrier_outside_mas(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_arbiter, "wait_for_external_vllm_idle_enabled", lambda: False)
    monkeypatch.setattr(vllm_arbiter, "wait_until_idle", lambda: calls.append("external_idle"))

    _trainer(calls).on_sample_end()
    assert calls == ["integrated_sleep"]


def test_on_sample_end_waits_after_integrated_sleep_in_mas(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_arbiter, "wait_for_external_vllm_idle_enabled", lambda: True)
    monkeypatch.setattr(vllm_arbiter, "wait_until_idle", lambda: calls.append("external_idle"))

    _trainer(calls).on_sample_end()
    assert calls == ["integrated_sleep", "external_idle"]
