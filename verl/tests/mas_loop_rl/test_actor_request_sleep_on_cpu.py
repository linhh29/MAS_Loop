from __future__ import annotations

import asyncio
from types import SimpleNamespace

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class _FakeEngine:
    def __init__(self):
        self.levels = []

    async def sleep(self, level: int):
        self.levels.append(level)


def test_request_level_hybrid_sleep_preserves_weights_with_level_one():
    """Per-request MAS switching must never use HYBRID's destructive level 2."""
    server = object.__new__(vLLMHttpServer)
    server.node_rank = 0
    server.config = SimpleNamespace(free_cache_engine=True)
    server.rollout_mode = RolloutMode.HYBRID
    server.engine = _FakeEngine()

    asyncio.run(server.sleep_preserve_weights())

    assert server.engine.levels == [1]


def test_native_hybrid_sleep_still_uses_training_level_two_for_full_weights():
    """Do not change veRL's native sleep->update_weights training contract."""
    server = object.__new__(vLLMHttpServer)
    server.node_rank = 0
    server.config = SimpleNamespace(free_cache_engine=True, mtp=None)
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.rollout_mode = RolloutMode.HYBRID
    server.engine = _FakeEngine()

    asyncio.run(server.sleep())

    assert server.engine.levels == [2]
