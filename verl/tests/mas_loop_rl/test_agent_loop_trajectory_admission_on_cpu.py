from __future__ import annotations

import asyncio

import pytest

from verl.experimental.agent_loop.agent_loop import (
    MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_ENV,
    _map_indexed_bounded,
    get_agent_loop_max_outstanding_per_worker,
)


def test_trajectory_admission_env_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_ENV, raising=False)
    assert get_agent_loop_max_outstanding_per_worker() == 0

    monkeypatch.setenv(MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_ENV, "128")
    assert get_agent_loop_max_outstanding_per_worker() == 128


@pytest.mark.parametrize("value", ["-1", "not-an-int"])
def test_trajectory_admission_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_ENV, value)
    with pytest.raises(ValueError, match=MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_ENV):
        get_agent_loop_max_outstanding_per_worker()


def test_bounded_map_limits_live_trajectories_and_preserves_order() -> None:
    async def exercise() -> tuple[list[int], int]:
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def run_one(index: int) -> int:
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.001 * (1 + index % 3))
            async with lock:
                active -= 1
            return index * 10

        outputs = await _map_indexed_bounded(23, 4, run_one)
        return outputs, peak

    outputs, peak = asyncio.run(exercise())
    assert outputs == [index * 10 for index in range(23)]
    assert peak == 4


def test_bounded_map_cancels_sibling_consumers_on_failure() -> None:
    async def exercise() -> None:
        async def run_one(index: int) -> int:
            if index == 3:
                raise RuntimeError("boom")
            await asyncio.sleep(0.05)
            return index

        with pytest.raises(RuntimeError, match="boom"):
            await _map_indexed_bounded(100, 8, run_one)

    asyncio.run(exercise())
