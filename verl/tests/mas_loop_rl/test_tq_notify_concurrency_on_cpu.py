import asyncio

from transfer_queue.storage.managers import base as storage_base


class _ConcreteStorageManager(storage_base.StorageManager):
    async def put_data(self, data, metadata, data_parser=None):
        raise NotImplementedError

    async def get_data(self, metadata):
        raise NotImplementedError

    async def clear_data(self, metadata):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


async def _exercise_limit(monkeypatch):
    monkeypatch.setattr(storage_base, "TQ_DATA_UPDATE_MAX_CONCURRENCY", 3)
    manager = object.__new__(_ConcreteStorageManager)
    manager.storage_manager_id = "test-storage"
    manager._notify_semaphore = None
    state = {"active": 0, "peak": 0}

    async def fake_transaction(_request):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1

    manager._notify_and_wait_limited = fake_transaction
    await asyncio.gather(*(manager._notify_and_wait([i]) for i in range(30)))
    assert state["peak"] == 3
    assert state["active"] == 0


def test_notify_socket_fanout_is_bounded(monkeypatch):
    asyncio.run(_exercise_limit(monkeypatch))
