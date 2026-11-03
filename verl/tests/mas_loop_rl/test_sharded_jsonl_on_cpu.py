from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from mas_loop_rl.build_editor_data_from_logs import build_rows
from mas_loop_rl.sharded_jsonl import append_jsonl_shard, close_jsonl_writers, iter_jsonl, resolve_jsonl_inputs


def _write_records(root: str, worker: int, count: int) -> None:
    for index in range(count):
        append_jsonl_shard(
            Path(root) / "canonical_rollouts",
            {
                "id": f"record-{worker}-{index}",
                "pair_id": f"pair-{worker}-{index}",
                "worker": worker,
                "index": index,
                "payload": "x" * 4096,
            },
            shard_prefix="orchestra",
        )


def _editor_record(index: int) -> dict:
    return {
        "id": f"record-{index}",
        "pair_id": f"pair-{index}",
        "role": "orchestra",
        "loop_id": "task-1:orchestra",
        "grpo_group_uid": "task-1",
        "rollout_sample_index": index,
        "time": float(index),
        "task": "2+2?",
        "expected": "4",
        "constraints": {},
        "src_mas": {"agents": [{"id": "solver", "role": "solve"}], "aggregation": "solver"},
        "src_trajectory": {"final_output": "4", "metadata": {"score": 1.0}},
        "peer_prompt": [{"role": "user", "content": "improve"}],
        "peer_output": "{}",
        "editor_score": 0.0,
        "src_score": 1.0,
        "upd_score": 1.0,
        "score_delta": 0.0,
        "editor_outcome": "tie",
    }


def test_process_shards_preserve_every_large_record_without_global_lock(tmp_path: Path):
    workers, count = 4, 40
    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=_write_records, args=(str(tmp_path), worker, count)) for worker in range(workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    shard_dir = tmp_path / "canonical_rollouts"
    shards = resolve_jsonl_inputs([shard_dir])
    records = list(iter_jsonl([shard_dir]))
    assert len(shards) == workers
    assert len(records) == workers * count
    assert len({record["pair_id"] for record in records}) == workers * count
    assert all(len(record["payload"]) == 4096 for record in records)


def test_editor_builder_reads_directory_of_canonical_shards(tmp_path: Path):
    shard_dir = tmp_path / "canonical_rollouts"
    for index in range(8):
        append_jsonl_shard(shard_dir, _editor_record(index), shard_prefix="orchestra")

    rows, rejected = build_rows([shard_dir], group_size=8)
    assert len(rows) == 8
    assert rejected == []
    assert {row["extra_info"]["pair_id"] for row in rows} == {f"pair-{i}" for i in range(8)}


def test_close_jsonl_writers_allows_immediate_readback_and_reopen(tmp_path: Path):
    stream_dir = tmp_path / "judge_results"
    shard = append_jsonl_shard(stream_dir, {"id": 1}, shard_prefix="judge")

    assert close_jsonl_writers(stream_dir) == 1
    assert list(iter_jsonl([stream_dir])) == [{"id": 1}]

    reopened = append_jsonl_shard(stream_dir, {"id": 2}, shard_prefix="judge")
    assert reopened == shard
    assert close_jsonl_writers(stream_dir, shard_prefix="judge") == 1
    assert list(iter_jsonl([stream_dir])) == [{"id": 1}, {"id": 2}]
