from __future__ import annotations

import json
from pathlib import Path

from mas_loop_rl.build_editor_data_from_logs import build_rows


def _source_record(index: int, *, group: str = "task-group") -> dict:
    source_mas = {"agents": [{"id": f"s{index}", "role": "solve"}], "aggregation": f"s{index}"}
    trajectory = {"final_output": str(index), "metadata": {"score": 0.0}}
    prompt = [{"role": "system", "content": "edit"}, {"role": "user", "content": "source"}]
    pair_id = f"pair-{index}"
    candidate = {
        "candidate_id": f"{pair_id}:refiner",
        "candidate_index": index,
        "editor_group_uid": f"editor-task:{group}:step-1",
        "prompt": prompt,
        "output": json.dumps({"diagnosis": {}, "patch": [{"op": "accept"}]}),
        "retry": {},
        "patch": {"diagnosis": {}, "patch": [{"op": "accept"}]},
        "updated_mas": source_mas,
        "updated_trajectory": trajectory,
        "upd_score": 0.0,
        "score_delta": 0.0,
        "reward": 0.0,
        "outcome": "accepted_incorrect",
        "trainable": True,
        "updated_valid": True,
    }
    return {
        "schema_version": "mas_loop_source_rollout_v6",
        "reward_design": "source_minus_refiner_exploit_v1",
        "id": f"record-{index}",
        "pair_id": pair_id,
        "role": "orchestra",
        "loop_id": f"{group}:orchestra",
        "grpo_group_uid": group,
        "rollout_sample_index": index,
        "optimizer_step": 1,
        "task": "task",
        "expected": "answer",
        "constraints": {},
        "src_mas": source_mas,
        "src_trajectory": trajectory,
        "src_score": 0.0,
        "peer_prompt": prompt,
        "peer_output": candidate["output"],
        "editor_score": 0.0,
        "editor_outcome": candidate["outcome"],
        "upd_score": 0.0,
        "score_delta": 0.0,
        "editor_candidates": [candidate],
        "editor_trainable": True,
    }


def test_v6_one_candidate_per_source_is_task_conditioned_not_same_source(tmp_path: Path) -> None:
    records = [_source_record(index) for index in range(8)]
    positive = records[1]["editor_candidates"][0]
    positive.update({
        "output": json.dumps({"diagnosis": {}, "patch": [{"op": "add_verifier"}]}),
        "patch": {"diagnosis": {}, "patch": [{"op": "add_verifier"}]},
        "upd_score": 1.0,
        "score_delta": 1.0,
        "reward": 1.0,
        "outcome": "improved",
    })
    path = tmp_path / "v6.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    rows, rejected = build_rows([path], group_size=8, accept_regret_weight=1.0)

    assert not rejected
    assert {row["extra_info"]["editor_group_semantics"] for row in rows} == {"task_conditioned_source_pairs"}
    assert all(row["extra_info"]["pairwise_task_group"] for row in rows)
    assert all(not row["extra_info"]["same_source_candidate"] for row in rows)
    assert len({row["extra_info"]["source_pair_id"] for row in rows}) == 8
    assert rows[0]["extra_info"]["accept_regret"] == 0.0


def test_accept_regret_applies_only_to_true_same_source_hypotheses(tmp_path: Path) -> None:
    record = _source_record(0)
    candidates = []
    for index in range(8):
        accepted = index == 0
        reward = 0.0 if accepted else (1.0 if index == 1 else 0.0)
        patch = {"diagnosis": {}, "patch": [{"op": "accept"} if accepted else {"op": "add_verifier"}]}
        candidates.append({
            "candidate_id": f"same-source:candidate-{index}",
            "candidate_index": index,
            "editor_group_uid": "editor-source:one-source",
            "prompt": record["peer_prompt"],
            "output": json.dumps(patch),
            "retry": {},
            "patch": patch,
            "updated_mas": record["src_mas"],
            "updated_trajectory": record["src_trajectory"],
            "upd_score": reward,
            "score_delta": reward,
            "reward": reward,
            "outcome": "accepted_incorrect" if accepted else ("improved" if reward > 0 else "no_gain"),
            "trainable": True,
            "updated_valid": True,
        })
    record["editor_candidates"] = candidates
    path = tmp_path / "same-source.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    rows, rejected = build_rows([path], group_size=8, accept_regret_weight=1.0)

    assert not rejected
    assert {row["extra_info"]["editor_group_semantics"] for row in rows} == {"same_source_repair_hypotheses"}
    assert all(row["extra_info"]["same_source_candidate"] for row in rows)
    assert all(not row["extra_info"]["pairwise_task_group"] for row in rows)
    assert len({row["extra_info"]["source_pair_id"] for row in rows}) == 1
    assert rows[0]["extra_info"]["accept_regret"] == 1.0
    assert rows[0]["extra_info"]["replay_editor_reward"] == -1.0
