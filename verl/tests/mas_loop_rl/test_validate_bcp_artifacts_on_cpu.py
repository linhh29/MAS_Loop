import json
from pathlib import Path

import pytest

from mas_loop_rl.validate_bcp_artifacts import (
    ORCHESTRA_ONLY_RL_METRICS,
    REQUIRED_RL_METRICS,
    validate_phase,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _metrics(phase: Path, *, orchestra: bool = False) -> None:
    required = set(REQUIRED_RL_METRICS)
    if orchestra:
        required.update(ORCHESTRA_ONLY_RL_METRICS)
    _write_jsonl(
        phase / "metrics" / "training_metrics.jsonl",
        [{"global_step": 1, "metrics": {key: 0.0 for key in required}}],
    )


def _judge(score: float = 1.0) -> dict:
    correct = "yes" if score == 1.0 else "no"
    return {
        "extracted_final_answer": "answer" if correct == "yes" else "wrong",
        "reasoning": "matches" if correct == "yes" else "does not match",
        "correct": correct,
        "confidence": 100.0,
        "parse_error": False,
        "score": score,
        "model": "judge",
        "max_tokens": 4096,
    }


def test_validate_complete_orchestra_artifacts(tmp_path: Path) -> None:
    phase = tmp_path / "round_1" / "orchestra"
    _metrics(phase, orchestra=True)
    pair = {
        "pair_id": "p1",
        "structured_pair": {
            "pair_id": "p1",
            "source": {"judge": _judge(1.0)},
            "updated": {"judge": _judge(0.0)},
            "editor": {
                "outcome": "error",
                "retry": {"errors": ["invalid JSON", "retry exhausted"]},
            },
            "errors": ["peer_edit_failed:LocalChatError"],
        },
    }
    _write_jsonl(phase / "rollouts" / "canonical_rollouts" / "orchestra-1.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts" / "judge_results" / "judge-1.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": _judge(1.0)},
            {"pair_id": "p1", "candidate": "updated", "judge": _judge(0.0)},
        ],
    )
    result = validate_phase(phase, "orchestra")
    assert result["valid"] is True
    assert result["rollouts"]["unique_pair_ids"] == 1
    assert result["rollouts"]["judged_pair_candidates"] == 2
    assert result["rollouts"]["pair_error_rows"] == 1
    assert result["rollouts"]["pair_error_entries"] == 1
    assert result["rollouts"]["editor_error_rows"] == 1
    assert result["rollouts"]["editor_retry_error_entries"] == 2
    assert result["rollouts"]["judge_negative_one_candidates"] == 0


def test_orchestra_validation_fails_when_any_candidate_lacks_judge(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    pair = {
        "pair_id": "p1",
        "structured_pair": {
            "pair_id": "p1",
            "source": {"judge": _judge()},
            "updated": {"judge": None},
        },
    }
    _write_jsonl(phase / "rollouts" / "canonical_rollouts" / "orchestra-1.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts" / "judge_results" / "judge-1.jsonl",
        [{"pair_id": "p1", "candidate": "source", "judge": _judge()}],
    )
    with pytest.raises(RuntimeError, match="lack valid Qwen judge metadata"):
        validate_phase(phase, "orchestra")


def test_validation_fails_when_required_rl_metric_is_missing(tmp_path: Path) -> None:
    phase = tmp_path / "editor"
    metrics = {key: 0.0 for key in REQUIRED_RL_METRICS}
    metrics.pop("actor/entropy")
    _write_jsonl(phase / "metrics" / "training_metrics.jsonl", [{"global_step": 1, "metrics": metrics}])
    _write_jsonl(phase / "rollouts" / "canonical_rollouts" / "editor-1.jsonl", [{"pair_id": "p1"}])
    with pytest.raises(RuntimeError, match="actor/entropy"):
        validate_phase(phase, "editor")


def test_validate_complete_editor_artifacts(tmp_path: Path) -> None:
    phase = tmp_path / "editor"
    _metrics(phase)
    _write_jsonl(
        phase / "rollouts" / "canonical_rollouts" / "editor-1.jsonl",
        [{"pair_id": "p1"}, {"pair_id": "p2"}],
    )
    result = validate_phase(phase, "editor")
    assert result["rollouts"] == {"canonical_rollout_rows": 2, "unique_pair_ids": 2}


def test_validate_editor_replay_consumed_artifacts(tmp_path: Path) -> None:
    phase = tmp_path / "editor"
    _metrics(phase)
    _write_jsonl(
        phase / "rollouts" / "editor_replay_consumed" / "editor-replay-1.jsonl",
        [{"pair_id": "p1"}, {"pair_id": "p2"}],
    )
    result = validate_phase(phase, "editor")
    assert result["rollouts"] == {"canonical_rollout_rows": 2, "unique_pair_ids": 2}


def test_orchestra_validation_requires_rollout_correlation_metrics(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase)
    pair = {
        "pair_id": "p1",
        "structured_pair": {
            "pair_id": "p1",
            "source": {"judge": _judge()},
            "updated": {"judge": _judge()},
        },
    }
    _write_jsonl(phase / "rollouts" / "canonical_rollouts" / "orchestra-1.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts" / "judge_results" / "judge-1.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": _judge()},
            {"pair_id": "p1", "candidate": "updated", "judge": _judge()},
        ],
    )
    with pytest.raises(RuntimeError, match="rollout_corr"):
        validate_phase(phase, "orchestra")


def test_editor_validation_does_not_require_unavailable_rollout_correlation_metrics(
    tmp_path: Path,
) -> None:
    phase = tmp_path / "editor"
    _metrics(phase)
    _write_jsonl(
        phase / "rollouts" / "canonical_rollouts" / "editor-1.jsonl",
        [{"pair_id": "p1"}],
    )
    result = validate_phase(phase, "editor")
    assert not (ORCHESTRA_ONLY_RL_METRICS & set(result["metrics"]["required_metric_keys"]))


def test_orchestra_validation_rejects_parse_error_sentinel(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    invalid = _judge(0.0)
    invalid.update({"parse_error": True, "score": -1.0})
    pair = {
        "pair_id": "p1",
        "structured_pair": {
            "pair_id": "p1",
            "source": {"judge": invalid},
            "updated": {"judge": _judge(0.0)},
        },
    }
    _write_jsonl(phase / "rollouts" / "canonical_rollouts" / "orchestra-1.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts" / "judge_results" / "judge-1.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": invalid},
            {"pair_id": "p1", "candidate": "updated", "judge": _judge(0.0)},
        ],
    )
    with pytest.raises(RuntimeError, match="lack valid Qwen judge metadata"):
        validate_phase(phase, "orchestra")


def test_orchestra_validation_accepts_audited_invalid_history_with_valid_replacement(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    invalid = _judge(0.0)
    invalid.update({"parse_error": True, "score": -1.0})
    pair = {
        "pair_id": "p1",
        "structured_pair": {
            "pair_id": "p1",
            "source": {"judge": _judge(1.0)},
            "updated": {"judge": _judge(0.0)},
        },
    }
    _write_jsonl(phase / "rollouts" / "canonical_rollouts" / "orchestra-1.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts" / "judge_results" / "judge-1.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": invalid},
            {"pair_id": "p1", "candidate": "updated", "judge": _judge(0.0)},
        ],
    )
    _write_jsonl(
        phase / "rollouts" / "judge_results" / "judge-backfill.jsonl",
        [{"pair_id": "p1", "candidate": "source", "judge": _judge(1.0)}],
    )
    result = validate_phase(phase, "orchestra")
    rollouts = result["rollouts"]
    assert rollouts["judge_result_rows"] == 3
    assert rollouts["valid_judge_result_rows"] == 2
    assert rollouts["historical_invalid_judge_result_rows"] == 1
    assert rollouts["historical_invalid_judge_keys"] == 1
    assert rollouts["superseded_invalid_judge_keys"] == 1
    assert rollouts["judged_pair_candidates"] == 2


def test_orchestra_validation_rejects_invalid_history_without_valid_replacement(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    invalid = _judge(0.0)
    invalid.update({"parse_error": True, "score": -1.0})
    pair = {
        "pair_id": "p1",
        "structured_pair": {
            "pair_id": "p1",
            "source": {"judge": _judge(1.0)},
            "updated": {"judge": _judge(0.0)},
        },
    }
    _write_jsonl(phase / "rollouts" / "canonical_rollouts" / "orchestra-1.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts" / "judge_results" / "judge-1.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": invalid},
            {"pair_id": "p1", "candidate": "updated", "judge": _judge(0.0)},
        ],
    )
    with pytest.raises(RuntimeError, match="lack persisted valid judge_results"):
        validate_phase(phase, "orchestra")


def test_orchestra_accept_reuses_source_standalone_judge_evidence(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    source = {"mas": {"agents": [{"id": "a"}]}, "trajectory": {"final_output": "x"}, "score": 1.0, "judge": _judge(1.0)}
    pair = {
        "pair_id": "accept-1",
        "structured_pair": {
            "pair_id": "accept-1",
            "source": source,
            "updated": dict(source),
            "editor": {"patch": {"diagnosis": {}, "patch": [{"op": "accept"}]}, "outcome": "accepted_correct"},
        },
    }
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts/judge_results/j.jsonl",
        [{"pair_id": "accept-1", "candidate": "source", "judge": _judge(1.0)}],
    )
    result = validate_phase(phase, "orchestra")
    assert result["rollouts"]["accepted_updated_judge_reuse"] == 1
    assert result["rollouts"]["required_standalone_judge_keys"] == 1


def test_orchestra_accept_rejects_changed_updated_evidence(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    pair = {
        "pair_id": "accept-1",
        "structured_pair": {
            "pair_id": "accept-1",
            "source": {"mas": {"agents": [{"id": "a"}]}, "trajectory": {"final_output": "x"}, "score": 1.0, "judge": _judge(1.0)},
            "updated": {"mas": {"agents": [{"id": "a"}]}, "trajectory": {"final_output": "different"}, "score": 1.0, "judge": _judge(1.0)},
            "editor": {"patch": {"diagnosis": {}, "patch": [{"op": "accept"}]}, "outcome": "accepted_correct"},
        },
    }
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts/judge_results/j.jsonl",
        [{"pair_id": "accept-1", "candidate": "source", "judge": _judge(1.0)}],
    )
    with pytest.raises(RuntimeError, match="accept/no-op changed source evidence"):
        validate_phase(phase, "orchestra")


def _v4_candidate(pair_id: str, index: int, *, outcome: str, judge=None, reward: float = 0.0, output: str = "patch") -> dict:
    trajectory = {"final_output": f"answer-{index}", "metadata": {}}
    if judge is not None:
        trajectory["metadata"]["judge"] = judge
    return {
        "candidate_id": f"{pair_id}:editor-{index}", "candidate_index": index,
        "prompt": [{"role": "user", "content": "repair"}], "output": output,
        "retry": {"raw_responses": [output], "errors": ["bad"] if outcome == "invalid_editor_output" else []},
        "patch": {"diagnosis": {}, "patch": [{"op": "rewrite_role", "agent_id": "a", "role": "solve"}]},
        "updated_mas": {"agents": [{"id": "a", "role": "solve"}]}, "updated_trajectory": trajectory,
        "src_score": 0.0, "upd_score": float(judge["score"]) if judge else 0.0,
        "score_delta": float(judge["score"]) if judge else 0.0, "reward": reward,
        "outcome": outcome, "trainable": True,
        "error": "LocalChatError:bad" if outcome == "invalid_editor_output" else "",
    }


def _complete_v4_pair(pair_id: str = "p1") -> dict:
    source_judge = _judge(0.0)
    updated_judge = _judge(1.0)
    executed = _v4_candidate(pair_id, 0, outcome="improved", judge=updated_judge, reward=1.0)
    malformed = _v4_candidate(pair_id, 1, outcome="invalid_editor_output", reward=-1.0, output="not-json")
    return {
        "pair_id": pair_id,
        "structured_pair": {
            "pair_id": pair_id,
            "source": {"mas": {"agents": [{"id": "a"}]}, "trajectory": {"final_output": "source", "metadata": {"judge": source_judge}}, "score": 0.0, "judge": source_judge},
            "editor_candidates": [executed, malformed],
            "updated": {"mas": executed["updated_mas"], "trajectory": executed["updated_trajectory"], "score": 1.0, "judge": updated_judge},
            "editor": {"prompt": executed["prompt"], "output": executed["output"], "patch": executed["patch"], "reward": 1.0, "outcome": "improved", "trainable": True, "retry": executed["retry"]},
            "comparison": {"score_delta": 1.0, "editor_reward": 1.0, "editor_outcome": "improved"},
            "errors": [],
        },
    }


def test_validate_v4_dynamic_candidate_contract_and_malformed_skip(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    pair = _complete_v4_pair()
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts/judge_results/j.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": _judge(0.0)},
            {"pair_id": "p1:editor-0", "candidate": "updated_0", "judge": _judge(1.0)},
        ],
    )

    result = validate_phase(phase, "orchestra")["rollouts"]

    assert result["editor_candidate_rows"] == 2
    assert result["executed_updated_candidates"] == 1
    assert result["invalid_editor_output_candidates"] == 1
    assert result["required_standalone_judge_keys"] == 2


def test_validate_v4_rejects_legacy_updated_key_for_dynamic_candidate(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [_complete_v4_pair()])
    _write_jsonl(
        phase / "rollouts/judge_results/j.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": _judge(0.0)},
            {"pair_id": "p1", "candidate": "updated", "judge": _judge(1.0)},
        ],
    )
    with pytest.raises(RuntimeError, match="unexpected judge candidate"):
        validate_phase(phase, "orchestra")


def test_validate_v4_rejects_malformed_candidate_without_raw_evidence(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    pair = _complete_v4_pair()
    malformed = pair["structured_pair"]["editor_candidates"][1]
    malformed.update({"output": "", "error": "", "retry": {}})
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [pair])
    _write_jsonl(
        phase / "rollouts/judge_results/j.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": _judge(0.0)},
            {"pair_id": "p1:editor-0", "candidate": "updated_0", "judge": _judge(1.0)},
        ],
    )
    with pytest.raises(RuntimeError, match="lacks raw output"):
        validate_phase(phase, "orchestra")


def _v5_pair(pair_id: str, index: int, *, source_score: float = 0.0, updated_score: float = 1.0) -> dict:
    from mas_loop_rl.pair_selector import select_pair_winner

    source_judge = _judge(source_score)
    updated_judge = _judge(updated_score)
    source_mas = {"agents": [{"id": "a", "budget": 2}], "edges": [], "aggregation": "a"}
    updated_mas = {"agents": [{"id": "a", "budget": 2, "role": "solve carefully"}], "edges": [], "aggregation": "a"}
    source_traj = {"final_output": "source", "metadata": {"judge": source_judge}}
    updated_traj = {"final_output": "updated", "metadata": {"judge": updated_judge}}
    delta = updated_score - source_score
    candidate_id = f"{pair_id}:refiner"
    candidate = {
        "candidate_id": candidate_id,
        "candidate_index": index,
        "editor_group_uid": "editor-task:task-group:step-1",
        "prompt": [{"role": "user", "content": "repair"}],
        "output": '{"stage1_localization":{},"stage2_causal_model":{},"stage3_repair_plan":{},"patch":[]}',
        "retry": {"raw_responses": []},
        "patch": {"diagnosis": {}, "patch": [{"op": "rewrite_role", "agent_id": "a", "role": "solve carefully"}]},
        "updated_mas": updated_mas,
        "updated_trajectory": updated_traj,
        "src_score": source_score,
        "upd_score": updated_score,
        "score_delta": delta,
        "reward": delta,
        "outcome": "improved" if delta > 0 else ("damaged" if delta < 0 else "no_gain"),
        "trainable": True,
        "updated_valid": True,
        "violations": [],
        "error": "",
    }
    selection = select_pair_winner(
        source_score=source_score,
        updated_score=updated_score,
        source_mas=source_mas,
        updated_mas=updated_mas,
        source_trajectory=source_traj,
        updated_trajectory=updated_traj,
    ).to_dict()
    winner_is_updated = selection["selected_origin"] == "updated"
    return {
        "schema_version": "mas_loop_source_rollout_v5",
        "pair_id": pair_id,
        "rollout_sample_index": index,
        "optimizer_step": 1,
        "structured_pair": {
            "schema_version": "mas_loop_pairwise_one_to_one_v5",
            "pair_id": pair_id,
            "task_group_uid": "task-group",
            "rollout_sample_index": index,
            "optimizer_step": 1,
            "task": {"task_text": "question", "expected_answer": "answer"},
            "source": {"mas": source_mas, "trajectory": source_traj, "score": source_score, "judge": source_judge},
            "editor_candidates": [candidate],
            "updated": {"mas": updated_mas, "trajectory": updated_traj, "score": updated_score, "judge": updated_judge},
            "editor": {
                "prompt": candidate["prompt"], "output": candidate["output"], "patch": candidate["patch"],
                "reward": delta, "outcome": candidate["outcome"], "trainable": True, "retry": candidate["retry"],
            },
            "comparison": {
                "score_delta": delta, "orchestra_reward": source_score + 0.3 * max(delta, 0.0),
                "editor_reward": delta, "orchestra_outcome": "pair_improved", "editor_outcome": candidate["outcome"],
            },
            "selection": selection,
            "winner": {
                "origin": selection["selected_origin"],
                "mas": updated_mas if winner_is_updated else source_mas,
                "trajectory": updated_traj if winner_is_updated else source_traj,
                "score": updated_score if winner_is_updated else source_score,
            },
            "errors": [],
        },
    }


def _write_v5_phase(phase: Path, count: int = 32) -> None:
    rows = [_v5_pair(f"v5-{index}", index) for index in range(count)]
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", rows)
    judges = []
    for index in range(count):
        judges.extend(
            [
                {"pair_id": f"v5-{index}", "candidate": "source", "judge": _judge(0.0)},
                {"pair_id": f"v5-{index}:refiner", "candidate": "updated", "judge": _judge(1.0)},
            ]
        )
    _write_jsonl(phase / "rollouts/judge_results/j.jsonl", judges)


def test_validate_v5_one_to_one_task_group_contract(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    _write_v5_phase(phase)

    result = validate_phase(phase, "orchestra")["rollouts"]

    assert result["v5_pair_rows"] == 32
    assert result["v5_task_groups"] == 1
    assert result["v5_expected_group_size"] == 32
    assert result["editor_candidate_rows"] == 32
    assert result["executed_updated_candidates"] == 32
    assert result["required_standalone_judge_keys"] == 64


def test_validate_v5_rejects_incomplete_task_group(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    _write_v5_phase(phase, count=31)
    with pytest.raises(RuntimeError, match="incomplete/non-contiguous v5 task group"):
        validate_phase(phase, "orchestra")


def test_validate_v5_rejects_old_dynamic_updated_label(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    _write_v5_phase(phase)
    judge_path = phase / "rollouts/judge_results/j.jsonl"
    rows = [json.loads(line) for line in judge_path.read_text().splitlines()]
    rows[1]["candidate"] = "updated_0"
    _write_jsonl(judge_path, rows)
    with pytest.raises(RuntimeError, match="unexpected judge candidate"):
        validate_phase(phase, "orchestra")


def test_validate_v5_rejects_winner_inconsistent_with_selection(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _metrics(phase, orchestra=True)
    _write_v5_phase(phase)
    canonical = phase / "rollouts/canonical_rollouts/o.jsonl"
    rows = [json.loads(line) for line in canonical.read_text().splitlines()]
    rows[0]["structured_pair"]["winner"]["origin"] = "source"
    _write_jsonl(canonical, rows)
    with pytest.raises(RuntimeError, match="winner state mismatch"):
        validate_phase(phase, "orchestra")
