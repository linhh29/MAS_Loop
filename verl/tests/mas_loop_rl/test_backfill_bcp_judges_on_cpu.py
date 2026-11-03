from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from mas_loop_rl.backfill_bcp_judges import JUDGE_MAX_TOKENS, backfill_phase, scan_missing_candidates
from mas_loop_rl.sharded_jsonl import append_jsonl_shard


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def valid_judge(score: float = 1.0) -> dict:
    correct = "yes" if score == 1.0 else "no"
    return {
        "extracted_final_answer": "answer" if correct == "yes" else "wrong",
        "reasoning": "matches" if correct == "yes" else "does not match",
        "correct": correct,
        "confidence": 100.0,
        "parse_error": False,
        "score": score,
        "model": "judge",
        "max_tokens": JUDGE_MAX_TOKENS,
        "raw_judgement": "{}",
        "error": "",
    }


def pair(
    pair_id: str,
    source_judge=None,
    updated_judge=None,
    *,
    editor_error: bool = False,
    editor_patch: dict | None = None,
) -> dict:
    errors = ["peer_edit_failed:LocalChatError"] if editor_error else []
    return {
        "pair_id": pair_id,
        "data_source": "bcp",
        "optimizer_step": 2,
        "structured_pair": {
            "pair_id": pair_id,
            "optimizer_step": 2,
            "outer_adversarial_round": 1,
            "task": {"task_text": "question", "expected_answer": "answer"},
            "source": {"trajectory": {"final_output": "source answer", "metadata": {}}, "judge": source_judge},
            "updated": {"trajectory": {"final_output": "", "metadata": {}}, "judge": updated_judge},
            "editor": {
                "outcome": "error" if editor_error else "ok",
                "patch": editor_patch if editor_patch is not None else {"operation": "replace", "agents": []},
            },
            "errors": errors,
        },
    }


class FakeJudge:
    calls: list[dict] = []

    def __init__(self, *, result_dir: Path, **kwargs):
        self.result_dir = Path(result_dir)

    def judge(self, **kwargs):
        self.__class__.calls.append(dict(kwargs))
        result = valid_judge(0.0 if kwargs["candidate"] == "updated" else 1.0)
        append_jsonl_shard(
            self.result_dir,
            {
                "pair_id": kwargs["pair_id"],
                "candidate": kwargs["candidate"],
                "question": kwargs["question"],
                "correct_answer": kwargs["correct_answer"],
                "response": kwargs["response"],
                "judge_prompt": "official prompt",
                "judge": result,
            },
            shard_prefix="judge-backfill-test",
        )
        return result


def test_noop_when_nested_and_standalone_judges_are_complete(tmp_path: Path) -> None:
    phase = tmp_path / "round_1" / "orchestra"
    canonical = phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl"
    write_jsonl(canonical, [pair("p1", valid_judge(), valid_judge(0.0))])
    write_jsonl(
        phase / "rollouts" / "judge_results" / "judge.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": valid_judge()},
            {"pair_id": "p1", "candidate": "updated", "judge": valid_judge(0.0)},
        ],
    )
    before = canonical.read_bytes()
    audit = backfill_phase(phase, judge_base_url="http://unused", judge_factory=FakeJudge)
    assert audit["state"] == "no_op"
    assert canonical.read_bytes() == before


def test_backfills_missing_source_and_updated_evidence_atomically(tmp_path: Path) -> None:
    FakeJudge.calls = []
    phase = tmp_path / "round_1" / "orchestra"
    canonical = phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl"
    # Source lacks nested evidence; updated has nested evidence but lacks its
    # standalone prompt/raw record. Both therefore require an independent call.
    write_jsonl(canonical, [pair("p1", None, valid_judge(), editor_error=True)])
    write_jsonl(
        phase / "rollouts" / "judge_results" / "judge.jsonl",
        [{"pair_id": "p1", "candidate": "source", "judge": valid_judge()}],
    )
    audit_path = tmp_path / "audit.json"
    audit = backfill_phase(
        phase,
        judge_base_url="http://127.0.0.1:8313/v1",
        max_workers=2,
        audit_path=audit_path,
        judge_factory=FakeJudge,
    )
    assert audit["state"] == "complete"
    assert audit["before"]["missing_candidates"] == 2
    assert audit["after"]["missing_candidates"] == 0
    assert {call["candidate"] for call in FakeJudge.calls} == {"source", "updated"}
    assert all(call["data_source"].endswith("_posttrain_backfill") for call in FakeJudge.calls)
    updated_row = json.loads(canonical.read_text(encoding="utf-8"))
    structured = updated_row["structured_pair"]
    assert structured["errors"] == ["peer_edit_failed:LocalChatError"]
    assert structured["source"]["judge"]["max_tokens"] == 4096
    assert structured["updated"]["judge"]["score"] == 0.0
    assert structured["updated"]["trajectory"]["metadata"]["scoring_method"] == "qwen_llm_judge"
    assert not list(canonical.parent.glob("*.tmp.*"))
    assert audit_path.is_file()


def test_accept_reuses_source_standalone_judge_without_backfilling_updated(tmp_path: Path) -> None:
    FakeJudge.calls = []
    phase = tmp_path / "round_1" / "orchestra"
    shared = valid_judge()
    row = pair("p1", shared, shared, editor_patch={"diagnosis": {}, "patch": [{"op": "accept"}]})
    row["structured_pair"]["updated"] = dict(row["structured_pair"]["source"])
    write_jsonl(phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl", [row])
    write_jsonl(
        phase / "rollouts" / "judge_results" / "judge.jsonl",
        [{"pair_id": "p1", "candidate": "source", "judge": shared}],
    )

    missing, report = scan_missing_candidates(phase)

    assert missing == []
    assert report["required_standalone_candidates"] == 1
    assert report["accepted_updated_judge_reuse"] == 1
    audit = backfill_phase(phase, judge_base_url="http://unused", judge_factory=FakeJudge)
    assert audit["state"] == "no_op"
    assert FakeJudge.calls == []


def test_scan_treats_empty_pipeline_output_as_judgeable_missing_candidate(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    write_jsonl(
        phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl",
        [pair("p1", valid_judge(), None, editor_error=True)],
    )
    write_jsonl(
        phase / "rollouts" / "judge_results" / "judge.jsonl",
        [{"pair_id": "p1", "candidate": "source", "judge": valid_judge()}],
    )
    missing, report = scan_missing_candidates(phase)
    assert [(x.pair_id, x.candidate, x.response) for x in missing] == [("p1", "updated", "")]
    assert report["empty_response_candidates"] == 1
    assert missing[0].pipeline_errors == ("peer_edit_failed:LocalChatError",)


def test_scan_rejects_fail_closed_parse_error_sentinel(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    invalid = valid_judge(0.0)
    invalid.update({"correct": "no", "parse_error": True, "score": -1.0})
    write_jsonl(
        phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl",
        [pair("p1", invalid, valid_judge(0.0))],
    )
    write_jsonl(
        phase / "rollouts" / "judge_results" / "judge.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": invalid},
            {"pair_id": "p1", "candidate": "updated", "judge": valid_judge(0.0)},
        ],
    )
    missing, report = scan_missing_candidates(phase)
    assert [(item.pair_id, item.candidate) for item in missing] == [("p1", "source")]
    assert report["nested_valid_candidates"] == 1
    assert report["standalone_valid_candidates"] == 1


def test_scan_rejects_score_inconsistent_with_correct_field(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    inconsistent = valid_judge(1.0)
    inconsistent["correct"] = "no"
    write_jsonl(
        phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl",
        [pair("p1", inconsistent, valid_judge(0.0))],
    )
    write_jsonl(
        phase / "rollouts" / "judge_results" / "judge.jsonl",
        [
            {"pair_id": "p1", "candidate": "source", "judge": inconsistent},
            {"pair_id": "p1", "candidate": "updated", "judge": valid_judge(0.0)},
        ],
    )
    missing, _ = scan_missing_candidates(phase)
    assert [(item.pair_id, item.candidate) for item in missing] == [("p1", "source")]


def test_scan_retries_transient_eio_without_duplicate_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phase = tmp_path / "orchestra"
    canonical = phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl"
    judge_shard = phase / "rollouts" / "judge_results" / "judge.jsonl"
    write_jsonl(canonical, [pair("p1", valid_judge(), valid_judge(0.0))])
    write_jsonl(
        judge_shard,
        [
            {"pair_id": "p1", "candidate": "source", "judge": valid_judge()},
            {"pair_id": "p1", "candidate": "updated", "judge": valid_judge(0.0)},
        ],
    )
    original_open = Path.open
    attempts = 0

    def flaky_open(path: Path, *args, **kwargs):
        nonlocal attempts
        if path == judge_shard and attempts < 2:
            attempts += 1
            raise OSError(errno.EIO, "transient test EIO", str(path))
        if path == judge_shard:
            attempts += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    missing, report = scan_missing_candidates(phase)
    assert attempts == 3
    assert missing == []
    assert report["canonical_pair_rows"] == 1
    assert report["standalone_valid_candidates"] == 2


def test_scan_does_not_retry_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phase = tmp_path / "orchestra"
    canonical = phase / "rollouts" / "canonical_rollouts" / "pairs.jsonl"
    judge_shard = phase / "rollouts" / "judge_results" / "judge.jsonl"
    write_jsonl(canonical, [pair("p1", valid_judge(), valid_judge(0.0))])
    judge_shard.parent.mkdir(parents=True, exist_ok=True)
    judge_shard.write_text("{not-json}\n", encoding="utf-8")
    original_open = Path.open
    opens = 0

    def counting_open(path: Path, *args, **kwargs):
        nonlocal opens
        if path == judge_shard:
            opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    with pytest.raises(RuntimeError, match="invalid Judge JSON"):
        scan_missing_candidates(phase)
    assert opens == 1


def v4_candidate(
    source_pair_id: str,
    index: int,
    *,
    outcome: str,
    judge=None,
    output: str = '{"diagnosis":{},"patch":[]}',
    reward: float = 0.0,
) -> dict:
    trajectory = {"final_output": f"updated answer {index}", "metadata": {}}
    if judge is not None:
        trajectory["metadata"]["judge"] = judge
        trajectory["metadata"]["score"] = judge["score"]
    return {
        "candidate_id": f"{source_pair_id}:editor-{index}",
        "candidate_index": index,
        "editor_group_uid": f"editor-source:{source_pair_id}",
        "prompt": [{"role": "user", "content": "repair"}],
        "output": output,
        "retry": {"attempt_count": 1, "raw_responses": [output], "errors": ["invalid"] if outcome == "invalid_editor_output" else []},
        "patch": {"diagnosis": {}, "patch": [{"op": "accept"}]} if outcome.startswith("accepted_") else {"diagnosis": {}, "patch": [{"op": "rewrite_role", "agent_id": "a", "role": "solve"}]},
        "updated_mas": {"agents": [{"id": "a"}]},
        "updated_trajectory": trajectory,
        "src_score": 0.0,
        "upd_score": float(judge["score"]) if judge is not None else 0.0,
        "score_delta": float(judge["score"]) if judge is not None else 0.0,
        "reward": reward,
        "outcome": outcome,
        "trainable": True,
        "violations": [],
        "error": "LocalChatError:invalid" if outcome == "invalid_editor_output" else "",
    }


def v4_pair(source_pair_id: str, candidates: list[dict]) -> dict:
    source_judge = valid_judge(0.0)
    source_trajectory = {"final_output": "source answer", "metadata": {"judge": source_judge, "score": 0.0}}
    primary = candidates[0]
    return {
        "pair_id": source_pair_id,
        "data_source": "bcp",
        "optimizer_step": 2,
        "structured_pair": {
            "pair_id": source_pair_id,
            "optimizer_step": 2,
            "outer_adversarial_round": 1,
            "task": {"task_text": "question", "expected_answer": "answer"},
            "source": {"mas": {"agents": [{"id": "a"}]}, "trajectory": source_trajectory, "score": 0.0, "judge": source_judge},
            "editor_candidates": candidates,
            "updated": {
                "mas": primary["updated_mas"],
                "trajectory": primary["updated_trajectory"],
                "score": primary["upd_score"],
                "judge": primary["updated_trajectory"]["metadata"].get("judge"),
            },
            "editor": {
                "prompt": primary["prompt"], "output": primary["output"], "patch": primary["patch"],
                "reward": primary["reward"], "outcome": primary["outcome"], "trainable": True,
                "retry": primary["retry"],
            },
            "comparison": {"score_delta": primary["score_delta"], "editor_reward": primary["reward"], "editor_outcome": primary["outcome"]},
            "errors": [],
        },
    }


def test_scan_v4_requires_dynamic_judge_only_for_executed_candidates(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    pair_id = "source-1"
    executed_judge = valid_judge(1.0)
    candidates = [
        v4_candidate(pair_id, 0, outcome="improved", judge=executed_judge, reward=1.0),
        v4_candidate(pair_id, 1, outcome="invalid_editor_output", reward=-1.0, output="not-json"),
    ]
    write_jsonl(phase / "rollouts/canonical_rollouts/pairs.jsonl", [v4_pair(pair_id, candidates)])
    write_jsonl(
        phase / "rollouts/judge_results/judge.jsonl",
        [
            {"pair_id": pair_id, "candidate": "source", "judge": valid_judge(0.0)},
            {"pair_id": f"{pair_id}:editor-0", "candidate": "updated_0", "judge": executed_judge},
        ],
    )

    missing, report = scan_missing_candidates(phase)

    assert missing == []
    assert report["executed_updated_candidates"] == 1
    assert report["skipped_unexecuted_candidates"] == 1
    assert report["required_standalone_candidates"] == 2


def test_backfill_v4_updates_exact_candidate_and_primary_fields(tmp_path: Path) -> None:
    FakeJudge.calls = []
    phase = tmp_path / "orchestra"
    pair_id = "source-1"
    candidates = [
        v4_candidate(pair_id, 0, outcome="no_gain", judge=None, reward=0.0),
        v4_candidate(pair_id, 1, outcome="invalid_editor_output", reward=-1.0, output="not-json"),
    ]
    canonical = phase / "rollouts/canonical_rollouts/pairs.jsonl"
    write_jsonl(canonical, [v4_pair(pair_id, candidates)])
    write_jsonl(
        phase / "rollouts/judge_results/judge.jsonl",
        [{"pair_id": pair_id, "candidate": "source", "judge": valid_judge(0.0)}],
    )

    audit = backfill_phase(
        phase,
        judge_base_url="http://127.0.0.1:8313/v1",
        max_workers=1,
        judge_factory=FakeJudge,
    )

    assert audit["state"] == "complete"
    assert [(call["pair_id"], call["candidate"]) for call in FakeJudge.calls] == [
        (f"{pair_id}:editor-0", "updated_0")
    ]
    structured = json.loads(canonical.read_text())["structured_pair"]
    repaired = structured["editor_candidates"][0]
    assert repaired["upd_score"] == 1.0
    assert repaired["score_delta"] == 1.0
    assert repaired["reward"] == 1.0
    assert repaired["outcome"] == "improved"
    assert structured["updated"]["trajectory"] == repaired["updated_trajectory"]
    assert structured["editor"]["output"] == repaired["output"]


def v5_pair(source_pair_id: str, index: int, *, updated_judge=None) -> dict:
    source_judge = valid_judge(0.0)
    source_mas = {"agents": [{"id": "a", "budget": 2}], "edges": [], "aggregation": "a"}
    updated_mas = {"agents": [{"id": "a", "budget": 2, "role": "solve"}], "edges": [], "aggregation": "a"}
    source_traj = {"final_output": "source answer", "metadata": {"judge": source_judge, "score": 0.0}}
    updated_traj = {"final_output": f"updated answer {index}", "metadata": {}}
    if updated_judge is not None:
        updated_traj["metadata"].update({"judge": updated_judge, "score": updated_judge["score"]})
    score = float(updated_judge["score"]) if updated_judge is not None else 0.0
    candidate = {
        "candidate_id": f"{source_pair_id}:refiner", "candidate_index": index,
        "editor_group_uid": "editor-task:task-group:step-2",
        "prompt": [{"role": "user", "content": "repair"}], "output": "patch",
        "retry": {}, "patch": {"diagnosis": {}, "patch": [{"op": "rewrite_role", "agent_id": "a", "role": "solve"}]},
        "updated_mas": updated_mas, "updated_trajectory": updated_traj,
        "src_score": 0.0, "upd_score": score, "score_delta": score, "reward": score,
        "outcome": "improved" if score > 0 else "no_gain", "trainable": True, "updated_valid": True,
    }
    return {
        "schema_version": "mas_loop_source_rollout_v5", "pair_id": source_pair_id,
        "data_source": "bcp", "optimizer_step": 2, "rollout_sample_index": index,
        "structured_pair": {
            "schema_version": "mas_loop_pairwise_one_to_one_v5", "pair_id": source_pair_id,
            "optimizer_step": 2, "outer_adversarial_round": 1, "rollout_sample_index": index,
            "task": {"task_text": "question", "expected_answer": "answer"},
            "source": {"mas": source_mas, "trajectory": source_traj, "score": 0.0, "judge": source_judge},
            "editor_candidates": [candidate],
            "updated": {"mas": updated_mas, "trajectory": updated_traj, "score": score, "judge": updated_judge},
            "editor": {"prompt": candidate["prompt"], "output": "patch", "patch": candidate["patch"], "reward": score, "outcome": candidate["outcome"], "trainable": True, "retry": {}},
            "comparison": {"score_delta": score, "editor_reward": score, "editor_outcome": candidate["outcome"], "orchestra_reward": 0.3 * score},
            "selection": {}, "winner": {}, "errors": [],
        },
    }


def test_scan_v5_uses_refiner_updated_key_and_complete_task_group(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    rows = [v5_pair(f"v5-{index}", index, updated_judge=valid_judge(1.0)) for index in range(32)]
    write_jsonl(phase / "rollouts/canonical_rollouts/pairs.jsonl", rows)
    judge_rows = []
    for index in range(32):
        judge_rows.extend([
            {"pair_id": f"v5-{index}", "candidate": "source", "judge": valid_judge(0.0)},
            {"pair_id": f"v5-{index}:refiner", "candidate": "updated", "judge": valid_judge(1.0)},
        ])
    write_jsonl(phase / "rollouts/judge_results/judge.jsonl", judge_rows)

    missing, report = scan_missing_candidates(phase)

    assert missing == []
    assert report["v5_pair_rows"] == 32
    assert report["v5_task_groups"] == 1
    assert report["required_standalone_candidates"] == 64


def test_scan_v6_accepts_rollout_aligned_nonzero_candidate_indices(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    rows = [v5_pair(f"v6-{index}", index, updated_judge=valid_judge(1.0)) for index in range(32)]
    judge_rows = []
    for index, row in enumerate(rows):
        row["schema_version"] = "mas_loop_source_rollout_v6"
        row["structured_pair"]["schema_version"] = "mas_loop_pairwise_one_to_one_v6"
        judge_rows.extend([
            {"pair_id": f"v6-{index}", "candidate": "source", "judge": valid_judge(0.0)},
            {"pair_id": f"v6-{index}:refiner", "candidate": "updated", "judge": valid_judge(1.0)},
        ])
    write_jsonl(phase / "rollouts/canonical_rollouts/pairs.jsonl", rows)
    write_jsonl(phase / "rollouts/judge_results/judge.jsonl", judge_rows)

    missing, report = scan_missing_candidates(phase)

    assert missing == []
    assert report["v5_pair_rows"] == 32
    assert report["v5_task_groups"] == 1
    # Regression: rollout 17 legitimately carries candidate_index=17, not 0.
    assert rows[17]["structured_pair"]["editor_candidates"][0]["candidate_index"] == 17


def test_backfill_v5_updates_unique_candidate_and_pair_state(tmp_path: Path) -> None:
    FakeJudge.calls = []
    phase = tmp_path / "orchestra"
    rows = [v5_pair(f"v5-{index}", index, updated_judge=valid_judge(1.0)) for index in range(32)]
    rows[0] = v5_pair("v5-0", 0, updated_judge=None)
    canonical = phase / "rollouts/canonical_rollouts/pairs.jsonl"
    write_jsonl(canonical, rows)
    judge_rows = []
    for index in range(32):
        judge_rows.append({"pair_id": f"v5-{index}", "candidate": "source", "judge": valid_judge(0.0)})
        if index:
            judge_rows.append({"pair_id": f"v5-{index}:refiner", "candidate": "updated", "judge": valid_judge(1.0)})
    write_jsonl(phase / "rollouts/judge_results/judge.jsonl", judge_rows)

    audit = backfill_phase(phase, judge_base_url="http://127.0.0.1:8313/v1", max_workers=1, judge_factory=FakeJudge)

    assert audit["state"] == "complete"
    assert [(call["pair_id"], call["candidate"]) for call in FakeJudge.calls] == [("v5-0:refiner", "updated")]
    first = json.loads(canonical.read_text().splitlines()[0])["structured_pair"]
    candidate = first["editor_candidates"][0]
    assert candidate["reward"] == 0.0
    assert first["updated"]["judge"]["score"] == 0.0
    assert first["selection"]["selected_origin"] == "source"
    assert first["winner"]["origin"] == "source"
    assert first["comparison"]["orchestra_reward"] == 0.0
