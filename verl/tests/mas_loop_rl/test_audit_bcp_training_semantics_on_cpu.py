import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "datasets" / "audit_bcp_training_semantics.py"
SPEC = importlib.util.spec_from_file_location("audit_bcp_training_semantics", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _judge(score: float) -> dict:
    return {
        "extracted_final_answer": "answer" if score else "wrong",
        "reasoning": "binary test verdict",
        "correct": "yes" if score else "no",
        "confidence": 100.0,
        "parse_error": False,
        "score": score,
        "model": "judge",
        "max_tokens": 4096,
    }


def _pair(pair_id: str, group: str, index: int, source: float, updated: float, *, accept: bool) -> dict:
    source_mas = {"agents": [{"id": "a"}], "edges": []}
    source_traj = {"final_output": "x", "metadata": {"judge": _judge(source)}}
    if accept:
        updated_mas = source_mas
        updated_traj = source_traj
        patch = {"diagnosis": {}, "patch": [{"op": "accept"}]}
        outcome = "accepted_correct" if source else "accepted_incorrect"
    else:
        updated_mas = {"agents": [{"id": "a"}, {"id": "b"}], "edges": []}
        updated_traj = {"final_output": "y", "metadata": {"judge": _judge(updated)}}
        patch = {"diagnosis": {}, "patch": [{"op": "add_verifier"}]}
        outcome = "improved" if updated > source else "no_gain"
    delta = updated - source
    structured = {
        "pair_id": pair_id,
        "task_group_uid": group,
        "rollout_sample_index": index,
        "source": {"mas": source_mas, "trajectory": source_traj, "score": source, "judge": _judge(source)},
        "editor": {"patch": patch, "reward": delta, "outcome": outcome},
        "updated": {"mas": updated_mas, "trajectory": updated_traj, "score": updated, "judge": _judge(updated)},
        "comparison": {"score_delta": delta, "editor_reward": delta},
        "errors": [],
    }
    return {
        "pair_id": pair_id,
        "grpo_group_uid": group,
        "rollout_sample_index": index,
        "src_score": source,
        "upd_score": updated,
        "score_delta": delta,
        "editor_score": delta,
        "editor_outcome": outcome,
        "structured_pair": structured,
    }


def _fixture(tmp_path: Path):
    root = tmp_path / "run"
    rows = [
        _pair("p0", "g0", 0, 0.0, 0.0, accept=True),
        _pair("p1", "g0", 1, 0.0, 1.0, accept=False),
        _pair("p2", "g1", 0, 1.0, 1.0, accept=True),
        _pair("p3", "g1", 1, 0.0, 0.0, accept=False),
    ]
    _write_jsonl(root / "round_1/orchestra/rollouts/canonical_rollouts/o.jsonl", rows)
    orchestra, pair_meta = AUDIT.audit_orchestra(
        root / "round_1/orchestra", group_size=2, expected_groups=2, regret_weight=1.0
    )
    extras = []
    for pair_id in ("p0", "p1", "p2", "p3"):
        meta = pair_meta[pair_id]
        best = meta["group_best_positive_patch_reward"]
        accepted = meta["accepted"]
        extras.append(
            {
                "pair_id": f"{pair_id}:refiner",
                "source_pair_id": pair_id,
                "editor_group_uid": f"editor-replay:{meta['group_uid']}",
                "rollout_sample_index": meta["sample_index"],
                "raw_editor_reward": meta["raw_editor_reward"],
                "accept_regret": best if accepted else 0.0,
                "group_best_positive_patch_reward": best,
                "editor_action": "accept" if accepted else "patch",
                "replay_editor_reward": meta["expected_replay_reward"],
            }
        )
    parquet = root / "data/editor_train.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"extra_info": extra} for extra in extras]), parquet)
    consumed = [
        {
            "pair_id": extra["pair_id"],
            "editor_group_uid": extra["editor_group_uid"],
            "rollout_sample_index": extra["rollout_sample_index"],
            "reward": extra["replay_editor_reward"],
        }
        for extra in extras
    ]
    _write_jsonl(root / "round_1/editor/rollouts/editor_replay_consumed/e.jsonl", consumed)
    return root, orchestra, pair_meta


def test_full_training_semantics_audit(tmp_path: Path) -> None:
    root, orchestra, pair_meta = _fixture(tmp_path)
    assert orchestra["rows"] == 4
    assert orchestra["groups"] == 2
    assert orchestra["accept_count"] == 2
    assert orchestra["judge_scores"] == {"0": 5, "1": 3}
    replay, rewards = AUDIT.audit_replay_parquet(
        root / "data/editor_train.parquet", pair_meta=pair_meta, group_size=2, regret_weight=1.0
    )
    assert replay["non_padding_rows"] == 4
    assert rewards == {
        "p0:refiner": -1.0,
        "p1:refiner": 1.0,
        "p2:refiner": 0.0,
        "p3:refiner": 0.0,
    }
    consumed = AUDIT.audit_editor_consumed(root / "round_1/editor", replay_rewards=rewards, group_size=2)
    assert consumed["non_padding_rows"] == 4
    assert consumed["groups"] == 2


def test_replay_requires_explicit_source_pair_foreign_key(tmp_path: Path) -> None:
    root, _, pair_meta = _fixture(tmp_path)
    table = pq.read_table(root / "data/editor_train.parquet")
    rows = table.to_pylist()
    rows[0]["extra_info"]["source_pair_id"] = ""
    pq.write_table(pa.Table.from_pylist(rows), root / "data/editor_train.parquet")
    with pytest.raises(RuntimeError, match="missing replay source_pair_id"):
        AUDIT.audit_replay_parquet(
            root / "data/editor_train.parquet", pair_meta=pair_meta, group_size=2, regret_weight=1.0
        )


def test_orchestra_rejects_incomplete_rollout_group(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    _write_jsonl(
        phase / "rollouts/canonical_rollouts/o.jsonl",
        [_pair("p0", "g0", 0, 0.0, 0.0, accept=True)],
    )
    with pytest.raises(RuntimeError, match="exact size/index/uniqueness"):
        AUDIT.audit_orchestra(phase, group_size=2, expected_groups=1, regret_weight=1.0)


def test_orchestra_rejects_accept_that_changes_trajectory(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    row = _pair("p0", "g0", 0, 1.0, 1.0, accept=True)
    row["structured_pair"]["updated"]["trajectory"] = {
        "final_output": "different",
        "metadata": {"judge": _judge(1.0)},
    }
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [row])
    with pytest.raises(RuntimeError, match="preserve exact MAS/trajectory"):
        AUDIT.audit_orchestra(phase, group_size=1, expected_groups=1, regret_weight=1.0)


def test_replay_rejects_missing_accept_regret(tmp_path: Path) -> None:
    root, _, pair_meta = _fixture(tmp_path)
    table = pq.read_table(root / "data/editor_train.parquet")
    rows = table.to_pylist()
    rows[0]["extra_info"]["accept_regret"] = 0.0
    pq.write_table(pa.Table.from_pylist(rows), root / "data/editor_train.parquet")
    with pytest.raises(RuntimeError, match="accept regret/replay reward mismatch"):
        AUDIT.audit_replay_parquet(
            root / "data/editor_train.parquet", pair_meta=pair_meta, group_size=2, regret_weight=1.0
        )


def test_orchestra_accepts_and_reports_posthoc_judge_backfill(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    row = _pair("p0", "g0", 0, 0.0, -1.0, accept=False)
    # A targeted post-rollout backfill repaired the nested Judge verdict, while
    # immutable top-level values retain the fail-closed reward used for training.
    row["structured_pair"]["updated"]["score"] = 0.0
    row["structured_pair"]["updated"]["judge"] = _judge(0.0)
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [row])

    summary, pair_meta = AUDIT.audit_orchestra(
        phase, group_size=1, expected_groups=1, regret_weight=1.0
    )

    assert summary["judge_backfill_score_disagreements"] == {"source": 0, "updated": 1, "total": 1}
    assert summary["training_scores"] == {"-1": 1, "0": 1}
    assert pair_meta["p0"]["raw_editor_reward"] == -1.0
    assert pair_meta["p0"]["final_updated_score"] == 0.0


def test_orchestra_accepts_evidenced_editor_error_penalty(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    row = _pair("p0", "g0", 0, 0.0, 0.0, accept=False)
    row["editor_score"] = -1.0
    row["editor_outcome"] = "error"
    pair = row["structured_pair"]
    pair["comparison"]["editor_reward"] = -1.0
    pair["editor"]["reward"] = -1.0
    pair["editor"]["outcome"] = "error"
    pair["errors"] = ["peer_edit_failed:LocalChatError:invalid patch"]
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [row])

    summary, pair_meta = AUDIT.audit_orchestra(
        phase, group_size=1, expected_groups=1, regret_weight=1.0
    )

    assert summary["editor_error_penalty_count"] == 1
    assert pair_meta["p0"]["raw_editor_reward"] == -1.0


def test_orchestra_rejects_unevidenced_editor_error_penalty(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    row = _pair("p0", "g0", 0, 0.0, 0.0, accept=False)
    row["editor_score"] = -1.0
    row["editor_outcome"] = "error"
    pair = row["structured_pair"]
    pair["comparison"]["editor_reward"] = -1.0
    pair["editor"]["reward"] = -1.0
    pair["editor"]["outcome"] = "error"
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [row])

    with pytest.raises(RuntimeError, match="invalid Editor error penalty"):
        AUDIT.audit_orchestra(phase, group_size=1, expected_groups=1, regret_weight=1.0)


def test_orchestra_rejects_training_time_delta_mismatch(tmp_path: Path) -> None:
    phase = tmp_path / "orchestra"
    row = _pair("p0", "g0", 0, 0.0, 1.0, accept=False)
    row["score_delta"] = 0.0
    row["structured_pair"]["comparison"]["score_delta"] = 0.0
    _write_jsonl(phase / "rollouts/canonical_rollouts/o.jsonl", [row])

    with pytest.raises(RuntimeError, match="training-time paired delta mismatch"):
        AUDIT.audit_orchestra(phase, group_size=1, expected_groups=1, regret_weight=1.0)
