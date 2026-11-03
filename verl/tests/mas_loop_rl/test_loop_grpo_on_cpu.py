import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mas_loop_rl.build_editor_data_from_logs import _pad_complete_groups, build_rows
from mas_loop_rl.loop_agent import MASLoopEditorReplayAgentLoop, MASLoopOrchestraAgentLoop
from mas_loop_rl.sharded_jsonl import iter_jsonl
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1.trainer_base import _validate_external_grpo_group_counts
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


@pytest.fixture(autouse=True)
def _disable_external_bcp_judge(monkeypatch: pytest.MonkeyPatch):
    # Unit-test fake trajectories carry deterministic metadata scores and do
    # not run an external Qwen service. Judge behavior has dedicated tests.
    monkeypatch.setenv("MAS_LOOP_USE_LLM_JUDGE", "0")


def test_external_group_batch_validation_rejects_split_task_group():
    _validate_external_grpo_group_counts(["task-a"] * 8 + ["task-b"] * 8, 8)
    with pytest.raises(ValueError, match="complete inside one optimizer batch"):
        _validate_external_grpo_group_counts(["task-a"] * 7 + ["task-b"] * 8, 8)


def test_eight_same_task_samples_form_one_grpo_group():
    rewards = torch.tensor([[float(i), 0] for i in range(1, 9)])
    mask = torch.tensor([[1, 0]] * 8)
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards, "response_mask": mask},
        non_tensors={"uid": np.array(["same-task"] * 8, dtype=object)},
    )
    result = compute_advantage_for_multi_trajectories(
        data,
        batch_keys=[f"unique-row-{i}_0_0" for i in range(8)],
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=8,
        multi_output_grpo_mode="external_group",
    )
    expected = (torch.arange(1, 9, dtype=torch.float32) - 4.5) / torch.std(
        torch.arange(1, 9, dtype=torch.float32), unbiased=True
    )
    assert torch.allclose(result.batch["advantages"][:, 0], expected, atol=1e-5)
    assert torch.equal(result.batch["advantages"], result.batch["returns"])


class _FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, task, mas, expected, **kwargs):
        self.calls.append({"task": task, "mas": mas, "expected": expected, "kwargs": kwargs})
        aggregation = str(mas.get("aggregation", "s0"))
        digits = "".join(ch for ch in aggregation if ch.isdigit())
        score = float(int(digits or 0)) / 10.0
        return {"final_output": f"answer-from-{aggregation}", "metadata": {"score": score}, "steps": []}


class _PatchPeer:
    def chat_validated(self, messages, *, validator, **kwargs):
        prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
        match = re.search(r'\"agent_id\"\s*:\s*\"([^\"]+)\"', prompt_text)
        agent_id = match.group(1) if match else "s1"
        text = (
            f'<refiner><stage1_localization source_status="actionable_failure" target_type="agent" '
            f'target_id="{agent_id}" failure_type="information_loss"><evidence><item agent_id="{agent_id}" '
            'execution_index="0" field="hard_mismatch_signals" '
            'location="node_cards[0].hard_mismatch_signals">The source execution trace is missing.</item>'
            '</evidence></stage1_localization><stage2_causal_model><causal_chain><step>The source lacks an execution trace.</step>'
            '</causal_chain><causal_explanation>The missing trace prevents localized repair.</causal_explanation>'
            '<counterfactual>A complete trace would support a targeted edit.</counterfactual>'
            '<downstream_impact>No safe structural change is justified.</downstream_impact></stage2_causal_model>'
            '<stage3_repair_plan confidence="0.8"><repair_goal>Preserve the source MAS.</repair_goal>'
            '<expected_effect>Avoid an unsupported edit.</expected_effect><risk>The source remains unchanged.</risk>'
            '</stage3_repair_plan><patch><operation op="accept" /></patch></refiner>'
        )
        return text, validator(text), {"attempt_count": 1, "retry_count": 0, "raw_responses": [text], "errors": []}


class _FakeOrchestraLoop(MASLoopOrchestraAgentLoop):
    def __init__(self, tmp_path: Path):
        self.log_dir = tmp_path / "logs"
        self.pair_dir = tmp_path / "pairs"
        self._fake_executor = _FakeExecutor()
        self._fake_peer = _PatchPeer()
        self.actor_messages = []
        self.actor_calls = 0
        self.peer_max_tokens = 128
        self.use_llm_judge = False

    def _executor(self, **kwargs):
        return self._fake_executor

    def _peer(self, **kwargs):
        return self._fake_peer

    def _subagent_lease_client(self):
        return None

    async def _managed_blocking(self, lease_client, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def _actor_generate(self, messages, sampling_params):
        self.actor_calls += 1
        self.actor_messages.append(messages)
        sid = f"s{self.actor_calls}"
        text = json.dumps(
            {
                "agents": [{"id": sid, "role": "solve", "tools": [], "budget": 1}],
                "edges": [],
                "aggregation": sid,
                "topology": "sequential",
                "max_cost": 1.0,
                "max_iterations": 1,
            }
        )
        return [1], [100 + self.actor_calls], [-0.1], text, {"generate_sequences": 0.01, "num_preempted": 0}, {}


def _loop_kwargs(session_id=0):
    return {
        "uid": "prompt-1",
        "session_id": session_id,
        "global_steps": 3,
        "data_source": "unit-test",
        "extra_info": {"task_text": "2+2?", "constraints": {}},
        "reward_model": {"ground_truth": "4"},
    }


def _read(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_orchestra_eight_sessions_only_see_initial_task_and_write_eight_one_to_one_pairs(tmp_path: Path):
    loop = _FakeOrchestraLoop(tmp_path)
    outputs = [asyncio.run(loop.run({}, **_loop_kwargs(i))) for i in range(8)]
    assert len(outputs) == 8
    assert [o.reward_score for o in outputs] == [i / 10.0 for i in range(1, 9)]
    assert all("Generate the initial MAS as one <mas> XML document" in messages[1]["content"] for messages in loop.actor_messages)
    assert all("previous_updated_mas" not in messages[1]["content"] for messages in loop.actor_messages)
    assert {o.extra_fields["grpo_group_uid"] for o in outputs} == {"prompt-1"}
    assert [o.extra_fields["rollout_sample_index"] for o in outputs] == list(range(8))
    # Each source has one source execution and one fresh updated execution,
    # including explicit Editor accept actions. No source trajectory is reused.
    assert len(loop._fake_executor.calls) == 16
    assert sum(call["kwargs"].get("profile_phase") == "source_subagent" for call in loop._fake_executor.calls) == 8
    assert sum(call["kwargs"].get("profile_phase") == "updated_subagent" for call in loop._fake_executor.calls) == 8

    canonical = list(iter_jsonl([loop.log_dir / "canonical_rollouts"]))
    assert len(canonical) == 8
    assert len({r["pair_id"] for r in canonical}) == 8
    assert all(r["structured_pair"]["editor"]["trainable"] for r in canonical)
    assert [r["rollout_sample_index"] for r in canonical] == list(range(8))
    assert all(r["canonical_record_type"] == "orchestra_editor_pair" for r in canonical)
    # Runtime no longer writes five full duplicate JSONL streams.
    assert not (loop.log_dir / "mas_loop_grpo_rewards.jsonl").exists()
    assert not (loop.pair_dir / "joint_orchestra_editor_pairs.jsonl").exists()
    assert not (loop.pair_dir / "editor_rollout_pairs.jsonl").exists()

    td = list_of_dict_to_tensordict([o.as_dict() for o in outputs])
    assert td.batch_size == torch.Size([8])


def _orchestra_record(stage: int) -> dict:
    prompt = [{"role": "system", "content": "edit"}, {"role": "user", "content": f"source {stage}"}]
    return {
        "id": f"record-{stage}",
        "pair_id": f"pair-{stage}",
        "role": "orchestra",
        "loop_id": "prompt-1:orchestra",
        "grpo_group_uid": "prompt-1",
        "rollout_sample_index": stage,
        "task": "2+2?",
        "expected": "4",
        "constraints": {},
        "src_mas": {"agents": [{"id": f"s{stage}", "role": "solve"}], "aggregation": f"s{stage}"},
        "src_trajectory": {"final_output": str(stage), "metadata": {"score": stage / 10}},
        "peer_prompt": prompt,
        "peer_output": json.dumps({"diagnosis": {"failure_type": "x"}, "patch": []}),
        "editor_score": stage / 100,
        "editor_outcome": "improved",
        "src_score": stage / 10,
        "upd_score": stage / 10 + stage / 100,
        "score_delta": stage / 100,
    }


def test_editor_replay_layout_is_complete_groups_of_eight(tmp_path: Path):
    complete = tmp_path / "complete-layout.jsonl"
    complete.write_text("".join(json.dumps(_orchestra_record(i)) + "\n" for i in range(8)))
    rows, rejected = build_rows([complete], group_size=8)
    assert not rejected
    padded = _pad_complete_groups(rows, multiple=64, group_size=8)
    for offset in range(0, len(padded), 64):
        counts = Counter(row["extra_info"]["editor_group_uid"] for row in padded[offset : offset + 64])
        assert counts
        assert set(counts.values()) == {8}


def test_editor_builder_keeps_all_eight_one_to_one_sources_and_rejects_incomplete(tmp_path: Path):
    complete = tmp_path / "complete.jsonl"
    complete.write_text("".join(json.dumps(_orchestra_record(i)) + "\n" for i in range(8)))
    rows, rejected = build_rows([complete], group_size=8)
    assert not rejected
    assert len(rows) == 8
    assert len({r["extra_info"]["editor_group_uid"] for r in rows}) == 1
    assert [r["extra_info"]["pair_id"] for r in rows] == [f"pair-{i}" for i in range(8)]
    assert [r["extra_info"]["replay_editor_reward"] for r in rows] == [i / 100 for i in range(8)]

    padded = _pad_complete_groups(rows.copy(), multiple=64, group_size=8)
    assert len(padded) == 64
    for i in range(0, len(padded), 8):
        assert len({r["extra_info"]["editor_group_uid"] for r in padded[i : i + 8]}) == 1

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text("".join(json.dumps(_orchestra_record(i)) + "\n" for i in range(7)))
    rows, rejected = build_rows([incomplete], group_size=8)
    assert rows == []
    assert any(r["reason"] == "incomplete_or_oversized_group" for r in rejected)


class _FakeTokenizer:
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 + 1 for c in text]


class _FakeReplay(MASLoopEditorReplayAgentLoop):
    def __init__(self, tmp_path: Path):
        self.log_dir = tmp_path
        self.response_length = 8192
        self.max_prompt_length = 15000
        self.tokenizer = _FakeTokenizer()

    async def apply_chat_template(self, messages):
        return [1, 2, 3]


def test_editor_replay_returns_exact_saved_response_without_generation(tmp_path: Path):
    rec = _orchestra_record(3)
    extra = {
        "pair_id": rec["pair_id"],
        "source_loop_id": rec["loop_id"],
        "source_grpo_group_uid": rec["grpo_group_uid"],
        "editor_group_uid": "editor-replay:prompt-1",
        "rollout_sample_index": 3,
        "replay_editor_prompt": rec["peer_prompt"],
        "replay_editor_output": rec["peer_output"],
        "replay_editor_reward": rec["editor_score"],
        "src_score": rec["src_score"],
        "upd_score": rec["upd_score"],
        "score_delta": rec["score_delta"],
        "editor_outcome": rec["editor_outcome"],
    }
    loop = _FakeReplay(tmp_path)
    output = asyncio.run(loop.run({}, extra_info=extra))
    assert output.reward_score == rec["editor_score"]
    assert output.response_logprobs is None
    assert output.extra_fields["pair_id"] == rec["pair_id"]
    assert output.extra_fields["grpo_group_uid"] == "editor-replay:prompt-1"
    consumed = []
    for shard in (tmp_path / "editor_replay_consumed").glob("*.jsonl"):
        consumed.extend(_read(shard))
    assert consumed[0]["pair_id"] == rec["pair_id"]


def test_actor_generate_inserts_eos_for_empty_vllm_response(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_ACTOR_GUIDED_JSON", "1")
    monkeypatch.setenv("MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL", "json")
    loop = object.__new__(MASLoopOrchestraAgentLoop)
    loop.response_length = 8
    loop.tokenizer = SimpleNamespace(eos_token_id=7, pad_token_id=0, decode=lambda ids, skip_special_tokens=True: "")

    async def apply_chat_template(messages):
        return [1, 2]

    class _Server:
        def __init__(self):
            self.sampling_params = None

        async def generate(self, **kwargs):
            self.sampling_params = kwargs["sampling_params"]
            return SimpleNamespace(token_ids=[], log_probs=[], num_preempted=0, extra_fields={})

    loop.apply_chat_template = apply_chat_template
    loop.server_manager = _Server()
    original_sampling = {"temperature": 0.7}
    _, response_ids, response_logprobs, _, metrics, _ = asyncio.run(
        loop._actor_generate([], original_sampling)
    )
    assert response_ids == [7]
    assert response_logprobs == [0.0]
    assert original_sampling == {"temperature": 0.7}
    guided = loop.server_manager.sampling_params["guided_decoding"]
    assert guided.json["required"] == ["agents", "edges", "aggregation", "topology", "max_cost", "max_iterations"]
    assert guided.disable_additional_properties is True
    assert metrics["actor_guided_json"] == 1.0
    assert metrics["actor_response_tokens"] == 1.0


def test_editor_action_space_has_explicit_accept_and_exactly_ten_ops():
    from mas_loop_rl.schema import (
        PATCH_GUIDED_JSON_SCHEMA,
        VALID_PATCH_OPS,
        apply_patch,
        is_accept_patch,
        normalize_editor_action,
    )

    assert len(VALID_PATCH_OPS) == 10
    assert "accept" in VALID_PATCH_OPS
    assert "remove_agent" not in VALID_PATCH_OPS
    assert "merge_agents" not in VALID_PATCH_OPS
    assert set(PATCH_GUIDED_JSON_SCHEMA["properties"]["patch"]["items"]["properties"]["op"]["enum"]) == VALID_PATCH_OPS

    accepted = normalize_editor_action({"diagnosis": {}, "patch": [{"op": "accept"}]})
    assert is_accept_patch(accepted)
    source = {"agents": [{"id": "solver", "role": "solve"}], "aggregation": "solver"}
    updated, errors = apply_patch(source, accepted)
    assert not errors
    assert updated["agents"][0]["id"] == "solver"
    assert updated["aggregation"] == "solver"

    mixed = normalize_editor_action({"patch": [{"op": "accept"}, {"op": "add_verifier"}]})
    assert [op["op"] for op in mixed["operations"]] == ["add_verifier"]
    with pytest.raises(ValueError, match="not translatable"):
        normalize_editor_action({"patch": []})
    with pytest.raises(ValueError, match=r"unsupported_operation:change_status") as exc_info:
        normalize_editor_action({"patch": [{"op": "change_status", "status": "success"}]})
    assert "not translatable" in str(exc_info.value)


@pytest.mark.parametrize("enabled", [False, True])
def test_external_editor_guided_json_is_opt_in(monkeypatch, enabled: bool):
    from mas_loop_rl.loop_agent import _editor_guided_json_schema
    from mas_loop_rl.schema import PATCH_GUIDED_JSON_SCHEMA

    monkeypatch.setenv("MAS_LOOP_EDITOR_GUIDED_JSON", "1" if enabled else "0")
    monkeypatch.setenv("MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL", "json")
    assert _editor_guided_json_schema() == (PATCH_GUIDED_JSON_SCHEMA if enabled else None)


def test_editor_patch_reports_only_new_or_worsened_budget_overflow():
    from mas_loop_rl.schema import apply_patch

    def source_with_budgets(budgets):
        agents = [
            {
                "id": f"agent_{index}",
                "role": f"role {index}",
                "tools": [],
                "budget": budget,
                "execution_mode": "llm_only",
            }
            for index, budget in enumerate(budgets)
        ]
        return {
            "agents": agents,
            "edges": [
                {"from": f"agent_{index}", "to": f"agent_{index + 1}", "type": "data_flow"}
                for index in range(len(agents) - 1)
            ],
            "aggregation": agents[-1]["id"],
            "topology": "sequential",
            "max_cost": sum(budgets),
            "max_iterations": 1,
        }

    overflow_source = source_with_budgets([4, 4, 4, 4, 1])
    _, errors = apply_patch(
        overflow_source,
        {"patch": [{"op": "rewrite_role", "agent_id": "agent_0", "role": "better role"}]},
    )
    assert "budget_overflow" not in errors

    valid_source = source_with_budgets([4, 4, 4, 3, 1])
    _, errors = apply_patch(
        valid_source,
        {"patch": [{"op": "change_budget", "agent_id": "agent_4", "budget": 2}]},
    )
    assert errors == []
    repaired, _ = apply_patch(valid_source, {"patch": [{"op": "change_budget", "agent_id": "agent_4", "budget": 2}]})
    assert sum(a["budget"] for a in repaired["agents"]) <= 16
    _, errors = apply_patch(
        valid_source,
        {"patch": [{"op": "add_agent", "agent_id": "extra", "role": "Provide an additional independent solution.", "budget": 2}]},
    )
    assert errors == []
    repaired, _ = apply_patch(valid_source, {"patch": [{"op": "change_budget", "agent_id": "agent_4", "budget": 2}]})
    assert sum(a["budget"] for a in repaired["agents"]) <= 16

    _, errors = apply_patch(
        overflow_source,
        {"patch": [{"op": "change_budget", "agent_id": "agent_4", "budget": 2}]},
    )
    assert errors == []
    repaired, _ = apply_patch(valid_source, {"patch": [{"op": "change_budget", "agent_id": "agent_4", "budget": 2}]})
    assert sum(a["budget"] for a in repaired["agents"]) <= 16

    overflow_reducible = source_with_budgets([4, 4, 4, 4, 4])
    _, errors = apply_patch(
        overflow_reducible,
        {"patch": [{"op": "change_budget", "agent_id": "agent_4", "budget": 3}]},
    )
    assert "budget_overflow" not in errors

    _, errors = apply_patch(
        overflow_source,
        {"patch": [{"op": "add_verifier", "agent_id": "verifier", "role": "Verify the candidate answer.", "budget": 1}]},
    )
    assert errors == []
    repaired, _ = apply_patch(valid_source, {"patch": [{"op": "change_budget", "agent_id": "agent_4", "budget": 2}]})
    assert sum(a["budget"] for a in repaired["agents"]) <= 16


def test_editor_builder_does_not_apply_cross_source_accept_regret(tmp_path: Path):
    records = [_orchestra_record(i) for i in range(8)]
    for rec in records:
        rec["editor_score"] = 0.0
        rec["score_delta"] = 0.0
        rec["upd_score"] = rec["src_score"]
    records[0]["peer_output"] = json.dumps({"diagnosis": {}, "patch": [{"op": "accept"}]})
    records[0]["edit_operation"] = {"diagnosis": {}, "patch": [{"op": "accept"}]}
    records[0]["editor_outcome"] = "accepted_incorrect"
    records[1]["peer_output"] = json.dumps({"diagnosis": {}, "patch": [{"op": "add_verifier"}]})
    records[1]["edit_operation"] = {"diagnosis": {}, "patch": [{"op": "add_verifier"}]}
    records[1]["editor_score"] = 1.0
    records[1]["score_delta"] = 1.0
    records[1]["upd_score"] = 1.0

    path = tmp_path / "accept-regret.jsonl"
    path.write_text("".join(json.dumps(rec) + "\n" for rec in records))
    rows, rejected = build_rows([path], group_size=8, accept_regret_weight=1.0)
    assert not rejected
    accept_meta = rows[0]["extra_info"]
    assert accept_meta["editor_action"] == "accept"
    assert accept_meta["raw_editor_reward"] == 0.0
    assert accept_meta["group_best_positive_patch_reward"] == 1.0
    assert accept_meta["accept_regret"] == 0.0
    assert accept_meta["replay_editor_reward"] == 0.0
    assert rows[1]["extra_info"]["replay_editor_reward"] == 1.0


def test_editor_builder_keeps_accept_zero_without_positive_sibling(tmp_path: Path):
    records = [_orchestra_record(i) for i in range(8)]
    for rec in records:
        rec["editor_score"] = 0.0
        rec["score_delta"] = 0.0
        rec["upd_score"] = rec["src_score"]
    records[0]["peer_output"] = json.dumps({"diagnosis": {}, "patch": [{"op": "accept"}]})
    records[0]["edit_operation"] = {"diagnosis": {}, "patch": [{"op": "accept"}]}
    records[0]["editor_outcome"] = "accepted_correct"

    path = tmp_path / "accept-no-regret.jsonl"
    path.write_text("".join(json.dumps(rec) + "\n" for rec in records))
    rows, rejected = build_rows([path], group_size=8, accept_regret_weight=1.0)
    assert not rejected
    assert rows[0]["extra_info"]["group_best_positive_patch_reward"] == 0.0
    assert rows[0]["extra_info"]["accept_regret"] == 0.0
    assert rows[0]["extra_info"]["replay_editor_reward"] == 0.0


def test_editor_replay_can_select_exact_optimizer_step(tmp_path: Path):
    path = tmp_path / "two-steps.jsonl"
    records = []
    for optimizer_step in (1, 2):
        for sample_index in range(8):
            rec = _orchestra_record(sample_index)
            rec["id"] = f"record-step-{optimizer_step}-{sample_index}"
            rec["pair_id"] = f"pair-step-{optimizer_step}-{sample_index}"
            rec["grpo_group_uid"] = f"prompt-step-{optimizer_step}"
            rec["optimizer_step"] = optimizer_step
            records.append(rec)
    path.write_text("".join(json.dumps(rec) + "\n" for rec in records))

    rows, rejected = build_rows([path], group_size=8, optimizer_step=2)

    assert not rejected
    assert len(rows) == 8
    assert {row["extra_info"]["source_grpo_group_uid"] for row in rows} == {"prompt-step-2"}
    assert {row["extra_info"]["pair_id"] for row in rows} == {
        f"pair-step-2-{sample_index}" for sample_index in range(8)
    }


def test_editor_replay_filters_retry_attempt_by_record_time(tmp_path: Path):
    path = tmp_path / "same-step-two-attempts.jsonl"
    records = []
    for attempt, record_time in (("stale", 100.0), ("current", 200.0)):
        for sample_index in range(8):
            rec = _orchestra_record(sample_index)
            rec["id"] = f"record-{attempt}-{sample_index}"
            rec["pair_id"] = f"pair-{attempt}-{sample_index}"
            rec["grpo_group_uid"] = f"prompt-{attempt}"
            rec["optimizer_step"] = 6
            rec["time"] = record_time + sample_index
            records.append(rec)
    path.write_text("".join(json.dumps(rec) + "\n" for rec in records))

    rows, rejected = build_rows(
        [path], group_size=8, optimizer_step=6, min_record_time=200.0
    )

    assert not rejected
    assert len(rows) == 8
    assert {row["extra_info"]["source_grpo_group_uid"] for row in rows} == {"prompt-current"}
    assert {row["extra_info"]["pair_id"] for row in rows} == {
        f"pair-current-{sample_index}" for sample_index in range(8)
    }


def _v5_orchestra_record(index: int, *, task_group: str = "v5-task", optimizer_step: int = 1, reward: float = 0.0) -> dict:
    rec = _orchestra_record(index)
    rec.update({
        "schema_version": "mas_loop_source_rollout_v5",
        "reward_design": "pairwise_source_plus_bounded_winner_credit_v5",
        "id": f"v5-record-{task_group}-{optimizer_step}-{index}",
        "pair_id": f"v5-pair-{task_group}-{optimizer_step}-{index}",
        "grpo_group_uid": task_group,
        "optimizer_step": optimizer_step,
        "rollout_sample_index": index,
    })
    candidate = {
        "candidate_id": f"{rec['pair_id']}:refiner", "candidate_index": index,
        "editor_group_uid": f"editor-task:{task_group}:step-{optimizer_step}",
        "prompt": rec["peer_prompt"], "output": rec["peer_output"], "retry": {},
        "patch": {"diagnosis": {"failure_type": "x"}, "patch": [{"op": "accept"}]},
        "updated_mas": rec["src_mas"], "updated_trajectory": rec["src_trajectory"],
        "src_score": rec["src_score"], "upd_score": rec["src_score"] + reward,
        "score_delta": reward, "reward": reward, "outcome": "no_gain" if reward == 0 else "improved",
        "trainable": True, "updated_valid": True,
    }
    rec["editor_candidates"] = [candidate]
    rec["editor_trainable"] = True
    return rec


def test_editor_builder_v5_forms_one_complete_32_member_task_group(tmp_path: Path):
    path = tmp_path / "v5-32.jsonl"
    records = [_v5_orchestra_record(index, reward=(index % 3 - 1) * 0.25) for index in range(32)]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    rows, rejected = build_rows([path], group_size=32, optimizer_step=1)

    assert not rejected
    assert len(rows) == 32
    assert len({row["extra_info"]["editor_group_uid"] for row in rows}) == 1
    assert [row["extra_info"]["rollout_sample_index"] for row in rows] == list(range(32))
    assert all(row["extra_info"]["pairwise_task_group"] for row in rows)
    assert all(not row["extra_info"]["same_source_candidate"] for row in rows)


def test_editor_builder_v5_rejects_incomplete_and_duplicate_rollout_indices(tmp_path: Path):
    incomplete = tmp_path / "v5-31.jsonl"
    incomplete.write_text("".join(json.dumps(_v5_orchestra_record(index)) + "\n" for index in range(31)))
    rows, rejected = build_rows([incomplete], group_size=32)
    assert rows == []
    assert any(item["reason"] == "incomplete_or_oversized_group" for item in rejected)

    duplicate = tmp_path / "v5-duplicate.jsonl"
    records = [_v5_orchestra_record(index) for index in range(32)]
    records[-1]["rollout_sample_index"] = 30
    records[-1]["editor_candidates"][0]["candidate_index"] = 30
    duplicate.write_text("".join(json.dumps(record) + "\n" for record in records))
    rows, rejected = build_rows([duplicate], group_size=32)
    assert rows == []
    assert any(item["reason"] == "non_contiguous_rollout_indices" for item in rejected)


def test_editor_builder_v5_allows_zero_advantage_group_without_fabrication(tmp_path: Path):
    path = tmp_path / "v5-all-tied.jsonl"
    path.write_text("".join(json.dumps(_v5_orchestra_record(index, reward=0.0)) + "\n" for index in range(32)))
    rows, rejected = build_rows([path], group_size=32)
    assert not rejected
    assert len(rows) == 32
    assert {row["extra_info"]["replay_editor_reward"] for row in rows} == {0.0}
    assert {row["extra_info"]["accept_regret"] for row in rows} == {0.0}

class _RewriteRolePeer:
    def chat_validated(self, messages, *, validator, **kwargs):
        text = (
            '<refiner><stage1_localization source_status="actionable_failure" '
            'target_type="agent" target_id="s1" failure_type="information_loss">'
            '<evidence><item agent_id="s1" execution_index="0" field="hard_mismatch_signals" location="node_cards[0].hard_mismatch_signals">source evidence gap</item></evidence>'
            '</stage1_localization><stage2_causal_model><causal_chain><step>gap</step></causal_chain>'
            '<causal_explanation>specific mechanism</causal_explanation><counterfactual>better role</counterfactual>'
            '<downstream_impact>better evidence</downstream_impact></stage2_causal_model>'
            '<stage3_repair_plan confidence="0.8"><repair_goal>repair source role</repair_goal>'
            '<expected_effect>better retrieval</expected_effect><risk>low</risk></stage3_repair_plan>'
            '<patch><operation op="rewrite_role" agent_id="s1">'
            '<role>UPDATED_MAS_ONLY_SENTINEL</role></operation></patch></refiner>'
        )
        return text, validator(text), {"attempt_count": 1, "retry_count": 0, "raw_responses": [text], "errors": []}


class _SourceTokenProofLoop(_FakeOrchestraLoop):
    def __init__(self, tmp_path: Path):
        super().__init__(tmp_path)
        self._fake_peer = _RewriteRolePeer()


def test_updated_mas_never_substitutes_orchestra_source_response_tokens(tmp_path: Path):
    loop = _SourceTokenProofLoop(tmp_path)
    output = asyncio.run(loop.run({}, **_loop_kwargs(0)))

    # The actor sampled source response id 101.  A later Refiner patch changes
    # the MAS role, but the on-policy Orchestra response remains exactly the
    # original actor token sequence.
    assert output.response_ids == [101]
    rows = list(iter_jsonl([loop.log_dir / "canonical_rollouts"]))
    assert len(rows) == 1
    row = rows[0]
    assert row["src_mas"]["agents"][0]["role"] == "solve"
    assert row["updated_mas"]["agents"][0]["role"] == "UPDATED_MAS_ONLY_SENTINEL"
    assert "UPDATED_MAS_ONLY_SENTINEL" not in row["actor_output"]
    assert row["solution"] == row["actor_output"]
    assert row["structured_pair"]["actor"]["output"] == row["actor_output"]


def test_editor_replay_tokenizes_full_refiner_xml_not_updated_mas(tmp_path: Path):
    xml = (
        '<refiner><stage1_localization source_status="actionable_failure" target_type="agent" '
        'target_id="agent_0" failure_type="information_loss"><evidence><item location="agent_0.output">gap</item>'
        '</evidence></stage1_localization><stage2_causal_model><causal_chain><step>cause</step></causal_chain>'
        '<causal_explanation>mechanism</causal_explanation><counterfactual>change</counterfactual>'
        '<downstream_impact>impact</downstream_impact></stage2_causal_model><stage3_repair_plan confidence="0.7">'
        '<repair_goal>repair</repair_goal><expected_effect>effect</expected_effect><risk>risk</risk>'
        '</stage3_repair_plan><patch><operation op="accept" /></patch></refiner>'
    )
    loop = _FakeReplay(tmp_path)
    output = asyncio.run(
        loop.run(
            {},
            extra_info={
                "pair_id": "xml-pair",
                "source_loop_id": "source-loop",
                "editor_group_uid": "editor-group",
                "rollout_sample_index": 0,
                "replay_editor_prompt": [{"role": "user", "content": "diagnose"}],
                "replay_editor_output": xml,
                "replay_editor_reward": 0.25,
            },
        )
    )
    assert output.response_ids == loop.tokenizer.encode(xml, add_special_tokens=False)
    assert output.reward_score == 0.25
