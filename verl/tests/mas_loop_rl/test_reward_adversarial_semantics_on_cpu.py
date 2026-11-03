from __future__ import annotations

import json

import mas_loop_rl.reward as reward_mod


VALID_MAS = (
    '{"agents":[{"id":"solver","role":"solve","tools":[],"budget":1}],'
    '"edges":[],"aggregation":"solver","topology":"sequential",'
    '"max_cost":10.0,"max_iterations":3}'
)


class _FakeExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, task, mas, expected, **kwargs):
        return {
            "final_output": "42",
            "metadata": {"score": 0.7},
            "agent_traces": [],
            "mas_config": mas,
        }


class _MalformedPeer:
    def __init__(self, *args, **kwargs):
        pass

    def chat(self, *args, **kwargs):
        return "I cannot provide a JSON patch."


def test_orchestra_is_not_penalized_when_peer_patch_is_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr(reward_mod, "MASExecutor", _FakeExecutor)
    monkeypatch.setattr(reward_mod, "LocalChatClient", _MalformedPeer)
    monkeypatch.setenv("MAS_LOOP_PAIR_DIR", str(tmp_path / "pairs"))

    result = reward_mod.compute_score(
        data_source="mas_loop_orchestra",
        solution_str=VALID_MAS,
        ground_truth="42",
        extra_info={"task_text": "Return 42", "constraints": {}},
        role="orchestra",
        peer_base_url="http://peer/v1",
        peer_model="editor-peer",
        subagent_base_url="http://subagent/v1",
        subagent_model="subagent",
        log_dir=str(tmp_path / "reward"),
        manage_vllm_sleep=False,
    )

    # A failed adversarial edit is a no-op.  It must not turn a valid source
    # MAS score into the generic -1 failure reward.
    assert result["score"] == 0.7
    assert result["src_score"] == 0.7
    assert result["upd_score"] == 0.7
    assert result["score_delta"] == 0.0
    assert result["outcome"] == "peer_failed"
    assert result["error"] == ""

    reward_row = json.loads((tmp_path / "reward/mas_loop_grpo_rewards.jsonl").read_text().strip())
    assert reward_row["score"] == 0.7
    assert reward_row["outcome"] == "peer_failed"
    assert reward_row["edit_operation"]["operations"] == []
    assert reward_row["violations"]
    assert reward_row["violations"][0].startswith("peer_edit_failed:ValueError:no valid <refiner> XML found; no JSON found")

    pair_row = json.loads((tmp_path / "pairs/orchestra_rollout_pairs.jsonl").read_text().strip())
    assert pair_row["comparison"]["reward"] == 0.7
    assert pair_row["comparison"]["outcome"] == "peer_failed"
    assert pair_row["error"] == ""
    assert pair_row["edit"]["violations"] == reward_row["violations"]


def test_invalid_orchestra_mas_still_receives_failure_reward(tmp_path, monkeypatch):
    monkeypatch.setattr(reward_mod, "MASExecutor", _FakeExecutor)
    monkeypatch.setattr(reward_mod, "LocalChatClient", _MalformedPeer)
    monkeypatch.setenv("MAS_LOOP_PAIR_DIR", str(tmp_path / "pairs"))

    result = reward_mod.compute_score(
        data_source="mas_loop_orchestra",
        solution_str="not json",
        ground_truth="42",
        extra_info={"task_text": "Return 42"},
        role="orchestra",
        peer_base_url="http://peer/v1",
        subagent_base_url="http://subagent/v1",
        log_dir=str(tmp_path / "reward"),
        manage_vllm_sleep=False,
    )

    assert result["score"] == -1.0
    assert result["outcome"] == "error"
    assert result["error"] == "no valid <mas> XML found; no JSON found"
