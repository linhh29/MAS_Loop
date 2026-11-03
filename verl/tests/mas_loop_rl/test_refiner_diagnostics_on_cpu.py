from __future__ import annotations

import json
from pathlib import Path

import pytest

from mas_loop_rl.build_editor_data_from_logs import build_rows
from mas_loop_rl.diagnostics import build_node_diagnostic_state
from mas_loop_rl.prompts import editor_prompt
from mas_loop_rl.schema import apply_patch, normalize_editor_action


def _source_mas() -> dict:
    return {
        "agents": [
            {
                "id": "researcher",
                "role": "Acquire and summarize task evidence.",
                "tools": ["search"],
                "budget": 1,
                "execution_mode": "multi_turn_search",
            },
            {
                "id": "final",
                "role": "Use upstream evidence and return the final answer.",
                "tools": [],
                "budget": 1,
                "execution_mode": "llm_only",
            },
        ],
        "edges": [{"from": "researcher", "to": "final", "type": "data_flow"}],
        "aggregation": "final",
        "topology": "dag",
        "max_cost": 2,
        "max_iterations": 3,
    }


def _trajectory() -> dict:
    return {
        "agent_traces": [
            {
                "agent_id": "researcher",
                "agent_role": "Acquire and summarize task evidence.",
                "output": "Found evidence A but omitted its qualifier.",
                "error": None,
                "search_rounds": [
                    {
                        "round": 1,
                        "search_query": "generic query",
                        "retrieved_docids": ["d1", "d2"],
                        "new_docids_first_seen_this_round": ["d1", "d2"],
                        "documents": ["raw document payload must not be copied"],
                    }
                ],
                "metadata": {
                    "execution_index": 0,
                    "declared_upstream_agent_ids": [],
                    "available_upstream_agent_ids": [],
                    "chat_diagnostics": {"secret_transport": "drop me"},
                    "dependency_prompt_mode": "declared_edges",
                },
            },
            {
                "agent_id": "final",
                "agent_role": "Use upstream evidence and return the final answer.",
                "output": "Answer A",
                "error": None,
                "metadata": {
                    "execution_index": 1,
                    "declared_upstream_agent_ids": ["researcher"],
                    "available_upstream_agent_ids": ["researcher"],
                    "dependency_prompt_mode": "declared_edges",
                },
            },
        ],
        "final_output": "Answer A",
        "success": False,
        "failure_nodes": [],
        "metadata": {"score": 0.0, "retrieved_docs": ["unbounded raw payload"]},
    }


def _diagnosis(target: str = "researcher") -> dict:
    return {
        "source_status": "actionable_failure",
        "target_type": "agent",
        "target_id": target,
        "failure_type": "information_loss",
        "evidence": [
            {
                "location": f"agent:{target}.output",
                "observation": "The output omitted a qualifier needed downstream.",
            }
        ],
        "causal_explanation": "The downstream final node cannot recover information absent from the handoff.",
        "repair_goal": "Require the target agent to preserve qualifiers in its evidence summary.",
        "confidence": 0.8,
    }


def test_node_diagnostic_state_is_bounded_generic_and_drops_gold_or_transport():
    state = build_node_diagnostic_state(
        _source_mas(),
        _trajectory(),
        task="Find the answer",
        constraints={"dataset": "arbitrary", "gold": "must not leak", "expected_answer": "secret"},
    )
    encoded = json.dumps(state)
    researcher = state["node_cards"][0]
    final = state["node_cards"][1]
    assert researcher["agent_id"] == "researcher"
    assert researcher["successors"] == ["final"]
    assert researcher["optional_capability_signals"]["retrieval"]["unique_retrieved_artifacts"] == 2
    assert final["upstream_received"] == ["researcher"]
    assert "raw document payload" not in encoded
    assert "unbounded raw payload" not in encoded
    assert "secret_transport" not in encoded
    assert "must not leak" not in encoded
    assert "expected_answer" not in encoded


def test_editor_prompt_trains_dataset_neutral_three_stage_reasoning():
    first = editor_prompt("Find the answer", _source_mas(), _trajectory(), {}, candidate_index=0, candidate_count=8)
    second = editor_prompt("Find the answer", _source_mas(), _trajectory(), {}, candidate_index=1, candidate_count=8)
    first_text = first[1]["content"]
    second_text = second[1]["content"]
    assert "node_diagnostic_state" in first_text
    assert "causal_explanation" in first_text
    assert "repair_goal" in first_text
    assert "stage1_localization" in first_text
    assert "stage2_causal_model" in first_text
    assert "stage3_repair_plan" in first_text
    assert "actionable_failure, preserve" in first_text
    assert "actionable_failure|preserve" not in first_text
    assert "candidate_sampling_focus" not in first_text
    assert first_text == second_text


def test_multi_operation_patch_is_allowed_but_every_operation_is_semantic():
    action = {
        "diagnosis": _diagnosis(),
        "patch": [
            {
                "op": "rewrite_role",
                "agent_id": "researcher",
                "new_role": "Acquire evidence and preserve every task-relevant qualifier.",
            },
            {"op": "change_budget", "agent_id": "researcher", "budget": 2},
        ],
    }
    normalized = normalize_editor_action(action, src_mas=_source_mas())
    assert len(normalized["operations"]) == 2
    updated, errors = apply_patch(_source_mas(), normalized)
    assert not errors
    assert updated["agents"][0]["budget"] == 2

    invalid = {"diagnosis": _diagnosis("missing"), "patch": [{"op": "rewrite_role", "agent_id": "missing", "new_role": "x"}]}
    with pytest.raises(ValueError, match="unknown_agent_reference"):
        # An all-untranslatable intent fails; mixed valid/invalid patches are partially retained.
        normalize_editor_action(invalid, src_mas=_source_mas())
    unchanged, errors = apply_patch(_source_mas(), invalid)
    assert unchanged["agents"][0]["id"] == "researcher"
    assert errors and errors[0].startswith("invalid_patch:")


def _candidate(index: int, reward: float, *, accept: bool = False) -> dict:
    if accept:
        diagnosis = {
            "source_status": "preserve",
            "target_type": "mas",
            "target_id": "mas",
            "failure_type": "none",
            "evidence": [{"location": "trajectory", "observation": "No supported failure was found."}],
            "causal_explanation": "The observed source behavior should be preserved.",
            "repair_goal": "Preserve the current MAS.",
            "confidence": 0.7,
        }
        patch = [{"op": "accept"}]
        outcome = "accepted_incorrect"
    else:
        diagnosis = _diagnosis()
        patch = [{"op": "rewrite_role", "agent_id": "researcher", "new_role": f"repair hypothesis {index}"}]
        outcome = "improved" if reward > 0 else "no_gain"
    action = {"diagnosis": diagnosis, "patch": patch, "operations": patch}
    return {
        "candidate_id": f"source-pair:editor-{index}",
        "candidate_index": index,
        "editor_group_uid": "editor-source:source-pair",
        "prompt": [{"role": "user", "content": f"candidate {index}"}],
        "output": json.dumps({"diagnosis": diagnosis, "patch": patch}),
        "patch": action,
        "reward": reward,
        "outcome": outcome,
        "trainable": True,
        "updated_mas": _source_mas(),
        "updated_trajectory": _trajectory(),
        "src_score": 0.0,
        "upd_score": reward,
        "score_delta": reward,
    }


def test_replay_groups_real_candidates_from_one_source_and_keeps_diagnosis(tmp_path: Path):
    record = {
        "id": "record",
        "pair_id": "source-pair",
        "role": "orchestra",
        "loop_id": "task:orchestra",
        "grpo_group_uid": "task-group",
        "rollout_sample_index": 0,
        "optimizer_step": 1,
        "task": "Find the answer",
        "expected": "gold is stored but never placed in diagnostic state",
        "constraints": {},
        "src_mas": _source_mas(),
        "src_trajectory": _trajectory(),
        "peer_prompt": [{"role": "user", "content": "primary"}],
        "peer_output": "primary",
        "editor_score": 0.2,
        "editor_candidates": [
            _candidate(0, 0.0, accept=True),
            _candidate(1, 0.0),
            _candidate(2, 0.2),
            _candidate(3, -0.1),
        ],
    }
    path = tmp_path / "same-source.jsonl"
    path.write_text(json.dumps(record) + "\n")
    rows, rejected = build_rows([path], group_size=4, optimizer_step=1)
    assert not rejected
    assert len(rows) == 4
    assert len({row["extra_info"]["editor_group_uid"] for row in rows}) == 1
    assert {row["extra_info"]["source_pair_id"] for row in rows} == {"source-pair"}
    assert [row["extra_info"]["replay_editor_reward"] for row in rows] == [0.0, 0.0, 0.2, -0.1]
    assert all(row["extra_info"]["same_source_candidate"] for row in rows)
    assert rows[2]["extra_info"]["diagnosis"]["repair_goal"]
    assert rows[0]["extra_info"]["accept_regret"] == 0.0


def test_editor_prompt_recursively_removes_benchmark_and_gold_fields_but_keeps_capabilities():
    constraints = {
        "dataset": "browsecomp_plus_secret_name",
        "evaluation_dataset": "hidden_benchmark",
        "requires_retrieval": True,
        "retriever": "local_search",
        "available_tools": ["local_search", "python"],
        "answer_format": "Return one concise entity name.",
        "metadata": {
            "expected_answer": "TOP_SECRET_ANSWER",
            "gold_docids": ["SECRET_DOC_1"],
            "nested": {
                "ground_truth": "NESTED_SECRET",
                "budget_limit": 3,
            },
        },
    }
    prompt = editor_prompt("Find the entity", _source_mas(), _trajectory(), constraints)[1]["content"]
    assert "browsecomp_plus_secret_name" not in prompt
    assert "hidden_benchmark" not in prompt
    assert "TOP_SECRET_ANSWER" not in prompt
    assert "SECRET_DOC_1" not in prompt
    assert "NESTED_SECRET" not in prompt
    assert '"requires_retrieval": true' in prompt
    assert '"retriever": "local_search"' in prompt
    assert '"available_tools": ["local_search", "python"]' in prompt
    assert '"answer_format": "Return one concise entity name."' in prompt
    assert '"budget_limit": 3' in prompt


def test_editor_prompt_is_dataset_neutral_for_math_code_and_non_retrieval_tasks():
    cases = [
        {
            "dataset": "private_math_benchmark",
            "domain": "math",
            "requires_retrieval": False,
            "answer_format": "Put the result in a box.",
            "reference_solution": "SECRET_MATH_SOLUTION",
        },
        {
            "benchmark": "private_code_benchmark",
            "domain": "code",
            "requires_retrieval": False,
            "available_tools": ["python"],
            "correct_answer": "SECRET_CODE_ANSWER",
        },
    ]
    texts = [editor_prompt("Solve the task", _source_mas(), _trajectory(), case)[1]["content"] for case in cases]
    assert "private_math_benchmark" not in texts[0]
    assert "SECRET_MATH_SOLUTION" not in texts[0]
    assert '"domain": "math"' in texts[0]
    assert '"requires_retrieval": false' in texts[0]
    assert "private_code_benchmark" not in texts[1]
    assert "SECRET_CODE_ANSWER" not in texts[1]
    assert '"domain": "code"' in texts[1]
    assert '"available_tools": ["python"]' in texts[1]
    for text in texts:
        assert "dataset-neutral MAS diagnosis-and-edit policy" not in text  # system message only
        assert "stage1_localization" in text
        assert "stage2_causal_model" in text
        assert "stage3_repair_plan" in text


def _grounded_action(state: dict, *, location: str = "node_cards[0].output_summary", **overrides) -> dict:
    card = state["node_cards"][0]
    evidence = {
        "agent_id": card["agent_id"],
        "execution_index": card["execution_index"],
        "field": location.rsplit(".", 1)[-1],
        "location": location,
        "observation": "The researcher output omits the qualifier required by the downstream answer node.",
    }
    evidence.update(overrides)
    return {
        "stage1_localization": {
            "source_status": "actionable_failure",
            "target_type": "agent",
            "target_id": "researcher",
            "failure_type": "information_loss",
            "evidence": [evidence],
        },
        "stage2_causal_model": {
            "causal_chain": ["The qualifier is absent from the handoff.", "The final node cannot recover it."],
            "causal_explanation": "Information lost in the researcher output is unavailable downstream.",
            "counterfactual": "Preserving the qualifier would make it available to the final node.",
            "downstream_impact": "The final answer can use the complete evidence.",
        },
        "stage3_repair_plan": {
            "repair_goal": "Require the researcher to preserve task-relevant qualifiers.",
            "expected_effect": "A complete evidence handoff.",
            "risk": "The summary may become slightly longer.",
            "confidence": 0.8,
        },
        "patch": [{"op": "rewrite_role", "agent_id": "researcher", "role": "Acquire evidence and preserve every task-relevant qualifier."}],
    }


def test_diagnostic_state_exposes_stable_node_edge_outcome_and_retrieval_trace_refs():
    state = build_node_diagnostic_state(_source_mas(), _trajectory(), task="Find the answer")
    researcher = state["node_cards"][0]
    retrieval_round = researcher["optional_capability_signals"]["retrieval"]["round_summaries"][0]
    assert researcher["trace_refs"]["output_summary"] == "node_cards[0].output_summary"
    assert retrieval_round["trace_refs"]["query"].endswith("round_summaries[0].query")
    assert state["edge_cards"][0]["trace_refs"]["source_executed"] == "edge_cards[0].source_executed"
    assert state["outcome"]["trace_refs"]["success"] == "outcome.success"


def test_grounded_refiner_accepts_exact_trace_reference_and_preserves_multi_operation_patch():
    state = build_node_diagnostic_state(_source_mas(), _trajectory(), task="Find the answer")
    action = _grounded_action(state)
    action["patch"].append({"op": "change_budget", "agent_id": "researcher", "budget": 2})
    normalized = normalize_editor_action(action, src_mas=_source_mas(), diagnostic_state=state)
    assert [op["op"] for op in normalized["operations"]] == ["rewrite_role", "change_budget"]
    assert normalized["stage1_localization"]["evidence"][0]["execution_index"] == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"location": "node_cards[99].output_summary"}, "trace_refs"),
        ({"location": "node_cards[0].agent_id", "field": "agent_id"}, "trace_refs"),
        ({"agent_id": "final"}, "agent_id does not match"),
        ({"execution_index": 7}, "execution_index does not match"),
        ({"field": "execution_status"}, "field does not match"),
        ({"observation": "Observed bounded evidence."}, "concrete trace signal"),
    ],
)
def test_grounded_refiner_rejects_ungrounded_evidence_metadata(overrides: dict, message: str):
    state = build_node_diagnostic_state(_source_mas(), _trajectory(), task="Find the answer")
    action = _grounded_action(state, **overrides)
    with pytest.raises(ValueError, match=message):
        normalize_editor_action(action, src_mas=_source_mas(), diagnostic_state=state)


def test_grounded_refiner_requires_matching_retrieval_round_and_target_support():
    state = build_node_diagnostic_state(_source_mas(), _trajectory(), task="Find the answer")
    location = state["node_cards"][0]["optional_capability_signals"]["retrieval"]["round_summaries"][0]["trace_refs"]["query"]
    action = _grounded_action(
        state,
        location=location,
        field="query",
        round=2,
        observation="The first retrieval query is generic and does not include the required qualifier.",
    )
    with pytest.raises(ValueError, match="round does not match"):
        normalize_editor_action(action, src_mas=_source_mas(), diagnostic_state=state)

    action = _grounded_action(state)
    action["stage1_localization"]["target_id"] = "final"
    with pytest.raises(ValueError, match="is not supported by any evidence"):
        normalize_editor_action(action, src_mas=_source_mas(), diagnostic_state=state)


def test_legacy_json_without_diagnostic_state_remains_compatible():
    legacy = {"diagnosis": _diagnosis(), "patch": [{"op": "rewrite_role", "agent_id": "researcher", "new_role": "Preserve qualifiers."}]}
    normalized = normalize_editor_action(legacy, src_mas=_source_mas())
    assert normalized["operations"][0]["op"] == "rewrite_role"


def test_editor_retry_instruction_requires_nonempty_grounded_observation():
    from mas_loop_rl.prompts import editor_retry_instruction

    instruction = editor_retry_instruction(_source_mas())
    assert "non-empty task-specific observation text" in instruction
    assert "empty item body" in instruction
