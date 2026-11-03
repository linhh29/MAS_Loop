import json

from mas_loop_rl.prompts import compact_trajectory_for_editor, editor_prompt, orchestra_prompt


def test_editor_prompt_drops_transport_payload_but_keeps_decision_evidence():
    trajectory = {
        "task_text": "duplicate task",
        "mas_config": {"large": "duplicate mas"},
        "agent_traces": [{
            "agent_id": "solver",
            "agent_role": "solve",
            "output": "answer evidence",
            "error": "",
            "usage": {"prompt_tokens": 999},
            "metadata": {"chat_diagnostics": {"attempts": ["x" * 10000]}},
            "search_rounds": [{"documents": ["large corpus payload"]}],
        }],
        "final_output": "final answer",
        "success": True,
        "failure_nodes": [],
        "metadata": {
            "score": 1.0,
            "rollout_timing": {"generation_sec": 10},
            "retrieved_docs": ["large corpus payload"],
        },
    }
    source_mas = {
        "agents": [{"id": "solver", "role": "solve", "tools": [], "budget": 1, "execution_mode": "llm_only"}],
        "edges": [],
        "aggregation": "solver",
        "topology": "sequential",
        "max_cost": 1,
        "max_iterations": 1,
    }
    compact = compact_trajectory_for_editor(trajectory, source_mas, "question")
    encoded = json.dumps(compact)
    assert compact["node_cards"][0]["output_summary"] == "answer evidence"
    assert compact["outcome"]["final_output"] == "final answer"
    assert compact["outcome"]["score"] == 1.0
    assert compact["final_dependency_path"] == ["solver"]
    assert "chat_diagnostics" not in encoded
    assert "prompt_tokens" not in encoded
    assert "large corpus payload" not in encoded
    assert "rollout_timing" not in encoded

    prompt = editor_prompt("question", source_mas, trajectory)[1]["content"]
    assert "answer evidence" in prompt
    assert "final answer" in prompt
    assert "large corpus payload" not in prompt
    assert len(prompt) < 5000


def test_bcp_orchestra_prompt_matches_neutral_mixed_routing_contract():
    messages = orchestra_prompt(
        "Find the entity matching several independent clues.",
        {"evaluation_dataset": "bcp", "max_agents": 8},
    )
    system = messages[0]["content"]
    prompt = messages[1]["content"]
    assert "Design a multi-agent system, not the final answer" in system
    assert "Use 2-8 agents" not in system
    assert "Use 3-6 agents" in prompt
    assert '3-6 clear self-contained sub-tasks' in prompt
    assert 'task-specific role' in prompt
    assert 'Generic or reusable placeholder roles are invalid' in prompt
    assert "specialized evidence-acquisition role and instructions" not in prompt
    assert "final synthesizer/verifier role and required final-answer format" not in prompt
    assert 'multi_turn_search' in prompt
    assert 'llm_only' in prompt
    assert "Do not make every agent a retrieval agent" in prompt
