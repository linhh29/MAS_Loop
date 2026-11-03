from __future__ import annotations

import contextlib
import threading
import time

from mas_loop_rl.executor import MASExecutor, _stable_execution_levels


class _ConcurrentClient:
    max_model_len = 32768
    truncate_safety_margin = 32

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=2.0)
        self.lock = threading.Lock()
        self.calls: list[str] = []
        self.message_roles: list[list[str]] = []
        self.root_active = 0
        self.root_active_peak = 0

    @contextlib.contextmanager
    def awake_session(self):
        yield

    def chat_once(self, messages, *, max_tokens, temperature, diagnostics_out=None):
        prompt = messages[-1]["content"]
        with self.lock:
            self.calls.append(prompt)
            self.message_roles.append([message["role"] for message in messages])
        if "ROOT_A" in prompt:
            with self.lock:
                self.root_active += 1
                self.root_active_peak = max(self.root_active_peak, self.root_active)
            try:
                self.barrier.wait()
                time.sleep(0.03)
                return "OUTPUT_A"
            finally:
                with self.lock:
                    self.root_active -= 1
        if "ROOT_B" in prompt:
            with self.lock:
                self.root_active += 1
                self.root_active_peak = max(self.root_active_peak, self.root_active)
            try:
                self.barrier.wait()
                return "OUTPUT_B"
            finally:
                with self.lock:
                    self.root_active -= 1
        assert "[A]:\nOUTPUT_A" in prompt
        assert "[B]:\nOUTPUT_B" in prompt
        return "FINAL"


def test_dependency_ready_agents_overlap_without_result_reuse(monkeypatch):
    monkeypatch.delenv("MAS_LOOP_RETRIEVER", raising=False)
    monkeypatch.setenv("MAS_LOOP_DEPENDENCY_PROMPT_MODE", "declared_edges")
    monkeypatch.setenv("MAS_LOOP_DAG_AGENT_PARALLEL", "1")
    mas = {
        "agents": [
            {"id": "a", "role": "ROOT_A", "tools": [], "budget": 1, "execution_mode": "llm_only"},
            {"id": "b", "role": "ROOT_B", "tools": [], "budget": 1, "execution_mode": "llm_only"},
            {"id": "final", "role": "FINAL_ROLE", "tools": [], "budget": 1, "execution_mode": "llm_only"},
        ],
        "edges": [
            {"from": "a", "to": "final", "type": "data_flow"},
            {"from": "b", "to": "final", "type": "data_flow"},
        ],
        "aggregation": "final",
        "topology": "dag",
        "max_cost": 3,
        "max_iterations": 1,
    }
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False)
    client = _ConcurrentClient()
    executor.client = client

    result = executor.execute("question", mas, "FINAL")

    assert client.root_active_peak == 2
    assert len(client.calls) == 3
    assert sum("ROOT_A" in prompt for prompt in client.calls) == 1
    assert sum("ROOT_B" in prompt for prompt in client.calls) == 1
    assert [trace["agent_id"] for trace in result["agent_traces"]] == ["a", "b", "final"]
    assert [trace["metadata"]["execution_level"] for trace in result["agent_traces"]] == [0, 0, 1]
    assert all(trace["metadata"]["dag_level_parallel"] for trace in result["agent_traces"])
    assert result["metadata"]["dependency_prompt_mode"] == "declared_edges"
    assert result["metadata"]["dag_agent_parallel"] is True
    assert result["metadata"]["graph_is_acyclic"] is True
    assert result["metadata"]["execution_level_count"] == 2
    assert result["metadata"]["execution_level_widths"] == [2, 1]
    assert result["metadata"]["execution_level_max_width"] == 2
    assert result["metadata"]["rollout_timing"]["execution_level_widths"] == [2, 1]
    assert result["final_output"] == "FINAL"


def test_reasoning_declared_edges_keeps_dataset_system_contract(monkeypatch):
    monkeypatch.delenv("MAS_LOOP_RETRIEVER", raising=False)
    monkeypatch.setenv("MAS_LOOP_DEPENDENCY_PROMPT_MODE", "declared_edges")
    monkeypatch.setenv("MAS_LOOP_DAG_AGENT_PARALLEL", "1")
    monkeypatch.setenv("MAS_LOOP_DATASET_NAME", "gpqa")
    monkeypatch.setenv("MAS_LOOP_REASONING_PROMPT_PROFILE", "reasoning_consensus_v1")
    mas = {
        "agents": [
            {"id": "a", "role": "ROOT_A", "tools": [], "budget": 1, "execution_mode": "llm_only"},
            {"id": "b", "role": "ROOT_B", "tools": [], "budget": 1, "execution_mode": "llm_only"},
            {"id": "final", "role": "FINAL_ROLE", "tools": [], "budget": 1, "execution_mode": "llm_only"},
        ],
        "edges": [
            {"from": "a", "to": "final", "type": "data_flow"},
            {"from": "b", "to": "final", "type": "data_flow"},
        ],
        "aggregation": "final", "topology": "dag", "max_cost": 3, "max_iterations": 1,
    }
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False)
    client = _ConcurrentClient()
    executor.client = client

    executor.execute("question", mas, "FINAL")

    assert client.message_roles == [["system", "user"], ["system", "user"], ["system", "user"]]


def test_cyclic_graph_has_deterministic_serial_levels():
    mas = {
        "agents": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {"from": "a", "to": "b", "type": "data_flow"},
            {"from": "b", "to": "a", "type": "data_flow"},
        ],
    }
    levels, acyclic = _stable_execution_levels(mas)
    assert not acyclic
    assert [[agent["id"] for agent in level] for level in levels] == [["a"], ["b"]]
