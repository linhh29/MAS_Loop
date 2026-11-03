from __future__ import annotations

import contextlib
import json
from pathlib import Path

import mas_loop_rl.bcp_retrieval as bcp
import mas_loop_rl.executor as executor_module
from mas_loop_rl.executor import MASExecutor
from mas_loop_rl.prompts import orchestra_prompt


class _PlannerAndAnswerClient:
    max_model_len = 8192
    truncate_safety_margin = 32

    def __init__(self):
        self.planner_calls = 0
        self.answer_calls = 0

    @contextlib.contextmanager
    def awake_session(self):
        yield

    def chat_once(self, messages, *, max_tokens, temperature, guided_json=None, response_format=None):
        if guided_json is not None:
            self.planner_calls += 1
            return json.dumps(
                {
                    "search_query": f"independent query {self.planner_calls}",
                    "reason": f"cover gap {self.planner_calls}",
                }
            )
        self.answer_calls += 1
        return "final researched answer"


def _one_agent_mas():
    return {
        "agents": [
            {
                "id": "researcher",
                "role": "independently research all constraints",
                "tools": ["bm25"],
                "budget": 1,
                "execution_mode": "multi_turn_search",
                "max_tokens": 256,
            }
        ],
        "edges": [],
        "aggregation": "researcher",
        "topology": "sequential",
        "max_cost": 1,
        "max_iterations": 1,
    }


def test_executor_runs_exactly_five_fresh_planner_bm25_turns_without_disk_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVER", "bm25")
    monkeypatch.setenv("MAS_LOOP_BCP_SEARCH_ROUNDS", "5")
    calls: list[str] = []

    def fake_retrieve(query: str, topk=None):
        calls.append(query)
        return [
            {"docid": f"doc-{len(calls)}", "text": f"evidence for {query}", "score": 1.0}
        ], {
            "cache_hit": False,
            "cache_key_scope": "exact_query_topk_index",
            "query": query,
            "topk": 5,
        }

    monkeypatch.setattr(executor_module, "retrieve_bcp_evidence_with_meta", fake_retrieve)
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False, max_model_len=32768)
    fake_client = _PlannerAndAnswerClient()
    executor.client = fake_client

    result = executor.execute("Who satisfies all clues?", _one_agent_mas(), "final researched answer")

    assert fake_client.planner_calls == 5
    assert fake_client.answer_calls == 1
    assert calls == [f"independent query {i}" for i in range(1, 6)]
    assert result["metadata"]["retrieval_mode"] == "per_agent_five_turn_llm_planned_bm25"
    assert result["metadata"]["retrieval_rounds"] == 5
    trace = result["agent_traces"][0]
    assert len(trace["search_rounds"]) == 5
    assert [x["round"] for x in trace["search_rounds"]] == [1, 2, 3, 4, 5]
    assert all("planner_prompt" not in x for x in trace["search_rounds"])
    assert all(x["planner_prompt_length"] > 0 for x in trace["search_rounds"])
    assert result["final_output"] == "final researched answer"

    summary = trace["retrieval_artifact"]
    assert summary["persisted"] is False
    assert summary["reason"] == "bm25_query_result_cache_disabled"
    assert summary["search_round_count"] == 5
    assert not (tmp_path / "artifacts").exists()


class _CountingRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, k):
        self.calls += 1
        return [bcp.EvidenceDoc(docid="same-doc", text="same deterministic evidence", score=3.0)]


def test_identical_turn_query_executes_bm25_again_without_reloading_index(tmp_path: Path, monkeypatch):
    retriever = _CountingRetriever()
    singleton_calls = 0

    def fake_singleton(*args, **kwargs):
        nonlocal singleton_calls
        singleton_calls += 1
        return retriever

    monkeypatch.setenv("MAS_LOOP_BM25_INDEX", str(tmp_path / "fake-index"))
    monkeypatch.setattr(bcp, "get_bm25_retriever", fake_singleton)

    first_docs, first_meta = bcp.retrieve_bcp_evidence_with_meta("exact same query", 5)
    second_docs, second_meta = bcp.retrieve_bcp_evidence_with_meta("exact same query", 5)

    assert first_docs == second_docs
    assert retriever.calls == 2
    assert singleton_calls == 2
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is False
    assert second_meta["cache_key_scope"] == "disabled"
    assert second_meta["query_result_cache_enabled"] is False
    assert second_meta["index_resident"] is True
    assert not (tmp_path / "query_cache").exists()


class _GatePlannerAndAnswerClient:
    max_model_len = 8192
    truncate_safety_margin = 32

    def __init__(self, decisions, initial_queries=None):
        self.decisions = list(decisions)
        self.initial_queries = list(initial_queries or ["Find the entity matching the role-specific clues."])
        self.initial_query_prompts: list[str] = []
        self.planner_prompts: list[str] = []
        self.answer_prompts: list[str] = []
        self.initial_query_calls: list[dict] = []
        self.planner_calls: list[dict] = []
        self.answer_calls: list[dict] = []

    @contextlib.contextmanager
    def awake_session(self):
        yield

    def chat_once(
        self, messages, *, max_tokens, temperature, guided_json=None,
        response_format=None, diagnostics_out=None,
    ):
        call = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "guided_json": guided_json,
            "response_format": response_format,
        }
        if response_format == {"type": "json_object"}:
            self.planner_prompts.append(messages[-1]["content"])
            self.planner_calls.append(call)
            return json.dumps(self.decisions.pop(0))
        if "search-query generator" in messages[-1]["content"]:
            self.initial_query_prompts.append(messages[-1]["content"])
            self.initial_query_calls.append(call)
            return self.initial_queries.pop(0)
        self.answer_prompts.append(messages[-1]["content"])
        self.answer_calls.append(call)
        return "answer only"


def test_controller_profile_generates_role_aware_query_then_early_stops(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVER", "bm25")
    monkeypatch.setenv("MAS_LOOP_BCP_RETRIEVAL_PROFILE", "controller")
    monkeypatch.setenv("MAS_LOOP_BCP_SEARCH_ROUNDS", "10")
    monkeypatch.setenv("MAS_LOOP_BCP_DOC_MAX_TOKENS", "3")
    monkeypatch.setenv("MAS_LOOP_BCP_MAX_FINAL_SNIPPETS", "2")
    monkeypatch.setenv("MAS_LOOP_BCP_SUBAGENT_TEMPERATURE", "0.2")
    calls: list[str] = []

    def fake_retrieve(query: str, topk=None):
        calls.append(query)
        suffix = len(calls)
        return [
            {"docid": "shared", "text": "one two three four five", "score": 2.0},
            {"docid": f"unique-{suffix}", "text": f"alpha beta gamma delta {suffix}", "score": 1.0},
        ], {"cache_hit": False, "query": query, "topk": 5}

    monkeypatch.setattr(executor_module, "retrieve_bcp_evidence_with_meta", fake_retrieve)
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False, max_model_len=32768)
    client = _GatePlannerAndAnswerClient([
        {"action": "search", "search_query": "second distinct facet", "reason": "missing clue"},
        {"action": "final", "search_query": "", "reason": "enough evidence"},
    ])
    executor.client = client

    result = executor.execute("Original question terms?", _one_agent_mas(), "answer only")

    assert calls == ["Find the entity matching the role-specific clues.", "second distinct facet"]
    assert len(client.initial_query_prompts) == 1
    initial_prompt = client.initial_query_prompts[0]
    assert "[Original question]\nOriginal question terms?" in initial_prompt
    assert "[Role / sub-agent purpose]\nindependently research all constraints" in initial_prompt
    assert (
        "[Current sub-task]\nRead GLOBAL TASK CONTEXT for the full task, constraints, "
        "and required output format. Use UPSTREAM DEPENDENCIES DATA when present. "
        "Execute only the sub-task described in ROLE INSTRUCTION."
    ) in initial_prompt
    assert "Keep the sentence under ~50 words" in initial_prompt
    initial_call = client.initial_query_calls[0]
    assert initial_call["messages"] == [{"role": "user", "content": initial_prompt}]
    assert initial_call["max_tokens"] == 8192
    assert initial_call["temperature"] == 0.2
    assert initial_call["guided_json"] is None
    assert initial_call["response_format"] is None
    assert len(client.planner_prompts) == 2
    controller_prompt = client.planner_prompts[0]
    assert "Top snippets this round" in controller_prompt
    assert "- shared: one two three four five" in controller_prompt
    assert "[BM25-1]" not in controller_prompt
    assert "[Role constraints]\nindependently research all constraints" in controller_prompt
    assert "[Sub-task input]\n--- ROLE INSTRUCTION ---" in controller_prompt
    assert "--- GLOBAL TASK CONTEXT ---\nOriginal question terms?" in controller_prompt
    assert "different entities, synonyms, facets, spellings, dates, OR complementary gaps" in controller_prompt
    assert "avoid overlapping keyword bundles" in controller_prompt
    assert "WHAT IS STILL MISSING" in controller_prompt
    assert "snippets already cover all constraints" in controller_prompt
    assert all(call["guided_json"] is None for call in client.planner_calls)
    assert all(call["response_format"] == {"type": "json_object"} for call in client.planner_calls)
    assert all([m["role"] for m in call["messages"]] == ["system", "user"] for call in client.planner_calls)
    assert result["metadata"]["retrieval_profile"] == "controller"
    assert result["metadata"]["retrieval_mode"] == "mixed_execution_mode_controller_controller_bm25"
    assert result["metadata"]["retrieval_evidence_policy"] == "accumulated"
    trace = result["agent_traces"][0]
    artifact = trace["retrieval_artifact"]
    assert artifact["initial_query_source"] == "controller_query_generator"
    assert artifact["initial_query"] == "Find the entity matching the role-specific clues."
    assert artifact["initial_query_raw_output"] == "Find the entity matching the role-specific clues."
    assert artifact["initial_query_error"] is None
    assert artifact["rounds_requested"] == 5
    assert artifact["rounds_completed"] == 2
    assert artifact["early_stop_reason"] == "planner_final"
    assert artifact["selected_document_count"] == 2
    assert artifact["doc_truncation_unit"] == "whitespace_words"
    assert artifact["doc_truncation_limit"] == 3
    answer_prompt = client.answer_prompts[0]
    assert "[Retrieved evidence]\nRetrieved evidence (ground answers in these snippets only):" in answer_prompt
    assert "[docid=shared] one two three" in answer_prompt
    assert "one two three four" not in answer_prompt
    assert "--- Multi-round retrieval trace (controller-guided BM25) ---" in answer_prompt
    assert "Round 1: planner_action='search'" in answer_prompt
    assert "new_distinct_snippets=2" in answer_prompt
    assert "Evidence snippets shown above (accumulated distinct snippets): 2" in answer_prompt
    assert "[Sub-agent role]\nresearcher" in answer_prompt
    assert "[Sub-task instruction]\nindependently research all constraints" in answer_prompt
    assert answer_prompt.endswith("Using the retrieved evidence as grounding, answer the initial question.")
    assert client.answer_calls[0]["messages"] == [{"role": "user", "content": answer_prompt}]
    assert client.answer_calls[0]["max_tokens"] == 8192
    assert client.answer_calls[0]["temperature"] == 0.2
    assert result["final_output"] == "answer only"


def test_controller_profile_rejects_normalized_duplicate_query(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVER", "bm25")
    monkeypatch.setenv("MAS_LOOP_BCP_RETRIEVAL_PROFILE", "controller")
    monkeypatch.setenv("MAS_LOOP_BCP_SEARCH_ROUNDS", "10")
    calls: list[str] = []

    def fake_retrieve(query: str, topk=None):
        calls.append(query)
        return [{"docid": "d1", "text": "evidence", "score": 1.0}], {
            "cache_hit": False, "query": query, "topk": 5,
        }

    monkeypatch.setattr(executor_module, "retrieve_bcp_evidence_with_meta", fake_retrieve)
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False, max_model_len=32768)
    executor.client = _GatePlannerAndAnswerClient(
        [{"action": "search", "search_query": "alpha beta gamma", "reason": "retry"}],
        initial_queries=["Alpha beta gamma"],
    )

    result = executor.execute("Alpha, beta gamma?", _one_agent_mas(), "answer only")

    assert calls == ["Alpha beta gamma"]
    assert len(executor.client.initial_query_prompts) == 1
    artifact = result["agent_traces"][0]["retrieval_artifact"]
    assert artifact["rounds_completed"] == 1
    assert artifact["early_stop_reason"] == "duplicate_normalized"
    turn = result["agent_traces"][0]["search_rounds"][0]
    assert turn["reject_reason"] == "duplicate_normalized"


def test_controller_uses_exact_search_synthesis_and_llm_only_prompts_without_custom_answer_contract(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVER", "bm25")
    monkeypatch.setenv("MAS_LOOP_BCP_RETRIEVAL_PROFILE", "controller")
    monkeypatch.setenv("MAS_LOOP_BCP_SEARCH_ROUNDS", "1")

    def fake_retrieve(query: str, topk=None):
        return [{"docid": "d1", "text": "evidence", "score": 1.0}], {
            "cache_hit": False, "query": query, "topk": 5,
        }

    monkeypatch.setattr(executor_module, "retrieve_bcp_evidence_with_meta", fake_retrieve)
    mas = {
        "agents": [
            {"id": "researcher", "role": "research clues", "tools": ["bm25"], "budget": 1,
             "execution_mode": "multi_turn_search"},
            {"id": "synthesizer", "role": "synthesize the answer", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
        ],
        "edges": [{"from": "researcher", "to": "synthesizer", "type": "data_flow"}],
        "aggregation": "synthesizer",
        "topology": "sequential",
        "max_cost": 2,
        "max_iterations": 1,
    }
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False, max_model_len=32768)
    client = _GatePlannerAndAnswerClient([
        {"action": "final", "search_query": "", "reason": "done"},
    ])
    executor.client = client

    executor.execute("Question?", mas, "answer only")

    assert len(client.answer_prompts) == 2
    search_prompt, llm_only_prompt = client.answer_prompts
    assert "[Retrieved evidence]" in search_prompt
    assert search_prompt.endswith("Using the retrieved evidence as grounding, answer the initial question.")
    assert "[Retrieved evidence]" not in llm_only_prompt
    assert "[Sub-agent role]\nsynthesizer" in llm_only_prompt
    assert "[Sub-task instruction]\nsynthesize the answer" in llm_only_prompt
    assert "[Task input]\n--- ROLE INSTRUCTION ---" in llm_only_prompt
    assert "[RESEARCHER]:\nanswer only" in llm_only_prompt
    assert "Output only the final answer text" not in search_prompt
    assert "Output only the final answer text" not in llm_only_prompt
    assert all(call["messages"] == [{"role": "user", "content": prompt}] for call, prompt in zip(client.answer_calls, client.answer_prompts))


def test_bcp_orchestra_prompt_requires_specialized_multi_turn_search_agents():
    prompt = orchestra_prompt(
        "Find the entity satisfying all clues.",
        {"dataset": "browsecomp_plus", "requires_retrieval": True, "max_agents": 8},
    )
    system = prompt[0]["content"]
    user = prompt[-1]["content"]
    assert "Design a multi-agent system, not the final answer" in system
    assert "Use 2-8 agents" not in system
    assert "Use 3-6 agents" in user
    assert "3-6 clear self-contained sub-tasks" in user
    assert "task-specific role" in user
    assert "Generic or reusable placeholder roles are invalid" in user
    assert "set execution_mode on every agent" in user
    assert "multi_turn_search" in user
    assert "llm_only" in user
    assert "edges must reference earlier agent ids" in user
    assert "NEW facts/documents" in user
    assert "Do not make every agent a retrieval agent" in user
    assert "aggregation agent should normally be llm_only" in user
    assert "synthesize only from retrieved upstream snippets" in user
    assert "Set budget=1 for every agent" in user
    assert "max_cost to the sum of budgets" in user


class _DependencyCapturingClient:
    max_model_len = 8192
    truncate_safety_margin = 32

    def __init__(self):
        self.answer_prompts: list[str] = []
        self.initial_query_prompts: list[str] = []
        self.planner_prompts: list[str] = []
        self.calls: list[dict] = []

    @contextlib.contextmanager
    def awake_session(self):
        yield

    def chat_once(
        self, messages, *, max_tokens, temperature, guided_json=None,
        response_format=None, diagnostics_out=None,
    ):
        user = messages[-1]["content"]
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "guided_json": guided_json,
            "response_format": response_format,
        })
        if response_format == {"type": "json_object"}:
            self.planner_prompts.append(user)
            return json.dumps({"action": "final", "search_query": "", "reason": "enough"})
        if "search-query generator" in messages[-1]["content"]:
            self.initial_query_prompts.append(user)
            return "declared dependency query"
        self.answer_prompts.append(user)
        if "ROLE_B_UNRELATED" in user:
            return "UNRELATED_B_MARKER"
        if "ROLE_A_SOURCE" in user:
            return "DECLARED_A_MARKER"
        if "ROLE_C_SEARCH" in user:
            return "SEARCH_C_MARKER"
        return "FINAL_MARKER"


def test_controller_dependency_prompts_follow_edges_and_topological_order(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVER", "bm25")
    monkeypatch.setenv("MAS_LOOP_BCP_RETRIEVAL_PROFILE", "controller")
    mas = {
        # Deliberately authored out of dependency order: A must still run before C.
        "agents": [
            {"id": "c", "role": "ROLE_C_REASON", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
            {"id": "b", "role": "ROLE_B_UNRELATED", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
            {"id": "a", "role": "ROLE_A_SOURCE", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
        ],
        "edges": [{"from": "a", "to": "c", "type": "data_flow"}],
        "aggregation": "c", "topology": "dag", "max_cost": 3, "max_iterations": 1,
    }
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False, max_model_len=32768)
    client = _DependencyCapturingClient()
    executor.client = client

    result = executor.execute("ORIGINAL_TASK_MARKER", mas, "FINAL_MARKER")

    traces = result["agent_traces"]
    assert [trace["agent_id"] for trace in traces] == ["b", "a", "c"]
    assert all("ORIGINAL_TASK_MARKER" in prompt for prompt in client.answer_prompts)
    b_prompt, a_prompt, c_prompt = client.answer_prompts
    assert "--- UPSTREAM DEPENDENCIES DATA ---" not in b_prompt
    assert "--- UPSTREAM DEPENDENCIES DATA ---" not in a_prompt
    assert "[A]:\nDECLARED_A_MARKER" in c_prompt
    assert "UNRELATED_B_MARKER" not in c_prompt
    assert traces[-1]["metadata"]["declared_upstream_agent_ids"] == ["a"]
    assert traces[-1]["metadata"]["available_upstream_agent_ids"] == ["a"]
    assert all(trace["metadata"]["dependency_prompt_mode"] == "declared_edges" for trace in traces)


def test_controller_controller_receives_only_declared_upstream_context(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVER", "bm25")
    monkeypatch.setenv("MAS_LOOP_BCP_RETRIEVAL_PROFILE", "controller")
    monkeypatch.setenv("MAS_LOOP_BCP_SEARCH_ROUNDS", "1")

    def fake_retrieve(query: str, topk=None):
        return [{"docid": "d1", "text": "retrieved evidence", "score": 1.0}], {
            "cache_hit": False, "query": query, "topk": 5,
        }

    monkeypatch.setattr(executor_module, "retrieve_bcp_evidence_with_meta", fake_retrieve)
    mas = {
        "agents": [
            {"id": "c", "role": "ROLE_C_SEARCH", "tools": ["bm25"], "budget": 1,
             "execution_mode": "multi_turn_search"},
            {"id": "b", "role": "ROLE_B_UNRELATED", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
            {"id": "a", "role": "ROLE_A_SOURCE", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
            {"id": "final", "role": "ROLE_FINAL", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
        ],
        "edges": [
            {"from": "a", "to": "c", "type": "data_flow"},
            {"from": "c", "to": "final", "type": "data_flow"},
        ],
        "aggregation": "final", "topology": "dag", "max_cost": 4, "max_iterations": 1,
    }
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False, max_model_len=32768)
    client = _DependencyCapturingClient()
    executor.client = client

    result = executor.execute("ORIGINAL_TASK_MARKER", mas, "FINAL_MARKER")

    assert len(client.initial_query_prompts) == 1
    initial_query_prompt = client.initial_query_prompts[0]
    assert "DECLARED_A_MARKER" in initial_query_prompt
    assert "UNRELATED_B_MARKER" not in initial_query_prompt
    assert len(client.planner_prompts) == 1
    controller_prompt = client.planner_prompts[0]
    assert "DECLARED_A_MARKER" in controller_prompt
    assert "UNRELATED_B_MARKER" not in controller_prompt
    final_trace = next(trace for trace in result["agent_traces"] if trace["agent_id"] == "final")
    final_prompt = client.answer_prompts[-1]
    assert "[C]:\nSEARCH_C_MARKER" in final_prompt
    assert "DECLARED_A_MARKER" not in final_prompt
    assert "UNRELATED_B_MARKER" not in final_prompt
    assert final_trace["metadata"]["declared_upstream_agent_ids"] == ["c"]


def test_mixed_routing_searches_only_multi_turn_agents(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVER", "bm25")
    monkeypatch.setenv("MAS_LOOP_BCP_RETRIEVAL_PROFILE", "controller")
    monkeypatch.setenv("MAS_LOOP_BCP_SEARCH_ROUNDS", "5")

    calls: list[str] = []
    def fake_retrieve(query: str, topk=None):
        calls.append(query)
        return [{"docid": "d1", "text": "evidence", "score": 1.0}], {
            "cache_hit": False, "query": query, "topk": 5,
        }

    monkeypatch.setattr(executor_module, "retrieve_bcp_evidence_with_meta", fake_retrieve)
    mas = {
        "agents": [
            {"id": "searcher", "role": "retrieve evidence", "tools": ["bm25"], "budget": 1,
             "execution_mode": "multi_turn_search"},
            {"id": "reasoner", "role": "reconcile evidence", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
            {"id": "final", "role": "return answer only", "tools": [], "budget": 1,
             "execution_mode": "llm_only"},
        ],
        "edges": [
            {"from": "searcher", "to": "reasoner", "type": "data_flow"},
            {"from": "reasoner", "to": "final", "type": "data_flow"},
        ],
        "aggregation": "final", "topology": "sequential", "max_cost": 3, "max_iterations": 1,
    }
    executor = MASExecutor("http://unused/v1", "subagent", max_agent_tokens=8192, manage_sleep=False, max_model_len=32768)
    client = _GatePlannerAndAnswerClient([
        {"action": "final", "search_query": "", "reason": "enough"},
    ])
    executor.client = client

    result = executor.execute("Question?", mas, "answer only")

    assert calls == ["Find the entity matching the role-specific clues."]
    assert len(client.initial_query_prompts) == 1
    assert len(result["metadata"]["retrieval_sessions"]) == 1
    assert result["metadata"]["retrieval_sessions"][0]["agent_id"] == "searcher"
    assert [trace["search_mode"] for trace in result["agent_traces"]] == [
        "bcp_multi_turn_bm25", "none", "none",
    ]
    assert len(client.planner_prompts) == 1
    assert len(client.answer_prompts) == 3
    assert "[Sub-agent role]\nfinal" in client.answer_prompts[-1]
    assert "Output only the final answer text" not in client.answer_prompts[-1]


def test_non_bcp_orchestra_prompt_does_not_inject_bcp_contract():
    prompt = orchestra_prompt("Solve 1+1.", {"dataset": "deepscaler"})
    assert "BrowseComp-Plus MAS contract" not in prompt[-1]["content"]
