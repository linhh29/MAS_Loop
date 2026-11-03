from __future__ import annotations

import contextlib

from mas_loop_rl.executor import MASExecutor
from mas_loop_rl.schema import MAS_GUIDED_JSON_SCHEMA, normalize_mas


class _FakeClient:
    max_model_len = 8192
    truncate_safety_margin = 32

    def __init__(self):
        self.max_tokens = []

    @contextlib.contextmanager
    def awake_session(self):
        yield

    def chat_once(self, messages, *, max_tokens, temperature):
        self.max_tokens.append(max_tokens)
        return "answer"


def test_model_authored_agent_token_budget_is_clamped():
    executor = MASExecutor(
        base_url="http://unused/v1",
        model="subagent",
        max_agent_tokens=1024,
        manage_sleep=False,
        max_model_len=8192,
    )
    fake = _FakeClient()
    executor.client = fake
    mas = {
        "agents": [
            {
                "id": "solver",
                "role": "solve",
                "tools": [],
                "budget": 1,
                "max_tokens": 1_000_000,
            }
        ],
        "edges": [],
        "aggregation": "solver",
        "topology": "sequential",
        "max_cost": 1,
        "max_iterations": 1,
    }

    result = executor.execute("question", mas)

    assert fake.max_tokens == [1024]
    assert result["final_output"] == "answer"
    # The persisted normalized MAS is bounded too, so pair logs cannot retain
    # an impossible million-token action as if it were executable.
    assert result["mas_config"]["agents"][0]["max_tokens"] == 8192


def test_guided_schema_and_normalizer_bound_declared_agent_tokens():
    maximum = MAS_GUIDED_JSON_SCHEMA["properties"]["agents"]["items"]["properties"]["max_tokens"]["maximum"]
    assert maximum == 8192
    normalized = normalize_mas(
        {
            "agents": [{"id": "a", "role": "r", "tools": [], "budget": 1, "max_tokens": 1_000_000}],
            "edges": [],
            "aggregation": "a",
            "topology": "sequential",
            "max_cost": 1,
            "max_iterations": 1,
        }
    )
    assert normalized["agents"][0]["max_tokens"] == 8192
