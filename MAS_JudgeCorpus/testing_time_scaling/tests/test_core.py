from pathlib import Path

from testing_time_scaling.accounting import normalize_usage, sum_usage
from testing_time_scaling.io_utils import append_jsonl, completed_keys
from testing_time_scaling.scoring import extract_answer, is_correct, majority_vote
from testing_time_scaling.selection import METHODS, cot_decisions, mas_baseline_decisions, parse_selected_index


def test_usage_normalization_and_sum():
    a=normalize_usage({"prompt_tokens":3,"completion_tokens":2})
    assert (a["input_tokens"],a["output_tokens"],a["total_tokens"])==(3,2,5)
    total=sum_usage([a,{"input_tokens":4,"output_tokens":1,"total_tokens":5,"usage_source":"api"}])
    assert total["total_tokens"]==10 and total["num_calls"]==2


def test_usage_sum_preserves_nested_sources():
    total = sum_usage([{
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "usage_sources": ["api", "tokenizer_exact"],
    }])
    assert total["usage_sources"] == ["api", "tokenizer_exact"]



def test_task_local_usage_capture_isolation():
    import asyncio
    from testing_time_scaling.clients import OpenAICompatibleClient

    async def run():
        client = OpenAICompatibleClient(
            name="test", model="dummy", base_url="http://127.0.0.1:1", api_key="x"
        )

        async def worker(role: str, tokens: int):
            with client.capture_usage() as ledger:
                await asyncio.sleep(0)
                client._record_usage(
                    role=role,
                    usage={"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1},
                    metadata={"role": role},
                )
                await asyncio.sleep(0)
            return ledger.totals()

        first, second = await asyncio.gather(worker("a", 3), worker("b", 7))
        assert first["total_tokens"] == 4 and first["num_calls"] == 1
        assert second["total_tokens"] == 8 and second["num_calls"] == 1
        assert client.ledger.totals()["total_tokens"] == 12
        await client.close()

    asyncio.run(run())

def test_answer_and_majority_tie_break():
    assert extract_answer("reason\n**Final Answer:** Paris") == "Paris"
    assert is_correct("Final Answer: The Eiffel Tower", "Eiffel Tower")
    idx,answer,votes=majority_vote(["Final Answer: A","Final Answer: B","Final Answer: B","Final Answer: A"])
    assert idx==0 and answer=="A" and votes["a"]==2


def test_judge_json_parser():
    assert parse_selected_index('```json\n{"selected_index": 7}\n```',16)==7
    assert parse_selected_index('selected index: 3',16)==3


def test_cot_token_attribution():
    rows=[]
    for i,ans in enumerate(["x","y","y"]):
        rows.append({"candidate_idx":i,"raw_output":f"Final Answer: {ans}","prediction":ans,"correct":ans=="y","gold":"y","shared_retrieval_usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12},"candidate_usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}})
    d=cot_decisions(rows)
    assert d["cot_lb"]["usage"]["total_tokens"]==18
    assert d["majority_voting"]["usage"]["total_tokens"]==30
    assert d["oracle_ub"]["correct"] is True


def test_mas_baseline_token_attribution():
    rows=[]
    for i,ans in enumerate(["x","y","y"]):
        rows.append({"candidate_idx":i,"raw_output":f"Final Answer: {ans}","prediction":ans,"correct":ans=="y","gold":"y","candidate_usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}})
    d=mas_baseline_decisions(rows)
    assert d["mas_lb"]["usage"]["total_tokens"]==6
    assert d["mas_majority_voting"]["usage"]["total_tokens"]==18
    assert d["mas_oracle_ub"]["correct"] is True


def test_resume_keys(tmp_path: Path):
    p=tmp_path/"x.jsonl"; append_jsonl(p,{"dataset":"bcp","sample_id":"s","candidate_idx":0})
    assert ("bcp","s",0) in completed_keys(p,("dataset","sample_id","candidate_idx"))


def test_hotpot_nested_lucene_document_extraction():
    from testing_time_scaling.retrieval import HotpotLuceneRetriever

    inner = '{"title":"Example","text":"Title: Example\\nExpected passage."}'
    outer = '{"id":"doc-1","contents":' + __import__('json').dumps(inner) + '}'
    assert HotpotLuceneRetriever._extract_text(inner) == "Title: Example\nExpected passage."
    assert HotpotLuceneRetriever._extract_text(outer) == "Title: Example\nExpected passage."


def test_vendored_pyserini_runtime_configuration(monkeypatch):
    import os
    import sys
    from testing_time_scaling.retrieval import _configure_vendored_pyserini

    monkeypatch.delenv("PYSERINI_DISABLE_VECTOR_MODULE", raising=False)
    monkeypatch.delenv("PYSERINI_SPARSE_ONLY", raising=False)
    _configure_vendored_pyserini()
    assert os.environ["PYSERINI_DISABLE_VECTOR_MODULE"] == "1"
    assert os.environ["PYSERINI_SPARSE_ONLY"] == "1"
    assert "jdk4py/java-runtime" in os.environ.get("JAVA_HOME", "")
    assert any(path.endswith("testing_time_scaling/vendor") for path in sys.path)


def test_added_qwen_judge_registered():
    assert "qwen3_8_2_4t_a95b" in METHODS
    assert "qwen3_8_2_4t_a95b_trajectory" in METHODS


def test_endpoint_alias_and_root_secret_shape(tmp_path: Path):
    import yaml
    from testing_time_scaling.config import load_config

    config = tmp_path / "config.yaml"
    secrets = tmp_path / "api_secrets.yaml"
    config.write_text("endpoints: {}\n")
    secrets.write_text(yaml.safe_dump({"endpoints": {"qwen3.8-2.4t-a95b": {"base_url": "https://x/v1", "model": "q"}}}))
    loaded = load_config(config, secrets)
    assert loaded["endpoints"]["qwen3_8_2_4t_a95b"]["model"] == "q"


def test_rate_limiter_spacing():
    import asyncio
    from testing_time_scaling.clients import AsyncRateLimiter

    async def run():
        limiter = AsyncRateLimiter(6000)  # 10 ms spacing keeps test fast.
        loop = asyncio.get_running_loop()
        starts = []
        async def take():
            await limiter.acquire()
            starts.append(loop.time())
        await asyncio.gather(*(take() for _ in range(3)))
        starts.sort()
        assert starts[1] - starts[0] >= 0.008
        assert starts[2] - starts[1] >= 0.008
    asyncio.run(run())


def test_summary_uses_selector_only_tokens():
    from testing_time_scaling.summarize import summarize_method

    rows = [{
        "correct": True,
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "usage_source": "api"},
        "selector_usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10, "usage_source": "tokenizer_exact"},
    }]
    summary = summarize_method(rows)
    assert summary["total_tokens"] == 10
    assert summary["token_m"] == 0.0
    assert summary["token_scope"] == "selector_only"
    assert summary["end_to_end_total_tokens"] == 120


def test_summary_missing_selector_usage_is_zero():
    from testing_time_scaling.summarize import summarize_method

    summary = summarize_method([{
        "correct": True,
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7, "usage_source": "api"},
    }])
    assert summary["total_tokens"] == 0
    assert summary["end_to_end_total_tokens"] == 7
