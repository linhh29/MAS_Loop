import json
from argparse import Namespace
from pathlib import Path

import pytest

from mas_loop_rl.bcp_judge import (
    BCPJudge,
    JUDGE_GUIDED_JSON_SCHEMA,
    _parse_judge_text,
    _validate_judge_text,
    grader_template_for,
    is_bcp_judge_enabled,
    is_llm_judge_enabled,
)
from mas_loop_rl.eval_bcp_with_judge import collect_candidates
from mas_loop_rl.sharded_jsonl import iter_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dataset_aware_prompt_selection_preserves_bcp_contract() -> None:
    hotpot, hotpot_version = grader_template_for("hotpotqa_test")
    bcp, bcp_version = grader_template_for("bcp")
    assert hotpot_version == "hotpotqa_semantic_judge_v1"
    assert "Do not solve the question" in hotpot
    assert "yes/no sentence" in hotpot
    assert bcp_version == "BrowseComp-Plus/judge.py::GRADER_TEMPLATE"
    assert "small margin of error for numerical problems" in bcp
    assert "For HotpotQA" not in bcp


def test_semantic_judge_enablement_is_dataset_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAS_LOOP_USE_LLM_JUDGE", raising=False)
    for dataset, retriever in (("hotpotqa", ""), ("", "hotpotqa_bm25"), ("bcp", "")):
        monkeypatch.setenv("MAS_LOOP_DATASET_NAME", dataset)
        monkeypatch.setenv("MAS_LOOP_RETRIEVER", retriever)
        assert is_llm_judge_enabled()
    for dataset in ("aime", "deepscaler"):
        monkeypatch.setenv("MAS_LOOP_DATASET_NAME", dataset)
        monkeypatch.setenv("MAS_LOOP_RETRIEVER", "")
        assert not is_llm_judge_enabled()
    monkeypatch.setenv("MAS_LOOP_DATASET_NAME", "hotpotqa")
    monkeypatch.setenv("MAS_LOOP_USE_LLM_JUDGE", "0")
    assert not is_llm_judge_enabled()


def test_judge_rejects_remote_url_and_accepts_loopback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        BCPJudge(base_url="https://example.com/v1", model="judge", result_dir=tmp_path / "bad")
    judge = BCPJudge(
        base_url="http://localhost:1/v1", model="judge", manage_sleep=False,
        result_dir=tmp_path / "ok",
    )
    assert judge.base_url == "http://localhost:1/v1"


def test_hotpotqa_trajectory_uses_semantic_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    judge = BCPJudge(
        base_url="http://127.0.0.1:1/v1", model="judge", manage_sleep=False,
        result_dir=tmp_path / "judge_results",
    )
    monkeypatch.setattr(
        judge.client,
        "chat_once",
        lambda *args, **kwargs: json.dumps({
            "extracted_final_answer": "no",
            "reasoning": "The explanatory sentence concludes no.",
            "correct": "yes",
            "confidence": 100,
        }),
    )
    trajectory = {"final_output": "No, they were not born in the same country.", "metadata": {"score": 0.0}}
    result = judge.judge_trajectory(
        trajectory, question="Were they born in the same country?", correct_answer="no",
        pair_id="hp-1", data_source="hotpotqa_train",
    )
    assert result["score"] == 1.0
    assert result["judge_dataset"] == "hotpotqa"
    assert result["prompt_version"] == "hotpotqa_semantic_judge_v1"
    assert trajectory["metadata"]["scoring_method"] == "hotpotqa_qwen_llm_judge"


def test_guided_schema_bounds_free_form_fields_before_required_verdict() -> None:
    properties = JUDGE_GUIDED_JSON_SCHEMA["properties"]
    assert properties["extracted_final_answer"]["maxLength"] == 1024
    assert properties["reasoning"]["maxLength"] == 2048
    assert JUDGE_GUIDED_JSON_SCHEMA["required"] == [
        "extracted_final_answer", "reasoning", "correct", "confidence"
    ]


def test_parse_json_yes_and_no() -> None:
    yes = _parse_judge_text(json.dumps({
        "extracted_final_answer": "Paris", "reasoning": "match", "correct": "yes", "confidence": "91%"
    }))
    no = _parse_judge_text(json.dumps({
        "extracted_final_answer": "London", "reasoning": "mismatch", "correct": "no", "confidence": 75
    }))
    assert yes["correct"] == "yes" and yes["confidence"] == 91 and not yes["parse_error"]
    assert no["correct"] == "no" and no["confidence"] == 75 and not no["parse_error"]


def test_parse_fenced_json_and_key_value_fallback() -> None:
    fenced = _parse_judge_text(
        '```json\n{"extracted_final_answer":"42","reasoning":"same","correct":"yes","confidence":100}\n```'
    )
    fallback = _parse_judge_text(
        "extracted_final_answer: 43\nreasoning: differs\ncorrect: no\nconfidence: 88%"
    )
    assert fenced["correct"] == "yes" and not fenced["parse_error"]
    assert fallback["correct"] == "no" and fallback["confidence"] == 88 and not fallback["parse_error"]


def test_parse_empty_malformed_and_confidence_clamp() -> None:
    assert _parse_judge_text("")["parse_error"] is True
    malformed = _parse_judge_text("not a judgement")
    assert malformed["correct"] == "no" and malformed["parse_error"] is True
    assert _parse_judge_text("correct: yes\nconfidence: 120%")["confidence"] == 100
    assert _parse_judge_text("correct: no\nconfidence: 0%")["confidence"] == 0


def test_parse_jsonish_output_with_unescaped_quotes_in_answer() -> None:
    malformed_json = '''{
  "extracted_final_answer": "Go Tell It on the Mountain" by James Baldwin,
  "reasoning": "It does not match the reference answer.",
  "correct": "no",
  "confidence": 100
}'''
    parsed = _parse_judge_text(malformed_json)
    assert parsed["extracted_final_answer"] == 'Go Tell It on the Mountain" by James Baldwin'
    assert parsed["reasoning"] == "It does not match the reference answer."
    assert parsed["correct"] == "no"
    assert parsed["confidence"] == 100
    assert parsed["parse_error"] is False


def test_jsonish_fallback_does_not_infer_verdict_from_reasoning() -> None:
    missing_top_level_correct = '''{
  "extracted_final_answer": "Paris",
  "reasoning": "The correct answer is Paris, so this should be yes.",
  "confidence": 100
}'''
    assert _parse_judge_text(missing_top_level_correct)["parse_error"] is True

    conflicting = '"correct": "yes",\n"correct": "no",\n"confidence": 100'
    assert _parse_judge_text(conflicting)["parse_error"] is True


def test_validate_rejects_json_missing_required_correct_field() -> None:
    incomplete = json.dumps({
        "extracted_final_answer": "None",
        "correct_answer": "Paris",
        "reasoning": "No exact answer was extracted.",
        "confidence": 100,
    })
    with pytest.raises(ValueError, match="required `correct: yes\\|no` field"):
        _validate_judge_text(incomplete)


def test_judge_guided_json_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAS_LOOP_JUDGE_GUIDED_JSON", raising=False)
    default_judge = BCPJudge(
        base_url="http://127.0.0.1:1/v1", model="judge", manage_sleep=False,
        result_dir=tmp_path / "default",
    )
    assert default_judge.guided_json_enabled is False

    monkeypatch.setenv("MAS_LOOP_JUDGE_GUIDED_JSON", "1")
    guided_judge = BCPJudge(
        base_url="http://127.0.0.1:1/v1", model="judge", manage_sleep=False,
        result_dir=tmp_path / "guided",
    )
    assert guided_judge.guided_json_enabled is True


def test_judge_retries_missing_correct_with_guided_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRIES", "2")
    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRY_DELAY_SEC", "0")
    monkeypatch.setenv("MAS_LOOP_JUDGE_GUIDED_JSON", "1")
    judge = BCPJudge(
        base_url="http://127.0.0.1:1/v1",
        model="judge",
        manage_sleep=False,
        result_dir=tmp_path / "judge_results",
    )
    replies = iter([
        json.dumps({
            "extracted_final_answer": "None",
            "correct_answer": "Paris",
            "reasoning": "Missing the required verdict.",
            "confidence": 100,
        }),
        json.dumps({
            "extracted_final_answer": "Paris",
            "reasoning": "Exact match.",
            "correct": "yes",
            "confidence": 100,
        }),
    ])
    calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat_once(messages, **kwargs):
        calls.append((messages, kwargs))
        kwargs["diagnostics_out"].update({"stream": True, "attempt_count": 1, "attempts": []})
        return next(replies)

    monkeypatch.setattr(judge.client, "chat_once", chat_once)
    result = judge.judge(question="Capital?", response="Paris", correct_answer="Paris")

    assert result["score"] == 1.0
    assert result["parse_error"] is False
    assert result["transport_diagnostics"]["attempt_count"] == 2
    assert result["transport_diagnostics"]["retry_count"] == 1
    assert len(calls) == 2
    assert all(kwargs["guided_json"] == JUDGE_GUIDED_JSON_SCHEMA for _, kwargs in calls)
    assert "all four keys" in calls[1][0][-1]["content"]


def test_judge_trajectory_replaces_score_and_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAS_LOOP_JUDGE_MAX_TOKENS", "4096")
    judge = BCPJudge(
        base_url="http://127.0.0.1:1/v1", model="judge", manage_sleep=False,
        max_model_len=23192, result_dir=tmp_path / "judge_results",
    )
    monkeypatch.setattr(
        judge.client,
        "chat_once",
        lambda *args, **kwargs: json.dumps({
            "extracted_final_answer": "Paris", "reasoning": "exact", "correct": "yes", "confidence": 100
        }),
    )
    trajectory = {"final_output": "Paris", "metadata": {"score": 0.25}}
    result = judge.judge_trajectory(
        trajectory, question="Capital?", correct_answer="Paris", pair_id="p1", candidate="source"
    )
    assert result["score"] == 1.0
    assert trajectory["metadata"]["score"] == 1.0
    assert trajectory["metadata"]["heuristic_score_before_llm_judge"] == 0.25
    assert trajectory["metadata"]["scoring_method"] == "qwen_llm_judge"
    records = list(iter_jsonl([tmp_path / "judge_results"]))
    assert len(records) == 1
    assert records[0]["pair_id"] == "p1"
    assert records[0]["judge"]["max_tokens"] == 4096


def test_judge_failure_is_fail_closed_not_heuristic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRIES", "2")
    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRY_DELAY_SEC", "0")
    judge = BCPJudge(
        base_url="http://127.0.0.1:1/v1", model="judge", manage_sleep=False,
        result_dir=tmp_path / "judge_results",
    )
    def fail(*args, **kwargs):
        raise RuntimeError("transport down")
    monkeypatch.setattr(judge.client, "chat_once", fail)
    trajectory = {"final_output": "Paris", "metadata": {"score": 1.0}}
    result = judge.judge_trajectory(trajectory, question="Capital?", correct_answer="Paris")
    assert result["score"] == -1.0
    assert "transport down" in result["error"]
    assert trajectory["metadata"]["score"] == -1.0
    assert trajectory["metadata"]["heuristic_score_before_llm_judge"] == 1.0


def test_training_launcher_requires_real_judge_completion_smoke_test() -> None:
    launcher = (REPO_ROOT / "scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    assert "smoke_test_bcp_judge" in launcher
    assert 'question="What is 2 + 2?"' in launcher
    assert 'correct_answer="4"' in launcher
    assert 'MAS_LOOP_JUDGE_MAX_TOKENS="${MAS_LOOP_JUDGE_MAX_TOKENS}"' in launcher
    assert 'max_tokens=${MAS_LOOP_JUDGE_MAX_TOKENS}' in launcher
    assert 'if result.get("score") != 1.0:' in launcher
    assert 'sleep_vllm "${JUDGE_PORT}" || true' in launcher
    assert launcher.count("runtime_env.env_vars.MAS_LOOP_JUDGE_MAX_TOKENS") == 2
    assert launcher.count("runtime_env.env_vars.MAS_LOOP_JUDGE_BASE_URL") == 2
    assert launcher.count("runtime_env.env_vars.MAS_LOOP_JUDGE_RESULTS_DIR") == 2


def test_bcp_enablement_is_explicit_and_non_bcp_can_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAS_LOOP_DATASET_NAME", "bcp")
    monkeypatch.delenv("MAS_LOOP_USE_LLM_JUDGE", raising=False)
    assert is_bcp_judge_enabled()
    monkeypatch.setenv("MAS_LOOP_USE_LLM_JUDGE", "0")
    assert not is_bcp_judge_enabled()
    monkeypatch.setenv("MAS_LOOP_DATASET_NAME", "deepscaler")
    assert not is_bcp_judge_enabled()


def test_collect_source_updated_from_canonical_pair(tmp_path: Path) -> None:
    shard = tmp_path / "canonical_rollouts" / "orchestra-1.jsonl"
    shard.parent.mkdir(parents=True)
    record = {
        "pair_id": "pair-1",
        "structured_pair": {
            "pair_id": "pair-1",
            "optimizer_step": 3,
            "outer_adversarial_round": 1,
            "task": {"task_text": "Q", "expected_answer": "A"},
            "source": {"trajectory": {"final_output": "source answer"}},
            "updated": {"trajectory": {"final_output": "updated answer"}},
        },
    }
    shard.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rows = collect_candidates([str(shard.parent)], {"source", "updated"})
    assert [(x["candidate"], x["response"]) for x in rows] == [
        ("source", "source answer"), ("updated", "updated answer")
    ]
    assert all(x["pair_id"] == "pair-1" and x["correct_answer"] == "A" for x in rows)
