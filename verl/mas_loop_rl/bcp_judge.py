"""BrowseComp-Plus Qwen LLM-as-judge scoring for MAS-Loop.

The grading prompt, system prompt, and yes/no semantics are copied from
``datasets/BrowseComp-Plus/judge.py`` so online GRPO reward and offline BCP
evaluation use the same criterion.  Judge failures are fail-closed: they are
persisted with score -1 and are never silently replaced by string matching.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .local_client import LocalChatClient, LocalChatError
from .sharded_jsonl import append_jsonl_shard


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()

HOTPOTQA_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct with respect to the precise [correct_answer].

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Extract exactly one final answer from [response], then compare only that extracted answer with [correct_answer]. Do not solve the question and do not use external knowledge.

For HotpotQA, answer 'yes' when the extracted answer is semantically equivalent to [correct_answer]. This includes an unambiguous short gold answer stated inside a complete sentence, equivalent entity aliases, equivalent dates or numbers, and an explanatory yes/no sentence whose final conclusion matches a gold answer of yes or no. Harmless supporting explanation is allowed when it does not contradict the answer.

Answer 'no' when no final answer can be extracted, multiple conflicting final answers are given, the answer is broader or narrower in a meaning-changing way, or any explanation contradicts the extracted final answer.

Return these fields:
extracted_final_answer: the single final answer, or 'None'.
reasoning: briefly explain only the equivalence or meaningful difference.
correct: exactly 'yes' or 'no'.
confidence: a number from 0 to 100; use 100 when the response gives no confidence.
""".strip()


def _normalize_data_source(data_source: str) -> str:
    value = (data_source or "").strip().lower().replace("-", "_")
    if "hotpot" in value:
        return "hotpotqa"
    if value in {"bcp", "browsecomp", "browsecomp_plus"} or "browsecomp" in value:
        return "bcp"
    return value or "bcp"


def grader_template_for(data_source: str) -> tuple[str, str]:
    if _normalize_data_source(data_source) == "hotpotqa":
        return HOTPOTQA_GRADER_TEMPLATE, "hotpotqa_semantic_judge_v1"
    return GRADER_TEMPLATE, "BrowseComp-Plus/judge.py::GRADER_TEMPLATE"


JUDGE_SYSTEM = (
    "You are a strict QA grader. "
    "Output only JSON with keys: extracted_final_answer, reasoning, correct, confidence."
)

JUDGE_GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        # Bound free-form fields so a pathological candidate cannot make the
        # Judge copy a repetitive answer until the full 4096-token completion
        # budget is exhausted before the required verdict is emitted.
        "extracted_final_answer": {"type": "string", "maxLength": 1024},
        "reasoning": {"type": "string", "maxLength": 2048},
        "correct": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": ["number", "string"]},
    },
    "required": ["extracted_final_answer", "reasoning", "correct", "confidence"],
    "additionalProperties": False,
}


def is_valid_bcp_judge_metadata(value: Any, *, max_tokens: int | None = None) -> bool:
    """Return whether persisted Judge metadata satisfies the grading contract.

    A transport/parser failure is represented by ``score=-1`` and must remain
    distinguishable from a valid incorrect answer (``correct=no, score=0``).
    This validator deliberately rejects fail-closed sentinel records so a
    post-training backfill cannot mistakenly treat them as completed evidence.
    """
    if not isinstance(value, dict) or value.get("parse_error") is not False:
        return False
    correct = value.get("correct")
    if correct not in {"yes", "no"}:
        return False
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    expected_score = 1.0 if correct == "yes" else 0.0
    if not math.isfinite(float(score)) or not math.isclose(float(score), expected_score, abs_tol=1e-9):
        return False
    if not isinstance(value.get("extracted_final_answer"), str):
        return False
    if not isinstance(value.get("reasoning"), str):
        return False
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 100.0:
        return False
    if not isinstance(value.get("model"), str) or not value.get("model"):
        return False
    if max_tokens is not None:
        persisted_max_tokens = value.get("max_tokens")
        if isinstance(persisted_max_tokens, bool) or not isinstance(persisted_max_tokens, int):
            return False
        if persisted_max_tokens != int(max_tokens):
            return False
    return True


def _parse_judge_text(text: str) -> dict[str, Any]:
    """Parse judge output with the official JSON-first/key-value fallback."""
    raw = (text or "").strip()
    if not raw:
        return {
            "extracted_final_answer": "None",
            "reasoning": "",
            "correct": "no",
            "confidence": 100.0,
            "raw_judgement": raw,
            "parse_error": True,
        }

    json_candidate = raw
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.I | re.S)
    if fenced:
        json_candidate = fenced.group(1).strip()
    try:
        obj = json.loads(json_candidate)
        if isinstance(obj, dict):
            ext = str(obj.get("extracted_final_answer", "None"))
            reason = str(obj.get("reasoning", ""))
            has_correct = "correct" in obj
            corr_raw = str(obj.get("correct", "no")).strip().lower()
            conf_raw = str(obj.get("confidence", "100"))
            conf_num = re.findall(r"\d+(?:\.\d+)?", conf_raw)
            conf = float(conf_num[0]) if conf_num else 100.0
            valid_correct = has_correct and corr_raw in {"yes", "no"}
            return {
                "extracted_final_answer": ext,
                "reasoning": reason,
                "correct": "yes" if corr_raw == "yes" else "no",
                "confidence": max(0.0, min(100.0, conf)),
                "raw_judgement": raw,
                "parse_error": not valid_correct,
            }
    except Exception:
        pass

    def _extract(prefix: str) -> str:
        # Qwen occasionally emits a JSON-looking object whose free-form answer
        # contains unescaped double quotes.  json.loads must reject that object,
        # but the later top-level fields are still independently recoverable.
        # Security: anchor the allow-listed field name at the start of a line so a
        # ``correct`` phrase inside free-form reasoning can never become a score.
        match = re.search(
            rf"(?im)^\s*[\"']?{re.escape(prefix)}[\"']?\s*:\s*(.*)$",
            raw,
        )
        return match.group(1).strip() if match else ""

    def _extract_required_correct() -> str:
        # Security: only an unambiguous, line-delimited top-level yes/no field is
        # accepted.  Missing, duplicated, or conflicting verdicts fail closed.
        matches = re.findall(
            r"(?im)^\s*[\"']?correct[\"']?\s*:\s*[\"']?(yes|no)[\"']?\s*,?\s*$",
            raw,
        )
        normalized = {value.lower() for value in matches}
        return normalized.pop() if len(matches) == 1 and len(normalized) == 1 else ""

    def _clean_jsonish_scalar(value: str) -> str:
        value = value.strip().rstrip(",").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value.lstrip('"\'').rstrip('"\'').strip()

    ext = _clean_jsonish_scalar(_extract("extracted_final_answer")) or "None"
    corr_text = _extract_required_correct()
    corr_raw = (corr_text or "no").strip().lower()
    reason = _clean_jsonish_scalar(_extract("reasoning"))
    conf_raw = _extract("confidence") or "100"
    conf_num = re.findall(r"\d+(?:\.\d+)?", conf_raw)
    conf = float(conf_num[0]) if conf_num else 100.0
    valid_correct = bool(corr_text) and corr_raw in {"yes", "no"}
    return {
        "extracted_final_answer": ext,
        "reasoning": reason,
        "correct": "yes" if corr_raw == "yes" else "no",
        "confidence": max(0.0, min(100.0, conf)),
        "raw_judgement": raw,
        "parse_error": not valid_correct,
    }


def _validate_judge_text(text: str) -> dict[str, Any]:
    """Require the complete grading contract before accepting a completion."""
    parsed = _parse_judge_text(text)
    if parsed.get("parse_error"):
        raise ValueError("judge response lacks a valid required `correct: yes|no` field")
    return parsed


def is_llm_judge_enabled() -> bool:
    dataset = os.getenv("MAS_LOOP_DATASET_NAME", "").strip().lower()
    # Mathematical datasets always use deterministic math equivalence, even if
    # a stale shell environment happens to request the retrieval-task judge.
    if dataset in {
        "aime", "hlemath", "deepscaler", "deep_scale_r", "deepscaler_math",
        "mas_loop_aime", "mas_loop_hlemath", "mas_loop_deepscaler",
    }:
        return False
    explicit = os.getenv("MAS_LOOP_USE_LLM_JUDGE", "").strip().lower()
    if explicit:
        return explicit not in {"0", "false", "no"}
    retriever = os.getenv("MAS_LOOP_RETRIEVER", "").strip().lower()
    return dataset in {"bcp", "browsecomp", "browsecomp_plus", "hotpotqa", "hotpot_qa"} or retriever in {
        "bm25", "bcp_bm25", "browsecomp_bm25", "hotpotqa_bm25"
    }


def is_bcp_judge_enabled() -> bool:
    """Backward-compatible alias for callers that still use the BCP name."""
    return is_llm_judge_enabled()


def _validate_loopback_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    # 安全说明：Judge 只能访问本机 vLLM，禁止环境变量把题目和答案转发到远端。
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.port is None or parsed.username is not None or parsed.password is not None):
        raise ValueError("MAS-Loop Judge base_url must be an explicit loopback HTTP endpoint")
    return base_url.rstrip("/")


class BCPJudge:
    """Synchronous local-vLLM BCP grader used after each completed MAS execution."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        manage_sleep: bool | None = None,
        max_model_len: int | None = None,
        result_dir: str | Path | None = None,
    ) -> None:
        self.base_url = _validate_loopback_base_url(str(base_url or os.getenv("MAS_LOOP_JUDGE_BASE_URL", "")))
        self.model = str(model or os.getenv("MAS_LOOP_JUDGE_MODEL", "judge"))
        if max_model_len is None:
            max_model_len = int(os.getenv("MAS_LOOP_JUDGE_MAX_MODEL_LEN", "23192"))
        self.max_tokens = int(os.getenv("MAS_LOOP_JUDGE_MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("MAS_LOOP_JUDGE_TEMPERATURE", "0"))
        guided_value = os.getenv("MAS_LOOP_JUDGE_GUIDED_JSON", "0").strip().lower()
        self.guided_json_enabled = guided_value not in {"0", "false", "no"}
        self.client = LocalChatClient(
            base_url=self.base_url,
            model=self.model,
            manage_sleep=manage_sleep,
            max_model_len=max_model_len,
            active_role="judge",
        )
        default_dir = Path(os.getenv("MAS_LOOP_LOG_DIR", "/path/to/checkpoints/unscoped/rollouts")) / "judge_results"
        self.result_dir = Path(result_dir or os.getenv("MAS_LOOP_JUDGE_RESULTS_DIR", str(default_dir)))

    def lease_client(self) -> LocalChatClient:
        """Return a lightweight sleep/admission client for async AgentLoop phases."""
        return LocalChatClient(
            base_url=self.base_url,
            model=self.model,
            manage_sleep=True,
            max_model_len=self.client.max_model_len,
            active_role="judge",
        )

    def judge(
        self,
        *,
        question: str,
        response: str,
        correct_answer: str,
        pair_id: str = "",
        candidate: str = "source",
        optimizer_step: int = 0,
        outer_adversarial_round: int = 1,
        data_source: str = "bcp",
    ) -> dict[str, Any]:
        template, prompt_version = grader_template_for(data_source)
        prompt = template.format(
            question=(question or "").strip(),
            response=(response or "").strip(),
            correct_answer=(correct_answer or "").strip(),
        )
        started = time.monotonic()
        diagnostics: dict[str, object] = {}
        raw = ""
        error = ""
        try:
            raw, parsed, diagnostics = self.client.chat_validated(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                validator=_validate_judge_text,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # The strict validator/retry path remains fail-closed when
                # guided JSON is disabled. This avoids xgrammar's single-CPU
                # scheduler bottleneck at high Judge concurrency.
                guided_json=JUDGE_GUIDED_JSON_SCHEMA if self.guided_json_enabled else None,
                retry_instruction=(
                    "The previous grading response violated the required schema. Return exactly one compact JSON "
                    "object with all four keys: extracted_final_answer, reasoning, correct, confidence. "
                    "The correct value must be exactly \"yes\" or \"no\". Do not add correct_answer or other keys."
                ),
            )
            score = 1.0 if parsed.get("correct") == "yes" else 0.0
        except Exception as exc:
            if isinstance(exc, LocalChatError):
                raw = exc.last_response
                diagnostics = {
                    "attempt_count": len(exc.diagnostics),
                    "retry_count": max(0, len(exc.diagnostics) - 1),
                    "attempts": list(exc.diagnostics),
                }
            parsed = _parse_judge_text(raw)
            error = f"{type(exc).__name__}:{exc}"
            score = -1.0

        elapsed = time.monotonic() - started
        result = {
            **parsed,
            "score": score,
            "error": error,
            "model": self.model,
            "base_url": self.base_url,
            "latency_sec": elapsed,
            "max_tokens": self.max_tokens,
            "max_model_len": self.client.max_model_len,
            "temperature": self.temperature,
            "transport_diagnostics": diagnostics,
            "prompt_version": prompt_version,
            "judge_dataset": _normalize_data_source(data_source),
        }
        record = {
            "schema_version": "mas_loop_semantic_llm_judge_v1",
            "time": time.time(),
            "pair_id": pair_id,
            "candidate": candidate,
            "optimizer_step": int(optimizer_step),
            "outer_adversarial_round": int(outer_adversarial_round),
            "data_source": data_source,
            "question": question,
            "correct_answer": correct_answer,
            "response": response,
            "judge_prompt": prompt,
            "judge": result,
        }
        append_jsonl_shard(self.result_dir, record, shard_prefix="judge")
        return result

    def judge_trajectory(
        self,
        trajectory: dict[str, Any],
        *,
        question: str,
        correct_answer: str,
        pair_id: str = "",
        candidate: str = "source",
        optimizer_step: int = 0,
        outer_adversarial_round: int = 1,
        data_source: str = "bcp",
    ) -> dict[str, Any]:
        result = self.judge(
            question=question,
            response=str(trajectory.get("final_output") or ""),
            correct_answer=correct_answer,
            pair_id=pair_id,
            candidate=candidate,
            optimizer_step=optimizer_step,
            outer_adversarial_round=outer_adversarial_round,
            data_source=data_source,
        )
        metadata = trajectory.setdefault("metadata", {})
        metadata["heuristic_score_before_llm_judge"] = metadata.get("score")
        metadata["score"] = float(result["score"])
        metadata["judge"] = result
        normalized_source = _normalize_data_source(data_source)
        metadata["scoring_method"] = (
            "qwen_llm_judge" if normalized_source == "bcp" else f"{normalized_source}_qwen_llm_judge"
        )
        trajectory["success"] = float(result["score"]) >= 0.999
        return result
