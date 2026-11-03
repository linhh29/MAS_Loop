from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.utils.profiler import simple_timer

from mas_loop_rl.bcp_judge import BCPJudge, is_bcp_judge_enabled
from mas_loop_rl.dispatch import run_blocking
from mas_loop_rl.executor import MASExecutor, score_output
from mas_loop_rl.local_client import LocalChatClient, LocalChatError, strip_reasoning
from mas_loop_rl.prompts import compact_trajectory_for_editor, editor_prompt, editor_retry_instruction, orchestra_prompt
from mas_loop_rl.schema import (
    MAS_GUIDED_JSON_SCHEMA, PATCH_GUIDED_JSON_SCHEMA, apply_patch, is_accept_patch,
    normalize_editor_action, normalize_mas, normalize_patch, validate_mas,
)
from mas_loop_rl.sharded_jsonl import append_jsonl_shard


def _python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value



def _expected(kwargs: dict[str, Any]) -> str:
    rm = _python(kwargs.get("reward_model")) or {}
    gt = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
    if isinstance(gt, dict):
        return str(gt.get("target") or gt.get("answer") or "")
    return str(gt or "")


def _extra_info(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = _python(kwargs.get("extra_info")) or {}
    return dict(extra) if isinstance(extra, dict) else {}


def _invalid_action_penalty() -> float:
    """Paper eta: invalid Refiner decisions and wrong Accept receive -eta."""
    try:
        value = float(os.getenv("MAS_LOOP_INVALID_ACTION_PENALTY", "0.5"))
    except (TypeError, ValueError):
        value = 0.5
    return max(0.0, value)


def _editor_guided_json_schema() -> dict[str, Any] | None:
    """Keep external Refiner guided decoding opt-in at scale.

    Strict schema validation and bounded structured retries remain active when
    guided decoding is disabled.  This mirrors the actor/Judge policy and
    avoids the vLLM V1 xgrammar scheduler path that can leave every request in
    ``running`` while making only a few decode steps per minute.
    """
    # XML-first output is incompatible with guided JSON decoding.
    if os.getenv("MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL", "xml").lower() == "xml":
        return None
    enabled = os.getenv("MAS_LOOP_EDITOR_GUIDED_JSON", "0") not in {"0", "false", "False", "no", "No"}
    return PATCH_GUIDED_JSON_SCHEMA if enabled else None


class _MASLoopBase(AgentLoopBase):
    """Common code for role-preserving, one-output MAS adversarial samples.

    A GRPO group is now produced by ``rollout.n`` independent sessions.  An
    AgentLoop invocation therefore returns exactly one trainable output.  The
    adversarial loop lives in parameter space: every optimizer batch updates a
    role, and the next batch is sampled from the updated role policy.
    """

    role = "unknown"
    actor_guided_json_schema: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.max_prompt_length = int(getattr(self.rollout_config, "max_prompt_length", 15000) or 15000)
        self.log_dir = Path(os.getenv("MAS_LOOP_LOG_DIR", "/path/to/checkpoints/unscoped/rollouts"))
        self.pair_dir = Path(os.getenv("MAS_LOOP_PAIR_DIR", str(self.log_dir / "pairs")))
        self.subagent_base_url = os.getenv("MAS_LOOP_SUBAGENT_BASE_URL", "")
        self.subagent_model = os.getenv("MAS_LOOP_SUBAGENT_MODEL", "subagent")
        self.peer_base_url = os.getenv("MAS_LOOP_PEER_BASE_URL", "")
        self.peer_model = os.getenv("MAS_LOOP_PEER_MODEL", "peer")
        self.manage_sleep = os.getenv("MAS_LOOP_VLLM_SLEEP", "1") not in {"0", "false", "False"}
        self.max_agent_tokens = int(os.getenv("MAS_LOOP_MAX_AGENT_TOKENS", "1024"))
        self.peer_max_tokens = int(os.getenv("MAS_LOOP_PEER_MAX_TOKENS", os.getenv("MAS_LOOP_EDITOR_MAX_TOKENS", "1024")))
        self.editor_candidates_per_source = max(1, int(os.getenv("MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE", "1")))
        self.editor_candidate_concurrency = max(
            1,
            min(
                self.editor_candidates_per_source,
                int(os.getenv("MAS_LOOP_EDITOR_CANDIDATE_CONCURRENCY", "2")),
            ),
        )
        self.subagent_max_model_len = int(os.getenv("MAS_LOOP_SUBAGENT_MAX_MODEL_LEN", "8192"))
        self.peer_max_model_len = int(os.getenv("MAS_LOOP_PEER_MAX_MODEL_LEN", str(15000 + 8192)))
        self.use_llm_judge = is_bcp_judge_enabled()
        self.judge_base_url = os.getenv("MAS_LOOP_JUDGE_BASE_URL", "")
        self.judge_model = os.getenv("MAS_LOOP_JUDGE_MODEL", "judge")
        self.judge_max_model_len = int(os.getenv("MAS_LOOP_JUDGE_MAX_MODEL_LEN", str(15000 + 8192)))

    async def _actor_generate(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]):
        prompt_ids = await self.apply_chat_template(messages)
        # Keep the configured 8192-token safety ceiling, but constrain MAS/patch
        # actors to finish as soon as one schema-valid JSON object is complete.
        # This removes pathological 8192-token tails without shortening the
        # user's maximum response length or changing GRPO sampling multiplicity.
        request_sampling_params = dict(sampling_params)
        # Keep actor structured decoding opt-in.  At several thousand concurrent
        # requests vLLM V1's guided-decoding path can stop dispatching model
        # steps while every request remains marked running.  Plain JSON prompting
        # plus the existing parser/retry path is both stochastic (important for
        # GRPO) and substantially more robust at MAS-Loop scale.
        guided_enabled = (
            os.getenv("MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL", "xml").lower() != "xml"
            and os.getenv("MAS_LOOP_ACTOR_GUIDED_JSON", "0") not in {"0", "false", "False"}
        )
        if guided_enabled and self.actor_guided_json_schema is not None:
            from vllm.sampling_params import GuidedDecodingParams

            request_sampling_params.setdefault(
                "guided_decoding",
                GuidedDecodingParams(
                    json=self.actor_guided_json_schema,
                    disable_additional_properties=True,
                ),
            )
        metrics: dict[str, Any] = {"actor_guided_json": float(guided_enabled and self.actor_guided_json_schema is not None)}
        with simple_timer("generate_sequences", metrics):
            token_output = await self.server_manager.generate(
                request_id=uuid.uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=request_sampling_params,
            )
        metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        response_ids = token_output.token_ids[: self.response_length]
        response_logprobs = token_output.log_probs[: self.response_length] if token_output.log_probs else None
        if not response_ids:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(self.tokenizer, "pad_token_id", None)
            response_ids = [int(eos_token_id if eos_token_id is not None else 0)]
            if token_output.log_probs is not None:
                response_logprobs = [0.0]
            metrics["empty_response_fallback"] = 1.0
        metrics["actor_response_tokens"] = float(len(response_ids))
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return prompt_ids, response_ids, response_logprobs, strip_reasoning(text), metrics, token_output.extra_fields

    def _executor(self, *, manage_sleep: bool | None = None) -> MASExecutor:
        if manage_sleep is None:
            manage_sleep = self.manage_sleep
        return MASExecutor(
            base_url=self.subagent_base_url,
            model=self.subagent_model,
            max_agent_tokens=self.max_agent_tokens,
            manage_sleep=manage_sleep,
            max_model_len=self.subagent_max_model_len,
        )

    def _subagent_lease_client(self) -> LocalChatClient:
        return LocalChatClient(
            base_url=self.subagent_base_url,
            model=self.subagent_model,
            manage_sleep=self.manage_sleep,
            max_model_len=self.subagent_max_model_len,
            active_role="subagent",
        )

    def _peer(self, *, manage_sleep: bool | None = None) -> LocalChatClient:
        if not self.peer_base_url:
            raise RuntimeError("MAS_LOOP_PEER_BASE_URL is required for Orchestra adversarial scoring")
        if manage_sleep is None:
            manage_sleep = self.manage_sleep
        return LocalChatClient(
            base_url=self.peer_base_url,
            model=self.peer_model,
            manage_sleep=manage_sleep,
            max_model_len=self.peer_max_model_len,
            active_role="editor",
        )

    def _judge(self, *, manage_sleep: bool | None = None) -> BCPJudge:
        if not self.judge_base_url:
            raise RuntimeError("MAS_LOOP_JUDGE_BASE_URL is required for BCP LLM-judge scoring")
        if manage_sleep is None:
            manage_sleep = self.manage_sleep
        return BCPJudge(
            base_url=self.judge_base_url,
            model=self.judge_model,
            manage_sleep=manage_sleep,
            max_model_len=self.judge_max_model_len,
            result_dir=self.log_dir / "judge_results",
        )

    def _judge_lease_client(self) -> LocalChatClient:
        return LocalChatClient(
            base_url=self.judge_base_url,
            model=self.judge_model,
            manage_sleep=self.manage_sleep,
            max_model_len=self.judge_max_model_len,
            active_role="judge",
        )

    async def _score_trajectory(
        self,
        trajectory: dict[str, Any],
        *,
        task: str,
        expected: str,
        record: dict[str, Any],
        candidate: str,
    ) -> float:
        """Apply the Qwen judge after sub-agent completion for BCP only."""
        if not self.use_llm_judge:
            return float(trajectory.get("metadata", {}).get("score", score_output(trajectory.get("final_output", ""), expected, os.getenv("MAS_LOOP_DATASET_NAME") or record.get("data_source"))))
        judge = self._judge(manage_sleep=False)
        result = await self._managed_blocking(
            self._judge_lease_client(),
            judge.judge_trajectory,
            trajectory,
            question=task,
            correct_answer=expected,
            pair_id=str(record.get("pair_id", "")),
            candidate=candidate,
            optimizer_step=int(record.get("optimizer_step", 0)),
            outer_adversarial_round=int(record.get("outer_adversarial_round", 1)),
            data_source=str(record.get("data_source") or "unknown"),
            phase=f"{candidate}_judge",
        )
        return float(result["score"])

    async def _blocking(self, fn, *args, phase: str = "", **kwargs):
        return await run_blocking(fn, *args, phase=phase, **kwargs)

    async def _managed_blocking(
        self,
        lease_client: LocalChatClient,
        fn,
        *args,
        phase: str = "",
        **kwargs,
    ):
        """Acquire model phase/admission asynchronously, then use dispatch.

        The blocking pool is reserved for real HTTP generation, BM25 and JSON
        work. Requests waiting for another sleeping model or for that model's
        active-request cap remain lightweight asyncio tasks instead of holding
        one worker thread each.
        """
        async with lease_client.async_awake_session():
            return await self._blocking(fn, *args, phase=phase, **kwargs)

    def _record(self, record: dict[str, Any]) -> None:
        # One canonical row contains the complete O_k -> E_k transition.  Ray
        # workers write separate process shards, eliminating the previous five
        # duplicate JSON serializations and the global OSSFS flock/flush tail.
        orchestra_pair = {
            "schema_version": "mas_loop_source_refiner_pair_v6",
            "id": record["id"],
            "pair_id": record["pair_id"],
            "task_group_uid": record.get("grpo_group_uid"),
            "rollout_sample_index": record.get("rollout_sample_index"),
            "optimizer_step": record.get("optimizer_step"),
            "outer_adversarial_round": record.get("outer_adversarial_round"),
            "role": record["role"],
            "task": {"task_text": record["task"], "expected_answer": record["expected"]},
            "actor": {"prompt": record["actor_prompt"], "output": record["actor_output"]},
            "source": {
                "mas": record.get("src_mas"),
                "trajectory": record.get("src_trajectory"),
                "score": record.get("src_score"),
                "judge": (record.get("src_trajectory") or {}).get("metadata", {}).get("judge"),
            },
            "editor": {
                "prompt": record.get("peer_prompt"),
                "output": record.get("peer_output"),
                "patch": record.get("edit_operation"),
                "reward": record.get("editor_score"),
                "outcome": record.get("editor_outcome"),
                "trainable": record.get("editor_trainable", False),
                "retry": record.get("peer_retry", {}),
            },
            "editor_candidates": record.get("editor_candidates", []),
            "updated": {
                "mas": record.get("updated_mas"),
                "trajectory": record.get("updated_trajectory"),
                "score": record.get("upd_score"),
                "judge": (record.get("updated_trajectory") or {}).get("metadata", {}).get("judge"),
            },
            "comparison": {
                "score_delta": record.get("score_delta"),
                "orchestra_reward": record.get("score"),
                "editor_reward": record.get("editor_score"),
                "orchestra_outcome": record.get("outcome"),
                "editor_outcome": record.get("editor_outcome"),
            },
            "patch_translation": (record.get("edit_operation") or {}).get("translation", {}),
            "errors": record.get("violations", []),
        }
        record["structured_pair"] = orchestra_pair
        record["canonical_record_type"] = "orchestra_editor_pair" if self.role == "orchestra" else f"{self.role}_rollout"
        append_jsonl_shard(self.log_dir / "canonical_rollouts", record, shard_prefix=self.role)

    def _output(self, prompt_ids, response_ids, response_logprobs, reward, metrics, token_extra, fields):
        extra = dict(token_extra or {})
        extra.update(
            {
                "pair_id": str(fields["pair_id"]),
                "loop_id": str(fields["loop_id"]),
                "rollout_sample_index": int(fields.get("rollout_sample_index", 0)),
                "role": self.role,
                # Trainer uses this only when external_group mode is enabled.
                "grpo_group_uid": str(fields.get("grpo_group_uid") or fields["loop_id"]),
            }
        )
        extra["reward_extra_info"] = {
            "pair_id": str(fields["pair_id"]),
            "loop_id": str(fields["loop_id"]),
            "rollout_sample_index": int(fields.get("rollout_sample_index", 0)),
            "role": self.role,
            "src_score": float(fields.get("src_score", 0.0)),
            "upd_score": float(fields.get("upd_score", 0.0)),
            "score_delta": float(fields.get("score_delta", 0.0)),
            "outcome": str(fields.get("outcome", "")),
            "error": str(fields.get("error", "")),
        }
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            reward_score=float(reward),
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=extra,
        )

    def _base_record(self, kwargs, loop_id, task, expected, prompt, output):
        extra = _extra_info(kwargs)
        session_id = int(_python(kwargs.get("session_id", 0)) or 0)
        pair_id = str(extra.get("pair_id") or f"{loop_id}:sample-{session_id}")
        return {
            "schema_version": "mas_loop_source_rollout_v6",
            "id": uuid.uuid4().hex,
            "pair_id": pair_id,
            "time": time.time(),
            "loop_id": loop_id,
            "dataset_sample_id": extra.get("query_id", extra.get("index", kwargs.get("index"))),
            "grpo_group_uid": str(extra.get("editor_group_uid") or _python(kwargs.get("uid", loop_id))),
            "rollout_sample_index": session_id,
            "optimizer_step": int(_python(kwargs.get("global_steps", 0)) or 0),
            "outer_adversarial_round": int(os.getenv("MAS_LOOP_OUTER_ROUND", "1")),
            "role": self.role,
            "data_source": str(_python(kwargs.get("data_source", ""))),
            "task": task,
            "expected": expected,
            "constraints": extra.get("constraints") or {},
            "actor_prompt": prompt,
            "actor_output": output,
            "solution": output,
            "reward_design": "source_minus_refiner_exploit_v1",
        }


class MASLoopOrchestraAgentLoop(_MASLoopBase):
    """Sample one source MAS and one Refiner action for each rollout member.

    ``rollout.n`` supplies the 32 source members of the task group.  Updated
    MASes are deterministic transforms of source tokens plus Refiner patches;
    they are comparison-only adversarial states and are never selected or
    substituted for Orchestra's on-policy response tokens.
    """

    role = "orchestra"
    actor_guided_json_schema = MAS_GUIDED_JSON_SCHEMA

    async def _evaluate_editor_pair(
        self,
        *,
        task: str,
        expected: str,
        constraints: dict[str, Any],
        src_mas: dict[str, Any],
        src_traj: dict[str, Any],
        src_score: float,
        rec: dict[str, Any],
    ) -> dict[str, Any]:
        rollout_index = int(rec.get("rollout_sample_index", 0))
        candidate_id = f"{rec['pair_id']}:refiner"
        editor_group_uid = f"editor-task:{rec['grpo_group_uid']}:step-{rec['optimizer_step']}"
        diagnostic_state = compact_trajectory_for_editor(src_traj, src_mas, task, constraints)
        messages = editor_prompt(task, src_mas, src_traj, constraints)
        result: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_index": rollout_index,
            "editor_group_uid": editor_group_uid,
            "prompt": messages,
            "output": "",
            "retry": {"attempt_count": 0, "retry_count": 0, "raw_responses": [], "errors": []},
            "patch": {"diagnosis": {}, "patch": [], "operations": []},
            "updated_mas": src_mas,
            "updated_trajectory": {},
            "src_score": src_score,
            "upd_score": 0.0,
            "score_delta": 0.0,
            "reward": -1.0,
            "outcome": "error",
            "trainable": False,
            "updated_valid": False,
            "violations": [],
            "error": "",
        }

        updated_mas = src_mas
        editor_action_valid = False
        editor_accepted = False
        editor_outcome = "error"
        try:
            peer_client = self._peer(manage_sleep=False)
            peer_text, patch, peer_retry = await self._managed_blocking(
                self._peer(manage_sleep=True),
                peer_client.chat_validated,
                messages,
                validator=lambda value: normalize_editor_action(
                    value, src_mas=src_mas, diagnostic_state=diagnostic_state
                ),
                guided_json=_editor_guided_json_schema(),
                retry_instruction=editor_retry_instruction(src_mas),
                max_tokens=self.peer_max_tokens,
                temperature=float(os.getenv("MAS_LOOP_EDITOR_TEMPERATURE", "0.7")),
                phase="editor_pair",
            )
            result.update({"output": peer_text, "retry": peer_retry, "patch": patch, "trainable": bool(peer_text.strip())})
            editor_action_valid = True
            editor_accepted = is_accept_patch(patch)
            if editor_accepted:
                editor_outcome = "accepted_correct" if src_score >= 0.999 else "accepted_incorrect"
            else:
                updated_mas, patch_errors = apply_patch(src_mas, patch)
                if patch_errors:
                    raise ValueError("invalid_patch_application:" + ";".join(patch_errors))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}:{exc}"
            result["violations"].append(result["error"])
            editor_outcome = "invalid_editor_output" if isinstance(exc, LocalChatError) else "error"
            if isinstance(exc, LocalChatError):
                result["output"] = exc.last_response
                result["retry"] = {
                    "attempt_count": len(exc.raw_responses),
                    "retry_count": max(0, len(exc.raw_responses) - 1),
                    "raw_responses": exc.raw_responses,
                    "errors": exc.errors,
                }
                # Keep malformed sampled text trainable with fail-closed reward,
                # but still execute a fresh updated trajectory below.
                result["trainable"] = bool(result["output"].strip())
            updated_mas = src_mas
            editor_action_valid = False
            editor_accepted = False

        # Every source member must have one independently executed updated
        # trajectory and Judge result. Accept and malformed Editor actions use
        # the unchanged MAS as the updated state, but never reuse src_traj.
        try:
            executor = self._executor(manage_sleep=False)
            updated_trajectory = await self._managed_blocking(
                self._subagent_lease_client(),
                executor.execute,
                task,
                updated_mas,
                expected,
                profile_phase="updated_subagent",
                pair_id=candidate_id,
                phase="updated_subagent",
            )
            updated_score = await self._score_trajectory(
                updated_trajectory,
                task=task,
                expected=expected,
                record={**rec, "pair_id": candidate_id},
                candidate="updated",
            )
            delta = updated_score - src_score
            if not editor_action_valid:
                reward = -_invalid_action_penalty()
                outcome = editor_outcome
            elif editor_accepted:
                reward = _invalid_action_penalty() if src_score >= 0.999 else -_invalid_action_penalty()
                outcome = editor_outcome
            else:
                reward = delta
                outcome = "improved" if delta > 0 else ("damaged" if delta < 0 else "no_gain")
            result.update(
                {
                    "updated_mas": updated_mas,
                    "updated_trajectory": updated_trajectory,
                    "upd_score": updated_score,
                    "score_delta": delta,
                    "reward": reward,
                    "outcome": outcome,
                    "updated_valid": True,
                }
            )
        except Exception as exc:
            execution_error = f"{type(exc).__name__}:{exc}"
            result["violations"].append(execution_error)
            result["error"] = ";".join(x for x in (result.get("error", ""), execution_error) if x)
            result["updated_mas"] = updated_mas
            result["outcome"] = "updated_execution_error"
        return result

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = _extra_info(kwargs)
        task = str(extra.get("task_text") or extra.get("question") or "")
        expected = _expected(kwargs)
        constraints = extra.get("constraints") or {}
        prompt_uid = str(_python(kwargs.get("uid", uuid.uuid4().hex)))
        loop_id = f"{prompt_uid}:orchestra"
        messages = orchestra_prompt(task, constraints)
        prompt_ids, response_ids, logprobs, actor_text, metrics, token_extra = await self._actor_generate(messages, sampling_params)
        rec = self._base_record(kwargs, loop_id, task, expected, messages, actor_text)
        rec["actor_response_tokens"] = len(response_ids)
        rec["actor_generate_seconds"] = float(metrics.get("generate_sequences", 0.0))
        rec["actor_guided_json"] = bool(metrics.get("actor_guided_json", 0.0))

        errors: list[str] = []
        src_mas: dict[str, Any] = {}
        src_traj: dict[str, Any] = {}
        src_score = 0.0
        orchestra_reward = -1.0
        orchestra_outcome = "error"
        error = ""
        pair: dict[str, Any] = {}

        try:
            src_mas = normalize_mas(actor_text)
            errors.extend(validate_mas(src_mas))
            if errors:
                raise ValueError("invalid_mas:" + ";".join(errors))
            executor = self._executor(manage_sleep=False)
            src_traj = await self._managed_blocking(
                self._subagent_lease_client(), executor.execute, task, src_mas, expected,
                profile_phase="source_subagent", pair_id=rec["pair_id"], phase="source_subagent",
            )
            src_score = await self._score_trajectory(src_traj, task=task, expected=expected, record=rec, candidate="source")
            pair = await self._evaluate_editor_pair(
                task=task, expected=expected, constraints=constraints, src_mas=src_mas,
                src_traj=src_traj, src_score=src_score, rec=rec,
            )
            lambda_adv = float(os.getenv("MAS_LOOP_ORCH_ADV_PENALTY", "0.3"))
            if not 0.0 <= lambda_adv <= 1.0:
                raise ValueError(f"MAS_LOOP_ORCH_ADV_PENALTY must be in [0, 1], got {lambda_adv}")
            delta = float(pair.get("score_delta", 0.0))
            # Orchestra owns only the sampled source MAS. Updated MAS is a
            # Refiner adversarial probe: if it improves the source, expose that
            # weakness as a penalty rather than rescue credit.
            orchestra_reward = src_score - lambda_adv * max(delta, 0.0)
            if str(pair.get("outcome")) in {"invalid_editor_output", "error"}:
                orchestra_outcome = "refiner_failed"
            elif delta > 0:
                orchestra_outcome = "source_exploited"
            elif src_score >= 0.999:
                orchestra_outcome = "source_robust"
            else:
                orchestra_outcome = "source_weak"
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            errors.append(error)

        updated_mas = pair.get("updated_mas", src_mas)
        updated_traj = pair.get("updated_trajectory", src_traj)
        upd_score = float(pair.get("upd_score", src_score))
        patch = pair.get("patch", {"diagnosis": {}, "patch": [], "operations": []})
        diagnosis = patch.get("diagnosis", {}) if isinstance(patch, dict) else {}

        rec.update(
            {
                "src_mas": src_mas,
                "src_trajectory": src_traj,
                "src_score": src_score,
                # Keep a one-element compatibility list while making the v5
                # one-source/one-Refiner pair explicit in primary fields.
                "editor_candidates": [pair] if pair else [],
                "editor_candidate_count": 1 if pair else 0,
                "editor_valid_candidate_count": int(bool(pair.get("updated_valid", False))) if pair else 0,
                "editor_candidate_diversity": {
                    "valid_candidates": int(bool(pair.get("trainable", False))) if pair else 0,
                    "unique_outputs": int(bool(str(pair.get("output", "")).strip())) if pair else 0,
                    "unique_diagnosis_targets": int(bool(str(diagnosis.get("target_id", "")).strip())),
                    "unique_failure_types": int(bool(str(diagnosis.get("failure_type", "")).strip())),
                    "unique_operation_sequences": int(bool((patch or {}).get("operations"))) if isinstance(patch, dict) else 0,
                },
                "peer_prompt": pair.get("prompt", []),
                "peer_output": pair.get("output", ""),
                "peer_retry": pair.get("retry", {}),
                "editor_trainable": bool(pair.get("trainable", False)),
                "editor_group_uid": pair.get("editor_group_uid", ""),
                "editor_score": float(pair.get("reward", -1.0)),
                "editor_outcome": str(pair.get("outcome", "error")),
                "edit_operation": patch,
                "updated_mas": updated_mas,
                "updated_trajectory": updated_traj,
                "upd_score": upd_score,
                "score_delta": float(pair.get("score_delta", 0.0)),
                "score": orchestra_reward,
                "outcome": orchestra_outcome,
                "violations": errors + list(pair.get("violations", [])),
                "error": error,
                "adversarial_role": "source_policy_penalized_by_refiner_exploit",
                "reward_design": "source_minus_refiner_exploit_v1",
            }
        )
        self._record(rec)
        return self._output(prompt_ids, response_ids, logprobs, orchestra_reward, metrics, token_extra, rec)


class MASLoopEditorAgentLoop(_MASLoopBase):
    """Direct one-source/one-patch Editor rollout (kept for diagnostics)."""

    role = "editor"
    actor_guided_json_schema = PATCH_GUIDED_JSON_SCHEMA

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = _extra_info(kwargs)
        task = str(extra.get("task_text") or extra.get("question") or "")
        expected = _expected(kwargs)
        constraints = extra.get("constraints") or {}
        loop_id = f"{_python(kwargs.get('uid', uuid.uuid4().hex))}:editor"
        src_mas = normalize_mas(extra.get("src_mas"))
        src_traj = extra.get("src_trajectory") or {}
        executor = self._executor(manage_sleep=False)
        subagent_lease = self._subagent_lease_client()
        if not isinstance(src_traj, dict) or "final_output" not in src_traj:
            src_traj = await self._managed_blocking(
                subagent_lease,
                executor.execute,
                task,
                src_mas,
                expected,
                profile_phase="source_subagent",
                pair_id=loop_id,
                phase="source_subagent",
            )
        diagnostic_state = compact_trajectory_for_editor(src_traj, src_mas, task, constraints)
        messages = editor_prompt(task, src_mas, src_traj, constraints)
        prompt_ids, response_ids, logprobs, actor_text, metrics, token_extra = await self._actor_generate(messages, sampling_params)
        rec = self._base_record(kwargs, loop_id, task, expected, messages, actor_text)
        rec["actor_response_tokens"] = len(response_ids)
        rec["actor_generate_seconds"] = float(metrics.get("generate_sequences", 0.0))
        rec["actor_guided_json"] = bool(metrics.get("actor_guided_json", 0.0))
        if self.use_llm_judge and not isinstance(src_traj.get("metadata", {}).get("judge"), dict):
            src_score = await self._score_trajectory(
                src_traj, task=task, expected=expected, record=rec, candidate="source"
            )
        else:
            src_score = float(src_traj.get("metadata", {}).get("score", score_output(src_traj.get("final_output", ""), expected, os.getenv("MAS_LOOP_DATASET_NAME") or rec.get("data_source"))))
        errors: list[str] = []
        try:
            patch = normalize_editor_action(actor_text, src_mas=src_mas, diagnostic_state=diagnostic_state)
            if is_accept_patch(patch):
                # Accept means rerun the unchanged source MAS; never reuse the
                # source trajectory because the paper evaluates two independent executions.
                upd_mas = src_mas
                upd_traj = await self._managed_blocking(
                    subagent_lease, executor.execute, task, upd_mas, expected,
                    profile_phase="updated_subagent", pair_id=rec["pair_id"], phase="updated_subagent",
                )
                upd_score = await self._score_trajectory(
                    upd_traj, task=task, expected=expected, record=rec, candidate="updated"
                )
                patch_errors = []
                delta = upd_score - src_score
                reward = _invalid_action_penalty() if src_score >= 0.999 else -_invalid_action_penalty()
                outcome = "accepted_correct" if src_score >= 0.999 else "accepted_incorrect"
            else:
                upd_mas, patch_errors = apply_patch(src_mas, patch)
                if patch_errors:
                    raise ValueError("invalid_patch_application:" + ";".join(patch_errors))
                upd_traj = await self._managed_blocking(
                    subagent_lease,
                    executor.execute,
                    task,
                    upd_mas,
                    expected,
                    profile_phase="updated_subagent",
                    pair_id=rec["pair_id"],
                    phase="updated_subagent",
                )
                upd_score = await self._score_trajectory(
                    upd_traj, task=task, expected=expected, record=rec, candidate="updated"
                )
                delta = upd_score - src_score
                reward = delta - float(os.getenv("MAS_LOOP_PATCH_ERROR_PENALTY", "0.2")) * len(patch_errors)
                outcome = "improved" if delta > 0 else ("damaged" if delta < 0 else "no_gain")
            error = ""
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            errors.append(error)
            patch = {"diagnosis": {}, "patch": [], "operations": []}
            upd_mas, upd_traj, upd_score, delta, reward, outcome = src_mas, src_traj, src_score, 0.0, -_invalid_action_penalty(), "error"
        rec.update(
            {
                "src_mas": src_mas,
                "src_trajectory": src_traj,
                "src_score": src_score,
                "edit_operation": patch,
                "updated_mas": upd_mas,
                "updated_trajectory": upd_traj,
                "upd_score": upd_score,
                "score_delta": delta,
                "score": reward,
                "outcome": outcome,
                "violations": errors,
                "error": error,
                "adversarial_role": "attacker_refiner",
            }
        )
        self._record(rec)
        return self._output(prompt_ids, response_ids, logprobs, reward, metrics, token_extra, rec)


class MASLoopEditorReplayAgentLoop(_MASLoopBase):
    """Train Editor on the exact E_k sampled for the corresponding O_k.

    The prompt/output/reward are collected while scoring Orchestra.  The current
    trainable Editor recomputes old log-probabilities for the stored response;
    no external vLLM generation or sub-agent execution happens in this phase.
    """

    role = "editor"

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = _extra_info(kwargs)
        messages = extra.get("replay_editor_prompt") or []
        output_text = str(extra.get("replay_editor_output") or "")
        if not isinstance(messages, list) or not messages or not output_text.strip():
            raise ValueError("Editor replay row requires non-empty replay_editor_prompt and replay_editor_output")

        prompt_ids = await self.apply_chat_template(messages)
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length :]
        response_ids = self.tokenizer.encode(output_text, add_special_tokens=False)[: self.response_length]
        if not response_ids:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            response_ids = [int(eos if eos is not None else 0)]

        reward = float(extra.get("replay_editor_reward", -1.0))
        pair_id = str(extra.get("pair_id") or uuid.uuid4().hex)
        group_uid = str(extra.get("editor_group_uid") or extra.get("source_grpo_group_uid") or pair_id)
        fields = {
            "pair_id": pair_id,
            "loop_id": str(extra.get("source_loop_id") or group_uid),
            "rollout_sample_index": int(extra.get("rollout_sample_index", 0)),
            "grpo_group_uid": group_uid,
            "src_score": float(extra.get("src_score", 0.0)),
            "upd_score": float(extra.get("upd_score", 0.0)),
            "score_delta": float(extra.get("score_delta", 0.0)),
            "outcome": str(extra.get("editor_outcome", "")),
            "error": str(extra.get("error", "")),
        }
        append_jsonl_shard(
            self.log_dir / "editor_replay_consumed",
            {
                "time": time.time(),
                "pair_id": pair_id,
                "editor_group_uid": group_uid,
                "rollout_sample_index": fields["rollout_sample_index"],
                "reward": reward,
                "source_loop_id": fields["loop_id"],
            },
            shard_prefix="editor-replay",
        )
        # This response was sampled by the external Editor service before the
        # current optimizer update.  Mark that exact pre-update policy version
        # so v1 off-policy diagnostics remain meaningful during replay.
        rollout_step = int(_python(kwargs.get("global_steps", 1)))
        policy_version = max(rollout_step - 1, 0)
        return self._output(
            prompt_ids,
            response_ids,
            None,
            reward,
            {"generate_sequences": 0.0, "num_preempted": -1},
            {"min_global_steps": policy_version, "max_global_steps": policy_version},
            fields,
        )
