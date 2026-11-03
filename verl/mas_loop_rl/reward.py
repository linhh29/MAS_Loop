from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from mas_loop_rl.bcp_judge import BCPJudge, is_llm_judge_enabled
from mas_loop_rl.executor import MASExecutor, score_output
from mas_loop_rl.local_client import LocalChatClient, strip_reasoning
from mas_loop_rl.prompts import editor_prompt
from mas_loop_rl.schema import (
    apply_patch, is_accept_patch, normalize_editor_action, normalize_mas, normalize_patch, validate_mas,
)


def _log(record: dict[str, Any], log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with (Path(log_dir) / "mas_loop_grpo_rewards.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _pair_log(record: dict[str, Any], log_dir: str):
    """Write one structured pair record per rollout sample.

    These pair files are deliberately separate from the reward JSONL so later
    dataset construction can consume a stable schema without parsing the raw
    training diagnostics. A record is written for successes and failures; failed
    rollouts keep raw actor output plus error so no sample silently disappears.
    """
    pair_dir = Path(os.getenv("MAS_LOOP_PAIR_DIR", str(Path(log_dir) / "pairs")))
    pair_dir.mkdir(parents=True, exist_ok=True)
    role = str(record.get("role") or "unknown")
    path = pair_dir / f"{role}_rollout_pairs.jsonl"
    pair = {
        "schema_version": "mas_loop_pair_v1",
        "id": record.get("id"),
        "time": record.get("time"),
        "role": record.get("role"),
        "data_source": record.get("data_source"),
        "reward_design": record.get("reward_design"),
        "adversarial_role": record.get("adversarial_role"),
        "task": {
            "task_text": record.get("task"),
            "expected_answer": record.get("expected"),
        },
        "model_output": record.get("solution"),
        "source": {
            "mas": record.get("src_mas"),
            "trajectory": record.get("src_trajectory"),
            "score": record.get("src_score"),
            "judge": (record.get("src_trajectory") or {}).get("metadata", {}).get("judge"),
        },
        "edit": {
            "operation": record.get("edit_operation"),
            "violations": record.get("violations", []),
        },
        "updated": {
            "mas": record.get("updated_mas"),
            "trajectory": record.get("updated_trajectory"),
            "score": record.get("upd_score"),
            "judge": (record.get("updated_trajectory") or {}).get("metadata", {}).get("judge"),
        },
        "comparison": {
            "score_delta": record.get("score_delta"),
            "outcome": record.get("outcome"),
            "reward": record.get("score"),
        },
        "error": record.get("error", ""),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False, default=str) + "\n")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _invalid_action_penalty() -> float:
    try:
        return max(0.0, float(os.getenv("MAS_LOOP_INVALID_ACTION_PENALTY", "0.5")))
    except (TypeError, ValueError):
        return 0.5


def _score_trajectory_with_optional_bcp_judge(
    trajectory: dict[str, Any],
    *,
    task: str,
    expected: str,
    rec: dict[str, Any],
    candidate: str,
    log_dir: str,
    manage_sleep: bool,
) -> float:
    if str(rec.get("dataset") or "").lower() == "alfworld":
        return float(trajectory.get("metadata", {}).get("score", 0.0) or 0.0)
    if not is_llm_judge_enabled():
        return float(trajectory.get("metadata", {}).get("score", score_output(trajectory.get("final_output", ""), expected, os.getenv("MAS_LOOP_DATASET_NAME") or rec.get("data_source"))))
    existing = trajectory.get("metadata", {}).get("judge")
    if isinstance(existing, dict):
        return float(existing.get("score", -1.0))
    judge = BCPJudge(manage_sleep=manage_sleep, result_dir=Path(log_dir) / "judge_results")
    with judge.client.awake_session():
        result = judge.judge_trajectory(
            trajectory,
            question=task,
            correct_answer=expected,
            pair_id=str(rec.get("id", "")),
            candidate=candidate,
            optimizer_step=int(rec.get("optimizer_step", 0)),
            outer_adversarial_round=int(os.getenv("MAS_LOOP_OUTER_ROUND", "1")),
            data_source=str(rec.get("data_source") or "unknown"),
        )
    return float(result["score"])



def _is_alfworld(extra_info: dict[str, Any]) -> bool:
    return str(extra_info.get("dataset") or "").strip().lower() == "alfworld"


def _execute_trajectory(
    executor: MASExecutor,
    *,
    task: str,
    mas: dict[str, Any],
    expected: str,
    extra_info: dict[str, Any],
    candidate: str,
    pair_id: str,
) -> dict[str, Any]:
    if not _is_alfworld(extra_info):
        return executor.execute(task, mas, expected, profile_phase=candidate, pair_id=pair_id)
    episode_row = {
        key: extra_info.get(key)
        for key in (
            "dataset", "id", "task_id", "game_file", "env_step_limit",
            "task_type", "official_split", "split", "task_descriptions",
            "floor_plan", "traj_file",
        )
        if key in extra_info
    }
    data_root = os.getenv(
        "MAS_LOOP_ALFWORLD_DATA_ROOT",
        str(Path(__file__).resolve().parents[3] / "json_2.1.1"),
    )
    return executor.execute_alfworld(
        task, mas, episode_row, data_root=data_root,
        profile_phase=candidate, pair_id=pair_id,
    )

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    role: str | None = None,
    peer_base_url: str | None = None,
    peer_model: str | None = None,
    subagent_base_url: str | None = None,
    subagent_model: str | None = None,
    log_dir: str | None = None,
    max_agent_tokens: int = 1024,
    manage_vllm_sleep: bool | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Adversarial MAS-Loop reward for Verl GRPO.

    The reward is intentionally simple and task-outcome based:

    * Refiner/Editor is the attacker/improver. It maximizes
      `delta = updated_score - source_score` by finding useful edits.
    * Orchestra is trained only on its sampled source MAS tokens. Its reward is
      source quality plus bounded credit when the paired Refiner produces a
      better state; the code-derived updated MAS is never treated as an
      Orchestra action.

    No hand-designed complexity score is used here. The logged MAS/trajectory
    fields are kept for later dataset construction and offline analysis.
    """
    extra_info = dict(extra_info or {})
    role = role or extra_info.get("role") or os.getenv("MAS_LOOP_ROLE", "orchestra")
    max_agent_tokens = int(os.getenv("MAS_LOOP_MAX_AGENT_TOKENS", str(max_agent_tokens)))
    log_dir = log_dir or os.getenv("MAS_LOOP_LOG_DIR", "/path/to/checkpoints/unscoped/rollouts")
    subagent_base_url = subagent_base_url or os.getenv("MAS_LOOP_SUBAGENT_BASE_URL") or os.getenv("MAS_LOOP_OPENAI_BASE_URL")
    subagent_model = subagent_model or os.getenv("MAS_LOOP_SUBAGENT_MODEL") or os.getenv("MAS_LOOP_MODEL", "local-model")
    peer_base_url = peer_base_url or os.getenv("MAS_LOOP_PEER_BASE_URL")
    peer_model = peer_model or os.getenv("MAS_LOOP_PEER_MODEL", "local-model")
    if manage_vllm_sleep is None:
        manage_vllm_sleep = os.getenv("MAS_LOOP_VLLM_SLEEP", "1") not in {"0", "false", "False"}

    task = extra_info.get("task_text") or extra_info.get("question") or ""
    expected = ground_truth if isinstance(ground_truth, str) else (ground_truth.get("target") if isinstance(ground_truth, dict) else str(ground_truth or ""))
    constraints = extra_info.get("constraints") or {}
    rec = {
        "id": uuid.uuid4().hex,
        "time": time.time(),
        "role": role,
        "data_source": data_source,
        "task": task,
        "expected": expected,
        "solution": solution_str,
        "reward_design": "adversarial_delta_v2_paper",
        "dataset": str(extra_info.get("dataset") or ""),
    }
    try:
        executor = MASExecutor(
            base_url=subagent_base_url,
            model=subagent_model,
            max_agent_tokens=max_agent_tokens,
            manage_sleep=manage_vllm_sleep,
        )
        if role == "orchestra":
            src_mas = normalize_mas(solution_str)
            src_errors = validate_mas(src_mas)
            if src_errors:
                raise ValueError("invalid_mas:" + ";".join(src_errors))
            src_traj = _execute_trajectory(executor, task=task, mas=src_mas, expected=expected, extra_info=extra_info, candidate="source", pair_id=rec["id"])
            src_score = _score_trajectory_with_optional_bcp_judge(
                src_traj, task=task, expected=expected, rec=rec, candidate="source",
                log_dir=log_dir, manage_sleep=bool(manage_vllm_sleep),
            )

            patch = {"diagnosis": {}, "patch": [], "operations": []}
            upd_mas = src_mas
            upd_traj = src_traj
            upd_score = src_score
            patch_errors: list[str] = []
            peer_error = ""
            if peer_base_url:
                # Current Refiner is the adversary: if it can substantially
                # improve Orchestra's MAS, Orchestra receives a lower reward.
                #
                # A malformed/empty peer response is an attack failure, not an
                # Orchestra failure.  The previous implementation let the
                # exception escape to the outer handler and assigned -1 to the
                # defender even when its MAS JSON and source trajectory were
                # valid.  Treat failed attacks as a no-op while retaining the
                # diagnostic in violations/pair logs.
                try:
                    peer = LocalChatClient(
                        base_url=peer_base_url,
                        model=peer_model,
                        manage_sleep=manage_vllm_sleep,
                        active_role="editor",
                    )
                    patch_text = peer.chat(
                        editor_prompt(task, src_mas, src_traj, constraints),
                        max_tokens=int(os.getenv("MAS_LOOP_EDITOR_MAX_TOKENS", "512")),
                        temperature=float(os.getenv("MAS_LOOP_EDITOR_TEMPERATURE", "0.7")),
                    )
                    patch = normalize_editor_action(strip_reasoning(patch_text), src_mas=src_mas)
                    upd_mas, patch_errors = apply_patch(src_mas, patch)
                    upd_traj = _execute_trajectory(executor, task=task, mas=upd_mas, expected=expected, extra_info=extra_info, candidate="updated", pair_id=rec["id"])
                    upd_score = _score_trajectory_with_optional_bcp_judge(
                        upd_traj, task=task, expected=expected, rec=rec, candidate="updated",
                        log_dir=log_dir, manage_sleep=bool(manage_vllm_sleep),
                    )
                except Exception as e:
                    peer_error = f"peer_edit_failed:{type(e).__name__}:{e}"
                    patch_errors.append(peer_error)
                    patch = {"diagnosis": {}, "patch": [], "operations": []}
                    upd_mas = src_mas
                    upd_traj = src_traj
                    upd_score = src_score

            delta = upd_score - src_score
            lambda_adv = _float_env("MAS_LOOP_ORCH_ADV_PENALTY", 0.3)
            if not 0.0 <= lambda_adv <= 1.0:
                raise ValueError(f"MAS_LOOP_ORCH_ADV_PENALTY must be in [0, 1], got {lambda_adv}")
            score = src_score - lambda_adv * max(delta, 0.0)
            outcome = (
                "peer_failed"
                if peer_error
                else ("robust" if src_score >= 0.999 and delta <= 0.0 else ("exploited" if delta > 0 else "weak"))
            )
            rec.update(
                {
                    "src_mas": src_mas,
                    "src_trajectory": src_traj,
                    "edit_operation": patch,
                    "updated_mas": upd_mas,
                    "updated_trajectory": upd_traj,
                    "src_score": src_score,
                    "upd_score": upd_score,
                    "score_delta": delta,
                    "adversarial_role": "source_policy_penalized_by_refiner_exploit",
                    "outcome": outcome,
                    "violations": patch_errors,
                    "score": score,
                }
            )
        else:
            src_mas = normalize_mas(extra_info.get("src_mas"))
            src_traj = extra_info.get("src_trajectory") or {}
            if _is_alfworld(extra_info) or not isinstance(src_traj, dict) or "final_output" not in src_traj:
                src_traj = _execute_trajectory(executor, task=task, mas=src_mas, expected=expected, extra_info=extra_info, candidate="source", pair_id=rec["id"])
            src_score = _score_trajectory_with_optional_bcp_judge(
                src_traj, task=task, expected=expected, rec=rec, candidate="source",
                log_dir=log_dir, manage_sleep=bool(manage_vllm_sleep),
            )

            patch = normalize_editor_action(strip_reasoning(solution_str))
            if is_accept_patch(patch):
                # Accept still requires an independent execution of the unchanged MAS.
                upd_mas = src_mas
                patch_errors = []
                upd_traj = _execute_trajectory(executor, task=task, mas=upd_mas, expected=expected, extra_info=extra_info, candidate="updated", pair_id=rec["id"])
                upd_score = _score_trajectory_with_optional_bcp_judge(
                    upd_traj, task=task, expected=expected, rec=rec, candidate="updated",
                    log_dir=log_dir, manage_sleep=bool(manage_vllm_sleep),
                )
                delta = upd_score - src_score
            else:
                upd_mas, patch_errors = apply_patch(src_mas, patch)
                if patch_errors:
                    raise ValueError("invalid_patch_application:" + ";".join(patch_errors))
                upd_traj = _execute_trajectory(executor, task=task, mas=upd_mas, expected=expected, extra_info=extra_info, candidate="updated", pair_id=rec["id"])
                upd_score = _score_trajectory_with_optional_bcp_judge(
                    upd_traj, task=task, expected=expected, rec=rec, candidate="updated",
                    log_dir=log_dir, manage_sleep=bool(manage_vllm_sleep),
                )
                delta = upd_score - src_score
            score = (_invalid_action_penalty() if src_score >= 0.999 else -_invalid_action_penalty()) if is_accept_patch(patch) else delta
            rec.update(
                {
                    "src_mas": src_mas,
                    "src_trajectory": src_traj,
                    "edit_operation": patch,
                    "updated_mas": upd_mas,
                    "updated_trajectory": upd_traj,
                    "src_score": src_score,
                    "upd_score": upd_score,
                    "score_delta": upd_score - src_score,
                    "adversarial_role": "attacker_refiner",
                    "outcome": ("accepted_correct" if src_score >= 0.999 else "accepted_incorrect") if is_accept_patch(patch) else ("improved" if upd_score > src_score else ("damaged" if upd_score < src_score else "no_gain")),
                    "violations": patch_errors,
                    "score": score,
                }
            )
        _log(rec, log_dir)
        _pair_log(rec, log_dir)
        # Verl concatenates reward extra_info dictionaries across samples and
        # requires identical non-tensor keys.  Always return the same compact
        # schema for both successful and failed samples; full variable-size
        # trajectories are written to the JSONL log above.
        return {
            "score": float(score),
            "src_score": float(rec.get("src_score", 0.0) or 0.0),
            "upd_score": float(rec.get("upd_score", 0.0) or 0.0),
            "score_delta": float(rec.get("score_delta", 0.0) or 0.0),
            "log_id": rec["id"],
            "outcome": str(rec.get("outcome", "")),
            "error": "",
        }
    except Exception as e:
        rec.update({
            "error": str(e),
            "score": -_invalid_action_penalty() if role != "orchestra" else -1.0,
            "src_score": float(rec.get("src_score", 0.0) or 0.0),
            "upd_score": float(rec.get("upd_score", 0.0) or 0.0),
            "score_delta": float(rec.get("score_delta", 0.0) or 0.0),
            "outcome": str(rec.get("outcome", "error")),
            "adversarial_role": "source_policy_penalized_by_refiner_exploit" if role == "orchestra" else "pairwise_refiner",
            "violations": rec.get("violations", []),
        })
        _log(rec, log_dir)
        _pair_log(rec, log_dir)
        return {
            "score": -_invalid_action_penalty() if role != "orchestra" else -1.0,
            "src_score": float(rec.get("src_score", 0.0) or 0.0),
            "upd_score": float(rec.get("upd_score", 0.0) or 0.0),
            "score_delta": float(rec.get("score_delta", 0.0) or 0.0),
            "log_id": rec["id"],
            "outcome": str(rec.get("outcome", "error")),
            "error": str(e),
        }
