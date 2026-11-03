"""Targeted recovery for one incomplete Orchestra/Refiner canonical pair.

The actor response is preserved verbatim.  This tool reparses that response with
current bounded XML repairs, then runs the missing source execution, Refiner,
updated execution and Judge calls.  Only after every required artifact is valid
does it atomically replace the selected JSONL row and publish temporary Judge
records into the run's judge_results directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .bcp_judge import BCPJudge, is_valid_bcp_judge_metadata
from .executor import MASExecutor
from .local_client import LocalChatClient, LocalChatError
from .prompts import editor_prompt, editor_retry_instruction
from .schema import apply_patch, is_accept_patch, normalize_editor_action, normalize_mas, validate_mas


def _structured_pair(record: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _load_target(shard: Path, *, line_number: int, pair_id: str, step: int) -> tuple[list[str], dict[str, Any]]:
    lines = shard.read_text(encoding="utf-8").splitlines()
    if not 1 <= line_number <= len(lines):
        raise RuntimeError(f"line {line_number} is outside {shard} ({len(lines)} rows)")
    row = json.loads(lines[line_number - 1])
    if str(row.get("pair_id")) != pair_id:
        raise RuntimeError(f"pair_id mismatch: expected={pair_id!r} actual={row.get('pair_id')!r}")
    if int(row.get("optimizer_step", -1)) != int(step):
        raise RuntimeError(f"optimizer_step mismatch: expected={step} actual={row.get('optimizer_step')}")
    if row.get("src_mas") or row.get("editor_candidates"):
        raise RuntimeError("target row is not incomplete; refusing to overwrite a completed pair")
    return lines, row


def _sleep_if_awake(client: LocalChatClient) -> None:
    try:
        if not client.is_sleeping():
            client.sleep(1)
    except Exception:
        pass


def _stage_cross_device_publish(source: Path, target: Path) -> Path:
    """Copy *source* to a same-filesystem temporary peer of *target*.

    ``os.replace`` cannot cross filesystem boundaries (the repair work dir is
    commonly under ``/tmp`` while checkpoints live on shared storage).  Copying
    first and fsyncing the destination lets the later publish remain an atomic
    same-directory rename.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with source.open("rb") as src, staged.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    shutil.copystat(source, staged)
    return staged


def repair(args: argparse.Namespace) -> dict[str, Any]:
    shard = args.shard.resolve()
    lines, rec = _load_target(shard, line_number=args.line, pair_id=args.pair_id, step=args.step)
    src_mas = normalize_mas(rec.get("actor_output") or "")
    errors = validate_mas(src_mas)
    if errors:
        raise RuntimeError("reparsed source MAS is invalid: " + ";".join(errors))

    repair_root = Path(tempfile.mkdtemp(prefix="mas_loop_pair_repair_", dir=str(args.work_dir.resolve())))
    temp_judges = repair_root / "judge_results"
    os.environ["MAS_LOOP_ROLLOUT_PROFILE_PATH"] = str(repair_root / "rollout_profile")
    os.environ["MAS_LOOP_JUDGE_RESULTS_DIR"] = str(temp_judges)

    helper_control = LocalChatClient(
        base_url=args.helper_url, model=args.helper_model, manage_sleep=False,
        max_model_len=args.max_model_len,
    )
    editor_control = LocalChatClient(
        base_url=args.editor_url, model=args.editor_model, manage_sleep=False,
        max_model_len=args.max_model_len,
    )
    executor = MASExecutor(
        base_url=args.helper_url, model=args.helper_model,
        max_agent_tokens=args.max_agent_tokens, manage_sleep=False,
        max_model_len=args.max_model_len,
    )
    judge = BCPJudge(
        base_url=args.helper_url, model=args.helper_model, manage_sleep=False,
        max_model_len=args.max_model_len, result_dir=temp_judges,
    )
    editor = LocalChatClient(
        base_url=args.editor_url, model=args.editor_model, manage_sleep=False,
        max_model_len=args.max_model_len, active_role="editor",
    )

    task = str(rec.get("task") or "")
    expected = str(rec.get("expected") or "")
    constraints = rec.get("constraints") or {}
    pair_id = str(rec["pair_id"])
    candidate_id = f"{pair_id}:refiner"
    common_judge = {
        "question": task,
        "correct_answer": expected,
        "optimizer_step": int(rec["optimizer_step"]),
        "outer_adversarial_round": int(rec.get("outer_adversarial_round", 1)),
        "data_source": str(rec.get("data_source") or "unknown"),
    }

    # The helper is expected awake and the Editor asleep at entry.
    if helper_control.is_sleeping():
        helper_control.wake_up()
    if not editor_control.is_sleeping():
        editor_control.sleep(1)

    try:
        src_traj = executor.execute(task, src_mas, expected, profile_phase="source_subagent", pair_id=pair_id)
        src_judge = judge.judge_trajectory(src_traj, pair_id=pair_id, candidate="source", **common_judge)
        if not is_valid_bcp_judge_metadata(src_judge, max_tokens=judge.max_tokens):
            raise RuntimeError(f"source Judge did not produce valid evidence: {src_judge.get('error')}")
        src_score = float(src_judge["score"])

        helper_control.sleep(1)
        editor_control.wake_up()
        messages = editor_prompt(task, src_mas, src_traj, constraints)
        try:
            peer_text, patch, peer_retry = editor.chat_validated(
                messages,
                validator=lambda value: normalize_editor_action(value, src_mas=src_mas),
                guided_json=None,
                retry_instruction=editor_retry_instruction(src_mas),
                max_tokens=args.editor_max_tokens,
                temperature=args.editor_temperature,
            )
        except LocalChatError as exc:
            raise RuntimeError(f"Refiner failed structured validation: {exc}; last={exc.last_response!r}") from exc
        if not peer_text.strip():
            raise RuntimeError("Refiner returned empty output")

        if is_accept_patch(patch):
            # Re-execute the unchanged graph for the independent updated sample.
            updated_mas = src_mas
            helper_control.sleep(1)
            editor_control.sleep(1)
            helper_control.wake_up()
            updated_traj = executor.execute(
                task, updated_mas, expected, profile_phase="updated_subagent", pair_id=candidate_id
            )
            updated_judge = judge.judge_trajectory(
                updated_traj, pair_id=candidate_id, candidate="updated", **common_judge
            )
            if not is_valid_bcp_judge_metadata(updated_judge, max_tokens=judge.max_tokens):
                raise RuntimeError(f"updated Judge did not produce valid evidence: {updated_judge.get('error')}")
            upd_score = float(updated_judge["score"])
            delta = upd_score - src_score
            eta = max(0.0, float(os.getenv("MAS_LOOP_INVALID_ACTION_PENALTY", "0.5")))
            editor_reward = eta if src_score >= 0.999 else -eta
            editor_outcome = "accepted_correct" if src_score >= 0.999 else "accepted_incorrect"
            updated_valid = True
        else:
            updated_mas, patch_errors = apply_patch(src_mas, patch)
            if patch_errors:
                raise RuntimeError("invalid_patch_application:" + ";".join(patch_errors))
            editor_control.sleep(1)
            helper_control.wake_up()
            updated_traj = executor.execute(
                task, updated_mas, expected, profile_phase="updated_subagent", pair_id=candidate_id
            )
            updated_judge = judge.judge_trajectory(
                updated_traj, pair_id=candidate_id, candidate="updated", **common_judge
            )
            if not is_valid_bcp_judge_metadata(updated_judge, max_tokens=judge.max_tokens):
                raise RuntimeError(f"updated Judge did not produce valid evidence: {updated_judge.get('error')}")
            upd_score = float(updated_judge["score"])
            delta = upd_score - src_score
            editor_reward = delta
            editor_outcome = "improved" if delta > 0 else ("damaged" if delta < 0 else "no_gain")
            updated_valid = True

        patch_diagnosis = patch.get("diagnosis", {}) if isinstance(patch, dict) else {}
        candidate = {
            "candidate_id": candidate_id,
            "candidate_index": int(rec.get("rollout_sample_index", 0)),
            "editor_group_uid": f"editor-task:{rec['grpo_group_uid']}:step-{rec['optimizer_step']}",
            "prompt": messages,
            "output": peer_text,
            "retry": peer_retry,
            "patch": patch,
            "updated_mas": updated_mas,
            "updated_trajectory": updated_traj,
            "src_score": src_score,
            "upd_score": upd_score,
            "score_delta": delta,
            "reward": editor_reward,
            "outcome": editor_outcome,
            "trainable": True,
            "updated_valid": updated_valid,
            "violations": [],
            "error": "",
        }
        lambda_adv = float(os.getenv("MAS_LOOP_ORCH_ADV_PENALTY", "0.3"))
        orchestra_reward = src_score - lambda_adv * max(delta, 0.0)
        orchestra_outcome = (
            "source_exploited" if delta > 0 else
            ("source_robust" if src_score >= 0.999 else "source_weak")
        )
        rec.update({
            "src_mas": src_mas,
            "src_trajectory": src_traj,
            "src_score": src_score,
            "editor_candidates": [candidate],
            "editor_candidate_count": 1,
            "editor_valid_candidate_count": 1,
            "editor_candidate_diversity": {
                "valid_candidates": 1,
                "unique_outputs": 1,
                "unique_diagnosis_targets": int(bool(str(patch_diagnosis.get("target_id", "")).strip())),
                "unique_failure_types": int(bool(str(patch_diagnosis.get("failure_type", "")).strip())),
                "unique_operation_sequences": int(bool((patch or {}).get("operations"))) if isinstance(patch, dict) else 0,
            },
            "peer_prompt": messages,
            "peer_output": peer_text,
            "peer_retry": peer_retry,
            "editor_trainable": True,
            "editor_group_uid": candidate["editor_group_uid"],
            "editor_score": editor_reward,
            "editor_outcome": editor_outcome,
            "edit_operation": patch,
            "updated_mas": updated_mas,
            "updated_trajectory": updated_traj,
            "upd_score": upd_score,
            "score_delta": delta,
            "score": orchestra_reward,
            "outcome": orchestra_outcome,
            "violations": [],
            "error": "",
            "adversarial_role": "source_policy_penalized_by_refiner_exploit",
            "reward_design": "source_minus_refiner_exploit_v1",
            "targeted_repair_audit": {
                "schema_version": "mas_loop_targeted_pair_repair_v1",
                "time": time.time(),
                "reason": "bounded_xml_reparse_then_real_downstream_execution",
                "preserved_actor_output": True,
                "source_judge_valid": True,
                "updated_judge_valid": True,
            },
        })
        rec["structured_pair"] = _structured_pair(rec)
        rec["canonical_record_type"] = "orchestra_editor_pair"

        if not all(math.isfinite(float(rec[k])) for k in ("src_score", "upd_score", "score_delta", "score", "editor_score")):
            raise RuntimeError("non-finite repaired score")

        # Stage Judge evidence onto the checkpoint filesystem before changing
        # the canonical row.  This catches ENOSPC/permission/cross-device
        # failures while the original row is still untouched.
        staged_judges: list[tuple[Path, Path, Path]] = []
        for source in sorted(temp_judges.glob("*.jsonl")):
            target = args.judge_dir / f"judge-targeted-repair-{os.getpid()}-{source.name}"
            staged_judges.append((_stage_cross_device_publish(source, target), target, source))

        backup = shard.with_name(f"{shard.name}.before-targeted-repair-{int(time.time())}.bak")
        shutil.copy2(shard, backup)
        lines[args.line - 1] = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        tmp = shard.with_name(f".{shard.name}.repair.{os.getpid()}.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, shard)

        published = []
        for staged, target, source in staged_judges:
            os.replace(staged, target)
            source.unlink(missing_ok=True)
            published.append(str(target))
        result = {
            "status": "repaired",
            "shard": str(shard),
            "line": args.line,
            "pair_id": pair_id,
            "backup": str(backup),
            "judge_shards": published,
            "src_score": src_score,
            "upd_score": upd_score,
            "score_delta": delta,
            "editor_outcome": editor_outcome,
            "orchestra_outcome": orchestra_outcome,
            "repair_root": str(repair_root),
        }
        (repair_root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        _sleep_if_awake(editor_control)
        _sleep_if_awake(helper_control)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--line", type=int, required=True)
    ap.add_argument("--pair-id", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--helper-url", required=True)
    ap.add_argument("--helper-model", default="judge")
    ap.add_argument("--editor-url", required=True)
    ap.add_argument("--editor-model", default="editor-peer")
    ap.add_argument("--judge-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp"))
    ap.add_argument("--max-model-len", type=int, default=23192)
    ap.add_argument("--max-agent-tokens", type=int, default=8192)
    ap.add_argument("--editor-max-tokens", type=int, default=768)
    ap.add_argument("--editor-temperature", type=float, default=0.7)
    args = ap.parse_args()
    print(json.dumps(repair(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
