"""Backfill missing Qwen Judge evidence in completed BCP canonical Pair shards.

This module is intentionally a *post-training* repair pass.  It never changes a
rollout trajectory or hides a pipeline error: it only sends candidates whose
nested Judge metadata and/or standalone ``judge_results`` evidence is missing
to the configured Qwen Judge, then atomically rewrites the affected canonical
JSONL shards.  The Judge prompt and raw response are persisted by
:class:`mas_loop_rl.bcp_judge.BCPJudge` in the normal ``judge_results`` stream.
"""
from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .bcp_judge import BCPJudge, is_valid_bcp_judge_metadata
from .pair_selector import select_pair_winner
from .schema import is_accept_patch, normalize_editor_action
from .sharded_jsonl import close_jsonl_writers

SCHEMA = "mas_loop_bcp_judge_backfill_v2"
JUDGE_MAX_TOKENS = 4096
LEGACY_CANDIDATES = ("source", "updated")
TRANSIENT_STORAGE_ERRNOS = frozenset(
    value
    for value in (
        errno.EIO,
        getattr(errno, "ESTALE", None),
        errno.ETIMEDOUT,
        errno.EAGAIN,
    )
    if value is not None
)


def _valid_judge(value: Any) -> bool:
    return is_valid_bcp_judge_metadata(value, max_tokens=JUDGE_MAX_TOKENS)


def _jsonl_files(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.jsonl") if p.is_file())


def _iter_text_lines_with_retry(
    path: Path,
    *,
    attempts: int = 8,
    initial_delay_sec: float = 0.25,
):
    """Yield numbered lines while tolerating bounded transient NFS read errors.

    A retry resumes after the last line successfully handed to the caller, so
    large canonical shards are not buffered in memory and callers do not see
    duplicate rows. JSON/schema failures happen outside this helper and remain
    fail-closed rather than being mistaken for storage failures.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    delay = max(0.0, float(initial_delay_sec))
    next_line_no = 1
    for attempt in range(1, attempts + 1):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if line_no < next_line_no:
                        continue
                    yield line_no, line
                    next_line_no = line_no + 1
            return
        except OSError as exc:
            if exc.errno not in TRANSIENT_STORAGE_ERRNOS or attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(max(delay * 2.0, 0.01), 4.0)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class MissingCandidate:
    shard: Path
    line_no: int
    pair_id: str
    candidate: str
    optimizer_step: int
    outer_adversarial_round: int
    data_source: str
    question: str
    correct_answer: str
    response: str
    nested_judge_valid: bool
    standalone_judge_present: bool
    pipeline_errors: tuple[str, ...]
    source_pair_id: str = ""
    candidate_index: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.pair_id, self.candidate


def _standalone_judged_keys(judge_dir: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for shard in _jsonl_files(judge_dir):
        for line_no, line in _iter_text_lines_with_retry(shard):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"invalid Judge JSON at {shard}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object Judge JSON at {shard}:{line_no}")
            pair_id = str(row.get("pair_id") or "")
            candidate = str(row.get("candidate") or "")
            # Do not pre-filter candidate labels here. The canonical scan is
            # authoritative for both legacy ``updated`` and v4 ``updated_i``
            # keys; retaining every valid key also lets strict validation reject
            # orphaned Judge evidence instead of silently ignoring it.
            if candidate and pair_id and _valid_judge(row.get("judge")):
                keys.add((pair_id, candidate))
    return keys


def _trajectory_judge(trajectory: Any) -> Any:
    if not isinstance(trajectory, dict):
        return None
    metadata = trajectory.get("metadata")
    return metadata.get("judge") if isinstance(metadata, dict) else None


def _is_accept_candidate(candidate: dict[str, Any]) -> bool:
    outcome = str(candidate.get("outcome") or "")
    if outcome.startswith("accepted_"):
        return True
    try:
        return is_accept_patch(normalize_editor_action(candidate.get("patch")))
    except (TypeError, ValueError):
        return False


def _is_executed_candidate(candidate: dict[str, Any]) -> bool:
    """Return whether v4 produced an independently judged updated execution."""
    if _is_accept_candidate(candidate):
        return False
    return str(candidate.get("outcome") or "") in {"improved", "damaged", "no_gain"}


def _legacy_accept(pair: dict[str, Any]) -> bool:
    editor = pair.get("editor") if isinstance(pair.get("editor"), dict) else {}
    try:
        return is_accept_patch(normalize_editor_action(editor.get("patch")))
    except (TypeError, ValueError):
        return False


def _append_missing(
    missing: list[MissingCandidate],
    *,
    shard: Path,
    line_no: int,
    pair_id: str,
    candidate: str,
    optimizer_step: int,
    outer_round: int,
    data_source: str,
    question: str,
    correct_answer: str,
    response: str,
    nested_ok: bool,
    standalone_ok: bool,
    pipeline_errors: tuple[str, ...],
    source_pair_id: str,
    candidate_index: int | None = None,
) -> None:
    missing.append(
        MissingCandidate(
            shard=shard,
            line_no=line_no,
            pair_id=pair_id,
            candidate=candidate,
            optimizer_step=optimizer_step,
            outer_adversarial_round=outer_round,
            data_source=data_source,
            question=question,
            correct_answer=correct_answer,
            response=response,
            nested_judge_valid=nested_ok,
            standalone_judge_present=standalone_ok,
            pipeline_errors=pipeline_errors,
            source_pair_id=source_pair_id,
            candidate_index=candidate_index,
        )
    )


def scan_missing_candidates(phase_dir: str | Path) -> tuple[list[MissingCandidate], dict[str, Any]]:
    """Return only source or actually executed updated candidates lacking Judge evidence.

    V4 rows use one source key ``(source_pair_id, source)`` and one dynamic key
    ``(candidate_id, updated_i)`` for every independently executed Editor
    candidate. Accept actions reuse the source verdict. Schema-invalid/error
    candidates were never executed and therefore must not be sent to Judge.
    Rows without ``editor_candidates`` retain the legacy source/updated contract.
    """
    phase_dir = Path(phase_dir).resolve()
    canonical_dir = phase_dir / "rollouts" / "canonical_rollouts"
    judge_dir = phase_dir / "rollouts" / "judge_results"
    shards = _jsonl_files(canonical_dir)
    if not shards:
        raise RuntimeError(f"no canonical Pair shards under {canonical_dir}")
    standalone = _standalone_judged_keys(judge_dir)
    missing: list[MissingCandidate] = []
    source_pair_ids: set[str] = set()
    candidate_ids: set[str] = set()
    rows = 0
    candidate_count = 0
    nested_valid = 0
    standalone_valid = 0
    required_standalone_candidates = 0
    accepted_updated_judge_reuse = 0
    skipped_unexecuted_candidates = 0
    executed_updated_candidates = 0
    v5_task_groups: dict[tuple[str, int], list[int]] = {}
    v5_pair_rows = 0

    for shard in shards:
        for line_no, line in _iter_text_lines_with_retry(shard):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"invalid canonical JSON at {shard}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object canonical JSON at {shard}:{line_no}")
            pair = row.get("structured_pair")
            if not isinstance(pair, dict):
                raise RuntimeError(f"missing structured_pair at {shard}:{line_no}")
            source_pair_id = str(pair.get("pair_id") or row.get("pair_id") or "")
            if not source_pair_id:
                raise RuntimeError(f"missing pair_id at {shard}:{line_no}")
            if source_pair_id in source_pair_ids:
                raise RuntimeError(f"duplicate canonical pair_id: {source_pair_id}")
            source_pair_ids.add(source_pair_id)
            rows += 1

            task = pair.get("task") if isinstance(pair.get("task"), dict) else {}
            question = str(task.get("task_text") or row.get("task") or "")
            correct_answer = str(task.get("expected_answer") or row.get("expected") or "")
            optimizer_step = int(pair.get("optimizer_step") or row.get("optimizer_step") or 0)
            outer_round = int(pair.get("outer_adversarial_round") or row.get("outer_adversarial_round") or 1)
            data_source = str(row.get("data_source") or "bcp")
            errors = pair.get("errors") if isinstance(pair.get("errors"), list) else []
            pipeline_errors = tuple(str(x) for x in errors)

            source = pair.get("source") if isinstance(pair.get("source"), dict) else {}
            source_traj = source.get("trajectory") if isinstance(source.get("trajectory"), dict) else {}
            source_response = str(source_traj.get("final_output") or source.get("final_output") or "")
            source_nested_ok = _valid_judge(source.get("judge"))
            source_standalone_ok = (source_pair_id, "source") in standalone
            candidate_count += 1
            required_standalone_candidates += 1
            nested_valid += int(source_nested_ok)
            standalone_valid += int(source_standalone_ok)
            if not source_nested_ok or not source_standalone_ok:
                _append_missing(
                    missing, shard=shard, line_no=line_no, pair_id=source_pair_id,
                    candidate="source", optimizer_step=optimizer_step, outer_round=outer_round,
                    data_source=data_source, question=question, correct_answer=correct_answer,
                    response=source_response, nested_ok=source_nested_ok,
                    standalone_ok=source_standalone_ok, pipeline_errors=pipeline_errors,
                    source_pair_id=source_pair_id,
                )

            editor_candidates = pair.get("editor_candidates")
            schema_version = str(pair.get("schema_version") or row.get("schema_version") or "")
            is_v5_pair = schema_version.endswith("v5")
            is_v6_pair = schema_version.endswith("v6")
            is_one_to_one = is_v5_pair or is_v6_pair
            if isinstance(editor_candidates, list):
                if not editor_candidates:
                    raise RuntimeError(f"empty editor_candidates at {shard}:{line_no}")
                if is_one_to_one and len(editor_candidates) != 1:
                    raise RuntimeError(f"v5/v6 pair must contain exactly one Refiner candidate at {shard}:{line_no}")
                observed_indices: list[int] = []
                for position, raw_candidate in enumerate(editor_candidates):
                    if not isinstance(raw_candidate, dict):
                        raise RuntimeError(f"non-object editor candidate at {shard}:{line_no}:{position}")
                    candidate_index = int(raw_candidate.get("candidate_index", -1))
                    observed_indices.append(candidate_index)
                    candidate_id = str(raw_candidate.get("candidate_id") or "")
                    if is_one_to_one:
                        rollout_index = int(pair.get("rollout_sample_index", row.get("rollout_sample_index", -1)))
                        if candidate_index != rollout_index:
                            raise RuntimeError(
                                f"v5/v6 candidate_index must equal rollout_sample_index for {source_pair_id}: "
                                f"candidate={candidate_index} rollout={rollout_index}"
                            )
                        expected_id = f"{source_pair_id}:refiner"
                        candidate_label = "updated"
                        group_uid = str(raw_candidate.get("editor_group_uid") or "")
                        if not group_uid:
                            raise RuntimeError(f"missing v5/v6 editor_group_uid for {candidate_id or source_pair_id}")
                        v5_task_groups.setdefault((group_uid, optimizer_step), []).append(rollout_index)
                        v5_pair_rows += 1
                    else:
                        expected_id = f"{source_pair_id}:editor-{candidate_index}"
                        candidate_label = f"updated_{candidate_index}"
                    if not candidate_id:
                        raise RuntimeError(f"missing candidate_id at {shard}:{line_no}:{position}")
                    if candidate_id != expected_id:
                        raise RuntimeError(
                            f"candidate_id mismatch at {shard}:{line_no}:{position}: "
                            f"expected={expected_id!r} actual={candidate_id!r}"
                        )
                    if candidate_id in candidate_ids:
                        raise RuntimeError(f"duplicate editor candidate_id: {candidate_id}")
                    candidate_ids.add(candidate_id)
                    candidate_count += 1

                    if _is_accept_candidate(raw_candidate):
                        accepted_updated_judge_reuse += 1
                        continue
                    if not _is_executed_candidate(raw_candidate):
                        skipped_unexecuted_candidates += 1
                        continue

                    executed_updated_candidates += 1
                    trajectory = raw_candidate.get("updated_trajectory")
                    trajectory = trajectory if isinstance(trajectory, dict) else {}
                    response = str(trajectory.get("final_output") or "")
                    nested_ok = _valid_judge(_trajectory_judge(trajectory))
                    standalone_ok = (candidate_id, candidate_label) in standalone
                    required_standalone_candidates += 1
                    nested_valid += int(nested_ok)
                    standalone_valid += int(standalone_ok)
                    if not nested_ok or not standalone_ok:
                        _append_missing(
                            missing, shard=shard, line_no=line_no, pair_id=candidate_id,
                            candidate=candidate_label, optimizer_step=optimizer_step,
                            outer_round=outer_round, data_source=data_source, question=question,
                            correct_answer=correct_answer, response=response, nested_ok=nested_ok,
                            standalone_ok=standalone_ok, pipeline_errors=pipeline_errors,
                            source_pair_id=source_pair_id, candidate_index=candidate_index,
                        )
                if not is_one_to_one and observed_indices != list(range(len(editor_candidates))):
                    raise RuntimeError(
                        f"non-contiguous editor candidate indices at {shard}:{line_no}: {observed_indices}"
                    )
                continue

            # Backward-compatible single-candidate rows.
            updated = pair.get("updated") if isinstance(pair.get("updated"), dict) else {}
            candidate_count += 1
            if _legacy_accept(pair):
                accepted_updated_judge_reuse += 1
                continue
            trajectory = updated.get("trajectory") if isinstance(updated.get("trajectory"), dict) else {}
            response = str(trajectory.get("final_output") or updated.get("final_output") or "")
            nested_ok = _valid_judge(updated.get("judge"))
            standalone_ok = (source_pair_id, "updated") in standalone
            required_standalone_candidates += 1
            nested_valid += int(nested_ok)
            standalone_valid += int(standalone_ok)
            executed_updated_candidates += 1
            if not nested_ok or not standalone_ok:
                _append_missing(
                    missing, shard=shard, line_no=line_no, pair_id=source_pair_id,
                    candidate="updated", optimizer_step=optimizer_step, outer_round=outer_round,
                    data_source=data_source, question=question, correct_answer=correct_answer,
                    response=response, nested_ok=nested_ok, standalone_ok=standalone_ok,
                    pipeline_errors=pipeline_errors, source_pair_id=source_pair_id,
                )

    expected_v5_group_size = int(os.getenv("MAS_LOOP_EDITOR_GROUP_SIZE", os.getenv("MAS_LOOP_ROLLOUT_N", "32")))
    if expected_v5_group_size <= 0:
        raise RuntimeError(f"invalid expected v5 group size: {expected_v5_group_size}")
    for (group_uid, step), indices in sorted(v5_task_groups.items()):
        observed = sorted(indices)
        expected_indices = list(range(expected_v5_group_size))
        if observed != expected_indices:
            raise RuntimeError(
                f"incomplete/non-contiguous v5 task group {group_uid!r} at optimizer_step={step}: "
                f"expected={expected_indices} actual={observed}"
            )

    report = {
        "phase_dir": str(phase_dir),
        "canonical_shards": len(shards),
        "canonical_pair_rows": rows,
        "candidate_count": candidate_count,
        "executed_updated_candidates": executed_updated_candidates,
        "skipped_unexecuted_candidates": skipped_unexecuted_candidates,
        "nested_valid_candidates": nested_valid,
        "standalone_valid_candidates": standalone_valid,
        "required_standalone_candidates": required_standalone_candidates,
        "accepted_updated_judge_reuse": accepted_updated_judge_reuse,
        "v5_pair_rows": v5_pair_rows,
        "v5_task_groups": len(v5_task_groups),
        "v5_expected_group_size": expected_v5_group_size,
        "missing_candidates": len(missing),
        "empty_response_candidates": sum(1 for target in missing if not target.response.strip()),
    }
    return missing, report

def _set_trajectory_judge(trajectory: dict[str, Any], judge: dict[str, Any]) -> None:
    metadata = trajectory.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        trajectory["metadata"] = metadata
    metadata.setdefault("heuristic_score_before_llm_judge", metadata.get("score"))
    metadata["score"] = float(judge["score"])
    metadata["judge"] = judge
    metadata["scoring_method"] = "qwen_llm_judge"
    trajectory["success"] = float(judge["score"]) >= 0.999


def _sync_v5_pair(pair: dict[str, Any]) -> None:
    """Recompute the unique v5 pair, selector state, and reward attribution."""
    source = pair.get("source") if isinstance(pair.get("source"), dict) else {}
    source_score = float(source.get("score", 0.0) or 0.0)
    candidates = pair.get("editor_candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise RuntimeError("v5 pair must contain exactly one Refiner candidate")
    candidate = candidates[0]
    candidate["src_score"] = source_score
    if _is_accept_candidate(candidate):
        candidate["updated_mas"] = copy.deepcopy(source.get("mas"))
        candidate["updated_trajectory"] = copy.deepcopy(source.get("trajectory"))
        candidate["upd_score"] = source_score
        candidate["score_delta"] = 0.0
        eta = max(0.0, float(os.getenv("MAS_LOOP_INVALID_ACTION_PENALTY", "0.5")))
        candidate["reward"] = eta if source_score >= 0.999 else -eta
        candidate["outcome"] = "accepted_correct" if source_score >= 0.999 else "accepted_incorrect"
        candidate["updated_valid"] = True
    elif _is_executed_candidate(candidate):
        judge = _trajectory_judge(candidate.get("updated_trajectory"))
        if _valid_judge(judge):
            updated_score = float(judge["score"])
            delta = updated_score - source_score
            candidate["upd_score"] = updated_score
            candidate["score_delta"] = delta
            candidate["reward"] = delta
            candidate["outcome"] = "improved" if delta > 0 else ("damaged" if delta < 0 else "no_gain")
            candidate["updated_valid"] = True
    else:
        candidate["updated_mas"] = copy.deepcopy(source.get("mas"))
        candidate["updated_trajectory"] = copy.deepcopy(source.get("trajectory"))
        candidate["upd_score"] = source_score
        candidate["score_delta"] = 0.0
        candidate["reward"] = -1.0
        candidate["updated_valid"] = False

    trajectory = candidate.get("updated_trajectory")
    pair["updated"] = {
        "mas": copy.deepcopy(candidate.get("updated_mas")),
        "trajectory": copy.deepcopy(trajectory),
        "score": float(candidate.get("upd_score", source_score) or 0.0),
        "judge": copy.deepcopy(_trajectory_judge(trajectory)),
    }
    pair["editor"] = {
        "prompt": copy.deepcopy(candidate.get("prompt", [])),
        "output": candidate.get("output", ""),
        "patch": copy.deepcopy(candidate.get("patch", {})),
        "reward": float(candidate.get("reward", -1.0)),
        "outcome": str(candidate.get("outcome") or "error"),
        "trainable": bool(candidate.get("trainable", False)),
        "retry": copy.deepcopy(candidate.get("retry", {})),
    }
    comparison = pair.setdefault("comparison", {})
    if not isinstance(comparison, dict):
        comparison = {}
        pair["comparison"] = comparison
    delta = float(candidate.get("score_delta", 0.0) or 0.0)
    comparison["score_delta"] = delta
    comparison["editor_reward"] = float(candidate.get("reward", -1.0))
    comparison["editor_outcome"] = str(candidate.get("outcome") or "error")
    is_v6 = str(pair.get("schema_version") or "").endswith("v6")
    if is_v6:
        lambda_adv = float(os.getenv("MAS_LOOP_ORCH_ADV_PENALTY", "0.3"))
        comparison["orchestra_reward"] = source_score - lambda_adv * max(delta, 0.0)
        pair.pop("selection", None)
        pair.pop("winner", None)
        pair["patch_translation"] = copy.deepcopy((candidate.get("patch") or {}).get("translation", {}))
        return
    alpha = float(os.getenv("MAS_LOOP_ORCH_WINNER_ALPHA", os.getenv("MAS_LOOP_ORCH_ADV_ALPHA", "0.3")))
    comparison["orchestra_reward"] = source_score + alpha * max(delta, 0.0)
    selection = select_pair_winner(
        source_score=source_score,
        updated_score=float(candidate.get("upd_score", source_score) or 0.0),
        source_mas=source.get("mas"),
        updated_mas=candidate.get("updated_mas"),
        source_trajectory=source.get("trajectory"),
        updated_trajectory=candidate.get("updated_trajectory"),
        source_valid=True,
        updated_valid=bool(candidate.get("updated_valid", False)),
        explicit_accept=_is_accept_candidate(candidate),
    ).to_dict()
    pair["selection"] = selection
    winner_state = source if selection["selected_origin"] == "source" else pair["updated"]
    pair["winner"] = {
        "origin": selection["selected_origin"],
        "mas": copy.deepcopy(winner_state.get("mas")),
        "trajectory": copy.deepcopy(winner_state.get("trajectory")),
        "score": float(winner_state.get("score", 0.0) or 0.0),
    }


def _sync_v4_primary(pair: dict[str, Any]) -> None:
    """Recompute candidate deltas and backward-compatible strongest fields."""
    source = pair.get("source") if isinstance(pair.get("source"), dict) else {}
    source_score = float(source.get("score", 0.0) or 0.0)
    source_mas = source.get("mas")
    source_trajectory = source.get("trajectory")
    candidates = pair.get("editor_candidates")
    if not isinstance(candidates, list):
        return

    selectable: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate["src_score"] = source_score
        if _is_accept_candidate(candidate):
            candidate["updated_mas"] = copy.deepcopy(source_mas)
            candidate["updated_trajectory"] = copy.deepcopy(source_trajectory)
            candidate["upd_score"] = source_score
            candidate["score_delta"] = 0.0
            candidate["reward"] = 0.0
            candidate["outcome"] = "accepted_correct" if source_score >= 0.999 else "accepted_incorrect"
        elif _is_executed_candidate(candidate):
            trajectory = candidate.get("updated_trajectory")
            judge = _trajectory_judge(trajectory)
            if _valid_judge(judge):
                updated_score = float(judge["score"])
                delta = updated_score - source_score
                candidate["upd_score"] = updated_score
                candidate["score_delta"] = delta
                candidate["reward"] = delta
                candidate["outcome"] = "improved" if delta > 0 else ("damaged" if delta < 0 else "no_gain")
        else:
            # No updated execution exists for malformed/error output. Keep it
            # fail-closed while ensuring compatibility fields cannot masquerade
            # as independent trajectory evidence.
            candidate["updated_mas"] = copy.deepcopy(source_mas)
            candidate["updated_trajectory"] = copy.deepcopy(source_trajectory)
            candidate["upd_score"] = source_score
            candidate["score_delta"] = 0.0
            candidate["reward"] = -1.0
        if candidate.get("trainable") and str(candidate.get("outcome") or "") != "error":
            selectable.append(candidate)

    primary = (
        max(selectable, key=lambda value: (float(value.get("score_delta", 0.0)), -int(value.get("candidate_index", 0))))
        if selectable
        else (candidates[0] if candidates else {})
    )
    if not isinstance(primary, dict):
        return
    trajectory = primary.get("updated_trajectory")
    primary_judge = _trajectory_judge(trajectory)
    pair["updated"] = {
        "mas": copy.deepcopy(primary.get("updated_mas")),
        "trajectory": copy.deepcopy(trajectory),
        "score": float(primary.get("upd_score", source_score) or 0.0),
        "judge": copy.deepcopy(primary_judge),
    }
    pair["editor"] = {
        "prompt": copy.deepcopy(primary.get("prompt", [])),
        "output": primary.get("output", ""),
        "patch": copy.deepcopy(primary.get("patch", {})),
        "reward": float(primary.get("reward", -1.0) or 0.0),
        "outcome": str(primary.get("outcome") or "error"),
        "trainable": bool(primary.get("trainable", False)),
        "retry": copy.deepcopy(primary.get("retry", {})),
    }
    comparison = pair.setdefault("comparison", {})
    if isinstance(comparison, dict):
        comparison["score_delta"] = float(primary.get("score_delta", 0.0) or 0.0)
        comparison["editor_reward"] = float(primary.get("reward", -1.0) or 0.0)
        comparison["editor_outcome"] = str(primary.get("outcome") or "error")


def _atomic_rewrite_shard(
    shard: Path,
    replacements: dict[tuple[str, str], dict[str, Any]],
) -> int:
    tmp = shard.with_name(f".{shard.name}.judge-backfill.tmp.{os.getpid()}")
    changed = 0
    try:
        with shard.open("r", encoding="utf-8") as source_handle, tmp.open("w", encoding="utf-8") as dest:
            for line_no, line in enumerate(source_handle, 1):
                if not line.strip():
                    dest.write(line)
                    continue
                row = json.loads(line)
                pair = row.get("structured_pair") if isinstance(row, dict) else None
                if not isinstance(pair, dict):
                    raise RuntimeError(f"missing structured_pair while rewriting {shard}:{line_no}")
                source_pair_id = str(pair.get("pair_id") or row.get("pair_id") or "")

                source_judge = replacements.get((source_pair_id, "source"))
                if source_judge is not None:
                    block = pair.get("source") if isinstance(pair.get("source"), dict) else {}
                    pair["source"] = block
                    block["judge"] = source_judge
                    block["score"] = float(source_judge["score"])
                    trajectory = block.get("trajectory")
                    if isinstance(trajectory, dict):
                        _set_trajectory_judge(trajectory, source_judge)
                    changed += 1

                editor_candidates = pair.get("editor_candidates")
                if isinstance(editor_candidates, list):
                    one_to_one_version = str(pair.get("schema_version") or row.get("schema_version") or "")
                    is_v5_pair = one_to_one_version.endswith("v5")
                    is_v6_pair = one_to_one_version.endswith("v6")
                    is_one_to_one = is_v5_pair or is_v6_pair
                    for candidate in editor_candidates:
                        if not isinstance(candidate, dict):
                            continue
                        candidate_id = str(candidate.get("candidate_id") or "")
                        candidate_index = int(candidate.get("candidate_index", -1))
                        candidate_label = "updated" if is_one_to_one else f"updated_{candidate_index}"
                        judge = replacements.get((candidate_id, candidate_label))
                        if judge is None:
                            continue
                        trajectory = candidate.get("updated_trajectory")
                        if not isinstance(trajectory, dict):
                            trajectory = {}
                            candidate["updated_trajectory"] = trajectory
                        _set_trajectory_judge(trajectory, judge)
                        changed += 1
                    if is_one_to_one:
                        _sync_v5_pair(pair)
                    else:
                        _sync_v4_primary(pair)
                else:
                    judge = replacements.get((source_pair_id, "updated"))
                    if judge is not None:
                        block = pair.get("updated") if isinstance(pair.get("updated"), dict) else {}
                        pair["updated"] = block
                        block["judge"] = judge
                        block["score"] = float(judge["score"])
                        trajectory = block.get("trajectory")
                        if isinstance(trajectory, dict):
                            _set_trajectory_judge(trajectory, judge)
                        changed += 1
                dest.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            dest.flush()
            os.fsync(dest.fileno())
        os.chmod(tmp, shard.stat().st_mode)
        os.replace(tmp, shard)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return changed

def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def backfill_phase(
    phase_dir: str | Path,
    *,
    judge_base_url: str,
    judge_model: str = "judge",
    max_model_len: int = 23192,
    max_workers: int = 256,
    audit_path: str | Path | None = None,
    judge_factory: Callable[..., Any] = BCPJudge,
) -> dict[str, Any]:
    """Judge all missing candidates and atomically patch their canonical rows."""
    phase_dir = Path(phase_dir).resolve()
    audit_path = Path(audit_path) if audit_path else phase_dir / "judge_backfill_audit.json"
    started = time.time()
    targets, before = scan_missing_candidates(phase_dir)
    base_audit: dict[str, Any] = {
        "schema_version": SCHEMA,
        "phase_dir": str(phase_dir),
        "started_at": started,
        "judge_base_url": judge_base_url,
        "judge_model": judge_model,
        "judge_max_model_len": int(max_model_len),
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "max_workers": int(max_workers),
        "before": before,
    }
    if not targets:
        audit = {**base_audit, "state": "no_op", "rewritten_shards": [], "results": [], "after": before, "completed_at": time.time()}
        _atomic_json(audit_path, audit)
        return audit

    os.environ["MAS_LOOP_JUDGE_MAX_TOKENS"] = str(JUDGE_MAX_TOKENS)
    os.environ["MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE"] = str(max(1, int(max_workers)))
    judge_results_dir = phase_dir / "rollouts" / "judge_results"
    judge = judge_factory(
        base_url=judge_base_url,
        model=judge_model,
        manage_sleep=False,
        max_model_len=int(max_model_len),
        result_dir=judge_results_dir,
    )

    def score(target: MissingCandidate) -> tuple[MissingCandidate, dict[str, Any]]:
        result = judge.judge(
            question=target.question,
            response=target.response,
            correct_answer=target.correct_answer,
            pair_id=target.pair_id,
            candidate=target.candidate,
            optimizer_step=target.optimizer_step,
            outer_adversarial_round=target.outer_adversarial_round,
            data_source=f"{target.data_source}_posttrain_backfill",
        )
        if not _valid_judge(result):
            raise RuntimeError(f"Judge returned invalid metadata for {target.pair_id}:{target.candidate}: {result}")
        result = dict(result)
        result["backfill"] = {
            "schema_version": SCHEMA,
            "time": time.time(),
            "reason": "missing_nested_or_standalone_judge_evidence",
            "previous_nested_valid": target.nested_judge_valid,
            "previous_standalone_present": target.standalone_judge_present,
        }
        return target, result

    replacements_by_shard: dict[Path, dict[tuple[str, str], dict[str, Any]]] = {}
    result_audit: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="bcp-judge-backfill") as pool:
            futures = [pool.submit(score, target) for target in targets]
            for future in as_completed(futures):
                target, result = future.result()
                replacements_by_shard.setdefault(target.shard, {})[target.key] = result
                result_audit.append({
                    "pair_id": target.pair_id,
                    "candidate": target.candidate,
                    "shard": str(target.shard),
                    "line_no": target.line_no,
                    "optimizer_step": target.optimizer_step,
                    "empty_response": not bool(target.response.strip()),
                    "pipeline_errors": list(target.pipeline_errors),
                    "previous_nested_valid": target.nested_judge_valid,
                    "previous_standalone_present": target.standalone_judge_present,
                    "score": float(result["score"]),
                    "judge_error": str(result.get("error") or ""),
                })
    finally:
        # Shared FUSE mounts can block a read-open of a newly appended shard
        # while this process still owns its append descriptor.  The backfill is
        # a bounded post-processing pass, so close all writers for this stream
        # before canonical rewrite and strict read-back verification.
        close_jsonl_writers(judge_results_dir)

    rewritten: list[dict[str, Any]] = []
    for shard in sorted(replacements_by_shard, key=str):
        expected = len(replacements_by_shard[shard])
        changed = _atomic_rewrite_shard(shard, replacements_by_shard[shard])
        if changed != expected:
            raise RuntimeError(f"canonical rewrite count mismatch for {shard}: expected={expected} changed={changed}")
        rewritten.append({"shard": str(shard), "replacements": changed})

    remaining, after = scan_missing_candidates(phase_dir)
    audit = {
        **base_audit,
        "state": "complete" if not remaining else "incomplete",
        "rewritten_shards": rewritten,
        "results": sorted(result_audit, key=lambda x: (x["pair_id"], x["candidate"])),
        "after": after,
        "completed_at": time.time(),
    }
    _atomic_json(audit_path, audit)
    if remaining:
        raise RuntimeError(f"Judge backfill left {len(remaining)} candidates incomplete; inspect {audit_path}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", required=True, type=Path)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:8313/v1")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--max-model-len", type=int, default=23192)
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    targets, report = scan_missing_candidates(args.phase_dir)
    if args.scan_only:
        payload = {"schema_version": SCHEMA, "state": "scan", **report}
        if args.audit:
            _atomic_json(args.audit, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    audit = backfill_phase(
        args.phase_dir,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        max_model_len=args.max_model_len,
        max_workers=args.max_workers,
        audit_path=args.audit,
    )
    print(json.dumps({"state": audit["state"], "before": audit["before"], "after": audit["after"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
