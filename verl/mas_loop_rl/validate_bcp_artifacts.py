"""Fail-closed validation of BCP training artifacts before phase hand-off.

The validator proves that successful BCP phases persisted the analysis data the
training contract requires.  It intentionally reads the on-disk JSONL shards,
rather than trusting process-local counters or launcher intent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .bcp_judge import is_valid_bcp_judge_metadata
from .pair_selector import SELECTOR_VERSION, select_pair_winner
from .schema import is_accept_patch, normalize_editor_action


# Core optimizer metrics are emitted by both the online Orchestra phase and
# the offline Editor replay phase.  Rollout-vs-training distribution metrics
# require actor rollout log-probabilities and therefore are authoritative only
# for Orchestra; Editor replay batches do not always carry those fields.
REQUIRED_RL_METRICS = {
    "actor/loss",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/kl_coef",
    "actor/entropy",
    "actor/grad_norm",
    "actor/lr",
    "critic/rewards/mean",
    "critic/score/mean",
}

ORCHESTRA_ONLY_RL_METRICS = {
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
}


def _jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("*.jsonl") if p.is_file())


def _iter_jsonl_strict(path: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    files = _jsonl_files(path)
    if not files:
        raise RuntimeError(f"no JSONL shards under {path}")
    for shard in files:
        with shard.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(f"invalid JSON at {shard}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise RuntimeError(f"non-object JSON at {shard}:{line_no}")
                yield shard, line_no, row


def _validate_metrics(metrics_file: Path, *, phase: str) -> dict[str, Any]:
    rows = list(_iter_jsonl_strict(metrics_file))
    if not rows:
        raise RuntimeError(f"no completed optimizer-step metrics in {metrics_file}")
    union: set[str] = set()
    steps: list[int] = []
    for _, _, row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError(f"metrics row is missing object field 'metrics': {metrics_file}")
        union.update(str(k) for k in metrics)
        step = row.get("global_step", metrics.get("training/global_step"))
        if step is not None:
            steps.append(int(step))
    required = set(REQUIRED_RL_METRICS)
    if phase == "orchestra":
        required.update(ORCHESTRA_ONLY_RL_METRICS)
    missing = sorted(required - union)
    if missing:
        raise RuntimeError(f"missing required RL metrics in {metrics_file}: {missing}")
    return {
        "rows": len(rows),
        "steps": sorted(set(steps)),
        "metric_keys": sorted(union),
        "required_metric_keys": sorted(required),
    }


def _valid_judge(judge: Any) -> bool:
    return is_valid_bcp_judge_metadata(judge, max_tokens=4096)




def _trajectory_judge(trajectory: Any) -> Any:
    if not isinstance(trajectory, dict):
        return None
    metadata = trajectory.get("metadata")
    return metadata.get("judge") if isinstance(metadata, dict) else None


def _is_accept_candidate(candidate: dict[str, Any]) -> bool:
    if str(candidate.get("outcome") or "").startswith("accepted_"):
        return True
    try:
        return is_accept_patch(normalize_editor_action(candidate.get("patch")))
    except (TypeError, ValueError):
        return False


def _is_executed_candidate(candidate: dict[str, Any]) -> bool:
    return not _is_accept_candidate(candidate) and str(candidate.get("outcome") or "") in {
        "improved", "damaged", "no_gain"
    }


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _legacy_accept(pair: dict[str, Any]) -> bool:
    editor = pair.get("editor") if isinstance(pair.get("editor"), dict) else {}
    try:
        return is_accept_patch(normalize_editor_action(editor.get("patch")))
    except (TypeError, ValueError):
        return False

def _validate_orchestra(phase_dir: Path) -> dict[str, Any]:
    rollout_dir = phase_dir / "rollouts"
    canonical_dir = rollout_dir / "canonical_rollouts"
    pair_ids: set[str] = set()
    candidate_ids: set[str] = set()
    missing_judge: list[str] = []
    rows = 0
    pair_error_rows = 0
    pair_error_entries = 0
    editor_error_rows = 0
    editor_retry_error_entries = 0
    judge_negative_one_candidates = 0
    standalone_expected: set[tuple[str, str]] = set()
    standalone_allowed: set[tuple[str, str]] = set()
    accepted_updated_judge_reuse = 0
    executed_updated_candidates = 0
    invalid_editor_output_candidates = 0
    skipped_unexecuted_candidates = 0
    total_editor_candidates = 0
    v5_task_groups: dict[tuple[str, int], list[int]] = {}
    v5_pair_rows = 0

    for shard, line_no, row in _iter_jsonl_strict(canonical_dir):
        rows += 1
        pair = row.get("structured_pair")
        if not isinstance(pair, dict):
            raise RuntimeError(f"missing structured_pair at {shard}:{line_no}")
        pair_id = str(pair.get("pair_id") or row.get("pair_id") or "")
        if not pair_id:
            raise RuntimeError(f"missing pair_id at {shard}:{line_no}")
        if pair_id in pair_ids:
            raise RuntimeError(f"duplicate canonical pair_id: {pair_id}")
        pair_ids.add(pair_id)

        pair_errors = pair.get("errors")
        if isinstance(pair_errors, list) and pair_errors:
            pair_error_rows += 1
            pair_error_entries += len(pair_errors)
        editor = pair.get("editor") if isinstance(pair.get("editor"), dict) else {}
        if str(editor.get("outcome") or "") == "error":
            editor_error_rows += 1
        retry = editor.get("retry")
        retry_errors = retry.get("errors") if isinstance(retry, dict) else None
        if isinstance(retry_errors, list):
            editor_retry_error_entries += len(retry_errors)

        source = pair.get("source") if isinstance(pair.get("source"), dict) else {}
        source_judge = source.get("judge")
        if not _valid_judge(source_judge):
            missing_judge.append(f"{pair_id}:source")
        elif float(source_judge.get("score")) == -1.0:
            judge_negative_one_candidates += 1
        source_key = (pair_id, "source")
        standalone_expected.add(source_key)
        standalone_allowed.add(source_key)
        source_score = float(source.get("score", source_judge.get("score", 0.0) if isinstance(source_judge, dict) else 0.0) or 0.0)

        editor_candidates = pair.get("editor_candidates")
        schema_version = str(pair.get("schema_version") or row.get("schema_version") or "")
        is_v5_pair = schema_version.endswith("v5")
        is_v6_pair = schema_version.endswith("v6")
        if isinstance(editor_candidates, list):
            if not editor_candidates:
                raise RuntimeError(f"empty editor_candidates at {shard}:{line_no}")

            if is_v5_pair or is_v6_pair:
                v5_pair_rows += 1
                if len(editor_candidates) != 1:
                    raise RuntimeError(
                        f"v5/v6 pair must contain exactly one Refiner candidate at {shard}:{line_no}; "
                        f"actual={len(editor_candidates)}"
                    )
                candidate = editor_candidates[0]
                if not isinstance(candidate, dict):
                    raise RuntimeError(f"non-object v5/v6 editor candidate at {shard}:{line_no}")
                total_editor_candidates += 1
                rollout_index = int(pair.get("rollout_sample_index", row.get("rollout_sample_index", -1)))
                candidate_index = int(candidate.get("candidate_index", -1))
                if candidate_index != rollout_index:
                    raise RuntimeError(
                        f"v5 candidate_index must equal rollout_sample_index for {pair_id}: "
                        f"candidate={candidate_index} rollout={rollout_index}"
                    )
                candidate_id = str(candidate.get("candidate_id") or "")
                expected_id = f"{pair_id}:refiner"
                if candidate_id != expected_id:
                    raise RuntimeError(
                        f"v5 candidate_id mismatch at {shard}:{line_no}: "
                        f"expected={expected_id!r} actual={candidate_id!r}"
                    )
                if candidate_id in candidate_ids:
                    raise RuntimeError(f"duplicate editor candidate_id: {candidate_id}")
                candidate_ids.add(candidate_id)
                if not _float_equal(candidate.get("src_score"), source_score):
                    raise RuntimeError(f"candidate src_score mismatch for {candidate_id}")

                group_uid = str(candidate.get("editor_group_uid") or "")
                if not group_uid:
                    raise RuntimeError(f"missing v5/v6 editor_group_uid for {candidate_id}")
                group_key = (group_uid, int(pair.get("optimizer_step", row.get("optimizer_step", 0)) or 0))
                v5_task_groups.setdefault(group_key, []).append(rollout_index)

                candidate_key = (candidate_id, "updated")
                if _is_accept_candidate(candidate):
                    accepted_updated_judge_reuse += 1
                    standalone_allowed.add(candidate_key)
                    if (
                        candidate.get("updated_mas") != source.get("mas")
                        or candidate.get("updated_trajectory") != source.get("trajectory")
                        or not _float_equal(candidate.get("upd_score"), source_score)
                        or not _float_equal(candidate.get("score_delta"), 0.0)
                        or not _float_equal(candidate.get("reward"), 0.0)
                    ):
                        raise RuntimeError(f"accept/no-op changed source evidence for {candidate_id}")
                elif _is_executed_candidate(candidate):
                    executed_updated_candidates += 1
                    standalone_expected.add(candidate_key)
                    standalone_allowed.add(candidate_key)
                    trajectory = candidate.get("updated_trajectory")
                    judge = _trajectory_judge(trajectory)
                    if not _valid_judge(judge):
                        missing_judge.append(f"{candidate_id}:updated")
                    else:
                        updated_score = float(judge["score"])
                        expected_delta = updated_score - source_score
                        if not _float_equal(candidate.get("upd_score"), updated_score):
                            raise RuntimeError(f"candidate upd_score mismatch for {candidate_id}")
                        if not _float_equal(candidate.get("score_delta"), expected_delta):
                            raise RuntimeError(f"candidate score_delta mismatch for {candidate_id}")
                        if not _float_equal(candidate.get("reward"), expected_delta):
                            raise RuntimeError(f"candidate reward mismatch for {candidate_id}")
                        expected_outcome = "improved" if expected_delta > 0 else ("damaged" if expected_delta < 0 else "no_gain")
                        if str(candidate.get("outcome") or "") != expected_outcome:
                            raise RuntimeError(f"candidate outcome mismatch for {candidate_id}")
                else:
                    skipped_unexecuted_candidates += 1
                    if str(candidate.get("outcome") or "") == "invalid_editor_output":
                        invalid_editor_output_candidates += 1
                    if not _float_equal(candidate.get("reward"), -1.0):
                        raise RuntimeError(f"unexecuted candidate has non-fail-closed reward for {candidate_id}")
                    if bool(candidate.get("updated_valid", False)):
                        raise RuntimeError(f"unexecuted candidate marked updated_valid for {candidate_id}")

                updated = pair.get("updated") if isinstance(pair.get("updated"), dict) else {}
                editor_block = pair.get("editor") if isinstance(pair.get("editor"), dict) else {}
                trajectory = candidate.get("updated_trajectory")
                if (
                    updated.get("mas") != candidate.get("updated_mas")
                    or updated.get("trajectory") != trajectory
                    or not _float_equal(updated.get("score"), candidate.get("upd_score"))
                    or updated.get("judge") != _trajectory_judge(trajectory)
                ):
                    raise RuntimeError(f"v5 compatibility updated is not the unique Refiner candidate for {pair_id}")
                if (
                    editor_block.get("prompt") != candidate.get("prompt")
                    or editor_block.get("output") != candidate.get("output")
                    or editor_block.get("patch") != candidate.get("patch")
                    or not _float_equal(editor_block.get("reward"), candidate.get("reward"))
                    or editor_block.get("outcome") != candidate.get("outcome")
                    or bool(editor_block.get("trainable", False)) != bool(candidate.get("trainable", False))
                ):
                    raise RuntimeError(f"v5 compatibility editor is not the unique Refiner candidate for {pair_id}")

                comparison = pair.get("comparison") if isinstance(pair.get("comparison"), dict) else {}
                if comparison and (
                    not _float_equal(comparison.get("score_delta"), candidate.get("score_delta"))
                    or not _float_equal(comparison.get("editor_reward"), candidate.get("reward"))
                    or comparison.get("editor_outcome") != candidate.get("outcome")
                ):
                    raise RuntimeError(f"one-to-one comparison is not aligned with unique candidate for {pair_id}")

                if is_v6_pair:
                    if "selection" in pair or "winner" in pair:
                        raise RuntimeError(f"v6 pair must not contain selection/winner for {pair_id}")
                    translation = pair.get("patch_translation")
                    if not isinstance(translation, dict) or translation.get("version") != "mas_patch_intent_compiler_v1":
                        raise RuntimeError(f"v6 patch translation audit missing for {pair_id}")
                    lambda_adv = float(os.getenv("MAS_LOOP_ORCH_ADV_PENALTY", "0.3"))
                    expected_orchestra_reward = source_score - lambda_adv * max(float(candidate.get("score_delta", 0.0) or 0.0), 0.0)
                    if comparison and not _float_equal(comparison.get("orchestra_reward"), expected_orchestra_reward):
                        raise RuntimeError(f"v6 Orchestra reward mismatch for {pair_id}")
                else:
                    explicit_accept = _is_accept_candidate(candidate)
                    expected_selection = select_pair_winner(
                        source_score=source_score,
                        updated_score=float(candidate.get("upd_score", source_score) or 0.0),
                        source_mas=source.get("mas"),
                        updated_mas=candidate.get("updated_mas"),
                        source_trajectory=source.get("trajectory"),
                        updated_trajectory=candidate.get("updated_trajectory"),
                        source_valid=True,
                        updated_valid=bool(candidate.get("updated_valid", False)),
                        explicit_accept=explicit_accept,
                    ).to_dict()
                    selection = pair.get("selection") if isinstance(pair.get("selection"), dict) else {}
                    for key, expected_value in expected_selection.items():
                        actual_value = selection.get(key)
                        if isinstance(expected_value, float):
                            if not _float_equal(actual_value, expected_value):
                                raise RuntimeError(f"v5 selection {key} mismatch for {pair_id}")
                        elif actual_value != expected_value:
                            raise RuntimeError(f"v5 selection {key} mismatch for {pair_id}")
                    if selection.get("selector_version") != SELECTOR_VERSION:
                        raise RuntimeError(f"v5 selector version mismatch for {pair_id}")
                    winner = pair.get("winner") if isinstance(pair.get("winner"), dict) else {}
                    selected_origin = str(selection.get("selected_origin") or "")
                    expected_state = source if selected_origin == "source" else {
                        "mas": candidate.get("updated_mas"), "trajectory": candidate.get("updated_trajectory"), "score": candidate.get("upd_score"),
                    }
                    if (winner.get("origin") != selected_origin or winner.get("mas") != expected_state.get("mas")
                        or winner.get("trajectory") != expected_state.get("trajectory")
                        or not _float_equal(winner.get("score"), expected_state.get("score"))):
                        raise RuntimeError(f"v5 winner state mismatch for {pair_id}")
                    alpha = float(os.getenv("MAS_LOOP_ORCH_WINNER_ALPHA", os.getenv("MAS_LOOP_ORCH_ADV_ALPHA", "0.3")))
                    expected_orchestra_reward = source_score + alpha * max(float(candidate.get("score_delta", 0.0) or 0.0), 0.0)
                    if comparison and not _float_equal(comparison.get("orchestra_reward"), expected_orchestra_reward):
                        raise RuntimeError(f"v5 Orchestra reward mismatch for {pair_id}")
                continue

            observed_indices: list[int] = []
            selectable: list[dict[str, Any]] = []
            for position, candidate in enumerate(editor_candidates):
                if not isinstance(candidate, dict):
                    raise RuntimeError(f"non-object editor candidate at {shard}:{line_no}:{position}")
                total_editor_candidates += 1
                candidate_index = int(candidate.get("candidate_index", -1))
                observed_indices.append(candidate_index)
                candidate_id = str(candidate.get("candidate_id") or "")
                expected_id = f"{pair_id}:editor-{candidate_index}"
                if candidate_id != expected_id:
                    raise RuntimeError(
                        f"candidate_id mismatch at {shard}:{line_no}:{position}: "
                        f"expected={expected_id!r} actual={candidate_id!r}"
                    )
                if candidate_id in candidate_ids:
                    raise RuntimeError(f"duplicate editor candidate_id: {candidate_id}")
                candidate_ids.add(candidate_id)
                candidate_label = f"updated_{candidate_index}"
                candidate_key = (candidate_id, candidate_label)

                if not _float_equal(candidate.get("src_score"), source_score):
                    raise RuntimeError(f"candidate src_score mismatch for {candidate_id}")
                if _is_accept_candidate(candidate):
                    accepted_updated_judge_reuse += 1
                    standalone_allowed.add(candidate_key)
                    if (
                        candidate.get("updated_mas") != source.get("mas")
                        or candidate.get("updated_trajectory") != source.get("trajectory")
                        or not _float_equal(candidate.get("upd_score"), source_score)
                        or not _float_equal(candidate.get("score_delta"), 0.0)
                        or not _float_equal(candidate.get("reward"), 0.0)
                    ):
                        raise RuntimeError(f"accept/no-op changed source evidence for {candidate_id}")
                elif _is_executed_candidate(candidate):
                    executed_updated_candidates += 1
                    standalone_expected.add(candidate_key)
                    standalone_allowed.add(candidate_key)
                    trajectory = candidate.get("updated_trajectory")
                    judge = _trajectory_judge(trajectory)
                    if not _valid_judge(judge):
                        missing_judge.append(f"{candidate_id}:{candidate_label}")
                    else:
                        updated_score = float(judge["score"])
                        expected_delta = updated_score - source_score
                        if not _float_equal(candidate.get("upd_score"), updated_score):
                            raise RuntimeError(f"candidate upd_score mismatch for {candidate_id}")
                        if not _float_equal(candidate.get("score_delta"), expected_delta):
                            raise RuntimeError(f"candidate score_delta mismatch for {candidate_id}")
                        if not _float_equal(candidate.get("reward"), expected_delta):
                            raise RuntimeError(f"candidate reward mismatch for {candidate_id}")
                        expected_outcome = "improved" if expected_delta > 0 else ("damaged" if expected_delta < 0 else "no_gain")
                        if str(candidate.get("outcome") or "") != expected_outcome:
                            raise RuntimeError(f"candidate outcome mismatch for {candidate_id}")
                else:
                    skipped_unexecuted_candidates += 1
                    outcome = str(candidate.get("outcome") or "")
                    if outcome == "invalid_editor_output":
                        invalid_editor_output_candidates += 1
                        retry = candidate.get("retry") if isinstance(candidate.get("retry"), dict) else {}
                        raw_responses = retry.get("raw_responses") if isinstance(retry.get("raw_responses"), list) else []
                        retry_errors = retry.get("errors") if isinstance(retry.get("errors"), list) else []
                        if not _float_equal(candidate.get("reward"), -1.0):
                            raise RuntimeError(f"malformed candidate reward is not fail-closed for {candidate_id}")
                        if not str(candidate.get("output") or "").strip():
                            raise RuntimeError(f"malformed candidate lacks raw output for {candidate_id}")
                        if not (str(candidate.get("error") or "").strip() or raw_responses or retry_errors):
                            raise RuntimeError(f"malformed candidate lacks retry/error evidence for {candidate_id}")
                    if _valid_judge(_trajectory_judge(candidate.get("updated_trajectory"))):
                        # Source reuse is expected in the compatibility fields,
                        # but it must not be represented as an independent Judge
                        # key or rewarded execution.
                        if not _float_equal(candidate.get("reward"), -1.0):
                            raise RuntimeError(f"unexecuted candidate has non-fail-closed reward for {candidate_id}")
                if candidate.get("trainable") and str(candidate.get("outcome") or "") != "error":
                    selectable.append(candidate)

            if observed_indices != list(range(len(editor_candidates))):
                raise RuntimeError(
                    f"non-contiguous editor candidate indices at {shard}:{line_no}: {observed_indices}"
                )

            primary = (
                max(selectable, key=lambda value: (float(value.get("score_delta", 0.0)), -int(value.get("candidate_index", 0))))
                if selectable
                else editor_candidates[0]
            )
            updated = pair.get("updated") if isinstance(pair.get("updated"), dict) else {}
            primary_trajectory = primary.get("updated_trajectory")
            if (
                updated.get("mas") != primary.get("updated_mas")
                or updated.get("trajectory") != primary_trajectory
                or not _float_equal(updated.get("score"), primary.get("upd_score"))
                or updated.get("judge") != _trajectory_judge(primary_trajectory)
            ):
                raise RuntimeError(f"backward-compatible primary updated is not strongest candidate for {pair_id}")
            if (
                editor.get("prompt") != primary.get("prompt")
                or editor.get("output") != primary.get("output")
                or editor.get("patch") != primary.get("patch")
                or not _float_equal(editor.get("reward"), primary.get("reward"))
                or editor.get("outcome") != primary.get("outcome")
                or bool(editor.get("trainable", False)) != bool(primary.get("trainable", False))
            ):
                raise RuntimeError(f"backward-compatible primary editor is not strongest candidate for {pair_id}")
            comparison = pair.get("comparison") if isinstance(pair.get("comparison"), dict) else {}
            if comparison and (
                not _float_equal(comparison.get("score_delta"), primary.get("score_delta"))
                or not _float_equal(comparison.get("editor_reward"), primary.get("reward"))
                or comparison.get("editor_outcome") != primary.get("outcome")
            ):
                raise RuntimeError(f"comparison is not aligned with strongest candidate for {pair_id}")
            continue

        # Legacy one-source/one-updated contract.
        updated = pair.get("updated") if isinstance(pair.get("updated"), dict) else {}
        updated_judge = updated.get("judge")
        if not _valid_judge(updated_judge):
            missing_judge.append(f"{pair_id}:updated")
        elif float(updated_judge.get("score")) == -1.0:
            judge_negative_one_candidates += 1
        updated_key = (pair_id, "updated")
        standalone_allowed.add(updated_key)
        if _legacy_accept(pair):
            if (
                source.get("mas") != updated.get("mas")
                or source.get("trajectory") != updated.get("trajectory")
                or source.get("score") != updated.get("score")
                or source.get("judge") != updated.get("judge")
            ):
                raise RuntimeError(f"accept/no-op changed source evidence for {pair_id}")
            accepted_updated_judge_reuse += 1
        else:
            standalone_expected.add(updated_key)
            executed_updated_candidates += 1

    if not rows:
        raise RuntimeError(f"no canonical Orchestra pairs under {canonical_dir}")
    expected_v5_group_size = int(os.getenv("MAS_LOOP_EDITOR_GROUP_SIZE", os.getenv("MAS_LOOP_ROLLOUT_N", "32")))
    if expected_v5_group_size <= 0:
        raise RuntimeError(f"invalid expected v5 group size: {expected_v5_group_size}")
    for (group_uid, optimizer_step), indices in sorted(v5_task_groups.items()):
        observed = sorted(indices)
        expected = list(range(expected_v5_group_size))
        if observed != expected:
            raise RuntimeError(
                f"incomplete/non-contiguous v5 task group {group_uid!r} at optimizer_step={optimizer_step}: "
                f"expected={expected} actual={observed}"
            )
    if missing_judge:
        raise RuntimeError(
            f"{len(missing_judge)} required candidates lack valid Qwen judge metadata; first={missing_judge[:20]}"
        )

    judge_dir = rollout_dir / "judge_results"
    judged: set[tuple[str, str]] = set()
    invalid_judged: set[tuple[str, str]] = set()
    judge_rows = 0
    valid_judge_rows = 0
    invalid_judge_rows = 0
    for shard, line_no, row in _iter_jsonl_strict(judge_dir):
        judge_rows += 1
        pair_id = str(row.get("pair_id") or "")
        candidate = str(row.get("candidate") or "")
        if not pair_id:
            raise RuntimeError(f"missing persisted judge pair_id at {shard}:{line_no}")
        if not candidate:
            raise RuntimeError(f"missing persisted judge candidate at {shard}:{line_no}")
        key = (pair_id, candidate)
        if key not in standalone_allowed:
            raise RuntimeError(f"unexpected judge candidate at {shard}:{line_no}: {key!r}")
        if _valid_judge(row.get("judge")):
            valid_judge_rows += 1
            judged.add(key)
        else:
            invalid_judge_rows += 1
            invalid_judged.add(key)

    missing_results = sorted(standalone_expected - judged)
    if missing_results:
        raise RuntimeError(
            f"{len(missing_results)} canonical candidates lack persisted valid judge_results rows; first={missing_results[:20]}"
        )
    invalid_without_replacement = sorted((invalid_judged & standalone_expected) - judged)
    if invalid_without_replacement:
        raise RuntimeError(
            f"{len(invalid_without_replacement)} invalid persisted judge_results keys lack valid replacements; "
            f"first={invalid_without_replacement[:20]}"
        )

    return {
        "canonical_pair_rows": rows,
        "unique_pair_ids": len(pair_ids),
        "editor_candidate_rows": total_editor_candidates,
        "v5_pair_rows": v5_pair_rows,
        "v5_task_groups": len(v5_task_groups),
        "v5_expected_group_size": int(os.getenv("MAS_LOOP_EDITOR_GROUP_SIZE", os.getenv("MAS_LOOP_ROLLOUT_N", "32"))),
        "executed_updated_candidates": executed_updated_candidates,
        "invalid_editor_output_candidates": invalid_editor_output_candidates,
        "skipped_unexecuted_candidates": skipped_unexecuted_candidates,
        "judge_result_rows": judge_rows,
        "valid_judge_result_rows": valid_judge_rows,
        "historical_invalid_judge_result_rows": invalid_judge_rows,
        "historical_invalid_judge_keys": len(invalid_judged),
        "superseded_invalid_judge_keys": len(invalid_judged & judged),
        "judged_pair_candidates": len(judged),
        "pair_error_rows": pair_error_rows,
        "pair_error_entries": pair_error_entries,
        "editor_error_rows": editor_error_rows,
        "editor_retry_error_entries": editor_retry_error_entries,
        "judge_negative_one_candidates": judge_negative_one_candidates,
        "accepted_updated_judge_reuse": accepted_updated_judge_reuse,
        "required_standalone_judge_keys": len(standalone_expected),
        "allowed_standalone_judge_keys": len(standalone_allowed),
    }

def _validate_editor(phase_dir: Path) -> dict[str, Any]:
    rollout_dir = phase_dir / "rollouts"
    canonical_dir = rollout_dir / "canonical_rollouts"
    # Editor replay does not generate a fresh model response: it records the
    # exact Orchestra-collected editor sample that was consumed for the PPO
    # update.  Those authoritative records are persisted under
    # ``editor_replay_consumed`` by MASLoopEditorReplayAgentLoop.  Prefer the
    # historical canonical path when present, but accept the replay path so
    # validation matches the writer's on-disk contract and remains fail-closed
    # when neither directory contains JSONL shards.
    if not _jsonl_files(canonical_dir):
        canonical_dir = rollout_dir / "editor_replay_consumed"
    rows = 0
    pair_ids: set[str] = set()
    for shard, line_no, row in _iter_jsonl_strict(canonical_dir):
        rows += 1
        pair_id = str(row.get("pair_id") or (row.get("structured_pair") or {}).get("pair_id") or "")
        if not pair_id:
            raise RuntimeError(f"missing Editor pair_id at {shard}:{line_no}")
        pair_ids.add(pair_id)
    if not rows:
        raise RuntimeError(f"no canonical Editor rollouts under {canonical_dir}")
    return {"canonical_rollout_rows": rows, "unique_pair_ids": len(pair_ids)}


def validate_phase(phase_dir: Path, phase: str) -> dict[str, Any]:
    phase_dir = phase_dir.resolve()
    if phase not in {"orchestra", "editor"}:
        raise ValueError(f"unsupported phase: {phase}")
    result: dict[str, Any] = {
        "schema": "mas_loop_bcp_artifact_validation_v1",
        "phase": phase,
        "phase_dir": str(phase_dir),
        "metrics": _validate_metrics(
            phase_dir / "metrics" / "training_metrics.jsonl", phase=phase
        ),
    }
    result["rollouts"] = _validate_orchestra(phase_dir) if phase == "orchestra" else _validate_editor(phase_dir)
    result["valid"] = True
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("orchestra", "editor"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_phase(args.phase_dir, args.phase)
    output = args.output or args.phase_dir / "artifact_validation.json"
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
