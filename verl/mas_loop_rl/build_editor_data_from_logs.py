from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from mas_loop_rl.prepare_data import _sanitize_rows_for_parquet, _write_parquet_via_local_staging, row_for
from mas_loop_rl.sharded_jsonl import iter_jsonl, resolve_jsonl_inputs



def _valid_editor_pair(rec: dict[str, Any]) -> tuple[bool, str]:
    if rec.get("role") != "orchestra":
        return False, "not_orchestra_record"
    if not rec.get("task"):
        return False, "missing_task"
    if not isinstance(rec.get("src_mas"), dict) or not rec.get("src_mas"):
        return False, "missing_src_mas"
    if not isinstance(rec.get("src_trajectory"), dict) or not rec.get("src_trajectory"):
        return False, "missing_src_trajectory"
    if not isinstance(rec.get("peer_prompt"), list) or not rec.get("peer_prompt"):
        return False, "missing_editor_prompt"
    if not str(rec.get("peer_output") or "").strip():
        return False, "missing_editor_output"
    if rec.get("editor_score") is None:
        return False, "missing_editor_reward"
    if not rec.get("grpo_group_uid"):
        return False, "missing_task_group_uid"
    return True, ""


def _is_accept_record(rec: dict[str, Any]) -> bool:
    """Identify an explicit accept action from canonical fields or model JSON."""
    outcome = str(rec.get("editor_outcome") or "")
    if outcome.startswith("accepted_"):
        return True
    action = rec.get("edit_operation")
    if not isinstance(action, (dict, str)):
        action = rec.get("peer_output")
    try:
        from mas_loop_rl.schema import is_accept_patch, normalize_editor_action

        return is_accept_patch(normalize_editor_action(action))
    except (TypeError, ValueError):
        return False


def _candidate_records(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten real same-source Editor samples while retaining legacy logs."""
    candidates = rec.get("editor_candidates")
    if not isinstance(candidates, list):
        return [rec]
    flattened: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        item = dict(rec)
        item.update(
            {
                "pair_id": str(candidate.get("candidate_id") or f"{rec.get('pair_id')}:editor-{index}"),
                "source_pair_id": str(rec.get("pair_id") or rec.get("id")),
                "peer_prompt": candidate.get("prompt"),
                "peer_output": candidate.get("output"),
                "peer_retry": candidate.get("retry", {}),
                "edit_operation": candidate.get("patch"),
                "editor_score": candidate.get("reward"),
                "editor_outcome": candidate.get("outcome"),
                "updated_mas": candidate.get("updated_mas"),
                "updated_trajectory": candidate.get("updated_trajectory"),
                "upd_score": candidate.get("upd_score"),
                "score_delta": candidate.get("score_delta"),
                "editor_trainable": candidate.get("trainable", False),
                "editor_group_uid": str(candidate.get("editor_group_uid") or f"editor-source:{rec.get('pair_id')}"),
                "editor_candidate_index": int(candidate.get("candidate_index", index)),
                "editor_candidate_count": len(candidates),
                "candidate_error": candidate.get("error", ""),
            }
        )
        flattened.append(item)
    return flattened


def build_rows(
    log_files: list[Path],
    group_size: int = 8,
    max_groups: int | None = None,
    accept_regret_weight: float = 0.0,
    optimizer_step: int | None = None,
    min_record_time: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build task-conditioned Editor GRPO groups from one-to-one pair logs.

    In the current one-to-one rollout, one task has ``group_size`` independently
    sampled source MASes and exactly one Refiner action per source. The Refiner
    group therefore compares 32 diagnosis/repair decisions under one task while
    each reward remains the local ``updated_score - source_score``. Legacy
    same-source multi-candidate logs remain readable for artifact compatibility.

    Group semantics are inferred from source identity, not a schema-version
    suffix. A v6 one-candidate record is flattened with ``editor_candidate_index``;
    using that field alone would incorrectly label 32 distinct sources as 32
    hypotheses for one source and could activate cross-source accept regret.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()

    for rec in iter_jsonl(log_files):
        if optimizer_step is not None and int(rec.get("optimizer_step", 0) or 0) != optimizer_step:
            continue
        if min_record_time is not None:
            # Security: Fail closed on missing or malformed provenance timestamps
            # whenever an attempt boundary is required; stale retry data must
            # never be admitted merely because it has the same optimizer step.
            try:
                record_time = float(rec.get("time"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(record_time) or record_time < min_record_time:
                continue
        for candidate_rec in _candidate_records(rec):
            valid, reason = _valid_editor_pair(candidate_rec)
            if not valid:
                rejected.append(
                    {
                        "reason": reason,
                        "record_id": rec.get("id"),
                        "pair_id": candidate_rec.get("pair_id"),
                        "candidate_error": candidate_rec.get("candidate_error", ""),
                    }
                )
                continue
            if candidate_rec.get("editor_trainable") is False:
                rejected.append(
                    {
                        "reason": "editor_candidate_not_trainable",
                        "record_id": rec.get("id"),
                        "pair_id": candidate_rec.get("pair_id"),
                        "candidate_error": candidate_rec.get("candidate_error", ""),
                    }
                )
                continue
            pair_id = str(candidate_rec.get("pair_id") or candidate_rec.get("id"))
            if pair_id in seen_pair_ids:
                continue
            seen_pair_ids.add(pair_id)
            group_uid = str(candidate_rec.get("editor_group_uid") or candidate_rec.get("grpo_group_uid"))
            grouped[group_uid].append(candidate_rec)

    rows: list[dict[str, Any]] = []
    complete_groups = 0
    for raw_group_uid, records in grouped.items():
        source_identities = {
            str(record.get("source_pair_id") or record.get("pair_id") or record.get("id") or "")
            for record in records
        }
        has_candidate_indices = all("editor_candidate_index" in record for record in records)
        is_same_source_group = len(source_identities) == 1 and has_candidate_indices
        is_pairwise_task_group = len(source_identities) == len(records) and not is_same_source_group
        index_key = "rollout_sample_index" if is_pairwise_task_group else ("editor_candidate_index" if is_same_source_group else "rollout_sample_index")
        records.sort(key=lambda record: (int(record.get(index_key, 0)), float(record.get("time", 0.0))))
        if len(records) != group_size:
            rejected.append(
                {
                    "reason": "incomplete_or_oversized_group",
                    "editor_group_uid": raw_group_uid,
                    "expected_group_size": group_size,
                    "actual_group_size": len(records),
                    "pair_ids": [record.get("pair_id") for record in records],
                }
            )
            continue
        sample_indices = [int(record.get(index_key, -1)) for record in records]
        if sample_indices != list(range(group_size)):
            rejected.append(
                {
                    "reason": "non_contiguous_candidate_indices" if is_same_source_group else "non_contiguous_rollout_indices",
                    "editor_group_uid": raw_group_uid,
                    "expected": list(range(group_size)),
                    "actual": sample_indices,
                }
            )
            continue

        accept_flags = [_is_accept_record(record) for record in records]
        best_positive_patch_reward = max(
            [0.0]
            + [
                float(record.get("editor_score", 0.0))
                for record, accepted in zip(records, accept_flags)
                if not accepted
            ]
        )
        editor_group_uid = f"editor-replay:{raw_group_uid}"
        for record, accepted in zip(records, accept_flags):
            idx = len(rows)
            raw_editor_reward = float(record["editor_score"])
            accept_regret = (
                accept_regret_weight * best_positive_patch_reward
                if accepted and is_same_source_group and accept_regret_weight > 0
                else 0.0
            )
            replay_editor_reward = -accept_regret if accepted and accept_regret else raw_editor_reward
            action = record.get("edit_operation") or {}
            diagnosis = action.get("diagnosis", {}) if isinstance(action, dict) else {}
            source_task_group_uid = str(record.get("grpo_group_uid") or "")
            row = row_for(
                "editor",
                task=str(record.get("task") or ""),
                answer=str(record.get("expected") or ""),
                idx=idx,
                src_mas=record["src_mas"],
                src_traj=record["src_trajectory"],
                constraints=record.get("constraints") or {"_": ""},
                extra_meta={
                    "pair_id": str(record.get("pair_id") or record.get("id")),
                    "source_pair_id": str(record.get("source_pair_id") or record.get("pair_id") or record.get("id")),
                    "source_record_id": record.get("id"),
                    "source_loop_id": record.get("loop_id"),
                    "source_grpo_group_uid": source_task_group_uid,
                    "editor_group_uid": editor_group_uid,
                    "rollout_sample_index": int(record.get(index_key, 0)),
                    "editor_candidate_index": int(record.get("editor_candidate_index", record.get("rollout_sample_index", 0))),
                    "editor_group_semantics": (
                        "task_conditioned_source_pairs"
                        if is_pairwise_task_group
                        else ("same_source_repair_hypotheses" if is_same_source_group else "legacy_task_level_sources")
                    ),
                    "replay_editor_prompt": record["peer_prompt"],
                    "replay_editor_output": str(record["peer_output"]),
                    "raw_editor_reward": raw_editor_reward,
                    "accept_regret": accept_regret,
                    "group_best_positive_patch_reward": best_positive_patch_reward,
                    "editor_action": "accept" if accepted else "patch",
                    "diagnosis": diagnosis,
                    "replay_editor_reward": replay_editor_reward,
                    "src_score": float(record.get("src_score", 0.0)),
                    "upd_score": float(record.get("upd_score", 0.0)),
                    "score_delta": float(record.get("score_delta", 0.0)),
                    "editor_outcome": str(record.get("editor_outcome", "")),
                    "same_source_candidate": is_same_source_group,
                    "pairwise_task_group": is_pairwise_task_group,
                },
            )
            rows.append(row)
        complete_groups += 1
        if max_groups and complete_groups >= max_groups:
            break
    return rows, rejected

def _pad_complete_groups(rows: list[dict[str, Any]], multiple: int, group_size: int) -> list[dict[str, Any]]:
    if multiple <= 0 or not rows:
        return rows
    if multiple % group_size != 0:
        raise ValueError(f"pad multiple {multiple} must be divisible by editor group size {group_size}")
    rem = len(rows) % multiple
    if rem == 0:
        return rows
    need = multiple - rem
    if need % group_size:
        raise AssertionError("padding must preserve complete Editor groups")
    original = list(rows)
    groups = [original[i : i + group_size] for i in range(0, len(original), group_size)]
    existing_uids = {
        str(group[0]["extra_info"]["editor_group_uid"])
        for group in groups
    }
    next_padding_index = 0
    for dup_group_index in range(need // group_size):
        source_group = groups[dup_group_index % len(groups)]
        source_uid = source_group[0]["extra_info"]["editor_group_uid"]
        duplicate_uid = f"{source_uid}:padding-{next_padding_index}"
        while duplicate_uid in existing_uids:
            next_padding_index += 1
            duplicate_uid = f"{source_uid}:padding-{next_padding_index}"
        existing_uids.add(duplicate_uid)
        next_padding_index += 1
        for item in source_group:
            dup = dict(item)
            extra = dict(dup.get("extra_info") or {})
            extra["is_padding_duplicate"] = True
            extra["padding_duplicate_of_group"] = source_uid
            extra["editor_group_uid"] = duplicate_uid
            extra["index"] = len(rows)
            dup["extra_info"] = extra
            rows.append(dup)
    return rows


def _pad_complete_groups_to_expected_rows(
    rows: list[dict[str, Any]], expected_rows: int, group_size: int
) -> list[dict[str, Any]]:
    """Pad only complete Editor groups to an exact, explicitly requested row count."""
    if expected_rows <= 0:
        raise ValueError("expected rows must be positive when exact replay padding is enabled")
    if group_size <= 0:
        raise ValueError("editor group size must be positive")
    if not rows:
        raise ValueError("cannot pad an empty Editor replay")
    if len(rows) % group_size != 0:
        raise ValueError(
            f"selected replay rows {len(rows)} must contain only complete groups of size {group_size}"
        )
    if expected_rows % group_size != 0:
        raise ValueError(
            f"expected rows {expected_rows} must be divisible by editor group size {group_size}"
        )
    if len(rows) > expected_rows:
        raise ValueError(
            f"selected replay rows {len(rows)} exceed expected rows {expected_rows}; refusing to truncate"
        )
    padded = _pad_complete_groups(rows, expected_rows, group_size)
    if len(padded) != expected_rows:
        raise AssertionError(
            f"exact Editor replay padding failed: expected_rows={expected_rows} actual_rows={len(padded)}"
        )
    return padded


def main() -> None:
    ap = argparse.ArgumentParser(description="Build task-conditioned one-to-one Refiner replay GRPO parquet from Orchestra logs.")
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--optimizer-step", type=int, default=0, help="Keep only canonical pairs from this optimizer step; 0 disables filtering.")
    ap.add_argument("--min-record-time", type=float, default=0.0, help="Keep only records at or after this Unix timestamp; 0 disables filtering.")
    ap.add_argument("--expected-groups", type=int, default=0, help="Fail closed unless this many complete source groups are selected.")
    ap.add_argument("--expected-rows", type=int, default=0, help="Fail closed unless this many one-to-one replay rows are selected after optional exact-target padding.")
    ap.add_argument(
        "--pad-selected-to-expected-rows",
        action="store_true",
        help=(
            "Explicitly duplicate complete Editor groups until --expected-rows is reached. "
            "Never truncates, splits, or admits an incomplete group; disabled by default."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad-train-to-multiple", type=int, default=0)
    ap.add_argument(
        "--accept-regret-weight",
        type=float,
        default=float(os.getenv("MAS_LOOP_ACCEPT_REGRET_WEIGHT", "0.0")),
    )
    args = ap.parse_args()
    if args.accept_regret_weight < 0:
        raise SystemExit("--accept-regret-weight must be non-negative")
    if args.min_record_time < 0 or not math.isfinite(args.min_record_time):
        raise SystemExit("--min-record-time must be a finite non-negative Unix timestamp")

    input_paths = resolve_jsonl_inputs(args.logs)
    if not input_paths:
        raise SystemExit(f"no JSONL inputs resolved from: {args.logs}")
    rows, rejected = build_rows(
        input_paths,
        group_size=args.group_size,
        max_groups=args.max_groups or None,
        accept_regret_weight=args.accept_regret_weight,
        optimizer_step=args.optimizer_step or None,
        min_record_time=args.min_record_time or None,
    )
    if not rows:
        raise SystemExit("no complete Editor pair groups found in logs")

    real_complete_rows = len(rows)
    real_complete_groups = real_complete_rows // args.group_size
    if args.pad_selected_to_expected_rows:
        if not args.expected_rows:
            raise SystemExit("--pad-selected-to-expected-rows requires --expected-rows")
        try:
            rows = _pad_complete_groups_to_expected_rows(rows, args.expected_rows, args.group_size)
        except (ValueError, AssertionError) as exc:
            raise SystemExit(str(exc)) from exc

    selected_groups = len(rows) // args.group_size
    if args.expected_groups and selected_groups != args.expected_groups:
        raise SystemExit(
            f"step-local replay group mismatch: optimizer_step={args.optimizer_step} "
            f"expected_groups={args.expected_groups} actual_groups={selected_groups}"
        )
    if args.expected_rows and len(rows) != args.expected_rows:
        raise SystemExit(
            f"step-local replay row mismatch: optimizer_step={args.optimizer_step} "
            f"expected_rows={args.expected_rows} actual_rows={len(rows)}"
        )

    # Shuffle whole task groups, never individual members.
    groups = [rows[i : i + args.group_size] for i in range(0, len(rows), args.group_size)]
    random.Random(args.seed).shuffle(groups)
    nval_groups = int(len(groups) * args.val_ratio) if args.val_ratio > 0 else 0
    if args.val_ratio > 0 and len(groups) > 1:
        nval_groups = max(1, nval_groups)
    val_groups = groups[:nval_groups]
    train_groups = groups[nval_groups:] or groups
    train = [row for group in train_groups for row in group]
    val = [row for group in val_groups for row in group]
    original_train_len = len(train)
    train = _pad_complete_groups(train, args.pad_train_to_multiple, args.group_size)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_parquet_via_local_staging(
        pd.DataFrame(_sanitize_rows_for_parquet(train)), out / "editor_train.parquet"
    )
    _write_parquet_via_local_staging(
        pd.DataFrame(_sanitize_rows_for_parquet(val or train[: args.group_size])),
        out / "editor_val.parquet",
    )
    rejected_path = out / "editor_replay_rejected.jsonl"
    with rejected_path.open("w", encoding="utf-8") as f:
        for rec in rejected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    explicit_rejected_records = sum(
        1 for rec in rejected if rec.get("reason") != "incomplete_or_oversized_group"
    )
    dropped_incomplete_group_rows = sum(
        int(rec.get("actual_group_size", 0) or 0)
        for rec in rejected
        if rec.get("reason") == "incomplete_or_oversized_group"
    )
    rejected_source_rows = explicit_rejected_records
    if args.expected_rows:
        rejected_source_rows = max(
            rejected_source_rows,
            args.expected_rows - real_complete_rows - dropped_incomplete_group_rows,
        )
    replay_audit = {
        "optimizer_step": args.optimizer_step,
        "min_record_time": args.min_record_time,
        "group_size": args.group_size,
        "real_complete_groups": real_complete_groups,
        "real_complete_rows": real_complete_rows,
        "padded_groups": selected_groups - real_complete_groups,
        "padded_rows": len(rows) - real_complete_rows,
        "rejected_source_rows": rejected_source_rows,
        "explicit_rejected_records": explicit_rejected_records,
        "dropped_incomplete_group_rows": dropped_incomplete_group_rows,
        "final_groups": selected_groups,
        "final_rows": len(rows),
        "exact_target_padding_enabled": bool(args.pad_selected_to_expected_rows),
    }
    (out / "editor_replay_audit.json").write_text(
        json.dumps(replay_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"editor-diagnostic-replay: train_rows={len(train)} train_groups={len(train)//args.group_size} "
        f"val_rows={len(val)} original_train_rows={original_train_len} rejected={len(rejected)} "
        f"real_complete_groups={real_complete_groups} padded_groups={selected_groups-real_complete_groups} "
        f"input_shards={len(input_paths)} -> {out}"
    )


if __name__ == "__main__":
    main()
