"""Utility functions for loading and analyzing RM training datasets."""

from __future__ import annotations
import json
from pathlib import Path

from .schema import TrainingPair, DatasetStatistics


def load_pairs(jsonl_path: Path) -> list[TrainingPair]:
    """Load RM training pairs from a JSONL file.

    Supports raw format (full metadata) and RM format ({chosen_text, rejected_text}).
    """
    pairs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "pair_id" in data:
                    pairs.append(TrainingPair.from_dict(data))
                elif "chosen_text" in data and "rejected_text" in data:
                    meta = data.get("metadata", {})
                    pairs.append(TrainingPair(
                        pair_id=meta.get("pair_id", f"pair_{line_num:04d}"),
                        task_id=meta.get("task_id", ""),
                        task_type=meta.get("task_type", ""),
                        chosen_score=meta.get("chosen_score", 0.0),
                        rejected_score=meta.get("rejected_score", 0.0),
                        chosen_source=meta.get("chosen_source", ""),
                        rejected_source=meta.get("rejected_source", ""),
                        score_delta=meta.get("score_delta", 0.0),
                    ))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: skipping invalid line {line_num}: {e}")
    return pairs


def compute_statistics(pairs: list[TrainingPair]) -> DatasetStatistics:
    from .constructor import _compute_statistics
    return _compute_statistics(pairs)


def print_statistics(stats: DatasetStatistics) -> None:
    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"  Total Pairs: {stats.total_pairs}")
    print(f"  Average Score Delta: {stats.average_score_delta:.4f}")
    print(f"  Average Chosen Agents: {stats.average_chosen_agents:.2f}")
    print(f"  Average Rejected Agents: {stats.average_rejected_agents:.2f}")
    print(f"  Orchestra Chosen: {stats.orchestra_chosen_count}")
    print(f"  Editor Chosen: {stats.editor_chosen_count}")
    print()
    print("  By Task Type:")
    for tt, count in sorted(stats.by_task_type.items()):
        print(f"    {tt}: {count}")
    print()
    print("  By Score Delta Bucket:")
    for bucket, count in sorted(stats.by_score_delta_bucket.items()):
        print(f"    {bucket}: {count}")
    print()
    print("  Editor Operations:")
    for op, count in sorted(stats.total_editor_operations.items()):
        print(f"    {op}: {count}")
    print("=" * 50)


def split_train_val(
    pairs: list[TrainingPair],
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_task_type: bool = True,
) -> tuple[list[TrainingPair], list[TrainingPair]]:
    import random
    random.seed(seed)
    if not stratify_by_task_type:
        pairs_copy = list(pairs)
        random.shuffle(pairs_copy)
        val_count = max(1, int(len(pairs_copy) * val_ratio))
        return pairs_copy[val_count:], pairs_copy[:val_count]
    by_type = {}
    for pair in pairs:
        by_type.setdefault(pair.task_type, []).append(pair)
    train_pairs, val_pairs = [], []
    for tt, type_pairs in by_type.items():
        random.shuffle(type_pairs)
        val_count = max(1, int(len(type_pairs) * val_ratio))
        train_pairs.extend(type_pairs[val_count:])
        val_pairs.extend(type_pairs[:val_count])
    return train_pairs, val_pairs