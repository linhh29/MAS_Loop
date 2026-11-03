"""Bradley-Terry RM training data constructor from MAS-Loop trajectories.

Scans outputs_formalized/{aime,drb}/tasks/ and constructs preference pairs
for reward model training. Each pair compares the orchestra-generated MAS
against the editor-modified MAS for the same task, ordered by score
(higher = chosen, lower = rejected).

Output: JSONL with {chosen_text, rejected_text, metadata} per line,
directly compatible with TRL's RewardTrainer (Bradley-Terry loss).

Inspired by OrchRM (Tsang et al., 2026): the RM operates directly at the
orchestration level, evaluating MAS graph text without sub-agent execution.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .schema import TrainingPair, DatasetStatistics


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _discover_task_dirs(output_dir: Path) -> list[Path]:
    task_dirs = []
    for dataset_dir in sorted(output_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        tasks_dir = dataset_dir / "tasks"
        if not tasks_dir.is_dir():
            continue
        for task_dir in sorted(tasks_dir.iterdir()):
            if task_dir.is_dir() and not task_dir.name.startswith("."):
                task_dirs.append(task_dir)
    return task_dirs


def _read_task_pair(task_dir: Path) -> dict[str, Any] | None:
    task_result = _load_json(task_dir / "task_result.json")
    if task_result is None or task_result.get("status") != "completed":
        return None

    orchestra_mas = _load_json(task_dir / "orchestra" / "mas_graph.json")
    editor_mas = _load_json(task_dir / "updated" / "mas_graph.json")
    if orchestra_mas is None:
        return None

    task_type = task_dir.parent.parent.name
    source_info = task_result.get("source", {})
    updated_info = task_result.get("updated", {})
    editor_info = task_result.get("editor", {})

    src_score = source_info.get("score", 0.0)
    upd_score = updated_info.get("score", 0.0)

    return {
        "task_id": task_result.get("task_id", task_dir.name),
        "task_text": task_result.get("task_text", ""),
        "task_type": task_type,
        "orchestra_mas": orchestra_mas,
        "orchestra_score": src_score,
        "orchestra_cost": source_info.get("cost_usd", 0.0),
        "orchestra_tokens": source_info.get("tokens", 0),
        "editor_mas": editor_mas if editor_mas is not None else {},
        "editor_score": upd_score,
        "editor_cost": updated_info.get("cost_usd", 0.0),
        "editor_tokens": updated_info.get("tokens", 0),
        "editor_operations": editor_info.get("operations", []),
        "comparison": task_result.get("comparison", {}),
    }


def _build_pair(task_data: dict[str, Any], pair_index: int) -> TrainingPair | None:
    """Build a score-based pair: higher-score MAS = chosen, lower-score = rejected."""
    if not task_data["editor_mas"]:
        return None

    orch_score = task_data["orchestra_score"]
    editor_score = task_data["editor_score"]

    if orch_score == editor_score:
        return None  # Skip ties

    if orch_score > editor_score:
        chosen_mas = task_data["orchestra_mas"]
        chosen_score = orch_score
        chosen_source = "orchestra"
        chosen_cost = task_data["orchestra_cost"]
        chosen_tokens = task_data["orchestra_tokens"]
        rejected_mas = task_data["editor_mas"]
        rejected_score = editor_score
        rejected_source = "editor"
        rejected_cost = task_data["editor_cost"]
        rejected_tokens = task_data["editor_tokens"]
    else:
        chosen_mas = task_data["editor_mas"]
        chosen_score = editor_score
        chosen_source = "editor"
        chosen_cost = task_data["editor_cost"]
        chosen_tokens = task_data["editor_tokens"]
        rejected_mas = task_data["orchestra_mas"]
        rejected_score = orch_score
        rejected_source = "orchestra"
        rejected_cost = task_data["orchestra_cost"]
        rejected_tokens = task_data["orchestra_tokens"]

    return TrainingPair(
        pair_id=f"{task_data['task_id']}_pair{pair_index:04d}",
        task_id=task_data["task_id"],
        task_type=task_data["task_type"],
        task_text=task_data["task_text"],
        chosen_mas=chosen_mas,
        chosen_score=chosen_score,
        chosen_source=chosen_source,
        chosen_cost_usd=chosen_cost,
        chosen_tokens=chosen_tokens,
        rejected_mas=rejected_mas,
        rejected_score=rejected_score,
        rejected_source=rejected_source,
        rejected_cost_usd=rejected_cost,
        rejected_tokens=rejected_tokens,
        editor_operations=task_data["editor_operations"],
        score_delta=abs(chosen_score - rejected_score),
        metadata={
            "orchestra_score": task_data["orchestra_score"],
            "editor_score": task_data["editor_score"],
            "comparison": task_data["comparison"],
        },
    )


def _compute_statistics(pairs: list[TrainingPair]) -> DatasetStatistics:
    stats = DatasetStatistics()
    stats.total_pairs = len(pairs)
    total_chosen, total_rejected = 0, 0
    for pair in pairs:
        stats.by_task_type[pair.task_type] = stats.by_task_type.get(pair.task_type, 0) + 1
        delta = pair.score_delta
        stats.average_score_delta += delta
        if delta == 0: bucket = "zero"
        elif delta <= 0.1: bucket = "0.0-0.1"
        elif delta <= 0.3: bucket = "0.1-0.3"
        elif delta <= 0.5: bucket = "0.3-0.5"
        elif delta <= 0.8: bucket = "0.5-0.8"
        else: bucket = "0.8-1.0"
        stats.by_score_delta_bucket[bucket] = stats.by_score_delta_bucket.get(bucket, 0) + 1
        total_chosen += len(pair.chosen_mas.get("nodes", []))
        total_rejected += len(pair.rejected_mas.get("nodes", []))
        if pair.chosen_source == "orchestra":
            stats.orchestra_chosen_count += 1
        elif pair.chosen_source == "editor":
            stats.editor_chosen_count += 1
        for op in pair.editor_operations:
            op_type = op.get("op", "unknown")
            stats.total_editor_operations[op_type] = stats.total_editor_operations.get(op_type, 0) + 1
    if stats.total_pairs > 0:
        stats.average_score_delta /= stats.total_pairs
        stats.average_chosen_agents = total_chosen / stats.total_pairs
        stats.average_rejected_agents = total_rejected / stats.total_pairs
    return stats


def construct_rm_dataset(
    output_dir: Path,
    output_path: Path | None = None,
    include_equal: bool = False,
    min_score_delta: float = 0.0,
    mas_text_format: str = "detailed",
) -> tuple[list[TrainingPair], DatasetStatistics]:
    """Construct Bradley-Terry RM training data from MAS-Loop trajectories.

    Scans outputs_formalized/{aime,drb}/tasks/ and builds preference pairs
    by comparing orchestra vs editor MAS scores. Higher-score = chosen,
    lower-score = rejected.

    Each chosen_text/rejected_text includes the task description, MAS source
    label (orchestra/editor), and the full MAS graph structure in text form.

    Output format: {chosen_text, rejected_text, metadata} per line.

    Args:
        output_dir: Root output directory (e.g., outputs_formalized/).
        output_path: Output JSONL path. Default: output_dir/rm_training.jsonl.
        include_equal: Include pairs where chosen_score == rejected_score.
        min_score_delta: Minimum absolute score difference.
        mas_text_format: "detailed", "compact", or "json".

    Returns:
        (pairs, statistics)
    """
    task_dirs = _discover_task_dirs(output_dir)
    pairs = []

    for task_dir in task_dirs:
        td = _read_task_pair(task_dir)
        if td is None:
            continue
        pair = _build_pair(td, len(pairs))
        if pair is not None:
            pairs.append(pair)

    if min_score_delta > 0:
        pairs = [p for p in pairs if p.score_delta >= min_score_delta]

    stats = _compute_statistics(pairs)

    if output_path is None:
        output_path = output_dir / "rm_training.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            entry = pair.to_rm_format(mas_text_format=mas_text_format)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Also save full metadata (raw format) for analysis
    raw_path = output_path.with_suffix(".raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    stats_path = output_path.with_suffix(".summary.json")
    stats_path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"RM training data: {len(pairs)} pairs -> {output_path}")
    print(f"  orchestra chosen: {stats.orchestra_chosen_count}")
    print(f"  editor chosen:    {stats.editor_chosen_count}")
    print(f"Raw pairs -> {raw_path}")
    print(f"Statistics -> {stats_path}")
    return pairs, stats