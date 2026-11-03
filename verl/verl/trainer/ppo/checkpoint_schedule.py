"""Checkpoint scheduling helpers for long-running PPO/GRPO jobs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


CHECKPOINT_COMPLETE_MARKER = ".mas_loop_checkpoint_complete.json"


def bounded_run_stop_step(total_steps: int, completed_steps: int, max_steps_this_run: int) -> int:
    """Return the inclusive optimizer step at which this trainer invocation stops.

    ``total_steps`` remains the full optimization horizon used by the optimizer
    and LR scheduler. ``max_steps_this_run`` only bounds one process lifetime,
    which lets MAS-Loop commit the paired peer-model update before another
    rollout batch can start.
    """

    total_steps = int(total_steps)
    completed_steps = int(completed_steps)
    max_steps_this_run = int(max_steps_this_run)
    if max_steps_this_run <= 0:
        return total_steps
    return min(total_steps, completed_steps + max_steps_this_run)


def checkpoint_complete_marker(global_step_folder: str | os.PathLike[str]) -> Path:
    """Return the out-of-band marker for a complete local checkpoint.

    The marker deliberately lives next to ``actor/`` rather than inside
    ``actor/huggingface`` so local checkpoint scanners can validate completeness
    without adding non-model files to the Transformers artifact.
    """

    return Path(global_step_folder) / CHECKPOINT_COMPLETE_MARKER


def clear_checkpoint_complete_marker(global_step_folder: str | os.PathLike[str]) -> None:
    """Remove a stale marker before a checkpoint directory is overwritten."""

    checkpoint_complete_marker(global_step_folder).unlink(missing_ok=True)


def write_checkpoint_complete_marker(
    global_step_folder: str | os.PathLike[str], global_step: int
) -> Path:
    """Atomically publish that all synchronous checkpoint writes completed."""

    folder = Path(global_step_folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = checkpoint_complete_marker(folder)
    temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
    payload = {
        "global_step": int(global_step),
        "completed_at_unix": time.time(),
        "protocol": "mas-loop-checkpoint-complete-v1",
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)
    return marker


def evenly_spaced_checkpoint_steps(total_steps: int, max_checkpoints: int = 10) -> tuple[int, ...]:
    """Return at most ``max_checkpoints`` steps, always including the final step.

    This is the MAS_Loop default checkpoint policy:

    * if a run has at most ``max_checkpoints`` optimizer steps, save every step;
    * otherwise choose ``max_checkpoints`` nearly-even positions across the
      complete run, including ``total_steps``.

    For example, a 27-step run with a limit of 10 saves exactly 10 checkpoints
    rather than interpreting ``27 // 10`` as a fixed interval (which would
    accidentally create 14 artifacts after adding the final step).
    """

    total_steps = int(total_steps)
    max_checkpoints = max(0, int(max_checkpoints))
    if total_steps <= 0:
        return ()
    if max_checkpoints <= 0:
        return ()
    if total_steps <= max_checkpoints:
        return tuple(range(1, total_steps + 1))

    steps = {(i * total_steps) // max_checkpoints for i in range(1, max_checkpoints + 1)}
    return tuple(sorted(steps))


def intermediate_plus_final_checkpoint_steps(
    total_steps: int, intermediate_checkpoints: int = 10
) -> tuple[int, ...]:
    """Return evenly spaced intermediate steps followed by the final step.

    The intermediate points divide the full optimization horizon into nearly
    equal intervals and are always strictly earlier than ``total_steps``. For
    example, 40 steps with ten intermediate artifacts yields 11 retained
    checkpoints in total. Short runs retain every available step rather than
    duplicating checkpoint positions.
    """

    total_steps = int(total_steps)
    intermediate_checkpoints = max(0, int(intermediate_checkpoints))
    if total_steps <= 0:
        return ()
    if intermediate_checkpoints <= 0:
        return (total_steps,)
    if total_steps <= intermediate_checkpoints + 1:
        return tuple(range(1, total_steps + 1))

    intervals = intermediate_checkpoints + 1
    intermediate = {(i * total_steps) // intervals for i in range(1, intervals)}
    intermediate.discard(0)
    intermediate.discard(total_steps)
    return (*sorted(intermediate), total_steps)


def configured_checkpoint_steps(config, total_steps: int) -> tuple[int, ...]:
    """Build an explicit schedule from the current or legacy trainer option.

    ``num_checkpoints`` is the preferred option and counts the final artifact.
    ``num_intermediate_checkpoints`` remains supported for old launch scripts;
    its historical meaning is preserved as intermediate artifacts plus final.
    """

    configured_total = config.trainer.get("num_checkpoints", None)
    if configured_total is not None:
        return evenly_spaced_checkpoint_steps(total_steps, int(configured_total or 0))

    legacy_intermediate = int(config.trainer.get("num_intermediate_checkpoints", 0) or 0)
    if legacy_intermediate <= 0:
        return ()
    return evenly_spaced_checkpoint_steps(total_steps, legacy_intermediate + 1)


def should_save_at_step(config, step: int, total_steps: int, explicit_steps: set[int] | frozenset[int]) -> bool:
    """Apply explicit schedule when configured, otherwise preserve save_freq."""

    if explicit_steps:
        return int(step) in explicit_steps
    save_freq = int(config.trainer.get("save_freq", -1) or -1)
    return save_freq > 0 and (int(step) >= int(total_steps) or int(step) % save_freq == 0)
