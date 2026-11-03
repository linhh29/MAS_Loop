"""Safe retention of inference checkpoints for strict MAS-Loop joint training."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Iterable

from verl.trainer.ppo.checkpoint_schedule import intermediate_plus_final_checkpoint_steps


EVALUATION_CHECKPOINT_MARKER = ".mas_loop_evaluation_checkpoint.json"
_STEP_DIRECTORY = re.compile(r"global_step_([1-9][0-9]*)")
_RESUME_SHARD_PATTERNS = (
    "model_world_size_*_rank_*.pt",
    "optim_world_size_*_rank_*.pt",
    "extra_state_world_size_*_rank_*.pt",
)


def retained_joint_checkpoint_steps(total_steps: int, total_checkpoints: int = 11) -> tuple[int, ...]:
    """Return the strict intermediate-plus-final retention schedule."""

    total_checkpoints = max(0, int(total_checkpoints))
    if total_checkpoints == 0:
        return ()
    return intermediate_plus_final_checkpoint_steps(total_steps, total_checkpoints - 1)


def _validated_step_directory(root: Path, step: int) -> Path:
    root = root.resolve(strict=True)
    checkpoint = root / f"global_step_{step}"
    if checkpoint.is_symlink() or checkpoint.parent != root:
        raise ValueError(f"unsafe checkpoint path: {checkpoint}")
    match = _STEP_DIRECTORY.fullmatch(checkpoint.name)
    if match is None or int(match.group(1)) != step:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint.name}")
    return checkpoint


def _validate_hf_artifact(checkpoint: Path) -> Path:
    hf = checkpoint / "actor" / "huggingface"
    if hf.is_symlink() or not (hf / "config.json").is_file():
        raise FileNotFoundError(f"missing local Hugging Face checkpoint: {hf}")
    weights = tuple(hf.glob("*.safetensors")) + tuple(hf.glob("pytorch_model*.bin"))
    if not weights or any(path.is_symlink() or not path.is_file() for path in weights):
        raise FileNotFoundError(f"missing or unsafe model weights: {hf}")
    return hf


def _write_evaluation_marker(checkpoint: Path, step: int, total_steps: int) -> None:
    marker = checkpoint / EVALUATION_CHECKPOINT_MARKER
    temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
    payload = {
        "protocol": "mas-loop-evaluation-checkpoint-v1",
        "global_step": step,
        "total_steps": total_steps,
        "retained_contents": ["actor/huggingface"],
        "compacted_at_unix": time.time(),
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)


def validate_completed_step_retention(
    role_roots: Iterable[str | os.PathLike[str]],
    step: int,
    total_steps: int,
    total_checkpoints: int = 11,
) -> bool:
    """Validate whether a completed historical step is present as required.

    Retained evaluation steps must exist for every role. Unselected steps may
    be absent after compaction; if they still exist (for example the active
    resume checkpoint), the caller can apply its normal checkpoint validator.
    """

    step = int(step)
    total_steps = int(total_steps)
    if total_steps <= 0 or not 1 <= step <= total_steps:
        raise ValueError(f"invalid completed step: step={step}, total={total_steps}")
    retained = step in frozenset(retained_joint_checkpoint_steps(total_steps, total_checkpoints))
    for root_text in role_roots:
        root = Path(root_text)
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"unsafe or missing checkpoint root: {root}")
        checkpoint = _validated_step_directory(root, step)
        if retained and not checkpoint.is_dir():
            raise FileNotFoundError(f"retained checkpoint is missing: {checkpoint}")
    return retained


def compact_joint_checkpoints(
    role_roots: Iterable[str | os.PathLike[str]],
    completed_step: int,
    total_steps: int,
    total_checkpoints: int = 11,
) -> tuple[int, ...]:
    """Compact or remove checkpoints older than the active resume checkpoint.

    The current completed step remains fully resumable until the next joint-step
    barrier. Once a newer barrier is durable, selected older checkpoints retain
    only their Hugging Face inference artifact and unselected older checkpoints
    are removed. At the final barrier, the final checkpoint is compacted too.
    """

    completed_step = int(completed_step)
    total_steps = int(total_steps)
    if total_steps <= 0 or not 1 <= completed_step <= total_steps:
        raise ValueError(
            f"invalid joint checkpoint boundary: completed={completed_step}, total={total_steps}"
        )
    retained = retained_joint_checkpoint_steps(total_steps, total_checkpoints)
    retained_set = frozenset(retained)
    cleanup_through = completed_step if completed_step == total_steps else completed_step - 1
    if cleanup_through <= 0:
        return retained

    for root_text in role_roots:
        root = Path(root_text)
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"unsafe or missing checkpoint root: {root}")
        for step in range(1, cleanup_through + 1):
            checkpoint = _validated_step_directory(root, step)
            if not checkpoint.exists():
                if step in retained_set:
                    raise FileNotFoundError(f"retained checkpoint is missing: {checkpoint}")
                continue
            if step in retained_set:
                _validate_hf_artifact(checkpoint)
                actor = checkpoint / "actor"
                # Security: deletion is restricted to fixed shard patterns beneath
                # a validated, non-symlink direct child of the configured root.
                for pattern in _RESUME_SHARD_PATTERNS:
                    for shard in actor.glob(pattern):
                        if shard.is_symlink() or not shard.is_file():
                            raise ValueError(f"unsafe resume shard path: {shard}")
                        shard.unlink()
                for metadata_name in ("data.pt", ".mas_loop_checkpoint_complete.json"):
                    metadata = checkpoint / metadata_name
                    if metadata.is_symlink():
                        raise ValueError(f"unsafe checkpoint metadata path: {metadata}")
                    metadata.unlink(missing_ok=True)
                _write_evaluation_marker(checkpoint, step, total_steps)
            else:
                # Security: recursive deletion is bounded to a validated generated
                # global_step_<integer> directory and never follows a symlink.
                shutil.rmtree(checkpoint)
    return retained


def main() -> int:
    """Command-line entry point used by the immutable joint-training wrapper."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-root", action="append", required=True)
    parser.add_argument("--completed-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--total-checkpoints", type=int, default=11)
    args = parser.parse_args()
    retained = compact_joint_checkpoints(
        args.role_root,
        args.completed_step,
        args.total_steps,
        args.total_checkpoints,
    )
    print(",".join(str(step) for step in retained))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
