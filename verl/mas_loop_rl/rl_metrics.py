"""Durable, lossless JSONL persistence for MAS-Loop RL optimizer metrics."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_scalar(value: Any) -> Any:
    """Convert tensor/numpy-like scalar values to finite JSON-compatible values."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError, RuntimeError):
            pass
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else str(converted)
    except (TypeError, ValueError):
        return str(value)


def persist_rl_metrics(metrics: dict[str, Any], step: int) -> None:
    """Append one complete optimizer-step metric record when configured.

    ``MAS_LOOP_RL_METRICS_FILE`` is set by the phase launcher to a file below
    the corresponding round/role checkpoint directory. Records are appended so
    a resumed run never destroys previous diagnostics.
    """
    raw_path = os.getenv("MAS_LOOP_RL_METRICS_FILE", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "mas_loop_rl_metrics_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": os.getenv("MAS_LOOP_DATASET_NAME", "unknown"),
        "role": os.getenv("MAS_LOOP_ROLE", "unknown"),
        "outer_round": int(os.getenv("MAS_LOOP_OUTER_ROUND", "0") or 0),
        "global_step": int(step),
        "metrics": {str(k): _json_scalar(v) for k, v in metrics.items()},
    }
    # A single trainer driver writes each phase file. One write call also keeps
    # individual JSONL records intact if a supervisor tails the file.
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
