"""BrowseComp-Plus dataset runner for MAS-Loop.

BrowseComp-Plus is an open-domain QA dataset with retrieval.
Scoring uses exact match with the ground truth answer string.

Wraps the existing datasets/BrowseComp-Plus/skill_mas_runner.py for
MAS generation and execution with BM25 retrieval.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_MAS_LOOP_ROOT = Path(__file__).resolve().parents[2]
if str(_MAS_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAS_LOOP_ROOT))

from mas_loop.mas_evo_traj import ExecutionTrajectory


@dataclass
class BcpConfig:
    """Configuration for BrowseComp-Plus dataset runner."""

    data_path: str = "datasets/BrowseComp-Plus/data/browsecomp_plus_test.jsonl"
    model: str = "deepseek-v4-flash"
    init_skill_path: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def bcp_score(final_output: str, expected_answer: str) -> float:
    """Score a BrowseComp-Plus answer: 1.0 for exact match, 0.0 otherwise.

    BrowseComp-Plus expects a single answer string. We normalize whitespace
    and compare case-insensitively.
    """
    out = (final_output or "").strip().lower()
    exp = (expected_answer or "").strip().lower()

    if not exp:
        return 0.0

    # Exact match after normalization
    if out == exp:
        return 1.0

    # Check if the expected answer is contained in the output
    if exp in out:
        return 0.5

    return 0.0


class BcpRunner:
    """BrowseComp-Plus dataset runner for MAS-Loop.

    Provides task loading and scoring compatible with the MAS-Loop
    execute-and-score pipeline. BrowseComp-Plus uses BM25 retrieval,
    so the tool_call_fn must be configured for multi_turn_search.
    """

    def __init__(self, config: BcpConfig):
        self.config = config

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load BCP tasks from the JSONL data file."""
        path = Path(self.config.data_path)
        if not path.is_absolute():
            path = _MAS_LOOP_ROOT / path

        tasks: list[dict[str, Any]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tasks.append({
                        "task_text": row.get("question", ""),
                        "expected_answer": str(row.get("answer", "")),
                        "task_id": str(row.get("id", "")),
                    })
        return tasks

    def score_trajectory(self, traj: ExecutionTrajectory, expected_answer: str) -> float:
        """Score a trajectory against the expected BCP answer."""
        return bcp_score(traj.final_output, expected_answer)

    def score_output(self, final_output: str, expected_answer: str) -> float:
        """Score a raw output string against the expected BCP answer."""
        return bcp_score(final_output, expected_answer)
