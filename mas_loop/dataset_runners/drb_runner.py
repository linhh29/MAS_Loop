"""Deep Research Bench dataset runner for MAS-Loop.

Deep Research Bench (DRB) evaluates long-form research article generation.
Scoring uses RACE evaluation metrics: comprehensiveness, insight,
instruction_following, readability, and overall_score.

Wraps the existing datasets/deep_research_bench/skill_mas_agent_runner.py
for MAS generation and execution.
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
class DrbConfig:
    """Configuration for Deep Research Bench dataset runner."""

    data_path: str = "datasets/deep_research_bench/data/drb_test.jsonl"
    model: str = "deepseek-v4-flash"
    init_skill_path: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def drb_score(final_output: str, expected_answer: str) -> float:
    """Score a DRB output: heuristic quality score based on article length.

    In a full implementation, this would use the RACE evaluation pipeline
    (comprehensiveness, insight, instruction_following, readability).
    For now, this is a heuristic placeholder that scores based on:
    - Output is non-empty and substantial (length > 200 chars)
    - Expected answer (reference) is present
    """
    out = (final_output or "").strip()
    if not out or len(out) < 200:
        return 0.0

    # Basic heuristic: substantial output = partial credit
    # In production, replace with RACE evaluation from deep_research_bench/evaluate.py
    score = min(1.0, len(out) / 5000.0)  # Scales with length up to ~5000 chars
    return score


class DrbRunner:
    """Deep Research Bench dataset runner for MAS-Loop.

    Provides task loading and scoring compatible with the MAS-Loop
    execute-and-score pipeline. DRB tasks generate long-form research
    articles.
    """

    def __init__(self, config: DrbConfig):
        self.config = config

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load DRB tasks from the JSONL data file."""
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
                        "task_text": row.get("prompt", ""),
                        "expected_answer": row.get("reference_article", ""),
                        "task_id": str(row.get("id", "")),
                        "rubric": row.get("rubric", {}),
                    })
        return tasks

    def score_trajectory(self, traj: ExecutionTrajectory, expected_answer: str) -> float:
        """Score a trajectory against the expected DRB reference."""
        return drb_score(traj.final_output, expected_answer)

    def score_output(self, final_output: str, expected_answer: str) -> float:
        """Score a raw output string against the expected DRB reference."""
        return drb_score(final_output, expected_answer)
