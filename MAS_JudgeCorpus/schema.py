"""Data structures for Bradley-Terry RM training pairs of MAS graphs.

Each training pair compares two MAS graph structures for the same task:
the higher-scoring one is "chosen", the lower-scoring is "rejected".

The RM learns to score MAS orchestration quality directly from the text
description of the MAS graph + task, without executing sub-agent rollouts.

Inspired by OrchRM (Tsang et al., 2026).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MasSource(str, Enum):
    ORCHESTRA = "orchestra"
    EDITOR = "editor"


@dataclass
class TrainingPair:
    """A single Bradley-Terry training pair for MAS reward model.

    Format: {chosen_text, rejected_text, metadata} where chosen_text and
    rejected_text are the textual representations of the MAS graph + task
    description. This format is directly compatible with TRL's RewardTrainer.

    Only MAS structure (topology, nodes, edges, roles, constraints) is
    included — no execution traces or runtime outputs.

    Attributes:
        pair_id: Unique identifier.
        task_id: Original task identifier.
        task_type: "aime" or "drb".
        task_text: The original task description.
        chosen_mas: Higher-scoring MAS graph.
        chosen_score: Score of the chosen MAS.
        chosen_source: "orchestra" or "editor" — where this MAS came from.
        chosen_cost_usd: Execution cost of chosen MAS.
        chosen_tokens: Tokens used by chosen MAS.
        rejected_mas: Lower-scoring MAS graph.
        rejected_score: Score of the rejected MAS.
        rejected_source: "orchestra" or "editor" — where this MAS came from.
        rejected_cost_usd: Execution cost of rejected MAS.
        rejected_tokens: Tokens used by rejected MAS.
        editor_operations: Editor operations applied to rejected MAS.
        score_delta: |chosen_score - rejected_score|.
        metadata: Additional metadata.
    """
    pair_id: str = ""
    task_id: str = ""
    task_type: str = ""
    task_text: str = ""

    chosen_mas: dict[str, Any] = field(default_factory=dict)
    chosen_score: float = 0.0
    chosen_source: str = ""
    chosen_cost_usd: float = 0.0
    chosen_tokens: int = 0

    rejected_mas: dict[str, Any] = field(default_factory=dict)
    rejected_score: float = 0.0
    rejected_source: str = ""
    rejected_cost_usd: float = 0.0
    rejected_tokens: int = 0

    editor_operations: list[dict[str, Any]] = field(default_factory=list)
    score_delta: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_text": self.task_text,
            "chosen_mas": self.chosen_mas,
            "chosen_score": self.chosen_score,
            "chosen_source": self.chosen_source,
            "chosen_cost_usd": self.chosen_cost_usd,
            "chosen_tokens": self.chosen_tokens,
            "rejected_mas": self.rejected_mas,
            "rejected_score": self.rejected_score,
            "rejected_source": self.rejected_source,
            "rejected_cost_usd": self.rejected_cost_usd,
            "rejected_tokens": self.rejected_tokens,
            "editor_operations": self.editor_operations,
            "score_delta": self.score_delta,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingPair:
        return cls(
            pair_id=d.get("pair_id", ""),
            task_id=d.get("task_id", ""),
            task_type=d.get("task_type", ""),
            task_text=d.get("task_text", ""),
            chosen_mas=d.get("chosen_mas", {}),
            chosen_score=d.get("chosen_score", 0.0),
            chosen_source=d.get("chosen_source", ""),
            chosen_cost_usd=d.get("chosen_cost_usd", 0.0),
            chosen_tokens=d.get("chosen_tokens", 0),
            rejected_mas=d.get("rejected_mas", {}),
            rejected_score=d.get("rejected_score", 0.0),
            rejected_source=d.get("rejected_source", ""),
            rejected_cost_usd=d.get("rejected_cost_usd", 0.0),
            rejected_tokens=d.get("rejected_tokens", 0),
            editor_operations=d.get("editor_operations", []),
            score_delta=d.get("score_delta", 0.0),
            metadata=d.get("metadata", {}),
        )

    def to_rm_format(
        self,
        mas_text_format: str = "detailed",
        max_role_length: int = 200,
    ) -> dict[str, Any]:
        """Convert to Bradley-Terry RM training format.

        Returns {chosen_text, rejected_text, metadata} where chosen_text
        and rejected_text are the textual representations of the MAS graph
        + task description. Each text explicitly marks the source
        (orchestra/editor) so the RM can learn source-specific patterns.
        """
        from .mas_formatter import format_mas_with_task
        chosen_text = format_mas_with_task(
            self.chosen_mas, self.task_text,
            fmt=mas_text_format, max_role_length=max_role_length,
        )
        rejected_text = format_mas_with_task(
            self.rejected_mas, self.task_text,
            fmt=mas_text_format, max_role_length=max_role_length,
        )
        return {
            "chosen_text": chosen_text,
            "rejected_text": rejected_text,
            "metadata": {
                "pair_id": self.pair_id,
                "task_id": self.task_id,
                "task_type": self.task_type,
                "chosen_score": self.chosen_score,
                "rejected_score": self.rejected_score,
                "chosen_source": self.chosen_source,
                "rejected_source": self.rejected_source,
                "score_delta": self.score_delta,
            },
        }


@dataclass
class DatasetStatistics:
    """Statistics for the constructed training dataset."""
    total_pairs: int = 0
    by_task_type: dict[str, int] = field(default_factory=dict)
    by_score_delta_bucket: dict[str, int] = field(default_factory=dict)
    average_score_delta: float = 0.0
    total_editor_operations: dict[str, int] = field(default_factory=dict)
    average_chosen_agents: float = 0.0
    average_rejected_agents: float = 0.0
    orchestra_chosen_count: int = 0
    editor_chosen_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pairs": self.total_pairs,
            "by_task_type": self.by_task_type,
            "by_score_delta_bucket": self.by_score_delta_bucket,
            "average_score_delta": self.average_score_delta,
            "total_editor_operations": self.total_editor_operations,
            "average_chosen_agents": round(self.average_chosen_agents, 2),
            "average_rejected_agents": round(self.average_rejected_agents, 2),
            "orchestra_chosen_count": self.orchestra_chosen_count,
            "editor_chosen_count": self.editor_chosen_count,
        }