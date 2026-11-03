"""MAS-EvoTraj: Structured representation of MAS evolution trajectories.

Each EvoTraj sample records the complete lifecycle of a MAS:
    (task, src_mas, src_trajectory, edit_operation, updated_mas,
     updated_trajectory, outcome, delta_summary)

Reference: MAS-Loop Expanded Idea, Section 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .mas_schema import MASConfig, Patch


@dataclass
class AgentTrajectory:
    """Execution trace of a single agent within a MAS run."""

    agent_id: str = ""
    agent_role: str = ""
    output: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "output": self.output,
            "usage": dict(self.usage),
            "error": self.error,
            "elapsed_sec": self.elapsed_sec,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTrajectory":
        return cls(
            agent_id=str(data.get("agent_id", "")),
            agent_role=str(data.get("agent_role", "")),
            output=str(data.get("output", "")),
            usage=dict(data.get("usage", {})),
            error=data.get("error"),
            elapsed_sec=float(data.get("elapsed_sec", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExecutionTrajectory:
    """Full execution trace of a MAS run."""

    task_id: str = ""
    task_text: str = ""
    mas_config: Optional[MASConfig] = None
    agent_traces: list[AgentTrajectory] = field(default_factory=list)
    final_output: str = ""
    success: bool = False
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    elapsed_sec: float = 0.0
    failure_nodes: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "mas_config": self.mas_config.to_dict() if self.mas_config else None,
            "agent_traces": [t.to_dict() for t in self.agent_traces],
            "final_output": self.final_output,
            "success": self.success,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "elapsed_sec": self.elapsed_sec,
            "failure_nodes": list(self.failure_nodes),
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionTrajectory":
        mc = data.get("mas_config")
        return cls(
            task_id=str(data.get("task_id", "")),
            task_text=str(data.get("task_text", "")),
            mas_config=MASConfig.from_dict(mc) if isinstance(mc, dict) else None,
            agent_traces=[AgentTrajectory.from_dict(t) for t in data.get("agent_traces", [])],
            final_output=str(data.get("final_output", "")),
            success=bool(data.get("success", False)),
            total_cost_usd=float(data.get("total_cost_usd", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            elapsed_sec=float(data.get("elapsed_sec", 0.0)),
            failure_nodes=list(data.get("failure_nodes", [])),
            error_message=data.get("error_message"),
            metadata=dict(data.get("metadata", {})),
        )

    def error_agents(self) -> list[str]:
        return [t.agent_id for t in self.agent_traces if t.error is not None]

    def redundant_agents(self) -> list[str]:
        outputs: dict[str, str] = {}
        redundant: list[str] = []
        for t in self.agent_traces:
            out = (t.output or "").strip()
            if not out:
                redundant.append(t.agent_id)
                continue
            for existing_id, existing_out in outputs.items():
                if out == existing_out:
                    redundant.append(t.agent_id)
                    break
            outputs[t.agent_id] = out
        return redundant


@dataclass
class EvoTrajSample:
    """A single MAS evolution trajectory sample.

    Fields (from Section 4.1 of the MAS-Loop Expanded Idea):
    - task: original task description including constraints
    - src_mas: the initial MAS configuration
    - src_trajectory: execution trace of the source MAS
    - edit_operation: the structured patch applied
    - updated_mas: the MAS after patching
    - updated_trajectory: execution trace of the updated MAS
    - outcome: task quality metrics (success, score, cost, latency)
    - edit_constraints: constraints that must be respected during editing
    - delta_summary: summary of changes and their impact
    """

    task: dict[str, Any] = field(default_factory=dict)
    src_mas: Optional[MASConfig] = None
    src_trajectory: Optional[ExecutionTrajectory] = None
    edit_operation: Optional[Patch] = None
    updated_mas: Optional[MASConfig] = None
    updated_trajectory: Optional[ExecutionTrajectory] = None
    outcome: dict[str, Any] = field(default_factory=dict)
    edit_constraints: dict[str, Any] = field(default_factory=dict)
    delta_summary: dict[str, Any] = field(default_factory=dict)
    sample_id: str = ""
    created_at: str = ""
    model_name: str = ""
    dataset_name: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "task": dict(self.task),
            "src_mas": self.src_mas.to_dict() if self.src_mas else None,
            "src_trajectory": self.src_trajectory.to_dict() if self.src_trajectory else None,
            "edit_operation": self.edit_operation.to_dict() if self.edit_operation else None,
            "updated_mas": self.updated_mas.to_dict() if self.updated_mas else None,
            "updated_trajectory": self.updated_trajectory.to_dict() if self.updated_trajectory else None,
            "outcome": dict(self.outcome),
            "edit_constraints": dict(self.edit_constraints),
            "delta_summary": dict(self.delta_summary),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvoTrajSample":
        sm = data.get("src_mas")
        st = data.get("src_trajectory")
        eo = data.get("edit_operation")
        um = data.get("updated_mas")
        ut = data.get("updated_trajectory")
        return cls(
            sample_id=str(data.get("sample_id", "")),
            created_at=str(data.get("created_at", "")),
            model_name=str(data.get("model_name", "")),
            dataset_name=str(data.get("dataset_name", "")),
            task=dict(data.get("task", {})),
            src_mas=MASConfig.from_dict(sm) if isinstance(sm, dict) else None,
            src_trajectory=ExecutionTrajectory.from_dict(st) if isinstance(st, dict) else None,
            edit_operation=Patch.from_dict(eo) if isinstance(eo, dict) else None,
            updated_mas=MASConfig.from_dict(um) if isinstance(um, dict) else None,
            updated_trajectory=ExecutionTrajectory.from_dict(ut) if isinstance(ut, dict) else None,
            outcome=dict(data.get("outcome", {})),
            edit_constraints=dict(data.get("edit_constraints", {})),
            delta_summary=dict(data.get("delta_summary", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "EvoTrajSample":
        return cls.from_dict(json.loads(text))

    def score_delta(self) -> float:
        src_score = float(self.outcome.get("src_score", 0.0))
        upd_score = float(self.outcome.get("upd_score", 0.0))
        return upd_score - src_score

    def cost_delta(self) -> float:
        src_cost = float(self.outcome.get("src_cost_usd", 0.0))
        upd_cost = float(self.outcome.get("upd_cost_usd", 0.0))
        return upd_cost - src_cost

    def is_improvement(self) -> bool:
        return self.score_delta() > 0

    def edit_size(self) -> int:
        if self.edit_operation is None:
            return 0
        return len(self.edit_operation.operations)


class EvoTrajDataset:
    """Reader / writer for MAS-EvoTraj JSONL files."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        self._samples: list[EvoTrajSample] = []

    @property
    def samples(self) -> list[EvoTrajSample]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def add(self, sample: EvoTrajSample) -> None:
        self._samples.append(sample)

    def load(self, path: str | Path) -> "EvoTrajDataset":
        self._path = Path(path)
        self._samples = []
        if not self._path.is_file():
            return self
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._samples.append(EvoTrajSample.from_json(line))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[EvoTrajDataset] skipped malformed line: {e}")
        return self

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("No output path specified")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for s in self._samples:
                f.write(s.to_json() + "\n")
        return target

    def filter_by_dataset(self, dataset_name: str) -> "EvoTrajDataset":
        result = EvoTrajDataset()
        for s in self._samples:
            if s.dataset_name == dataset_name:
                result.add(s)
        return result

    def filter_by_improvement(self, min_delta: float = 0.0) -> "EvoTrajDataset":
        result = EvoTrajDataset()
        for s in self._samples:
            if s.score_delta() >= min_delta:
                result.add(s)
        return result

    def statistics(self) -> dict[str, Any]:
        if not self._samples:
            return {"count": 0}
        count = len(self._samples)
        improvements = sum(1 for s in self._samples if s.is_improvement())
        avg_score_delta = sum(s.score_delta() for s in self._samples) / count
        avg_cost_delta = sum(s.cost_delta() for s in self._samples) / count
        avg_edit_size = sum(s.edit_size() for s in self._samples) / count
        return {
            "count": count,
            "improvement_rate": improvements / count if count > 0 else 0.0,
            "avg_score_delta": avg_score_delta,
            "avg_cost_delta": avg_cost_delta,
            "avg_edit_size": avg_edit_size,
            "dataset_names": list({s.dataset_name for s in self._samples if s.dataset_name}),
        }