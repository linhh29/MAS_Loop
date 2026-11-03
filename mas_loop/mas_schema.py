"""MAS Schema: structured representation of Multi-Agent System configurations.

Defines the canonical MAS schema, patch operations, topology types, and
validation logic used throughout the MAS-Loop framework (Orchestra, Editor,
Executor, MAS-EvoTraj).

Schema reference (from MAS-Loop Expanded Idea, Section 2):
```json
{
  "agents": [
    {"id": "planner", "role": "decompose task", "tools": [], "budget": 1},
    {"id": "solver", "role": "solve subtasks", "tools": ["calculator"], "budget": 3},
    {"id": "verifier", "role": "check final answer", "tools": [], "budget": 1}
  ],
  "edges": [
    {"from": "planner", "to": "solver", "type": "subtask"},
    {"from": "solver", "to": "verifier", "type": "answer_check"}
  ],
  "aggregation": "verifier_final"
}
```
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TopologyType(str, Enum):
    """MAS topology styles supported by the framework."""

    SEQUENTIAL = "sequential"
    DAG = "dag"  # Multi-dependency / diamond
    HIERARCHICAL = "hierarchical"
    ITERATIVE = "iterative"  # Loop-based


class EdgeType(str, Enum):
    """Semantic edge types between agents."""

    SUBTASK = "subtask"  # Parent delegates a sub-task
    ANSWER_CHECK = "answer_check"  # Verification edge
    CRITIQUE = "critique"  # Critique / improvement
    SUMMARIZE = "summarize"  # Merging / summarisation
    DATA_FLOW = "data_flow"  # Generic data flow


class ExecutionMode(str, Enum):
    """How each agent should be executed at runtime."""

    LLM_ONLY = "llm_only"
    MULTI_TURN_SEARCH = "multi_turn_search"
    VITA_TOOL = "vita_tool"


class FailureType(str, Enum):
    """Diagnosis failure categories for Editor."""

    MISSING_VERIFICATION = "missing_verification"
    REDUNDANT_AGENT = "redundant_agent"
    BROKEN_EDGE = "broken_edge"
    ROLE_MISMATCH = "role_mismatch"
    BUDGET_OVERFLOW = "budget_overflow"
    TOOL_MISMATCH = "tool_mismatch"
    AGGREGATION_ERROR = "aggregation_error"
    COMMUNICATION_NOISE = "communication_noise"
    SINGLE_POINT_FAILURE = "single_point_failure"


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


class PatchOp(str, Enum):
    """Structured edit operations for MAS modification."""

    ADD_AGENT = "add_agent"
    REMOVE_AGENT = "remove_agent"
    MERGE_AGENTS = "merge_agents"
    REWRITE_ROLE = "rewrite_role"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    ADD_VERIFIER = "add_verifier"
    CHANGE_AGGREGATION = "change_aggregation"
    CHANGE_BUDGET = "change_budget"
    REPLACE_TOOL = "replace_tool"
    EARLY_STOP_RULE = "early_stop_rule"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """A single agent in the MAS."""

    id: str
    role: str
    tools: list[str] = field(default_factory=list)
    budget: int = 1
    execution_mode: ExecutionMode = ExecutionMode.LLM_ONLY
    max_tokens: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "tools": list(self.tools),
            "budget": self.budget,
            "execution_mode": self.execution_mode.value,
        }
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        em = data.get("execution_mode", "llm_only")
        return cls(
            id=str(data["id"]),
            role=str(data["role"]),
            tools=list(data.get("tools", [])),
            budget=int(data.get("budget", 1)),
            execution_mode=ExecutionMode(em) if em in {e.value for e in ExecutionMode} else ExecutionMode.LLM_ONLY,
            max_tokens=data.get("max_tokens"),
        )


@dataclass
class EdgeConfig:
    """A directed communication edge between two agents."""

    from_agent: str
    to_agent: str
    edge_type: EdgeType = EdgeType.DATA_FLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.edge_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeConfig":
        et = data.get("type", "data_flow")
        return cls(
            from_agent=str(data["from"]),
            to_agent=str(data["to"]),
            edge_type=EdgeType(et) if et in {e.value for e in EdgeType} else EdgeType.DATA_FLOW,
        )


@dataclass
class MASConfig:
    """Full Multi-Agent System configuration."""

    agents: list[AgentConfig] = field(default_factory=list)
    edges: list[EdgeConfig] = field(default_factory=list)
    aggregation: str = ""  # ID of the agent whose output is the final answer
    topology: TopologyType = TopologyType.SEQUENTIAL
    max_cost: float = 10.0
    max_iterations: int = 3

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "edges": [e.to_dict() for e in self.edges],
            "aggregation": self.aggregation,
            "topology": self.topology.value,
            "max_cost": self.max_cost,
            "max_iterations": self.max_iterations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MASConfig":
        topo = data.get("topology", "sequential")
        return cls(
            agents=[AgentConfig.from_dict(a) for a in data.get("agents", [])],
            edges=[EdgeConfig.from_dict(e) for e in data.get("edges", [])],
            aggregation=str(data.get("aggregation", "")),
            topology=TopologyType(topo) if topo in {t.value for t in TopologyType} else TopologyType.SEQUENTIAL,
            max_cost=float(data.get("max_cost", 10.0)),
            max_iterations=int(data.get("max_iterations", 3)),
        )

    @classmethod
    def from_json(cls, text: str) -> "MASConfig":
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def agent_ids(self) -> set[str]:
        return {a.id for a in self.agents}

    def agent_count(self) -> int:
        return len(self.agents)

    def edge_count(self) -> int:
        return len(self.edges)

    def total_budget(self) -> int:
        return sum(a.budget for a in self.agents)

    def is_connected(self) -> bool:
        """Check if the MAS graph is weakly connected."""
        if not self.agents:
            return True
        ids = self.agent_ids()
        adj: dict[str, set[str]] = {aid: set() for aid in ids}
        for e in self.edges:
            if e.from_agent in ids:
                adj[e.from_agent].add(e.to_agent)
            if e.to_agent in ids:
                adj[e.to_agent].add(e.from_agent)
        if not ids:
            return True
        start = next(iter(ids))
        visited: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nb in adj.get(cur, set()):
                if nb not in visited:
                    stack.append(nb)
        return visited == ids

    def has_cycle(self) -> bool:
        """Detect if the MAS graph has a directed cycle."""
        ids = self.agent_ids()
        adj: dict[str, set[str]] = {aid: set() for aid in ids}
        for e in self.edges:
            if e.from_agent in ids and e.to_agent in ids:
                adj[e.from_agent].add(e.to_agent)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {aid: WHITE for aid in ids}

        def _dfs(u: str) -> bool:
            color[u] = GRAY
            for v in adj.get(u, set()):
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and _dfs(v):
                    return True
            color[u] = BLACK
            return False

        for aid in ids:
            if color[aid] == WHITE and _dfs(aid):
                return True
        return False

    def copy(self) -> "MASConfig":
        return self.from_dict(self.to_dict())


# ---------------------------------------------------------------------------
# Patch definition
# ---------------------------------------------------------------------------


@dataclass
class PatchOperation:
    """A single structured edit operation on a MAS."""

    op: PatchOp
    agent_id: str = ""
    agent_role: str = ""
    from_agent: str = ""
    to_agent: str = ""
    edge_type: EdgeType = EdgeType.DATA_FLOW
    aggregation: str = ""
    budget: int = 0
    tool_name: str = ""
    merge_target: str = ""  # For merge_agents: which agent absorbs the other
    old_role: str = ""  # For rewrite_role
    new_role: str = ""  # For rewrite_role

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op.value}
        if self.agent_id:
            d["agent_id"] = self.agent_id
        if self.agent_role:
            d["agent_role"] = self.agent_role
        if self.from_agent:
            d["from"] = self.from_agent
        if self.to_agent:
            d["to"] = self.to_agent
        if self.edge_type != EdgeType.DATA_FLOW:
            d["type"] = self.edge_type.value
        if self.aggregation:
            d["aggregation"] = self.aggregation
        if self.budget:
            d["budget"] = self.budget
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.merge_target:
            d["merge_target"] = self.merge_target
        if self.old_role:
            d["old_role"] = self.old_role
        if self.new_role:
            d["new_role"] = self.new_role
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatchOperation":
        op = PatchOp(data["op"])
        return cls(
            op=op,
            agent_id=str(data.get("agent_id", "")),
            agent_role=str(data.get("agent_role", "")),
            from_agent=str(data.get("from", "")),
            to_agent=str(data.get("to", "")),
            edge_type=EdgeType(data.get("type", "data_flow")),
            aggregation=str(data.get("aggregation", "")),
            budget=int(data.get("budget", 0)),
            tool_name=str(data.get("tool_name", "")),
            merge_target=str(data.get("merge_target", "")),
            old_role=str(data.get("old_role", "")),
            new_role=str(data.get("new_role", "")),
        )


@dataclass
class Patch:
    """A collection of patch operations with diagnosis metadata."""

    diagnosis: dict[str, Any] = field(default_factory=dict)
    operations: list[PatchOperation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": dict(self.diagnosis),
            "patch": [op.to_dict() for op in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Patch":
        return cls(
            diagnosis=dict(data.get("diagnosis", {})),
            operations=[PatchOperation.from_dict(op) for op in data.get("patch", [])],
        )

    def is_empty(self) -> bool:
        return len(self.operations) == 0

    def changed(self) -> bool:
        """Return True if the patch has at least one operation that actually changes the MAS."""
        return len(self.operations) > 0

    def edit_type(self) -> str:
        """Categorize the patch into a high-level edit type.

        Based on Section 4.3 of MAS-Loop Expanded Idea.
        """
        ops = {op.op for op in self.operations}
        if not ops:
            return "no_op"

        # constraining: change_budget, early_stop_rule
        if ops <= {PatchOp.CHANGE_BUDGET, PatchOp.EARLY_STOP_RULE}:
            return "constraining"

        # simplifying: remove_agent, remove_edge, merge_agents
        if ops <= {PatchOp.REMOVE_AGENT, PatchOp.REMOVE_EDGE, PatchOp.MERGE_AGENTS}:
            return "simplifying"

        # repairing: rewrite_role, replace_tool, add_verifier
        if ops <= {PatchOp.REWRITE_ROLE, PatchOp.REPLACE_TOOL, PatchOp.ADD_VERIFIER}:
            return "repairing"

        # upgrading: add_agent, add_edge, add_verifier
        if ops <= {PatchOp.ADD_AGENT, PatchOp.ADD_EDGE, PatchOp.ADD_VERIFIER}:
            return "upgrading"

        # restructuring: add_edge, remove_edge, change_aggregation
        if ops <= {PatchOp.ADD_EDGE, PatchOp.REMOVE_EDGE, PatchOp.CHANGE_AGGREGATION}:
            return "restructuring"

        # mixed: combination of multiple types
        if ops & {PatchOp.ADD_AGENT, PatchOp.ADD_VERIFIER}:
            return "extending"
        if ops & {PatchOp.REMOVE_AGENT, PatchOp.MERGE_AGENTS}:
            return "compressing"
        if ops & {PatchOp.REWRITE_ROLE}:
            return "refining"

        return "mixed"


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def apply_patch(mas: MASConfig, patch: Patch) -> MASConfig:
    """Apply a patch to a MAS config, returning a new MASConfig.

    The original ``mas`` is never modified — a deep copy is made first.
    """
    result = mas.copy()

    for op in patch.operations:
        if op.op == PatchOp.ADD_AGENT:
            result.agents.append(
                AgentConfig(
                    id=op.agent_id or f"agent_{len(result.agents)+1}",
                    role=op.agent_role or "new agent",
                )
            )
        elif op.op == PatchOp.REMOVE_AGENT:
            result.agents = [a for a in result.agents if a.id != op.agent_id]
            result.edges = [
                e
                for e in result.edges
                if e.from_agent != op.agent_id and e.to_agent != op.agent_id
            ]
            if result.aggregation == op.agent_id:
                result.aggregation = ""
        elif op.op == PatchOp.MERGE_AGENTS:
            target = op.merge_target or op.to_agent
            source = op.agent_id
            result.agents = [a for a in result.agents if a.id != source]
            # Rewire edges: source -> target
            result.edges = [
                EdgeConfig(
                    from_agent=target if e.from_agent == source else e.from_agent,
                    to_agent=target if e.to_agent == source else e.to_agent,
                    edge_type=e.edge_type,
                )
                for e in result.edges
            ]
            # Remove self-loops created by rewiring
            result.edges = [
                e for e in result.edges if e.from_agent != e.to_agent
            ]
            if result.aggregation == source:
                result.aggregation = target
        elif op.op == PatchOp.REWRITE_ROLE:
            new_role_text = op.new_role or op.agent_role
            for a in result.agents:
                if a.id == op.agent_id:
                    if new_role_text:
                        a.role = new_role_text
        elif op.op == PatchOp.ADD_EDGE:
            result.edges.append(
                EdgeConfig(
                    from_agent=op.from_agent,
                    to_agent=op.to_agent,
                    edge_type=op.edge_type,
                )
            )
        elif op.op == PatchOp.REMOVE_EDGE:
            result.edges = [
                e
                for e in result.edges
                if not (e.from_agent == op.from_agent and e.to_agent == op.to_agent)
            ]
        elif op.op == PatchOp.ADD_VERIFIER:
            verifier_id = op.agent_id or "verifier"
            if verifier_id not in {a.id for a in result.agents}:
                result.agents.append(
                    AgentConfig(
                        id=verifier_id,
                        role="verify consistency and correctness",
                        budget=1,
                    )
                )
            if op.from_agent and op.from_agent in {a.id for a in result.agents}:
                result.edges.append(
                    EdgeConfig(
                        from_agent=op.from_agent,
                        to_agent=verifier_id,
                        edge_type=EdgeType.ANSWER_CHECK,
                    )
                )
            if op.to_agent and op.to_agent in {a.id for a in result.agents}:
                result.edges.append(
                    EdgeConfig(
                        from_agent=verifier_id,
                        to_agent=op.to_agent,
                        edge_type=EdgeType.ANSWER_CHECK,
                    )
                )
        elif op.op == PatchOp.CHANGE_AGGREGATION:
            result.aggregation = op.aggregation
        elif op.op == PatchOp.CHANGE_BUDGET:
            for a in result.agents:
                if a.id == op.agent_id:
                    if op.budget > 0:
                        a.budget = op.budget
                    else:
                        a.budget = a.budget + 1
        elif op.op == PatchOp.REPLACE_TOOL:
            for a in result.agents:
                if a.id == op.agent_id and op.tool_name:
                    if op.old_role and op.old_role in a.tools:
                        idx = a.tools.index(op.old_role)
                        a.tools[idx] = op.tool_name
                    else:
                        a.tools = [op.tool_name]
        elif op.op == PatchOp.EARLY_STOP_RULE:
            result.max_iterations = max(1, op.budget if op.budget else result.max_iterations)

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class PatchValidationError(Exception):
    """Raised when a patch is invalid for a given MAS."""

    pass


def validate_patch(mas: MASConfig, patch: Patch) -> list[str]:
    """Validate a patch against a MAS config, returning a list of issues.

    Returns an empty list if the patch is valid.
    """
    issues: list[str] = []

    for op in patch.operations:
        # Check agent existence for operations that reference an agent
        if op.op in (
            PatchOp.REMOVE_AGENT, PatchOp.REWRITE_ROLE, PatchOp.CHANGE_BUDGET,
            PatchOp.REPLACE_TOOL, PatchOp.MERGE_AGENTS,
        ):
            if op.agent_id and op.agent_id not in mas.agent_ids():
                issues.append(f"{op.op.value}: agent '{op.agent_id}' not found in MAS")

        if op.op == PatchOp.ADD_AGENT:
            if op.agent_id and op.agent_id in mas.agent_ids():
                issues.append(f"add_agent: agent '{op.agent_id}' already exists in MAS")

        if op.op == PatchOp.ADD_EDGE:
            if op.from_agent and op.from_agent not in mas.agent_ids():
                issues.append(f"add_edge: from_agent '{op.from_agent}' not found")
            if op.to_agent and op.to_agent not in mas.agent_ids():
                issues.append(f"add_edge: to_agent '{op.to_agent}' not found")

        if op.op == PatchOp.REMOVE_EDGE:
            if not any(
                e.from_agent == op.from_agent and e.to_agent == op.to_agent
                for e in mas.edges
            ):
                issues.append(
                    f"remove_edge: edge ({op.from_agent} -> {op.to_agent}) not found"
                )

        if op.op == PatchOp.ADD_VERIFIER:
            verifier_id = op.agent_id or "verifier"
            if verifier_id in mas.agent_ids():
                issues.append(f"add_verifier: verifier '{verifier_id}' already exists")

        if op.op == PatchOp.CHANGE_AGGREGATION:
            if op.aggregation and op.aggregation not in mas.agent_ids():
                issues.append(
                    f"change_aggregation: agent '{op.aggregation}' not found"
                )

        if op.op == PatchOp.CHANGE_BUDGET:
            if op.budget and op.budget < 1:
                issues.append(f"change_budget: budget must be >= 1, got {op.budget}")

        if op.op == PatchOp.REWRITE_ROLE:
            if not op.new_role and not op.agent_role:
                issues.append(f"rewrite_role: no new role provided for '{op.agent_id}'")

        if op.op == PatchOp.REPLACE_TOOL:
            if not op.tool_name:
                issues.append(f"replace_tool: no tool_name provided for '{op.agent_id}'")

    return issues


def validate_mas_config(mas: MASConfig) -> list[str]:
    """Validate a MAS config for structural correctness."""
    issues: list[str] = []

    if not mas.agents:
        issues.append("MAS must have at least one agent")
    if mas.aggregation and mas.aggregation not in mas.agent_ids():
        issues.append(f"aggregation agent '{mas.aggregation}' not found in agents")
    for e in mas.edges:
        if e.from_agent not in mas.agent_ids():
            issues.append(f"edge 'from' agent '{e.from_agent}' not found")
        if e.to_agent not in mas.agent_ids():
            issues.append(f"edge 'to' agent '{e.to_agent}' not found")
    if not mas.is_connected():
        issues.append("MAS graph is not connected")
    if mas.has_cycle():
        issues.append("MAS graph has a directed cycle")

    return issues