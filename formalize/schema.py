from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import re

class Topology(str, Enum):
    SEQUENTIAL = "sequential"
    DAG = "dag"

class EdgeType(str, Enum):
    DATA = "data"
    CONTROL = "control_flow"
    FEEDBACK = "feedback"
    VERIFY = "verify"

class PatchOp(str, Enum):
    ADD_AGENT = "add_agent"
    REMOVE_AGENT = "remove_agent"
    MERGE_AGENTS = "merge_agents"
    SPLIT_AGENT = "split_agent"
    REWRITE_ROLE = "rewrite_role"
    ENHANCE_ROLE_DEPTH = "enhance_role_depth"
    SHORTEN_ROLE = "shorten_role"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    ADD_VERIFIER = "add_verifier"
    ADD_TOOL = "add_tool"
    REPLACE_TOOL = "replace_tool"
    ADD_FALLBACK = "add_fallback"
    ADD_RETRY_LOGIC = "add_retry_logic"
    ADD_CONSISTENCY_CHECK = "add_consistency_check"
    SET_FINAL = "set_final"
    CHANGE_AGGREGATION = "change_aggregation"
    CHANGE_TOPOLOGY = "change_topology"
    REORDER_AGENTS = "reorder_agents"
    FIX_OUTPUT_FORMAT = "fix_output_format"
    REDUCE_ITERATIONS = "reduce_iterations"
    CHANGE_BUDGET = "change_budget"
    EARLY_STOP_RULE = "early_stop_rule"
    ENFORCE_TOKEN_LIMIT = "enforce_token_limit"
    ENFORCE_STEP_LIMIT = "enforce_step_limit"

@dataclass
class SubTask:
    idx: int
    desc: str
    dependencies: list[str] = field(default_factory=list)

@dataclass
class TaskDecomposition:
    goal: str
    sub_tasks: list[SubTask] = field(default_factory=list)

@dataclass
class AgentSpec:
    id: str
    role: str
    sub_task_idx: int = 0
    dependencies: list[str] = field(default_factory=list)
    is_final: bool = False
    constraints: list[str] = field(default_factory=list)

@dataclass
class AgentSpecs:
    specs: list[AgentSpec] = field(default_factory=list)

@dataclass
class AgentNode:
    id: str
    role: str
    task: str = ""
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    is_final: bool = False
    execution_mode: str = "llm_only"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "role": self.role, "task": self.task, "goal": self.goal,
            "constraints": self.constraints, "is_final": self.is_final,
            "execution_mode": self.execution_mode, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentNode:
        return cls(
            id=d.get("id", ""), role=d.get("role", ""), task=d.get("task", ""),
            goal=d.get("goal", ""), constraints=d.get("constraints", []),
            is_final=d.get("is_final", False),
            execution_mode=d.get("execution_mode", "llm_only"), meta=d.get("meta", {}),
        )

    def to_xml(self) -> str:
        cs = "".join(f"<constraint>{c}</constraint>" for c in self.constraints)
        return f"<agent id=\"{self.id}\" is_final=\"{str(self.is_final).lower()}\" execution_mode=\"{self.execution_mode}\"><role>{self.role}</role><task>{self.task}</task><goal>{self.goal}</goal><constraints>{cs}</constraints></agent>"

    @classmethod
    def from_xml(cls, elem: dict) -> AgentNode:
        def _text(v):
            if isinstance(v, dict) and "#text" in v: return v["#text"]
            return str(v) if v else ""
        return cls(
            id=elem.get("@id", ""), role=_text(elem.get("role", "")),
            task=_text(elem.get("task", "")), goal=_text(elem.get("goal", "")),
            constraints=[_text(c) for c in (elem.get("constraints", {}).get("constraint", []) if isinstance(elem.get("constraints", {}), dict) else [])] if isinstance(elem.get("constraints"), dict) else (elem.get("constraints", []) if isinstance(elem.get("constraints"), list) else []),
            is_final=elem.get("@is_final", "false") == "true",
            execution_mode=elem.get("@execution_mode", "llm_only"),
        )

@dataclass
class GraphEdge:
    f: str
    t: str
    type: EdgeType = EdgeType.DATA

    def to_dict(self) -> dict[str, Any]:
        return {"f": self.f, "t": self.t, "type": self.type.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphEdge:
        return cls(f=d.get("f", ""), t=d.get("t", ""), type=EdgeType(d.get("type", "data")))

    def to_xml(self) -> str:
        return f"<edge f=\"{self.f}\" t=\"{self.t}\" type=\"{self.type.value}\"/>"

    @classmethod
    def from_xml(cls, elem: dict) -> GraphEdge:
        return cls(f=elem.get("@f", ""), t=elem.get("@t", ""), type=EdgeType(elem.get("@type", "data")))

@dataclass
class MASGraph:
    nodes: list[AgentNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    topology: Topology = Topology.SEQUENTIAL

    def node_by_id(self, nid: str) -> Optional[AgentNode]:
        for n in self.nodes:
            if n.id == nid: return n
        return None

    def final_node(self) -> Optional[AgentNode]:
        for n in self.nodes:
            if n.is_final: return n
        return self.nodes[-1] if self.nodes else None

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes], "edges": [e.to_dict() for e in self.edges], "topology": self.topology.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MASGraph:
        return cls(
            nodes=[AgentNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[GraphEdge.from_dict(e) for e in d.get("edges", [])],
            topology=Topology(d.get("topology", "sequential")),
        )

    def to_xml(self) -> str:
        ns = "".join(n.to_xml() for n in self.nodes)
        es = "".join(e.to_xml() for e in self.edges)
        return f"<mas_graph topology=\"{self.topology.value}\"><nodes>{ns}</nodes><edges>{es}</edges></mas_graph>"

    @classmethod
    def from_xml(cls, elem: dict) -> MASGraph:
        nodes = []
        if isinstance(elem.get("nodes"), dict):
            nlist = elem["nodes"].get("agent", [])
            if isinstance(nlist, dict): nlist = [nlist]
            nodes = [AgentNode.from_xml(n) for n in nlist]
        edges = []
        if isinstance(elem.get("edges"), dict):
            elist = elem["edges"].get("edge", [])
            if isinstance(elist, dict): elist = [elist]
            edges = [GraphEdge.from_xml(e) for e in elist]
        return cls(nodes=nodes, edges=edges, topology=Topology(elem.get("@topology", "sequential")))

@dataclass
class GraphPatchOp:
    op: PatchOp
    target: str = ""
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op.value, "target": self.target, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphPatchOp:
        return cls(op=PatchOp(d.get("op", "")), target=d.get("target", ""), value=d.get("value"))

    def to_xml(self) -> str:
        v = str(self.value) if self.value else ""
        return f"<operation op=\"{self.op.value}\" target=\"{self.target}\">{v}</operation>"

    @classmethod
    def from_xml(cls, elem: dict) -> GraphPatchOp:
        op = PatchOp(elem.get("@op", ""))
        target = elem.get("@target", "")
        v = elem.get("#text", "") if isinstance(elem, dict) else ""
        return cls(op=op, target=target, value=v)

@dataclass
class GraphPatch:
    diagnosis: dict[str, Any] = field(default_factory=dict)
    operations: list[GraphPatchOp] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.operations) == 0

    def to_dict(self) -> dict[str, Any]:
        return {"diagnosis": self.diagnosis, "ops": [o.to_dict() for o in self.operations]}

    def to_xml(self) -> str:
        diag = json.dumps(self.diagnosis, ensure_ascii=False)
        ops = "".join(o.to_xml() for o in self.operations)
        return f"<editor_output><diagnosis>{diag}</diagnosis><operations>{ops}</operations></editor_output>"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphPatch:
        diagnosis = d.get("diagnosis", {})
        ops_raw = d.get("operations", d.get("ops", []))
        ops = [GraphPatchOp.from_dict(o) if isinstance(o, dict) and "op" in o else GraphPatchOp.from_xml(o) for o in ops_raw]
        return cls(diagnosis=diagnosis, operations=ops)

    @classmethod
    def from_xml(cls, elem: dict) -> GraphPatch:
        diagnosis = {}
        if isinstance(elem.get("diagnosis"), dict):
            diag = elem["diagnosis"]
            if "#text" in diag:
                try: diagnosis = json.loads(diag["#text"])
                except: diagnosis = {"raw": diag["#text"]}
            else:
                diagnosis = {k: (v.get("#text", v) if isinstance(v, dict) else v) for k, v in diag.items()}
        ops = []
        if isinstance(elem.get("operations"), dict):
            olist = elem["operations"].get("operation", [])
            if isinstance(olist, dict): olist = [olist]
            ops = [GraphPatchOp.from_xml(o) for o in olist]
        return cls(diagnosis=diagnosis, operations=ops)

def extract_xml(text: str) -> dict[str, Any]:
    """Extract and parse XML from LLM response using xml.etree.ElementTree."""
    import xml.etree.ElementTree as ET
    m = re.search(r"<(editor_output|mas_graph|task_decomposition|agent_specs)[^>]*>[\s\S]*?</\1>", text, re.DOTALL)
    if not m:
        return {}
    try:
        root = ET.fromstring(m.group(0))
        return {root.tag: _etree_to_dict(root)}
    except ET.ParseError:
        return {}

def _etree_to_dict(element) -> dict[str, Any]:
    """Convert an ElementTree element to a dict compatible with our from_xml methods."""
    result = {}
    for k, v in element.attrib.items():
        result[f"@{k}"] = v
    text = (element.text or "").strip()
    if text and not len(element):
        result["#text"] = text
    for child in element:
        ctag = child.tag
        child_dict = _etree_to_dict(child)
        if ctag in result:
            if not isinstance(result[ctag], list):
                result[ctag] = [result[ctag]]
            result[ctag].append(child_dict)
        else:
            result[ctag] = child_dict
    return result