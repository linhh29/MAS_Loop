from __future__ import annotations

import dataclasses
import re
from typing import Any

from build_task_filtered_rm import mas_to_orchestra_xml


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value): return {k: to_jsonable(v) for k,v in dataclasses.asdict(value).items()}
    if isinstance(value, dict): return {str(k): to_jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


def artifacts_to_mas(artifacts: Any, mas_code: str) -> dict[str, Any]:
    raw_agents = list(getattr(artifacts, "normalized_sub_agents", []) or []) if artifacts else []
    agents = []
    for idx, item in enumerate(raw_agents):
        item = dict(item)
        role_name = str(item.get("role_name") or f"agent_{idx}")
        tc = item.get("tool_context") or {}
        agents.append({
            "id": role_name,
            "role": item.get("role_instruction", ""),
            "task": item.get("user_prompt", ""),
            "tools": ["bm25"] if str(tc.get("execution_mode", "")) == "multi_turn_search" else [],
            "execution_mode": tc.get("execution_mode", "llm_only"),
            "budget": item.get("max_tokens", ""),
        })
    ids = {x["id"] for x in agents}
    edges: list[dict[str, str]] = []
    # Generated templates wire upstream state outputs as state["out_ROLE"].
    for target in ids:
        target_pos = mas_code.find(f'out_{target}')
        prefix = mas_code[:target_pos] if target_pos >= 0 else mas_code
        for source in ids:
            if source != target and re.search(rf'out_{re.escape(source)}\b', prefix[-5000:]):
                edge = {"from": source, "to": target, "type": "data_flow"}
                if edge not in edges: edges.append(edge)
    return {"topology": "generated_dag", "agents": agents, "edges": edges}


def state_to_trajectory(state: dict[str, Any], retrieval_sessions: list[dict[str, Any]], final_output: str, success: bool) -> dict[str, Any]:
    traces = []
    session_by_role = {str(x.get("role_name", "")): x for x in retrieval_sessions}
    for key, output in state.items():
        if not str(key).startswith("out_"): continue
        aid = str(key)[4:]
        session = session_by_role.get(aid, {})
        traces.append({
            "agent_id": aid, "agent_role": aid, "output": str(output or ""),
            "search_mode": "bm25" if session else "",
            "search_rounds": session.get("trajectory", []),
            "error": "",
        })
    return {"agent_traces": traces, "final_output": final_output, "success": success}


def rm_text(question: str, mas_xml: str, *, trajectory: bool) -> str:
    label = "Candidate MAS with per-agent execution:" if trajectory else "Candidate MAS:"
    return f"Task:\n{question}\n\n{label}\n{mas_xml}"


def make_xmls(question: str, mas: dict[str, Any], trajectory: dict[str, Any]) -> tuple[str, str, str, str]:
    orch_xml = mas_to_orchestra_xml(mas)
    traj_xml = mas_to_orchestra_xml(mas, trajectory, include_retrieval=True)
    return orch_xml, traj_xml, rm_text(question, orch_xml, trajectory=False), rm_text(question, traj_xml, trajectory=True)
