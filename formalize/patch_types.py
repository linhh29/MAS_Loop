from __future__ import annotations
from enum import Enum

class PatchCategory(str, Enum):
    SIMPLIFY = "simplify"
    REPAIR = "repair"
    UPGRADE = "upgrade"
    COMPRESS = "compress"
    REALLOCATE = "reallocate"
    CONSTRAIN = "constrain"
    STABILIZE = "stabilize"

OPERATION_REGISTRY = {
    "remove_agent": {"category": PatchCategory.SIMPLIFY, "desc": "Remove a redundant agent", "risk": "medium", "requires_trace": False, "target_format": "agent_id", "value_format": "reason"},
    "merge_agents": {"category": PatchCategory.SIMPLIFY, "desc": "Merge two agents into one", "risk": "medium", "requires_trace": False, "target_format": "agent_id,agent_id", "value_format": "merged role description"},
    "remove_edge": {"category": PatchCategory.SIMPLIFY, "desc": "Remove an unnecessary edge", "risk": "low", "requires_trace": True, "target_format": "edge_id", "value_format": "reason"},
    "remove_constraint": {"category": PatchCategory.SIMPLIFY, "desc": "Remove an overly restrictive constraint", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "constraint text to remove"},
    "rewrite_role": {"category": PatchCategory.REPAIR, "desc": "Rewrite an agent's role instruction", "risk": "low", "requires_trace": True, "target_format": "agent_id", "value_format": "complete new role text"},
    "add_edge": {"category": PatchCategory.REPAIR, "desc": "Add a missing dependency edge", "risk": "low", "requires_trace": True, "target_format": "from_id>to_id", "value_format": "edge type: data|control|feedback"},
    "change_topology": {"category": PatchCategory.REPAIR, "desc": "Change graph topology", "risk": "high", "requires_trace": False, "target_format": "graph", "value_format": "sequential|dag"},
    "fix_output_format": {"category": PatchCategory.REPAIR, "desc": "Fix output format issues", "risk": "low", "requires_trace": True, "target_format": "agent_id", "value_format": "format specification"},
    "add_agent": {"category": PatchCategory.UPGRADE, "desc": "Add a new agent to the workflow", "risk": "medium", "requires_trace": False, "target_format": "after which agent", "value_format": "new agent role description"},
    "add_verifier": {"category": PatchCategory.UPGRADE, "desc": "Add a verification agent that becomes the FINAL agent: receives upstream output, verifies correctness, and outputs the final \\boxed{answer} (corrected if wrong)", "risk": "low", "requires_trace": False, "target_format": "agent_id to verify (target node's output is verified, then verifier becomes final)", "value_format": "verifier role text — must include 'output the final answer in \\boxed{} format'"},
    "add_constraint": {"category": PatchCategory.UPGRADE, "desc": "Add a hard constraint to an agent", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "constraint text"},
    "add_tool": {"category": PatchCategory.UPGRADE, "desc": "Give an agent a new tool capability", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "tool name and description"},
    "replace_tool": {"category": PatchCategory.UPGRADE, "desc": "Replace an agent's tool", "risk": "medium", "requires_trace": True, "target_format": "agent_id", "value_format": "new tool description"},
    "enhance_role_depth": {"category": PatchCategory.UPGRADE, "desc": "Enhance agent role with more detailed instructions", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "additional role details"},
    "reduce_iterations": {"category": PatchCategory.COMPRESS, "desc": "Reduce planning/thinking iterations", "risk": "medium", "requires_trace": False, "target_format": "agent_id", "value_format": "max iterations"},
    "change_budget": {"category": PatchCategory.COMPRESS, "desc": "Change token/compute budget", "risk": "medium", "requires_trace": True, "target_format": "agent_id", "value_format": "new budget"},
    "early_stop_rule": {"category": PatchCategory.COMPRESS, "desc": "Add early stopping condition", "risk": "medium", "requires_trace": False, "target_format": "agent_id", "value_format": "stop condition"},
    "shorten_role": {"category": PatchCategory.COMPRESS, "desc": "Shorten an agent's role instruction", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "shortened role text"},
    "set_final": {"category": PatchCategory.REALLOCATE, "desc": "Change which agent is the final output", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "reason"},
    "change_aggregation": {"category": PatchCategory.REALLOCATE, "desc": "Change output aggregation method", "risk": "medium", "requires_trace": False, "target_format": "agent_id", "value_format": "aggregation method (verifier_final|last_agent|majority_vote)"},
    "split_agent": {"category": PatchCategory.REALLOCATE, "desc": "Split an agent into multiple", "risk": "high", "requires_trace": True, "target_format": "agent_id", "value_format": "split specification"},
    "reorder_agents": {"category": PatchCategory.REALLOCATE, "desc": "Reorder agent execution sequence", "risk": "low", "requires_trace": False, "target_format": "graph", "value_format": "new order: a1,a3,a2,..."},
    "enforce_token_limit": {"category": PatchCategory.CONSTRAIN, "desc": "Enforce a token limit on an agent", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "max tokens"},
    "enforce_step_limit": {"category": PatchCategory.CONSTRAIN, "desc": "Enforce a step limit on an agent", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "max steps"},
    "add_fallback": {"category": PatchCategory.STABILIZE, "desc": "Add fallback behavior for failures", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "fallback description"},
    "add_retry_logic": {"category": PatchCategory.STABILIZE, "desc": "Add retry logic to an agent", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "max retries, retry condition"},
    "add_consistency_check": {"category": PatchCategory.STABILIZE, "desc": "Add a consistency check between agents", "risk": "low", "requires_trace": False, "target_format": "agent_id", "value_format": "check description"},
}

def get_operations_by_category(cat: PatchCategory) -> list[dict]:
    return [{"name": k, **v} for k, v in OPERATION_REGISTRY.items() if v["category"] == cat]

def get_operation_meta(name: str) -> dict:
    return OPERATION_REGISTRY.get(name, {})

def format_operations_for_prompt() -> str:
    lines = []
    for cat in PatchCategory:
        ops = get_operations_by_category(cat)
        lines.append(f"\n  **{cat.value}** ({cat.name}):")
        for op in ops:
            lines.append(f"    - {op['name']}: {op['desc']}")
    return "\n".join(lines)

def format_operations_for_xml() -> str:
    lines = []
    for cat in PatchCategory:
        ops = get_operations_by_category(cat)
        lines.append(f"  <!-- {cat.value} -->")
        for op in ops:
            lines.append(f"  <operation op=\"{op['name']}\" target=\"agent_id\">{op['value_format']}</operation>")
    return "\n".join(lines)