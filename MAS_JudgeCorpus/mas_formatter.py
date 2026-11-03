"""MAS graph to text representation formatters.

This module provides multiple text representations of MAS graph structures
for training a judge/value model. The key idea is to capture the structural
properties of the MAS graph (topology, nodes, edges, roles, constraints)
in a format that an LLM can reason about for scoring.

Three format types are provided:
- "detailed": Full human-readable text with all node/edge details.
- "compact": Concise key-value format with structural statistics.
- "json": Raw JSON representation of the graph structure.
"""

from __future__ import annotations
from typing import Any, Literal

MAS_FORMAT_TYPE = Literal["detailed", "compact", "json"]


def format_mas_to_text(
    mas: dict[str, Any],
    fmt: MAS_FORMAT_TYPE = "detailed",
    max_role_length: int = 200,
) -> str:
    """Format a MAS graph to a text representation.

    Args:
        mas: MAS graph dict with "nodes", "edges", "topology" keys.
        fmt: Format type: "detailed", "compact", or "json".
        max_role_length: Maximum length of role/task text in detailed format.

    Returns:
        Formatted text string.
    """
    if fmt == "json":
        return _format_json(mas, max_role_length)
    elif fmt == "compact":
        return _format_compact(mas, max_role_length)
    else:
        return _format_detailed(mas, max_role_length)


def _format_detailed(mas: dict[str, Any], max_role_length: int) -> str:
    """Full detailed text representation of a MAS graph."""
    nodes = mas.get("nodes", [])
    edges = mas.get("edges", [])
    topology = mas.get("topology", "unknown")

    lines = []
    lines.append("=" * 60)
    lines.append("MAS GRAPH STRUCTURE")
    lines.append("=" * 60)

    # Topology info
    lines.append(f"Topology: {topology}")
    lines.append(f"Total Agents: {len(nodes)}")
    lines.append(f"Total Edges: {len(edges)}")

    # Structural statistics
    stats = _compute_stats(mas)
    lines.append(f"Max Depth: {stats['max_depth']}")
    lines.append(f"Max Branching Factor: {stats['max_branching']}")
    lines.append(f"Final Agents: {stats['final_agent_count']}")
    lines.append(f"Verify Edges: {stats['verify_edge_count']}")

    # Execution modes
    exec_modes = set(n.get("execution_mode", "llm_only") for n in nodes)
    lines.append(f"Execution Modes: {', '.join(sorted(exec_modes)) if exec_modes else 'llm_only'}")

    # Agent connectivity
    lines.append(f"Root Agents: {', '.join(stats['root_agents']) if stats['root_agents'] else 'none'}")
    lines.append(f"Leaf Agents: {', '.join(stats['leaf_agents']) if stats['leaf_agents'] else 'none'}")

    lines.append("")

    # Node details
    lines.append(f"AGENTS ({len(nodes)}):")
    lines.append("-" * 40)
    for i, node in enumerate(nodes):
        node_id = node.get("id", f"unknown_{i}")
        role = node.get("role", "")
        task = node.get("task", "")
        goal = node.get("goal", "")
        constraints = node.get("constraints", [])
        is_final = node.get("is_final", False)
        exec_mode = node.get("execution_mode", "llm_only")

        lines.append(f"Agent [{node_id}]:")
        lines.append(f"  Role: {_truncate(role, max_role_length)}")
        if task and task != role:
            lines.append(f"  Task: {_truncate(task, max_role_length)}")
        if goal:
            lines.append(f"  Goal: {_truncate(goal, max_role_length)}")
        if exec_mode != "llm_only":
            lines.append(f"  Execution: {exec_mode}")
        if is_final:
            lines.append(f"  Status: FINAL OUTPUT")
        if constraints:
            lines.append(f"  Constraints ({len(constraints)}):")
            for c in constraints:
                lines.append(f"    - {_truncate(c, max_role_length)}")
        lines.append("")

    # Edge details
    lines.append(f"DEPENDENCIES ({len(edges)}):")
    lines.append("-" * 40)

    # Group edges by type
    data_edges = [e for e in edges if e.get("type") == "data"]
    verify_edges = [e for e in edges if e.get("type") == "verify"]

    if data_edges:
        lines.append("Data Flow:")
        for e in data_edges:
            lines.append(f"  {e['f']} -> {e['t']}")

    if verify_edges:
        lines.append("Verification Flow:")
        for e in verify_edges:
            lines.append(f"  {e['f']} -> {e['t']}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _format_compact(mas: dict[str, Any], max_role_length: int) -> str:
    """Compact key-value text representation of a MAS graph."""
    nodes = mas.get("nodes", [])
    edges = mas.get("edges", [])
    topology = mas.get("topology", "unknown")
    stats = _compute_stats(mas)

    lines = []
    lines.append(f"MAS(topology={topology}, agents={len(nodes)}, edges={len(edges)}, "
                 f"depth={stats['max_depth']}, branching={stats['max_branching']}, "
                 f"final_agents={stats['final_agent_count']}, verify_edges={stats['verify_edge_count']})")

    lines.append("Agents:")
    for node in nodes:
        flags = "FINAL" if node.get("is_final") else ""
        constraints_count = len(node.get("constraints", []))
        cstr = f" constraints={constraints_count}" if constraints_count else ""
        lines.append(f"  [{node['id']}]{flags} role: {_truncate(node.get('role', ''), max_role_length)}{cstr}")

    lines.append("Edges:")
    for edge in edges:
        lines.append(f"  {edge['f']} -> {edge['t']} [{edge.get('type', 'data')}]")

    return "\n".join(lines)


def _format_json(mas: dict[str, Any], max_role_length: int) -> str:
    """JSON representation of the MAS graph (truncated for readability)."""
    import json
    # Make a lightweight copy with truncated text fields
    lightweight = {"topology": mas.get("topology", "unknown")}
    lightweight["nodes"] = []
    for node in mas.get("nodes", []):
        lightweight["nodes"].append({
            "id": node.get("id", ""),
            "role": _truncate(node.get("role", ""), max_role_length),
            "is_final": node.get("is_final", False),
            "constraints_count": len(node.get("constraints", [])),
        })
    lightweight["edges"] = [
        {"f": e["f"], "t": e["t"], "type": e.get("type", "data")}
        for e in mas.get("edges", [])
    ]
    return json.dumps(lightweight, ensure_ascii=False, indent=2)


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _compute_stats(mas: dict[str, Any]) -> dict[str, Any]:
    """Compute structural statistics of a MAS graph.

    Args:
        mas: MAS graph dict with nodes, edges, topology.

    Returns:
        Dict with structural statistics.
    """
    nodes = mas.get("nodes", [])
    edges = mas.get("edges", [])
    node_ids = {n["id"] for n in nodes}

    # Build adjacency
    adjacency: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    out_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    verify_edge_count = 0

    for edge in edges:
        f, t = edge["f"], edge["t"]
        if f in adjacency and t in node_ids:
            adjacency[f].append(t)
            out_degree[f] += 1
            in_degree[t] = in_degree.get(t, 0) + 1
            if edge.get("type") == "verify":
                verify_edge_count += 1

    # Root agents (no incoming edges, or incoming from outside)
    root_agents = [nid for nid in node_ids if in_degree.get(nid, 0) == 0]

    # Leaf agents (no outgoing edges)
    leaf_agents = [nid for nid in node_ids if out_degree.get(nid, 0) == 0]

    # Max depth via BFS from root agents
    max_depth = 0
    if root_agents:
        visited = set()
        # (node_id, depth)
        queue = [(r, 0) for r in root_agents]
        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            max_depth = max(max_depth, depth)
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

    max_branching = max(out_degree.values()) if out_degree else 0

    final_agent_count = sum(1 for n in nodes if n.get("is_final"))

    return {
        "root_agents": root_agents,
        "leaf_agents": leaf_agents,
        "max_depth": max_depth,
        "max_branching": max_branching,
        "final_agent_count": final_agent_count,
        "verify_edge_count": verify_edge_count,
    }


def format_mas_with_task(
    mas: dict[str, Any],
    task_text: str,
    fmt: MAS_FORMAT_TYPE = "detailed",
    max_role_length: int = 200,
) -> str:
    """Format a MAS graph together with the task description.

    Args:
        mas: MAS graph dict.
        task_text: The task description.
        fmt: Format type for the MAS graph.
        max_role_length: Maximum role/task text length.

    Returns:
        Formatted prompt string.
    """
    parts = []
    parts.append("TASK DESCRIPTION:")
    parts.append("-" * 40)
    parts.append(task_text)
    parts.append("")
    parts.append(format_mas_to_text(mas, fmt=fmt, max_role_length=max_role_length))
    return "\n".join(parts)


def format_mas_pair_with_task(
    mas_a: dict[str, Any],
    mas_b: dict[str, Any],
    task_text: str,
    fmt: MAS_FORMAT_TYPE = "detailed",
    max_role_length: int = 200,
) -> str:
    """Format two MAS graphs with a task for pairwise comparison.

    Args:
        mas_a: First MAS graph.
        mas_b: Second MAS graph.
        task_text: The task description.
        fmt: Format type for MAS graphs.
        max_role_length: Maximum role/task text length.

    Returns:
        Formatted comparison prompt string.
    """
    parts = []
    parts.append("TASK DESCRIPTION:")
    parts.append("-" * 40)
    parts.append(task_text)
    parts.append("")
    parts.append("MAS A:")
    parts.append(format_mas_to_text(mas_a, fmt=fmt, max_role_length=max_role_length))
    parts.append("")
    parts.append("MAS B:")
    parts.append(format_mas_to_text(mas_b, fmt=fmt, max_role_length=max_role_length))
    parts.append("")
    parts.append("Which MAS is better designed for this task? Answer with 'MAS A' or 'MAS B'.")
    return "\n".join(parts)