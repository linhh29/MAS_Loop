from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional

from .schema import AgentNode, GraphEdge, MASGraph, Topology, EdgeType, PatchOp, GraphPatchOp

_CLASS_TEMPLATE = """from __future__ import annotations
from typing import Any
import json
from template.sub_agent import SubAgentRequest, build_workflow_subagent_prompt

class {class_name}:
    def __init__(self, text_call_fn, tool_call_fn=None, **kw):
        self.text_call_fn = text_call_fn
        self.tool_call_fn = tool_call_fn
{agent_initializations}

    async def forward_async(self, task_text: str) -> dict[str, Any]:
        state: dict[str, Any] = {{}}
        task_info = task_text
{forward_body}
        return state
"""

def _generate_agent_init(node: AgentNode) -> str:
    user_prompt = "Read GLOBAL TASK CONTEXT for the full task, constraints, and required output format. Use UPSTREAM DEPENDENCIES DATA when present. Execute only the sub-task described in ROLE INSTRUCTION."
    execution_mode = node.execution_mode or "llm_only"
    role_text = node.role
    if node.constraints:
        constraints_text = "\n".join(f"- {c}" for c in node.constraints)
        role_text = f"{role_text}\n\n--- CONSTRAINTS ---\n{constraints_text}"
    return f"""        # Agent: {node.id}
        self.{node.id} = SubAgentRequest(
            role_name="{node.id}",
            role_instruction={json.dumps(role_text)},
            user_prompt={json.dumps(user_prompt)},
            tool_context={{"execution_mode": "{execution_mode}"}}
        )"""

def _generate_sequential_forward(nodes: list[AgentNode]) -> str:
    lines = ['        upstream_outputs: dict[str, str] = {}']
    lines.append("")
    for i, node in enumerate(nodes):
        lines.append(f"        # Agent {node.id}")
        if i == 0:
            lines.append(f"        prompt_{node.id} = build_workflow_subagent_prompt(self.{node.id}.role_instruction, self.{node.id}.user_prompt, task_info)")
        else:
            lines.append(f"        prompt_{node.id} = build_workflow_subagent_prompt(self.{node.id}.role_instruction, self.{node.id}.user_prompt, task_info, upstream_outputs)")
        lines.append(f"        result_{node.id} = await self.{node.id}.execute_async(text_call_fn=self.text_call_fn, tool_call_fn=self.tool_call_fn, prompt_override=prompt_{node.id})")
        lines.append(f'        state["out_{node.id}"] = result_{node.id}.text')
        lines.append(f'        state["usage_{node.id}"] = {{')
        lines.append(f'            "input_tokens": result_{node.id}.usage.get("input_tokens", 0),')
        lines.append(f'            "output_tokens": result_{node.id}.usage.get("output_tokens", 0),')
        lines.append(f'            "cost_usd": result_{node.id}.usage.get("estimated_cost_usd", 0.0),')
        lines.append(f"        }}")
        lines.append(f'        upstream_outputs["{node.id}"] = result_{node.id}.text')
        lines.append("")
    # Set final output — find the node marked is_final, not just the last one
    final = nodes[-1]
    for n in nodes:
        if n.is_final:
            final = n
            break
    lines.append(f'        state["final_output"] = state["out_{final.id}"]')
    lines.append(f'        state["usage_final"] = state["usage_{final.id}"]')
    return "\n".join(lines)

def _generate_dag_forward(nodes: list[AgentNode], edges: list[GraphEdge]) -> str:
    # Filter out edges with empty source/target (caused by editor bugs)
    valid_ids = {n.id for n in nodes}
    edges = [e for e in edges if e.f in valid_ids and e.t in valid_ids]
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in nodes}
    for e in edges:
        in_degree[e.t] = in_degree.get(e.t, 0) + 1
        adj.setdefault(e.f, []).append(e.t)
    queue = [nid for nid, d in in_degree.items() if d == 0]
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for to_id in adj.get(nid, []):
            in_degree[to_id] -= 1
            if in_degree[to_id] == 0:
                queue.append(to_id)
    for n in nodes:
        if n.id not in order:
            order.append(n.id)
    node_map = {n.id: n for n in nodes}
    lines = ['        upstream_outputs: dict[str, str] = {}']
    lines.append("")
    for nid in order:
        node = node_map[nid]
        lines.append(f"        # Agent {node.id}")
        is_final_node = node.is_final
        if is_final_node:
            # Final node receives ALL upstream outputs, not just direct edge dependencies
            lines.append(f"        upstream_{node.id} = dict(upstream_outputs)")
            lines.append(f"        prompt_{node.id} = build_workflow_subagent_prompt(self.{node.id}.role_instruction, self.{node.id}.user_prompt, task_info, upstream_{node.id})")
        else:
            lines.append(f"        upstream_{node.id} = {{}}")
            has_upstream = False
            for e in edges:
                if e.t == node.id:
                    lines.append(f'        upstream_{node.id}["{e.f}"] = state.get("out_{e.f}", "")')
                    has_upstream = True
            if has_upstream:
                lines.append(f"        prompt_{node.id} = build_workflow_subagent_prompt(self.{node.id}.role_instruction, self.{node.id}.user_prompt, task_info, upstream_{node.id})")
            else:
                lines.append(f"        prompt_{node.id} = build_workflow_subagent_prompt(self.{node.id}.role_instruction, self.{node.id}.user_prompt, task_info)")
        lines.append(f"        result_{node.id} = await self.{node.id}.execute_async(text_call_fn=self.text_call_fn, tool_call_fn=self.tool_call_fn, prompt_override=prompt_{node.id})")
        lines.append(f'        state["out_{node.id}"] = result_{node.id}.text')
        lines.append(f'        state["usage_{node.id}"] = {{')
        lines.append(f'            "input_tokens": result_{node.id}.usage.get("input_tokens", 0),')
        lines.append(f'            "output_tokens": result_{node.id}.usage.get("output_tokens", 0),')
        lines.append(f'            "cost_usd": result_{node.id}.usage.get("estimated_cost_usd", 0.0),')
        lines.append(f"        }}")
        lines.append(f'        upstream_outputs["{node.id}"] = result_{node.id}.text')
        lines.append("")
    final_node = None
    for n in nodes:
        if n.is_final:
            final_node = n
            break
    if not final_node:
        final_node = node_map[order[-1]]
    lines.append(f'        state["final_output"] = state["out_{final_node.id}"]')
    lines.append(f'        state["usage_final"] = state["usage_{final_node.id}"]')
    return "\n".join(lines)

def compile_graph_to_code(graph: MASGraph, class_name: str = "MASWorkflow") -> str:
    agent_inits = [_generate_agent_init(node) for node in graph.nodes]
    agent_init_block = "\n\n".join(agent_inits)
    if graph.topology == Topology.SEQUENTIAL or not graph.edges:
        forward_body = _generate_sequential_forward(graph.nodes)
    else:
        forward_body = _generate_dag_forward(graph.nodes, graph.edges)
    return _CLASS_TEMPLATE.format(class_name=class_name, agent_initializations=agent_init_block, forward_body=forward_body)

def compile_and_validate(graph: MASGraph, class_name: str = "MASWorkflow") -> tuple[str, bool, Optional[str]]:
    code = compile_graph_to_code(graph, class_name)
    try:
        import ast
        ast.parse(code)
        return code, True, None
    except SyntaxError as e:
        return code, False, str(e)

def apply_graph_patch(graph: MASGraph, patch, dataset: str = "aime") -> MASGraph:
    from .schema import GraphPatch
    new_graph = MASGraph(nodes=[n for n in graph.nodes], edges=[e for e in graph.edges], topology=graph.topology)
    if isinstance(patch, GraphPatch):
        ops = patch.operations
    elif hasattr(patch, 'operations'):
        ops = patch.operations
    else:
        ops = []
    for op in ops:
        if op.op == PatchOp.REWRITE_ROLE:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.role = str(op.value) if op.value else n.role
        elif op.op == PatchOp.ENHANCE_ROLE_DEPTH:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.role = n.role + "\n" + str(op.value) if op.value else n.role
        elif op.op == PatchOp.SHORTEN_ROLE:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.role = str(op.value) if op.value else n.role
        elif op.op == PatchOp.ADD_CONSTRAINT:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.constraints.append(str(op.value))
        elif op.op == PatchOp.REMOVE_CONSTRAINT:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.constraints = [c for c in n.constraints if c != str(op.value)]
        elif op.op == PatchOp.SET_FINAL:
            matched = False
            for n in new_graph.nodes:
                if n.id == op.target:
                    matched = True
            if matched:
                for n in new_graph.nodes:
                    n.is_final = (n.id == op.target)
            # If target doesn't match any node, skip (no-op)
        elif op.op == PatchOp.ADD_VERIFIER:
            verifier_id = f"v{op.target}" if op.target else "v1"
            counter = 1
            while new_graph.node_by_id(verifier_id):
                counter += 1
                verifier_id = f"v{op.target}_{counter}"
            # Inherit execution_mode from the graph's existing nodes
            inherit_mode = new_graph.nodes[0].execution_mode if new_graph.nodes else "llm_only"
            # Use value as role only if it looks like a real description (not just an ID)
            role_text = str(op.value) if op.value else ""
            if len(role_text) < 10 or role_text == str(op.target) or role_text == verifier_id:
                role_text = "Verify the result for correctness and output the final answer"
            verifier_node = AgentNode(
                id=verifier_id,
                role=role_text,
                constraints=[],
                is_final=True,
                execution_mode=inherit_mode,
            )
            for n in new_graph.nodes:
                if n.is_final:
                    n.is_final = False
            new_graph.nodes.append(verifier_node)
            new_graph.edges.append(GraphEdge(f=op.target, t=verifier_id, type=EdgeType.VERIFY))
        elif op.op == PatchOp.REMOVE_AGENT:
            new_graph.nodes = [n for n in new_graph.nodes if n.id != op.target]
            new_graph.edges = [e for e in new_graph.edges if e.f != op.target and e.t != op.target]
        elif op.op == PatchOp.ADD_AGENT:
            new_id = f"a{len(new_graph.nodes)}"
            inherit_mode = new_graph.nodes[0].execution_mode if new_graph.nodes else "llm_only"
            new_node = AgentNode(id=new_id, role=str(op.value) if op.value else "New agent", execution_mode=inherit_mode)
            new_graph.nodes.append(new_node)
            if op.target:
                new_graph.edges.append(GraphEdge(f=op.target, t=new_id, type=EdgeType.DATA))
        elif op.op == PatchOp.ADD_EDGE:
            parts = op.target.split(">") if ">" in (op.target or "") else [op.target, ""]
            if parts[0] and len(parts) > 1 and parts[1]:
                new_graph.edges.append(GraphEdge(f=parts[0], t=parts[1], type=EdgeType.DATA))
        elif op.op == PatchOp.REMOVE_EDGE:
            new_graph.edges = [e for e in new_graph.edges if not (e.f == op.target)]
        elif op.op == PatchOp.CHANGE_TOPOLOGY:
            new_graph.topology = Topology(str(op.value)) if op.value else new_graph.topology
        elif op.op == PatchOp.ADD_CONSISTENCY_CHECK:
            target = op.target if op.target else new_graph.nodes[-1].id
            for n in new_graph.nodes:
                if n.id == target:
                    n.constraints.append(f"Consistency check: {op.value}")
        elif op.op == PatchOp.ADD_FALLBACK:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.constraints.append(f"Fallback: {op.value}")
        elif op.op == PatchOp.ADD_RETRY_LOGIC:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.constraints.append(f"Retry: {op.value}")
        elif op.op == PatchOp.FIX_OUTPUT_FORMAT:
            for n in new_graph.nodes:
                if n.id == op.target:
                    n.constraints.append(f"Output format: {op.value}")
        elif op.op == PatchOp.CHANGE_AGGREGATION:
            pass
        elif op.op == PatchOp.REORDER_AGENTS:
            if op.value:
                new_order = [x.strip() for x in str(op.value).split(",")]
                node_map = {n.id: n for n in new_graph.nodes}
                new_graph.nodes = [node_map[nid] for nid in new_order if nid in node_map]
    # Post-processing: add dataset-aware constraints to ALL final nodes
    for n in new_graph.nodes:
        if n.is_final:
            n.constraints.append("CRITICAL: After completing your own task, you MUST synthesize ALL upstream content into a single, complete final output. Do NOT output only a verification report or summary — include the full upstream content in your final output along with your own contribution.")
            if dataset == "aime":
                n.constraints.append("Output the final answer in \\boxed{answer} format.")
            elif dataset == "bcp":
                n.constraints.append("Output ONLY the final answer string, no extra explanation or formatting.")
            break
    return new_graph