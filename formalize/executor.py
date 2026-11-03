from __future__ import annotations
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

_BUILDING_ROOT = Path(__file__).resolve().parents[1] / "building"
if str(_BUILDING_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUILDING_ROOT))

from .schema import MASGraph
from .compiler import compile_graph_to_code

async def execute_mas_graph(graph: MASGraph, text_call_fn, tool_call_fn=None, class_name: str = "MASWorkflow") -> dict[str, Any]:
    code = compile_graph_to_code(graph, class_name)
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return {"success": False, "final_output": "", "agent_traces": {}, "costs": {},
                "error": f"Compile error: {e}", "traceback": traceback.format_exc()}
    cls = namespace.get(class_name)
    if cls is None:
        return {"success": False, "final_output": "", "error": f"Class {class_name} not found in compiled code"}
    instance = cls(text_call_fn=text_call_fn, tool_call_fn=tool_call_fn)
    task_text = ""
    for node in graph.nodes:
        if node.goal:
            task_text = node.goal
            break
    if not task_text:
        for node in graph.nodes:
            if node.task:
                task_text = node.task
                break
    try:
        out = await instance.forward_async(task_text or "Execute the task")
    except Exception as e:
        return {"success": False, "final_output": "", "agent_traces": {}, "costs": {},
                "error": str(e), "traceback": traceback.format_exc()}
    final_output = out.get("final_output", "")
    success = bool(final_output) and len(final_output) > 10
    agent_traces = {}
    costs = {}
    total_tokens = 0
    total_cost = 0.0
    for node in graph.nodes:
        key = f"out_{node.id}"
        if key in out:
            agent_traces[node.id] = out[key]
        usage_key = f"usage_{node.id}"
        if usage_key in out:
            u = out[usage_key]
            costs[node.id] = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                              "cost_usd": u.get("cost_usd", 0.0)}
            total_tokens += u.get("input_tokens", 0) + u.get("output_tokens", 0)
            total_cost += u.get("cost_usd", 0.0)
    return {"success": success, "final_output": final_output, "agent_traces": agent_traces,
            "costs": costs, "total_tokens": total_tokens, "total_cost_usd": total_cost, "error": None}

async def execute_graph_and_score(graph: MASGraph, text_call_fn, expected: str, class_name: str = "MASWorkflow") -> dict[str, Any]:
    result = await execute_mas_graph(graph, text_call_fn, class_name)
    final_output = result.get("final_output", "")
    import re
    m = re.search(r"\\boxed\{([^}]+)\}", final_output or "")
    predicted = m.group(1).strip() if m else ""
    score = 1.0 if predicted.strip() == expected.strip() else 0.0
    result["predicted"] = predicted
    result["score"] = score
    return result