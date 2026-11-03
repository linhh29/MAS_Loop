from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from building.skill_mas.openai_async_client import AsyncOpenAIClient
from .schema import MASGraph, GraphPatch, GraphPatchOp, PatchOp, extract_xml
from .prompts import EDITOR_SYSTEM, EDITOR_TEMPLATE, EDITOR_MINIMAL_TEMPLATE

class Editor:
    def __init__(self, client: AsyncOpenAIClient):
        self.client = client

    async def edit(self, graph: MASGraph, task_text: str, expected: str, actual_score: float, final_output: str = "", agent_traces: dict[str, str] | None = None) -> tuple[GraphPatch, dict[str, Any]]:
        graph_xml = graph.to_xml()
        traces_str = ""
        if agent_traces:
            traces_str = "\n".join(f"[{aid}]:\n{out}" for aid, out in agent_traces.items())
        user_prompt = EDITOR_TEMPLATE.format(task_text=task_text, expected=expected, graph_xml=graph_xml, actual_score=actual_score, final_output=final_output, agent_traces=traces_str)
        return await self._do_edit(user_prompt)

    async def edit_minimal(self, graph: MASGraph, task_text: str, expected: str) -> tuple[GraphPatch, dict[str, Any]]:
        user_prompt = EDITOR_MINIMAL_TEMPLATE.format(task_text=task_text, expected=expected, graph_xml=graph.to_xml())
        return await self._do_edit(user_prompt)

    async def _do_edit(self, user_prompt: str) -> tuple[GraphPatch, dict[str, Any]]:
        text, usage = await self.client.generate(user_prompt=user_prompt, system_prompt=EDITOR_SYSTEM)
        data = extract_xml(text)
        if "editor_output" in data:
            return GraphPatch.from_xml(data["editor_output"]), usage
        # Fallback: add a default improvement
        patch = GraphPatch(diagnosis={"failure_type": "upgrade", "evidence": "No XML output parsed"}, operations=[GraphPatchOp(op=PatchOp.ADD_VERIFIER, target="a5", value="Verify the result and output the final answer in \\boxed{} format")])
        return patch, usage

    async def edit_with_raw(self, graph: MASGraph, task_text: str, expected: str, actual_score: float, final_output: str = "", agent_traces: dict[str, str] | None = None) -> tuple[GraphPatch, str, dict[str, Any]]:
        graph_xml = graph.to_xml()
        traces_str = ""
        if agent_traces:
            traces_str = "\n".join(f"[{aid}]:\n{out}" for aid, out in agent_traces.items())
        user_prompt = EDITOR_TEMPLATE.format(task_text=task_text, expected=expected, graph_xml=graph_xml, actual_score=actual_score, final_output=final_output, agent_traces=traces_str)
        text, usage = await self.client.generate(user_prompt=user_prompt, system_prompt=EDITOR_SYSTEM)
        data = extract_xml(text)
        if "editor_output" in data:
            return GraphPatch.from_xml(data["editor_output"]), text, usage
        patch = GraphPatch(diagnosis={"failure_type": "upgrade", "evidence": "No XML output parsed"}, operations=[GraphPatchOp(op=PatchOp.ADD_VERIFIER, target="a5", value="Verify the result and output the final answer in \\boxed{} format")])
        return patch, text, usage