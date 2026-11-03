from __future__ import annotations
from .schema import (
    AgentNode, AgentSpec, AgentSpecs, GraphEdge, MASGraph, GraphPatch, GraphPatchOp,
    SubTask, TaskDecomposition, Topology, EdgeType, PatchOp, extract_xml,
)
from .patch_types import (
    PatchCategory, OPERATION_REGISTRY, get_operations_by_category,
    get_operation_meta, format_operations_for_prompt, format_operations_for_xml,
)
from .compiler import compile_graph_to_code, compile_and_validate, apply_graph_patch
from .executor import execute_mas_graph, execute_graph_and_score
from .orchestra import run_formalized_orchestra
from .editor import Editor
from .prompts import (
    STAGE1_SYSTEM, STAGE1_TEMPLATE, STAGE2_SYSTEM, STAGE2_TEMPLATE,
    EDITOR_SYSTEM, EDITOR_TEMPLATE, EDITOR_MINIMAL_TEMPLATE,
    get_dataset_requirement,
)