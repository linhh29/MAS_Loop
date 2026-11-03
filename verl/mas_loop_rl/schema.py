from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

# Canonical Refiner action space from the latest paper.  A rollout contains
# exactly one action: Accept, or one bounded edit.
VALID_PATCH_OPS = {
    "accept", "add_agent", "rewrite_role", "add_edge", "remove_edge",
    "change_budget", "replace_tool",
}
LEGACY_PATCH_OPS = {"add_verifier", "change_aggregation", "early_stop_rule"}
VALID_EDGE_TYPES = {"subtask", "answer_check", "critique", "summarize", "data_flow"}
VALID_TOPOLOGIES = {"sequential", "dag", "hierarchical", "iterative"}
VALID_EXECUTION_MODES = {"llm_only", "multi_turn_search"}
# A model-authored MAS is untrusted control data.  In particular, an Orchestra
# once emitted ``max_tokens=1000000`` for a sub-agent, which made vLLM reject
# the request before generation.  Keep the schema/model representation within
# the configured sub-agent context; the executor applies the tighter runtime
# budget (normally 1024) as a second line of defence.
MAX_DECLARED_AGENT_TOKENS = 8192


MAS_GUIDED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agents": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "role": {"type": "string", "minLength": 1, "maxLength": 512},
                    "tools": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 64}},
                    "budget": {"type": "integer", "minimum": 1, "maximum": 4},
                    "execution_mode": {"type": "string", "enum": sorted(VALID_EXECUTION_MODES)},
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_DECLARED_AGENT_TOKENS,
                    },
                },
                "required": ["id", "role", "tools", "budget", "execution_mode"],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array", "maxItems": 32,
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "maxLength": 64}, "to": {"type": "string", "maxLength": 64},
                    "type": {"type": "string", "enum": sorted(VALID_EDGE_TYPES)},
                },
                "required": ["from", "to", "type"],
                "additionalProperties": False,
            },
        },
        "aggregation": {"type": "string", "minLength": 1, "maxLength": 64},
        "topology": {"type": "string", "enum": sorted(VALID_TOPOLOGIES)},
        "max_cost": {"type": "number"},
        "max_iterations": {"type": "integer", "minimum": 1},
    },
    "required": ["agents", "edges", "aggregation", "topology", "max_cost", "max_iterations"],
    "additionalProperties": False,
}

PATCH_GUIDED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stage1_localization": {
            "type": "object",
            "properties": {
                "source_status": {"type": "string", "enum": ["actionable_failure", "preserve"]},
                "target_type": {"type": "string", "enum": ["agent", "edge", "aggregation", "mas"]},
                "target_id": {"type": "string", "maxLength": 128},
                "failure_type": {"type": "string", "maxLength": 128},
                "evidence": {
                    "type": "array", "minItems": 1, "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "maxLength": 128},
                            "execution_index": {"type": "integer", "minimum": -1},
                            "round": {"type": "integer", "minimum": 1},
                            "field": {"type": "string", "maxLength": 96},
                            "location": {"type": "string", "maxLength": 240},
                            "observation": {"type": "string", "maxLength": 600},
                        },
                        "required": ["agent_id", "execution_index", "field", "location", "observation"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["source_status", "target_type", "target_id", "failure_type", "evidence"],
            "additionalProperties": False,
        },
        "stage2_causal_model": {
            "type": "object",
            "properties": {
                "causal_chain": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string", "maxLength": 300}},
                "causal_explanation": {"type": "string", "maxLength": 800},
                "counterfactual": {"type": "string", "maxLength": 600},
                "downstream_impact": {"type": "string", "maxLength": 600},
            },
            "required": ["causal_chain", "causal_explanation", "counterfactual", "downstream_impact"],
            "additionalProperties": False,
        },
        "stage3_repair_plan": {
            "type": "object",
            "properties": {
                "repair_goal": {"type": "string", "maxLength": 600},
                "expected_effect": {"type": "string", "maxLength": 600},
                "risk": {"type": "string", "maxLength": 500},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["repair_goal", "expected_effect", "risk", "confidence"],
            "additionalProperties": False,
        },
        "patch": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": sorted(VALID_PATCH_OPS)},
                    "agent_id": {"type": "string", "maxLength": 64},
                    "id": {"type": "string", "maxLength": 64},
                    "role": {"type": "string", "maxLength": 1000},
                    "new_role": {"type": "string", "maxLength": 1000},
                    "from": {"type": "string", "maxLength": 64},
                    "to": {"type": "string", "maxLength": 64},
                    "type": {"type": "string", "enum": sorted(VALID_EDGE_TYPES)},
                    "value": {"type": "string", "maxLength": 512},
                    "aggregation": {"type": "string", "maxLength": 64},
                    "budget": {"type": "integer", "minimum": 1, "maximum": 4},
                    "tool": {"type": "string", "maxLength": 64},
                    "old_tool": {"type": "string", "maxLength": 64},
                    "new_tool": {"type": "string", "maxLength": 64},
                    "rule": {"type": "string", "maxLength": 256},
                    "max_iterations": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["op"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["stage1_localization", "stage2_causal_model", "stage3_repair_plan", "patch"],
    "additionalProperties": False,
}

def _balanced_json_spans(text: str, open_ch: str, close_ch: str):
    """Yield balanced candidate JSON substrings, ignoring braces in strings."""
    starts = [i for i, ch in enumerate(text) if ch == open_ch]
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    yield start, i + 1, text[start : i + 1]
                    break




def _json_loads_lenient(fragment: str) -> Any:
    """Parse model-emitted JSON with a tiny leniency layer.

    Qwen-style responses sometimes escape apostrophes inside double-quoted JSON
    strings as '. That is valid in Python strings but invalid JSON, and
    caused the scanner to skip the full MAS object and accidentally return an
    inner edge object. Keep strict parsing first, then retry this safe repair.
    """
    try:
        return json.loads(fragment)
    except Exception as first:
        repaired = fragment.replace(chr(92) + chr(39), chr(39))
        if repaired != fragment:
            try:
                return json.loads(repaired)
            except Exception:
                pass
        raise first

def extract_json(text: str) -> Any:
    """Extract the most relevant JSON object/array from a model response.

    GRPO models often prepend long ``<think>`` reasoning that may itself quote
    small JSON fragments such as ``[]`` or the prompt schema before emitting the
    final MAS/patch JSON.  The old implementation took the first balanced JSON
    span, which turned many otherwise-correct rollouts into ``MAS must be a JSON
    object`` errors.  We now parse every balanced candidate and prefer full
    MAS/patch-shaped objects, then otherwise return the last valid JSON value.
    """
    if not text:
        raise ValueError("empty model output")
    t = text.strip()

    candidates: list[Any] = []

    # Markdown JSON fences are usually intentional final answers; inspect all
    # of them but still fall back to free-form scanning below.
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", t, flags=re.S | re.I):
        block = m.group(1).strip()
        try:
            candidates.append(_json_loads_lenient(block))
        except Exception:
            pass

    try:
        candidates.append(_json_loads_lenient(t))
    except Exception:
        pass

    spans = list(_balanced_json_spans(t, "{", "}")) + list(_balanced_json_spans(t, "[", "]"))
    # Stable text order prevents an early [] from beating a later final object.
    for _, _, frag in sorted(spans, key=lambda x: x[0]):
        try:
            candidates.append(_json_loads_lenient(frag))
        except Exception:
            continue

    if not candidates:
        raise ValueError("no JSON found")

    # Prefer the actual structured objects used by this project.  Use the last
    # occurrence because the final answer commonly appears after reasoning that
    # quotes examples/schemas from the prompt.
    for obj in reversed(candidates):
        if isinstance(obj, dict) and "agents" in obj:
            return obj
    for obj in reversed(candidates):
        if isinstance(obj, dict) and ("patch" in obj or "operations" in obj or "diagnosis" in obj):
            return obj
    for obj in reversed(candidates):
        if isinstance(obj, dict):
            return obj
    return candidates[-1]



XML_PROTOCOL_VERSION = "mas_loop_xml_v1"
_MAX_XML_CHARS = 200_000
_MAX_XML_ELEMENTS = 512


_REDUNDANT_SELF_CLOSE_RE = re.compile(
    r"(?P<self><(?P<tag>[A-Za-z_][\w:.-]*)(?:\s[^<>]*?)?/\s*>)\s*</(?P=tag)\s*>",
    flags=re.I,
)
_BARE_XML_AMP_RE = re.compile(
    r"&(?!#\d+;|#x[0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]+;)"
)
# An XML element name cannot begin with a digit, so literals such as
# ``<250 centuries`` and ``4 < 6`` are unambiguously comparison text rather
# than markup. Keep this repair deliberately narrower than a generic
# stray-``<`` recovery; optional whitespace is accepted only before a number.
_UNESCAPED_NUMERIC_LT_RE = re.compile(r"<(?=\s*[+-]?(?:\d|\.\d))")
# XML never permits whitespace between ``<`` and an opening element name.
# Therefore comparisons such as ``λ2 < λ1`` and ``x < y`` are also
# unambiguous text when the right operand begins with a Unicode identifier.
# Requiring at least one whitespace character prevents valid tags such as
# ``<role>`` from being rewritten.
_UNESCAPED_SPACED_NAME_LT_RE = re.compile(r"<(?=\s+[^\W\d]\w*)")


def _repair_near_xml(fragment: str, root_name: str) -> tuple[str, list[str]]:
    """Repair a very small set of syntax-only model XML mistakes.

    This is intentionally not a general HTML/XML recovery parser.  It only
    accepts transformations that preserve the represented MAS/patch meaning:

    * ``<tools /></tools>`` -> ``<tools />`` (and the same pattern for any tag)
    * one omitted final ``</agent>`` immediately before ``</agents>``
    * a bare ampersand -> ``&amp;``
    * an unescaped comparison such as ``<250`` or ``λ2 < λ1`` -> ``&lt;``
    * one omitted ``</downstream_impact>`` immediately before the enclosing
      ``</stage2_causal_model>`` in a Refiner response

    The MAS structural repairs are applied only when tag counts prove
    that exactly one agent close is missing.  Keeping the repair set bounded
    prevents malformed structure from being silently reinterpreted.
    """
    repaired, redundant_count = _REDUNDANT_SELF_CLOSE_RE.subn(
        lambda match: match.group("self"), fragment
    )
    repairs: list[str] = []
    if redundant_count:
        repairs.append(f"redundant_self_close:{redundant_count}")

    if root_name.lower() == "mas":
        open_agents = len(re.findall(r"<agent(?:\s|>)", repaired, flags=re.I))
        close_agents = len(re.findall(r"</agent\s*>", repaired, flags=re.I))
        if open_agents == close_agents + 1 and re.search(r"</agents\s*>", repaired, flags=re.I):
            repaired, count = re.subn(
                r"</agents\s*>", "</agent></agents>", repaired,
                count=1, flags=re.I,
            )
            if count:
                repairs.append("missing_final_agent_close:1")

        # A frequent near-XML completion closes every <agent> but omits only
        # the surrounding </agents> immediately before <edges>.  This repair
        # is syntax-only: require exactly one opening container, no closing
        # container, balanced agent tags, and the canonical sibling boundary.
        open_agent_containers = len(re.findall(r"<agents(?:\s|>)", repaired, flags=re.I))
        close_agent_containers = len(re.findall(r"</agents\s*>", repaired, flags=re.I))
        if (
            open_agent_containers == 1
            and close_agent_containers == 0
            and open_agents == close_agents
            and re.search(r"</agent\s*>\s*<edges(?:\s|>)", repaired, flags=re.I)
        ):
            repaired, count = re.subn(
                r"(</agent\s*>)\s*(<edges(?:\s|>))",
                r"\1</agents>\2",
                repaired,
                count=1,
                flags=re.I,
            )
            if count:
                repairs.append("missing_agents_container_close:1")

    if root_name.lower() == "refiner":
        # Keep recovery syntax-only and narrowly bounded: the model sometimes
        # emits the complete downstream-impact text but closes its parent
        # directly. Exactly one unmatched leaf at the canonical parent boundary
        # is unambiguous; other malformed Refiner structures remain rejected.
        open_impacts = len(re.findall(r"<downstream_impact(?:\s|>)", repaired, flags=re.I))
        close_impacts = len(re.findall(r"</downstream_impact\s*>", repaired, flags=re.I))
        if open_impacts == 1 and close_impacts == 0:
            repaired, count = re.subn(
                r"</stage2_causal_model\s*>",
                "</downstream_impact></stage2_causal_model>",
                repaired,
                count=1,
                flags=re.I,
            )
            if count:
                repairs.append("missing_downstream_impact_close:1")

    repaired, amp_count = _BARE_XML_AMP_RE.subn("&amp;", repaired)
    if amp_count:
        repairs.append(f"bare_ampersand:{amp_count}")
    repaired, numeric_lt_count = _UNESCAPED_NUMERIC_LT_RE.subn("&lt;", repaired)
    if numeric_lt_count:
        repairs.append(f"unescaped_numeric_lt:{numeric_lt_count}")
    repaired, spaced_name_lt_count = _UNESCAPED_SPACED_NAME_LT_RE.subn("&lt;", repaired)
    if spaced_name_lt_count:
        repairs.append(f"unescaped_spaced_name_lt:{spaced_name_lt_count}")
    return repaired, repairs


def _xml_text(element: ET.Element | None, default: str = "") -> str:
    if element is None:
        return default
    return "".join(element.itertext()).strip() or default


def _extract_xml_root(text: str, root_name: str) -> ET.Element:
    if not text or not str(text).strip():
        raise ValueError("empty model output")
    raw = str(text).strip()
    if len(raw) > _MAX_XML_CHARS:
        raise ValueError("XML output exceeds size limit")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", raw, flags=re.I):
        raise ValueError("DTD/entity declarations are not allowed")
    # Prefer the final complete expected root: prompts/examples may quote an
    # earlier fragment before the actual answer.
    pattern = re.compile(rf"<{root_name}\b[^>]*>.*?</{root_name}\s*>", flags=re.I | re.S)
    matches = list(pattern.finditer(raw))
    candidates = [m.group(0) for m in matches]
    if not candidates and raw.lower().startswith(f"<{root_name}"):
        candidates = [raw]
    last_error: Exception | None = None
    for fragment in reversed(candidates):
        attempts = [(fragment, [])]
        repaired, repairs = _repair_near_xml(fragment, root_name)
        if repairs and repaired != fragment:
            attempts.append((repaired, repairs))
        for candidate, _repairs in attempts:
            try:
                root = ET.fromstring(candidate)
                if root.tag.split("}")[-1].lower() != root_name.lower():
                    continue
                if sum(1 for _ in root.iter()) > _MAX_XML_ELEMENTS:
                    raise ValueError("XML output has too many elements")
                return root
            except Exception as exc:
                last_error = exc
    raise ValueError(f"no valid <{root_name}> XML found" + (f": {last_error}" if last_error else ""))


def parse_mas_xml(text: str) -> dict[str, Any]:
    root = _extract_xml_root(text, "mas")
    agents_parent = root.find("agents")
    agent_elements = list(agents_parent.findall("agent")) if agents_parent is not None else list(root.findall("agent"))
    agents: list[dict[str, Any]] = []
    for idx, elem in enumerate(agent_elements[:8]):
        tools_parent = elem.find("tools")
        tools = [_xml_text(tool) for tool in tools_parent.findall("tool")] if tools_parent is not None else []
        role = elem.get("role") or _xml_text(elem.find("role"), "Solve the task carefully.")
        agent: dict[str, Any] = {
            "id": elem.get("id") or f"agent_{idx}",
            "role": role,
            "tools": [tool for tool in tools if tool],
            "budget": elem.get("budget") or _xml_text(elem.find("budget"), "1"),
            "execution_mode": elem.get("execution_mode") or _xml_text(elem.find("execution_mode"), "llm_only"),
        }
        max_tokens = elem.get("max_tokens") or _xml_text(elem.find("max_tokens"))
        if max_tokens:
            agent["max_tokens"] = max_tokens
        agents.append(agent)
    edges_parent = root.find("edges")
    edge_elements = list(edges_parent.findall("edge")) if edges_parent is not None else list(root.findall("edge"))
    edges = [
        {
            "from": elem.get("from") or elem.get("source") or _xml_text(elem.find("from")),
            "to": elem.get("to") or elem.get("target") or _xml_text(elem.find("to")),
            "type": elem.get("type") or elem.get("edge_type") or _xml_text(elem.find("type"), "data_flow"),
        }
        for elem in edge_elements[:32]
    ]
    return {
        "agents": agents,
        "edges": edges,
        "aggregation": root.get("aggregation") or _xml_text(root.find("aggregation")),
        "topology": root.get("topology") or _xml_text(root.find("topology"), "sequential"),
        "max_cost": root.get("max_cost") or _xml_text(root.find("max_cost"), "10"),
        "max_iterations": root.get("max_iterations") or _xml_text(root.find("max_iterations"), "3"),
    }


def _operation_from_xml(elem: ET.Element) -> dict[str, Any]:
    op = dict(elem.attrib)
    if elem.tag.split("}")[-1].lower() not in {"operation", "op", "edit"}:
        op.setdefault("op", elem.tag.split("}")[-1])
    for child in elem:
        key = child.tag.split("}")[-1]
        value = _xml_text(child)
        if value and key not in op:
            op[key] = value
    return op


def parse_refiner_xml(text: str) -> dict[str, Any]:
    root = _extract_xml_root(text, "refiner")
    s1 = root.find("stage1_localization")
    evidence: list[dict[str, Any]] = []
    if s1 is not None:
        evidence_parent = s1.find("evidence")
        evidence_elements = list(evidence_parent) if evidence_parent is not None else list(s1.findall("item"))
        for item in evidence_elements[:4]:
            evidence_item: dict[str, Any] = {
                "agent_id": item.get("agent_id") or _xml_text(item.find("agent_id")),
                "execution_index": item.get("execution_index") or _xml_text(item.find("execution_index")),
                "field": item.get("field") or _xml_text(item.find("field")),
                "location": item.get("location") or _xml_text(item.find("location"), "trajectory"),
                "observation": item.get("observation") or _xml_text(item.find("observation")) or _xml_text(item),
            }
            round_value = item.get("round") or _xml_text(item.find("round"))
            if round_value:
                evidence_item["round"] = round_value
            evidence.append(evidence_item)
    stage1 = {
        "source_status": (s1.get("source_status") if s1 is not None else "") or _xml_text(s1.find("source_status") if s1 is not None else None),
        "target_type": (s1.get("target_type") if s1 is not None else "") or _xml_text(s1.find("target_type") if s1 is not None else None),
        "target_id": (s1.get("target_id") if s1 is not None else "") or _xml_text(s1.find("target_id") if s1 is not None else None),
        "failure_type": (s1.get("failure_type") if s1 is not None else "") or _xml_text(s1.find("failure_type") if s1 is not None else None),
        "evidence": evidence,
    }
    s2 = root.find("stage2_causal_model")
    chain_parent = s2.find("causal_chain") if s2 is not None else None
    chain_elements = list(chain_parent) if chain_parent is not None else []
    chain = [_xml_text(step) for step in chain_elements if _xml_text(step)]
    stage2 = {
        "causal_chain": chain,
        "causal_explanation": _xml_text(s2.find("causal_explanation") if s2 is not None else None),
        "counterfactual": _xml_text(s2.find("counterfactual") if s2 is not None else None),
        "downstream_impact": _xml_text(s2.find("downstream_impact") if s2 is not None else None),
    }
    s3 = root.find("stage3_repair_plan")
    stage3 = {
        "repair_goal": _xml_text(s3.find("repair_goal") if s3 is not None else None),
        "expected_effect": _xml_text(s3.find("expected_effect") if s3 is not None else None),
        "risk": _xml_text(s3.find("risk") if s3 is not None else None),
        "confidence": (s3.get("confidence") if s3 is not None else "") or _xml_text(s3.find("confidence") if s3 is not None else None, "0.5"),
    }
    patch_parent = root.find("patch")
    operations = [_operation_from_xml(elem) for elem in (list(patch_parent) if patch_parent is not None else [])]
    return {
        "stage1_localization": stage1,
        "stage2_causal_model": stage2,
        "stage3_repair_plan": stage3,
        "patch": operations,
        "output_protocol": XML_PROTOCOL_VERSION,
    }


def extract_structured(text: str, expected_kind: str | None = None) -> Any:
    """Parse XML-first model output, retaining legacy JSON compatibility."""
    errors: list[str] = []
    kinds = [expected_kind] if expected_kind in {"mas", "refiner"} else ["refiner", "mas"]
    for kind in kinds:
        try:
            return parse_mas_xml(text) if kind == "mas" else parse_refiner_xml(text)
        except ValueError as exc:
            errors.append(str(exc))
    try:
        return extract_json(text)
    except ValueError as exc:
        errors.append(str(exc))
    raise ValueError("; ".join(errors))

def normalize_mas(obj: Any) -> dict[str, Any]:
    if isinstance(obj, str):
        obj = extract_structured(obj, "mas")
    if not isinstance(obj, dict):
        raise ValueError("MAS must be an XML <mas> document or compatible JSON object")
    agents = obj.get("agents")
    if agents is None:
        agents = []
    if hasattr(agents, "tolist"):
        agents = agents.tolist()
    if not isinstance(agents, list) or not agents:
        raise ValueError("MAS requires a non-empty agents list")
    out_agents = []
    seen = set()
    for idx, a in enumerate(agents[:8]):
        if not isinstance(a, dict):
            continue
        aid = str(a.get("id") or f"agent_{idx}").strip().replace(" ", "_")[:48]
        if not aid or aid in seen:
            aid = f"agent_{idx}"
        seen.add(aid)
        role = str(a.get("role") or "Solve the task carefully.").strip()[:1000]
        tools = a.get("tools")
        if hasattr(tools, "tolist"):
            tools = tools.tolist()
        tools = tools if isinstance(tools, list) else []
        try:
            budget = max(1, min(4, int(a.get("budget", 1))))
        except Exception:
            budget = 1
        execution_mode = str(a.get("execution_mode") or "llm_only").strip().lower()
        if execution_mode not in VALID_EXECUTION_MODES:
            execution_mode = "llm_only"
        out_agents.append({
            "id": aid,
            "role": role,
            "tools": [str(x)[:80] for x in tools[:8]],
            "budget": budget,
            "execution_mode": execution_mode,
            **(
                {
                    "max_tokens": max(
                        1,
                        min(MAX_DECLARED_AGENT_TOKENS, int(a["max_tokens"])),
                    )
                }
                if str(a.get("max_tokens", "")).isdigit()
                else {}
            ),
        })
    if not out_agents:
        raise ValueError("MAS has no valid agents")
    ids = {a["id"] for a in out_agents}
    edges = []
    edges_in = obj.get("edges")
    if edges_in is None:
        edges_in = []
    if hasattr(edges_in, "tolist"):
        edges_in = edges_in.tolist()
    for e in edges_in:
        if not isinstance(e, dict):
            continue
        fr, to = str(e.get("from", "")), str(e.get("to", ""))
        if fr in ids and to in ids and fr != to:
            et = str(e.get("type") or "data_flow")
            edges.append({"from": fr, "to": to, "type": et if et in VALID_EDGE_TYPES else "data_flow"})
    if len(out_agents) > 1 and not edges:
        edges = [{"from": out_agents[i]["id"], "to": out_agents[i+1]["id"], "type": "data_flow"} for i in range(len(out_agents)-1)]
    aggregation = str(obj.get("aggregation") or out_agents[-1]["id"])
    if aggregation not in ids:
        aggregation = out_agents[-1]["id"]
    topo = str(obj.get("topology") or "sequential")
    return {
        "agents": out_agents,
        "edges": edges[:16],
        "aggregation": aggregation,
        "topology": topo if topo in VALID_TOPOLOGIES else "sequential",
        "max_cost": float(obj.get("max_cost", 10.0) or 10.0),
        "max_iterations": int(obj.get("max_iterations", 3) or 3),
    }


def _bounded(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _bounded_list(value: Any, *, limit: int, item_limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [text for item in value[:limit] if (text := _bounded(item, item_limit))]


def _normalize_refiner_reasoning(raw: Any, operations: list[dict[str, Any]]) -> dict[str, Any]:
    obj = dict(raw) if isinstance(raw, dict) else {}
    # Legacy diagnosis-only outputs remain readable for checkpoint/log
    # compatibility. New prompts and guided decoding emit the three stages.
    legacy = dict(obj.get("diagnosis")) if isinstance(obj.get("diagnosis"), dict) else {}
    stage1_raw = obj.get("stage1_localization")
    stage2_raw = obj.get("stage2_causal_model")
    stage3_raw = obj.get("stage3_repair_plan")
    stage1 = dict(stage1_raw) if isinstance(stage1_raw, dict) else legacy
    stage2 = dict(stage2_raw) if isinstance(stage2_raw, dict) else legacy
    stage3 = dict(stage3_raw) if isinstance(stage3_raw, dict) else legacy

    accepted = len(operations) == 1 and operations[0].get("op") == "accept"
    source_status = str(stage1.get("source_status") or ("preserve" if accepted else "actionable_failure"))
    if source_status not in {"actionable_failure", "preserve"}:
        source_status = "preserve" if accepted else "actionable_failure"
    target_type = str(stage1.get("target_type") or "mas")
    if target_type not in {"agent", "edge", "aggregation", "mas"}:
        target_type = "mas"
    target_id = _bounded(stage1.get("target_id", stage1.get("target_component", "mas")), 128) or "mas"
    failure_type = _bounded(stage1.get("failure_type", "none" if accepted else "unspecified"), 128)

    evidence_raw = stage1.get("evidence")
    if isinstance(evidence_raw, str):
        evidence_raw = [{"location": "trajectory", "observation": evidence_raw}]
    evidence: list[dict[str, Any]] = []
    if isinstance(evidence_raw, list):
        for item in evidence_raw[:4]:
            if isinstance(item, dict):
                location = _bounded(item.get("location"), 240)
                observation = _bounded(item.get("observation"), 600)
                evidence_item: dict[str, Any] = {
                    "location": location or "trajectory",
                    "observation": observation,
                }
                for key, limit in (("agent_id", 128), ("field", 96)):
                    if value := _bounded(item.get(key), limit):
                        evidence_item[key] = value
                for key in ("execution_index", "round"):
                    value = item.get(key)
                    if value not in (None, ""):
                        try:
                            evidence_item[key] = int(value)
                        except (TypeError, ValueError):
                            evidence_item[key] = value
            else:
                location, observation = "trajectory", _bounded(item, 600)
                evidence_item = {"location": location, "observation": observation}
            if observation:
                evidence.append(evidence_item)
    if not evidence:
        evidence = [{"location": "trajectory", "observation": "source preserved" if accepted else "bounded evidence not specified"}]

    causal_explanation = _bounded(
        stage2.get("causal_explanation", stage2.get("mechanism", "no supported failure" if accepted else "unspecified causal mechanism")),
        800,
    )
    causal_chain = _bounded_list(stage2.get("causal_chain"), limit=5, item_limit=300)
    if not causal_chain:
        causal_chain = [causal_explanation]
    counterfactual = _bounded(stage2.get("counterfactual", "no edit should improve the source" if accepted else "repairing the target should interrupt the failure chain"), 600)
    downstream_impact = _bounded(stage2.get("downstream_impact", "preserve downstream behavior" if accepted else "improve downstream evidence use and final behavior"), 600)
    repair_goal = _bounded(stage3.get("repair_goal", "preserve the current behavior" if accepted else "improve the diagnosed component"), 600)
    expected_effect = _bounded(stage3.get("expected_effect", "no behavioral regression" if accepted else "resolve the diagnosed causal failure"), 600)
    risk = _bounded(stage3.get("risk", "unnecessary change" if accepted else "the patch may not realize the intended mechanism"), 500)
    try:
        confidence = max(0.0, min(1.0, float(stage3.get("confidence", legacy.get("confidence", 0.5)))))
    except (TypeError, ValueError):
        confidence = 0.5

    stage1_out = {
        "source_status": source_status,
        "target_type": target_type,
        "target_id": target_id,
        "failure_type": failure_type or ("none" if accepted else "unspecified"),
        "evidence": evidence,
    }
    stage2_out = {
        "causal_chain": causal_chain,
        "causal_explanation": causal_explanation,
        "counterfactual": counterfactual,
        "downstream_impact": downstream_impact,
    }
    stage3_out = {
        "repair_goal": repair_goal,
        "expected_effect": expected_effect,
        "risk": risk,
        "confidence": confidence,
    }
    # diagnosis is a compatibility projection for existing analysis code.
    diagnosis = {**stage1_out, "causal_explanation": causal_explanation, "repair_goal": repair_goal, "confidence": confidence}
    return {
        "stage1_localization": stage1_out,
        "stage2_causal_model": stage2_out,
        "stage3_repair_plan": stage3_out,
        "diagnosis": diagnosis,
    }

OPERATION_ALIASES = {
    "accept": "accept", "keep": "accept", "preserve": "accept", "no_change": "accept", "noop": "accept",
    "add_agent": "add_agent", "create_agent": "add_agent", "insert_agent": "add_agent",
    "add_verifier": "legacy_add_verifier", "add_checker": "legacy_add_verifier", "add_critic": "legacy_add_verifier",
    "rewrite_role": "rewrite_role", "change_role": "rewrite_role", "update_role": "rewrite_role", "modify_role": "rewrite_role", "set_role": "rewrite_role",
    "add_edge": "add_edge", "connect": "add_edge", "connect_agents": "add_edge", "link_agents": "add_edge",
    "remove_edge": "remove_edge", "delete_edge": "remove_edge", "disconnect": "remove_edge", "unlink_agents": "remove_edge",
    "change_aggregation": "legacy_change_aggregation", "set_aggregation": "legacy_change_aggregation", "set_aggregator": "legacy_change_aggregation", "change_aggregator": "legacy_change_aggregation",
    "change_budget": "change_budget", "set_budget": "change_budget", "update_budget": "change_budget",
    "replace_tool": "replace_tool", "add_tool": "replace_tool", "set_tool": "replace_tool", "update_tool": "replace_tool", "change_tool": "replace_tool",
    "early_stop_rule": "legacy_early_stop_rule", "set_max_iterations": "legacy_early_stop_rule", "change_max_iterations": "legacy_early_stop_rule", "set_iterations": "legacy_early_stop_rule",
}

FIELD_ALIASES = {
    "agent": "agent_id", "node": "agent_id", "node_id": "agent_id", "target_agent": "agent_id", "target_agent_id": "agent_id",
    "source": "from", "src": "from", "from_agent": "from", "source_agent": "from",
    "target": "to", "dst": "to", "to_agent": "to", "destination": "to", "target_node": "to",
    "new_aggregation": "aggregation", "aggregator": "aggregation", "aggregation_agent": "aggregation",
    "tool_name": "new_tool", "replacement_tool": "new_tool", "target_tool": "new_tool",
    "iterations": "max_iterations", "max_iteration": "max_iterations",
}

EDGE_TYPE_ALIASES = {
    "flow": "data_flow", "data": "data_flow", "information_flow": "data_flow",
    "verify": "answer_check", "verification": "answer_check", "check": "answer_check",
    "review": "critique", "critic": "critique", "summary": "summarize",
}

PATCH_TRANSLATOR_VERSION = "mas_patch_intent_compiler_v2_single_edit"


def _canonical_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _unique_agent_id(requested: Any, known_ids: set[str], default: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(requested or default).strip()).strip("_")[:48] or default
    candidate = base
    suffix = 2
    while candidate in known_ids:
        candidate = f"{base}_{suffix}"[:64]
        suffix += 1
    return candidate


def _resolve_agent_id(value: Any, agents: list[dict[str, Any]], *, allow_new: bool = False) -> tuple[str, str | None]:
    raw = str(value or "").strip()
    if not raw:
        return "", "missing_agent_reference"
    ids = [str(agent.get("id") or "") for agent in agents]
    if raw in ids:
        return raw, None
    folded = [aid for aid in ids if aid.casefold() == raw.casefold()]
    if len(folded) == 1:
        return folded[0], f"resolved_agent_case:{raw}->{folded[0]}"
    token = _canonical_token(raw)
    normalized = [aid for aid in ids if _canonical_token(aid) == token]
    if len(normalized) == 1:
        return normalized[0], f"resolved_agent_token:{raw}->{normalized[0]}"
    role_matches = [
        str(agent.get("id") or "") for agent in agents
        if token and token in _canonical_token(agent.get("role", ""))
    ]
    if len(role_matches) == 1:
        return role_matches[0], f"resolved_agent_role:{raw}->{role_matches[0]}"
    if allow_new:
        return raw, None
    if len(normalized) > 1 or len(role_matches) > 1:
        return "", f"ambiguous_agent_reference:{raw}"
    return "", f"unknown_agent_reference:{raw}"


def _raw_operations(obj: dict[str, Any]) -> list[Any]:
    ops = obj.get("patch", obj.get("operations", obj.get("edits", obj.get("changes", []))))
    if isinstance(ops, dict):
        ops = [ops]
    return ops if isinstance(ops, list) else []


def _translate_operations(raw_ops: list[Any], src_mas: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = normalize_mas(src_mas) if src_mas is not None else {"agents": [], "edges": [], "aggregation": ""}
    agents = [dict(agent) for agent in source.get("agents", [])]
    known_ids = {str(agent.get("id") or "") for agent in agents}
    edges = {(str(e.get("from") or ""), str(e.get("to") or ""), str(e.get("type") or "data_flow")) for e in source.get("edges", [])}
    canonical: list[dict[str, Any]] = []
    warnings: list[str] = []
    dropped: list[dict[str, Any]] = []
    requested: list[dict[str, Any]] = []

    # The paper deliberately bounds each intervention to one edit.  Do not
    # silently truncate a multi-edit completion: doing so changes the action
    # whose log-probability is being trained.
    if len(raw_ops) > 1:
        return [], {
            "version": PATCH_TRANSLATOR_VERSION, "status": "invalid",
            "requested_operations": copy.deepcopy(raw_ops), "canonical_operations": [],
            "applied_operations": [], "noop_operations": [],
            "dropped_operations": [{"reason": "too_many_operations", "count": len(raw_ops)}],
            "warnings": ["paper_action_space_allows_exactly_one_operation"],
            "semantic_intent_preserved": False,
        }

    for index, raw in enumerate(raw_ops):
        if not isinstance(raw, dict):
            dropped.append({"index": index, "reason": "operation_not_object", "operation": raw})
            continue
        requested.append(copy.deepcopy(raw))
        expanded = dict(raw)
        for alias, target in FIELD_ALIASES.items():
            if target not in expanded and alias in expanded:
                expanded[target] = expanded[alias]
                warnings.append(f"field_alias:{alias}->{target}")
        raw_name = str(expanded.get("op") or expanded.get("operation") or expanded.get("action") or expanded.get("type") or "").strip()
        name = OPERATION_ALIASES.get(_canonical_token(raw_name), "")
        if not name:
            dropped.append({"index": index, "reason": f"unsupported_operation:{raw_name[:64]}", "operation": raw})
            continue
        if _canonical_token(raw_name) != name:
            warnings.append(f"operation_alias:{raw_name}->{name}")
        op: dict[str, Any] = {"op": name}

        if name == "accept":
            canonical.append(op)
            continue

        if name == "add_agent":
            requested_id = expanded.get("agent_id", expanded.get("id"))
            default = "agent_new"
            aid = _unique_agent_id(requested_id, known_ids, default)
            if requested_id and str(requested_id).strip() != aid:
                warnings.append(f"renamed_new_agent:{requested_id}->{aid}")
            role = str(expanded.get("role") or expanded.get("new_role") or "Assist solving the task.").strip()
            try:
                raw_budget = int(expanded.get("budget", 1) or 1)
            except (TypeError, ValueError):
                raw_budget = 1
                warnings.append(f"invalid_budget_defaulted:{expanded.get('budget')}->1")
            if not 1 <= raw_budget <= 4:
                dropped.append({"index": index, "reason": "budget_out_of_bounds", "operation": raw})
                continue
            budget = raw_budget
            op.update({"agent_id": aid, "role": role[:1000] or "Assist solving the task.", "budget": budget})
            if expanded.get("from"):
                resolved, note = _resolve_agent_id(expanded.get("from"), agents)
                if note:
                    warnings.append(note)
                if resolved:
                    op["from"] = resolved
            canonical.append(op)
            known_ids.add(aid)
            agents.append({"id": aid, "role": op["role"]})
            continue

        if name in {"rewrite_role", "change_budget", "replace_tool"}:
            ref = expanded.get("agent_id", expanded.get("id"))
            aid, note = _resolve_agent_id(ref, agents)
            if note:
                warnings.append(note)
            if not aid:
                dropped.append({"index": index, "reason": note or "missing_agent_reference", "operation": raw})
                continue
            op["agent_id"] = aid
            if name == "rewrite_role":
                role = str(expanded.get("role") or expanded.get("new_role") or expanded.get("value") or "").strip()
                if not role:
                    dropped.append({"index": index, "reason": "missing_role", "operation": raw})
                    continue
                op["role"] = role[:1000]
            elif name == "change_budget":
                try:
                    raw_budget = int(expanded.get("budget", expanded.get("value")))
                except (TypeError, ValueError):
                    dropped.append({"index": index, "reason": "invalid_budget", "operation": raw})
                    continue
                if not 1 <= raw_budget <= 4:
                    dropped.append({"index": index, "reason": "budget_out_of_bounds", "operation": raw})
                    continue
                op["budget"] = raw_budget
            else:
                new_tool = str(expanded.get("new_tool") or expanded.get("value") or expanded.get("tool") or "").strip()
                old_tool = str(expanded.get("old_tool") or "").strip()
                if not new_tool and not old_tool:
                    dropped.append({"index": index, "reason": "missing_tool_change", "operation": raw})
                    continue
                if new_tool:
                    op["new_tool"] = new_tool[:64]
                if old_tool:
                    op["old_tool"] = old_tool[:64]
            canonical.append(op)
            continue

        if name in {"add_edge", "remove_edge"}:
            source_id, source_note = _resolve_agent_id(expanded.get("from"), agents)
            target_id, target_note = _resolve_agent_id(expanded.get("to"), agents)
            for note in (source_note, target_note):
                if note:
                    warnings.append(note)
            if not source_id or not target_id or source_id == target_id:
                dropped.append({"index": index, "reason": "invalid_edge_endpoints", "operation": raw})
                continue
            edge_type_raw = str(expanded.get("edge_type") or (expanded.get("type") if expanded.get("type") != raw_name else "") or "data_flow")
            edge_type = EDGE_TYPE_ALIASES.get(_canonical_token(edge_type_raw), _canonical_token(edge_type_raw))
            if edge_type not in VALID_EDGE_TYPES:
                # Edge type carries no core intent; preserve connect intent with
                # safe default when model emits punctuation/unknown synonym.
                warnings.append(f"invalid_edge_type_defaulted:{edge_type_raw}->data_flow")
                edge_type = "data_flow"
            op.update({"from": source_id, "to": target_id, "type": edge_type})
            canonical.append(op)
            if name == "add_edge":
                edges.add((source_id, target_id, edge_type))
            else:
                edges = {edge for edge in edges if not (edge[0] == source_id and edge[1] == target_id)}
            continue

        if name == "change_aggregation":
            target, note = _resolve_agent_id(expanded.get("aggregation", expanded.get("value")), agents)
            if note:
                warnings.append(note)
            if not target:
                dropped.append({"index": index, "reason": note or "missing_aggregation", "operation": raw})
                continue
            op["aggregation"] = target
            canonical.append(op)
            continue

        if name == "early_stop_rule":
            raw_value = expanded.get("max_iterations", expanded.get("value", expanded.get("rule")))
            match = re.search(r"-?\d+", str(raw_value or ""))
            if not match:
                dropped.append({"index": index, "reason": "invalid_max_iterations", "operation": raw})
                continue
            raw_iterations = int(match.group(0))
            iterations = max(1, min(8, raw_iterations))
            if iterations != raw_iterations:
                warnings.append(f"clamped_max_iterations:{raw_iterations}->{iterations}")
            op["max_iterations"] = iterations
            canonical.append(op)

    if any(op["op"] == "accept" for op in canonical) and len(canonical) > 1:
        canonical = [op for op in canonical if op["op"] != "accept"]
        warnings.append("dropped_accept_mixed_with_edits")
    audit = {
        "version": PATCH_TRANSLATOR_VERSION,
        "status": "translated" if canonical else "untranslatable",
        "requested_operations": requested,
        "canonical_operations": copy.deepcopy(canonical),
        "applied_operations": [],
        "noop_operations": [],
        "dropped_operations": dropped,
        "warnings": list(dict.fromkeys(warnings)),
        "semantic_intent_preserved": bool(canonical),
    }
    return canonical, audit


_REFINER_PLACEHOLDER_OBSERVATIONS = {
    "observed bounded evidence", "bounded evidence not specified", "earliest failure",
    "downstream consequence", "mechanism", "falsifiable expected change",
    "minimal coherent repair", "expected effect", "regression risk",
    "improved task specific role", "source preserved",
}
_TRACE_LOCATION_RE = re.compile(r"^(?:node_cards\[\d+\]|edge_cards\[\d+\]|outcome)(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*$")


def _normalized_placeholder_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _resolve_diagnostic_location(state: dict[str, Any], location: str) -> Any:
    # 安全说明：只接受诊断器生成的受限字段路径，不执行 eval/XPath，防止注入和越界读取。
    if not _TRACE_LOCATION_RE.fullmatch(location):
        raise ValueError(f"evidence location has invalid syntax: {location}")
    current: Any = state
    for name, index_text in re.findall(r"\.?(node_cards|edge_cards|outcome|[A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", location):
        if name:
            if not isinstance(current, dict) or name not in current:
                raise ValueError(f"evidence location does not exist: {location}")
            current = current[name]
        else:
            index = int(index_text)
            if not isinstance(current, list) or index >= len(current):
                raise ValueError(f"evidence location does not exist: {location}")
            current = current[index]
    return current


def _diagnostic_trace_ref_fields(state: dict[str, Any]) -> dict[str, str]:
    """Return the diagnostic-owned location whitelist and its declared field names."""
    refs: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            trace_refs = value.get("trace_refs")
            if isinstance(trace_refs, dict):
                for field, location in trace_refs.items():
                    if isinstance(location, str) and _TRACE_LOCATION_RE.fullmatch(location):
                        refs[location] = str(field)
            for key, child in value.items():
                if key != "trace_refs":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(state)
    return refs


def _grounding_context_for_location(state: dict[str, Any], location: str) -> dict[str, Any]:
    node_match = re.match(r"^node_cards\[(\d+)\]", location)
    if node_match:
        cards = state.get("node_cards") or []
        index = int(node_match.group(1))
        if index >= len(cards) or not isinstance(cards[index], dict):
            raise ValueError(f"evidence location does not exist: {location}")
        return {"kind": "node", **cards[index]}
    edge_match = re.match(r"^edge_cards\[(\d+)\]", location)
    if edge_match:
        cards = state.get("edge_cards") or []
        index = int(edge_match.group(1))
        if index >= len(cards) or not isinstance(cards[index], dict):
            raise ValueError(f"evidence location does not exist: {location}")
        return {"kind": "edge", **cards[index]}
    return {"kind": "outcome", "agent_id": "mas", "execution_index": -1}


def _repair_refiner_evidence_grounding(normalized: dict[str, Any], diagnostic_state: dict[str, Any]) -> list[str]:
    """Repair only evidence metadata that can be recomputed from diagnostics.

    Models often name a real diagnostic field that is not itself exposed as a
    ``trace_refs`` entry (for example ``retrieval.errors``). Map it to a
    diagnostic-owned summary carrying same observed signal, then recompute all
    dependent metadata. Never invent evidence when no non-empty grounded value
    exists.
    """
    stage1 = normalized.get("stage1_localization") or {}
    evidence = stage1.get("evidence") or []
    refs = _diagnostic_trace_ref_fields(diagnostic_state)
    repairs: list[str] = []

    def value_for(location: str) -> Any:
        try:
            return _resolve_diagnostic_location(diagnostic_state, location)
        except Exception:
            return None

    def candidate_location(location: str) -> str | None:
        if location in refs:
            return location
        node = re.match(r"^node_cards\[(\d+)\]", location)
        edge = re.match(r"^edge_cards\[(\d+)\]", location)
        if node:
            node_index = int(node.group(1))
            cards = diagnostic_state.get("node_cards") or []
            if node_index >= len(cards) or not isinstance(cards[node_index], dict):
                return None
            prefix = f"node_cards[{node.group(1)}]"
            card = cards[node_index]
            leaf = location.rsplit(".", 1)[-1]
            if leaf in {"errors", "error", "retrieval_errors"} and "retrieval_error" in (card.get("hard_mismatch_signals") or []):
                target = f"{prefix}.hard_mismatch_signals"
                if target in refs:
                    return target
            same_leaf = [path for path in refs if path.startswith(prefix + ".") and path.rsplit(".", 1)[-1] == leaf]
            if same_leaf:
                return same_leaf[0]
            if f"{prefix}.hard_mismatch_signals" in refs and card.get("hard_mismatch_signals"):
                return f"{prefix}.hard_mismatch_signals"
            if f"{prefix}.card" in refs:
                return f"{prefix}.card"
        if edge:
            prefix = f"edge_cards[{edge.group(1)}]"
            leaf = location.rsplit(".", 1)[-1]
            same_leaf = [path for path in refs if path.startswith(prefix + ".") and path.rsplit(".", 1)[-1] == leaf]
            if same_leaf:
                return same_leaf[0]
            if f"{prefix}.card" in refs:
                return f"{prefix}.card"
        if location.startswith("outcome."):
            for path in ("outcome.error_message", "outcome.global_signals", "outcome.final_output", "outcome.success"):
                if path in refs and value_for(path) not in (None, "", [], {}):
                    return path
        return None

    def concrete_observation(location: str, value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                text = str(value)
        text = text[:560]
        return f"Observed diagnostic signal at {location}: {text}" if text else ""

    def needs_grounding(observation: Any) -> bool:
        text = str(observation or "").strip()
        if not text or _normalized_placeholder_text(text) in _REFINER_PLACEHOLDER_OBSERVATIONS:
            return True
        # Reject vague model prose when grounded diagnostic value exists.
        normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        vague = (
            r"^(?:earliest )?(?:failure|error|issue|problem|signal)(?: found| detected)?$",
            r"^(?:retrieval|execution|agent|tool) (?:failure|error|problem|issue)$",
            r"^(?:see|refer to|check) (?:trace|diagnostics?|logs?)$",
            r"^(?:there is|has) (?:a|an) (?:failure|error|problem|issue)$",
        )
        return any(re.fullmatch(pattern, normalized) for pattern in vague)

    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue
        old_location = str(item.get("location") or "").strip()
        new_location = candidate_location(old_location)
        if not new_location:
            # Model may emit a near-miss path or placeholder location. Recover
            # with first non-empty trace-owned signal; validation still checks
            # target/agent metadata after this rewrite.
            fallback_candidates = [
                path for path in refs
                if value_for(path) not in (None, "", [], {})
            ]
            if fallback_candidates:
                new_location = fallback_candidates[0]
                item["location"] = new_location
                repairs.append(f"evidence[{index}].location:{old_location}->{new_location}")
            else:
                continue
        if new_location != old_location:
            item["location"] = new_location
            repairs.append(f"evidence[{index}].location:{old_location}->{new_location}")
        context = _grounding_context_for_location(diagnostic_state, new_location)
        if context["kind"] == "node":
            item["agent_id"] = str(context.get("agent_id") or "")
            item["execution_index"] = int(context.get("execution_index", -1))
        elif context["kind"] == "edge":
            item["agent_id"] = str(context.get("edge_id") or "")
            item["execution_index"] = -1
        else:
            item["agent_id"] = "mas"
            item["execution_index"] = -1
        expected_field = refs.get(new_location)
        if expected_field and item.get("field") != expected_field:
            item["field"] = expected_field
            repairs.append(f"evidence[{index}].field->{expected_field}")
        if "round_summaries[" not in new_location:
            item.pop("round", None)
        value = value_for(new_location)
        # Pick another trace-owned field when model chose valid but empty field.
        # Empty fields cannot support editor diagnosis; execution_status/card are
        # stable non-empty fallbacks on same node/edge/outcome context.
        if value in (None, "", [], {}):
            prefix = new_location.split(".", 1)[0]
            fallback_candidates = [
                path for path in refs
                if path.startswith(prefix + ".") and value_for(path) not in (None, "", [], {})
            ]
            if prefix == "outcome":
                fallback_candidates = [
                    path for path in (
                        "outcome.error_message", "outcome.global_signals",
                        "outcome.final_output", "outcome.success",
                    )
                    if path in refs and value_for(path) not in (None, "", [], {})
                ]
            if fallback_candidates:
                fallback = fallback_candidates[0]
                if fallback != new_location:
                    repairs.append(f"evidence[{index}].location:{new_location}->{fallback}")
                    new_location = fallback
                    item["location"] = fallback
                    context = _grounding_context_for_location(diagnostic_state, new_location)
                    if context["kind"] == "node":
                        item["agent_id"] = str(context.get("agent_id") or "")
                        item["execution_index"] = int(context.get("execution_index", -1))
                    elif context["kind"] == "edge":
                        item["agent_id"] = str(context.get("edge_id") or "")
                        item["execution_index"] = -1
                    else:
                        item["agent_id"] = "mas"
                        item["execution_index"] = -1
                    expected_field = refs.get(new_location)
                    if expected_field:
                        item["field"] = expected_field
                    value = value_for(new_location)
        # Re-render every grounded observation from diagnostic value. This prevents
        # vague model prose (including arbitrary synonyms) from passing as trace
        # evidence while preserving only real, non-empty diagnostic signals.
        if value not in (None, "", [], {}):
            observation = concrete_observation(new_location, value)
            if observation and observation != str(item.get("observation") or "").strip():
                item["observation"] = observation
                repairs.append(f"evidence[{index}].observation:derived")
    return repairs


def _validate_refiner_grounding(normalized: dict[str, Any], diagnostic_state: dict[str, Any]) -> None:
    if not isinstance(diagnostic_state, dict):
        raise ValueError("diagnostic_state must be an object")
    stage1 = normalized.get("stage1_localization") or {}
    evidence = stage1.get("evidence") or []
    if not evidence:
        raise ValueError("stage1_localization requires grounded evidence")
    target_type = str(stage1.get("target_type") or "")
    target_id = str(stage1.get("target_id") or "")
    target_grounded = target_type not in {"agent", "aggregation", "edge"}
    trace_ref_fields = _diagnostic_trace_ref_fields(diagnostic_state)
    if not trace_ref_fields:
        raise ValueError("diagnostic_state contains no trace_refs")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError(f"evidence[{index}] must be an object")
        location = str(item.get("location") or "").strip()
        observation = str(item.get("observation") or "").strip()
        if not observation or _normalized_placeholder_text(observation) in _REFINER_PLACEHOLDER_OBSERVATIONS:
            raise ValueError(f"evidence[{index}].observation must describe a concrete trace signal")
        if location not in trace_ref_fields:
            raise ValueError(f"evidence[{index}].location must be copied exactly from diagnostic trace_refs: {location}")
        _resolve_diagnostic_location(diagnostic_state, location)
        context = _grounding_context_for_location(diagnostic_state, location)
        expected_agent = (
            str(context.get("agent_id") or "") if context["kind"] == "node"
            else str(context.get("edge_id") or "") if context["kind"] == "edge"
            else "mas"
        )
        if context["kind"] == "node":
            try:
                expected_execution = int(context.get("execution_index", -1))
            except (TypeError, ValueError):
                expected_execution = -1
        else:
            expected_execution = -1
        if str(item.get("agent_id") or "") != expected_agent:
            raise ValueError(f"evidence[{index}].agent_id does not match {location}: expected {expected_agent}")
        try:
            actual_execution = int(item.get("execution_index"))
        except (TypeError, ValueError):
            raise ValueError(f"evidence[{index}].execution_index must be an integer") from None
        if actual_execution != expected_execution:
            raise ValueError(
                f"evidence[{index}].execution_index does not match {location}: expected {expected_execution}"
            )
        field = str(item.get("field") or "").strip()
        expected_field = trace_ref_fields[location]
        if field != expected_field:
            raise ValueError(f"evidence[{index}].field does not match location: expected {expected_field}")
        round_match = re.search(r"round_summaries\[(\d+)\]", location)
        if round_match:
            rounds = (((context.get("optional_capability_signals") or {}).get("retrieval") or {}).get("round_summaries") or [])
            round_index = int(round_match.group(1))
            expected_round = int(rounds[round_index].get("round")) if round_index < len(rounds) else None
            try:
                actual_round = int(item.get("round"))
            except (TypeError, ValueError):
                raise ValueError(f"evidence[{index}].round is required for retrieval evidence") from None
            if expected_round is None or actual_round != expected_round:
                raise ValueError(f"evidence[{index}].round does not match {location}: expected {expected_round}")
        if target_type in {"agent", "aggregation"} and context["kind"] == "node" and expected_agent == target_id:
            target_grounded = True
        if target_type == "edge" and context["kind"] == "edge" and expected_agent == target_id:
            target_grounded = True
    if not target_grounded:
        raise ValueError(f"stage1 target {target_type}:{target_id} is not supported by any evidence item")


def normalize_patch(obj: Any, src_mas: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(obj, str):
        obj = extract_structured(obj, "refiner")
    if not isinstance(obj, dict):
        raise ValueError("patch must be an XML <refiner> document or compatible JSON object")
    clean, audit = _translate_operations(_raw_operations(obj), src_mas)
    reasoning = _normalize_refiner_reasoning(obj, clean)
    accepted = len(clean) == 1 and clean[0]["op"] == "accept"
    desired_status = "preserve" if accepted else "actionable_failure"
    if reasoning["stage1_localization"]["source_status"] != desired_status:
        old = reasoning["stage1_localization"]["source_status"]
        reasoning["stage1_localization"]["source_status"] = desired_status
        reasoning["diagnosis"]["source_status"] = desired_status
        audit["warnings"].append(f"normalized_source_status:{old}->{desired_status}")
    return {**reasoning, "patch": clean, "operations": clean, "translation": audit}


def is_accept_patch(patch: Any) -> bool:
    normalized = normalize_patch(patch)
    return len(normalized["operations"]) == 1 and normalized["operations"][0]["op"] == "accept"


def normalize_editor_action(
    obj: Any,
    src_mas: dict[str, Any] | None = None,
    *,
    diagnostic_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a model-authored edit intent into safe canonical operations.

    Minor vocabulary, field, identifier and bounded-value differences are
    translated rather than treated as policy failures.  A response fails only
    when no explicit accept or safely executable edit intent can be recovered.
    """
    parsed = extract_structured(obj, "refiner") if isinstance(obj, str) else obj
    normalized = normalize_patch(parsed, src_mas=src_mas)
    if diagnostic_state is not None:
        repairs = _repair_refiner_evidence_grounding(normalized, diagnostic_state)
        if repairs:
            normalized.setdefault("translation", {}).setdefault("warnings", []).extend(
                f"grounding_repair:{item}" for item in repairs
            )
        # Edge targets are often hallucinated even when evidence is valid.
        # Rebind target to grounded edge evidence rather than rejecting whole run.
        stage1 = normalized.get("stage1_localization") or {}
        if str(stage1.get("target_type") or "") == "edge":
            edge_ids = []
            for item in stage1.get("evidence") or []:
                loc = str(item.get("location") or "")
                if loc.startswith("edge_cards["):
                    try:
                        edge_index = int(loc.split("[", 1)[1].split("]", 1)[0])
                        edge_cards = diagnostic_state.get("edge_cards") or []
                        if 0 <= edge_index < len(edge_cards):
                            edge_ids.append(str(edge_cards[edge_index].get("edge_id") or ""))
                    except (ValueError, TypeError):
                        pass
            if edge_ids and str(stage1.get("target_id") or "") not in edge_ids:
                stage1["target_id"] = edge_ids[0]
                normalized.setdefault("translation", {}).setdefault("warnings", []).append(
                    "grounding_repair:stage1.target_id->" + edge_ids[0]
                )
        _validate_refiner_grounding(normalized, diagnostic_state)
    if not normalized["operations"]:
        audit = normalized.get("translation", {})
        reasons = [str(item.get("reason")) for item in audit.get("dropped_operations", []) if isinstance(item, dict)]
        # Invalid editor syntax must not abort Vita task. Preserve source MAS.
        if diagnostic_state is not None:
            normalized["operations"] = [{"op": "accept"}]
            normalized["patch"] = [{"op": "accept"}]
            audit["status"] = "fallback_accept"
            audit.setdefault("warnings", []).append("untranslatable_editor_intent_fallback_accept")
            return normalized
        suffix = "; ".join(reasons[:5]) or "no recognizable operation"
        raise ValueError(f"Editor intent is not translatable: {suffix}")
    return normalized

def validate_mas(mas: dict[str, Any]) -> list[str]:
    errors = []
    try:
        norm = normalize_mas(mas)
    except Exception as e:
        return [str(e)]
    ids = {a["id"] for a in norm["agents"]}
    if norm["aggregation"] not in ids:
        errors.append("aggregation_not_agent")
    for e in norm["edges"]:
        if e["from"] not in ids or e["to"] not in ids:
            errors.append("edge_unknown_agent")
    if sum(a.get("budget", 1) for a in norm["agents"]) > 16:
        errors.append("budget_overflow")
    if _mas_has_cycle(norm):
        errors.append("cycle_detected")
    return errors


def _mas_has_cycle(mas: dict[str, Any]) -> bool:
    graph = {str(a["id"]): [] for a in mas.get("agents", [])}
    for edge in mas.get("edges", []):
        fr, to = str(edge.get("from", "")), str(edge.get("to", ""))
        if fr in graph and to in graph:
            graph[fr].append(to)
    visiting, visited = set(), set()
    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(dfs(child) for child in graph[node]):
            return True
        visiting.remove(node); visited.add(node)
        return False
    return any(dfs(node) for node in graph)


def apply_patch(src_mas: dict[str, Any], patch: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Compile one strict, bounded Refiner edit; reject rather than repair it."""
    source = normalize_mas(copy.deepcopy(src_mas))
    try:
        p = normalize_editor_action(patch, src_mas=source)
    except (TypeError, ValueError) as exc:
        return source, [f"invalid_patch:{exc}"]
    ops = p.get("operations") or []
    audit = copy.deepcopy(p.get("translation") or {})
    if len(ops) != 1:
        return source, ["invalid_patch:exactly_one_operation_required"]
    op = ops[0]
    name = str(op.get("op") or "")
    if name == "accept":
        audit["status"] = "accepted"
        audit["noop_operations"] = [{"index": 0, "reason": "explicit_accept", "operation": copy.deepcopy(op)}]
        if isinstance(patch, dict): patch["translation"] = audit
        return source, []
    mas = copy.deepcopy(source)
    agents = {str(a["id"]): a for a in mas["agents"]}
    try:
        if name == "add_agent":
            aid = str(op["agent_id"])
            if aid in agents or len(mas["agents"]) >= 8:
                raise ValueError("agent_already_exists_or_limit_reached")
            mas["agents"].append({"id": aid, "role": str(op.get("role") or "Assist solving the task.")[:1000],
                                  "tools": [], "budget": int(op.get("budget", 1)), "execution_mode": "llm_only"})
        elif name == "rewrite_role":
            if str(op["agent_id"]) not in agents: raise ValueError("unknown_agent_reference")
            role = str(op.get("role") or "").strip()
            if not role: raise ValueError("missing_role")
            agents[str(op["agent_id"])] ["role"] = role[:1000]
        elif name == "change_budget":
            if str(op["agent_id"]) not in agents: raise ValueError("unknown_agent_reference")
            budget = int(op["budget"])
            if not 1 <= budget <= 4: raise ValueError("budget_out_of_bounds")
            agents[str(op["agent_id"])] ["budget"] = budget
        elif name == "replace_tool":
            if str(op["agent_id"]) not in agents: raise ValueError("unknown_agent_reference")
            agent = agents[str(op["agent_id"])]
            tools = [str(x) for x in agent.get("tools", [])]
            old, new = str(op.get("old_tool") or ""), str(op.get("new_tool") or "")
            if old:
                if old not in tools: raise ValueError("old_tool_not_found")
                if new: tools[tools.index(old)] = new
                else: tools.remove(old)
            elif new and new not in tools:
                tools.append(new)
            else:
                raise ValueError("missing_tool_change")
            if len(tools) > 8: raise ValueError("tool_limit_exceeded")
            agent["tools"] = tools
        elif name in {"add_edge", "remove_edge"}:
            fr, to = str(op["from"]), str(op["to"])
            if fr not in agents or to not in agents or fr == to: raise ValueError("invalid_edge_endpoints")
            edge = {"from": fr, "to": to, "type": str(op.get("type") or "data_flow")}
            if name == "add_edge":
                if edge in mas["edges"]: raise ValueError("edge_already_exists")
                if len(mas["edges"]) >= 32: raise ValueError("edge_limit_exceeded")
                mas["edges"].append(edge)
                if _mas_has_cycle(mas): raise ValueError("edge_would_create_cycle")
            else:
                before = len(mas["edges"])
                mas["edges"] = [e for e in mas["edges"] if not (e["from"] == fr and e["to"] == to and (not op.get("type") or e["type"] == edge["type"]))]
                if len(mas["edges"]) == before: raise ValueError("edge_not_found")
        else:
            raise ValueError(f"unsupported canonical operation {name}")
    except (KeyError, TypeError, ValueError) as exc:
        return source, [f"invalid_patch:{exc}"]
    errors = validate_mas(mas)
    if _mas_has_cycle(mas): errors.append("cycle_detected")
    if errors:
        return source, [f"final_validation:{item}" for item in errors]
    audit["applied_operations"] = [copy.deepcopy(op)]
    audit["status"] = "applied"
    audit["semantic_intent_preserved"] = True
    if isinstance(patch, dict): patch["translation"] = audit
    return normalize_mas(mas), []

