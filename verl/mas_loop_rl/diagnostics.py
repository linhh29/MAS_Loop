from __future__ import annotations

import os
import re
from typing import Any


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


_REFINER_SENSITIVE_CONSTRAINT_KEYS = {
    "answer",
    "correct_answer",
    "expected",
    "expected_answer",
    "gold",
    "ground_truth",
    "label",
    "reference_answer",
    "reference_solution",
    "solution",
    "target",
    "dataset",
    "evaluation_dataset",
    "benchmark",
    "benchmark_name",
    "query_id",
    "sample_id",
    "gold_docs",
    "gold_docids",
    "evidence_docs",
    "evidence_docids",
}


def _is_refiner_sensitive_constraint_key(key: Any) -> bool:
    normalized = str(key or "").strip().lower().replace("-", "_")
    if normalized in _REFINER_SENSITIVE_CONSTRAINT_KEYS:
        return True
    return normalized.startswith(("gold_", "ground_truth_", "expected_answer_", "correct_answer_", "reference_answer_"))


def sanitize_refiner_constraints(value: Any, *, depth: int = 0) -> Any:
    """Keep operational capabilities while removing benchmark and answer leakage.

    Refiner should see constraints that affect executable MAS design (available
    tools, retrieval requirements, output format, budgets), but not dataset
    names, labels, reference answers/solutions, or gold evidence identifiers.
    The traversal is recursive because dataset loaders often nest those fields
    under metadata or evaluation dictionaries.
    """
    if depth >= 6:
        return _bounded_text(value, 400)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:48]:
            if _is_refiner_sensitive_constraint_key(raw_key):
                continue
            key = str(raw_key)[:96]
            sanitized = sanitize_refiner_constraints(raw_value, depth=depth + 1)
            if sanitized not in (None, "", [], {}):
                cleaned[key] = sanitized
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize_refiner_constraints(item, depth=depth + 1) for item in list(value)[:32]]
    if isinstance(value, (str, bytes)):
        return _bounded_text(value.decode(errors="replace") if isinstance(value, bytes) else value, 400)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value, 400)


def _string_list(value: Any, *, limit: int, item_chars: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_bounded_text(item, item_chars) for item in list(value)[:limit] if str(item or "").strip()]


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w-]+", text or "") if len(token) > 2}


def _utilization_signal(output: str, final_output: str) -> dict[str, Any]:
    output_tokens = _token_set(output)
    final_tokens = _token_set(final_output)
    if not output_tokens or not final_tokens:
        return {"token_overlap": 0.0, "signal": "unavailable"}
    overlap = len(output_tokens & final_tokens) / max(1, len(final_tokens))
    return {
        "token_overlap": round(overlap, 3),
        "signal": "high" if overlap >= 0.5 else "partial" if overlap >= 0.15 else "low",
    }


def _retrieval_summary(trace: dict[str, Any], *, text_limit: int, card_index: int) -> dict[str, Any] | None:
    rounds = trace.get("search_rounds")
    artifact = trace.get("retrieval_artifact")
    errors = _string_list(trace.get("search_errors"), limit=5, item_chars=240)
    if not isinstance(rounds, list) and not isinstance(artifact, dict) and not errors:
        return None

    compact_rounds: list[dict[str, Any]] = []
    seen_docids: set[str] = set()
    repeated_queries = 0
    for index, raw_round in enumerate((rounds or [])[:8]):
        if not isinstance(raw_round, dict):
            continue
        query = raw_round.get("search_query", raw_round.get("query", ""))
        docids = raw_round.get("retrieved_docids") or raw_round.get("docids") or []
        new_docids = raw_round.get("new_docids_first_seen_this_round") or []
        docids = _string_list(docids, limit=12, item_chars=96)
        seen_docids.update(docids)
        repeated = bool(raw_round.get("query_repeated_from_prior_turn", False))
        repeated_queries += int(repeated)
        round_path = (
            f"node_cards[{card_index}].optional_capability_signals."
            f"retrieval.round_summaries[{len(compact_rounds)}]"
        )
        compact_rounds.append(
            {
                "round": int(raw_round.get("round", index + 1) or index + 1),
                "query": _bounded_text(query, min(text_limit, 320)),
                "retrieved_count": len(docids),
                "new_docids_count": len(new_docids) if isinstance(new_docids, list) else None,
                "query_repeated": repeated,
                "trace_refs": {
                    "query": f"{round_path}.query",
                    "retrieved_count": f"{round_path}.retrieved_count",
                    "new_docids_count": f"{round_path}.new_docids_count",
                    "query_repeated": f"{round_path}.query_repeated",
                },
            }
        )

    summary: dict[str, Any] = {
        "rounds_completed": len(compact_rounds),
        "unique_retrieved_artifacts": len(seen_docids),
        "repeated_query_rounds": repeated_queries,
        "round_summaries": compact_rounds,
    }
    if isinstance(artifact, dict):
        summary["artifact_summary"] = {
            key: artifact[key]
            for key in (
                "selected_document_count",
                "retrieval_profile",
                "doc_truncation_unit",
                "doc_truncation_limit",
            )
            if key in artifact
        }
    if errors:
        summary["errors"] = errors
    return summary


def build_node_diagnostic_state(
    src_mas: dict[str, Any],
    src_trajectory: dict[str, Any],
    task: str = "",
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build bounded, dataset-agnostic evidence for a Refiner decision.

    The state intentionally excludes expected/gold answers and raw transport
    diagnostics. Dataset-specific execution artifacts are included only when
    feature-detected, so non-retrieval tasks receive the same generic node and
    data-flow representation.
    """
    if not isinstance(src_mas, dict):
        src_mas = {}
    if not isinstance(src_trajectory, dict):
        src_trajectory = {}
    constraints = constraints if isinstance(constraints, dict) else {}

    text_limit = max(256, int(os.getenv("MAS_LOOP_EDITOR_DIAGNOSTIC_TEXT_CHARS", "1800")))
    role_limit = max(128, int(os.getenv("MAS_LOOP_EDITOR_DIAGNOSTIC_ROLE_CHARS", "800")))
    final_limit = max(256, int(os.getenv("MAS_LOOP_EDITOR_FINAL_OUTPUT_CHARS", "3000")))

    all_agents = [agent for agent in (src_mas.get("agents") or []) if isinstance(agent, dict)]
    aggregation = str(src_mas.get("aggregation") or "")
    # Keep diagnostics bounded while never dropping the aggregation node. Some
    # imported baseline MASes contain >8 agents; blindly slicing the first 8
    # falsely reported a valid aggregation as an unknown/unexecuted agent.
    agents = all_agents[:8]
    if aggregation and aggregation not in {str(agent.get("id") or "") for agent in agents}:
        aggregation_agent = next((agent for agent in all_agents if str(agent.get("id") or "") == aggregation), None)
        if aggregation_agent is not None:
            agents = (agents[:7] + [aggregation_agent]) if len(agents) >= 8 else (agents + [aggregation_agent])
    edges = [edge for edge in (src_mas.get("edges") or []) if isinstance(edge, dict)][:32]
    traces = [trace for trace in (src_trajectory.get("agent_traces") or []) if isinstance(trace, dict)]
    trace_by_id = {str(trace.get("agent_id") or ""): trace for trace in traces}
    final_output = _bounded_text(src_trajectory.get("final_output"), final_limit)

    predecessors: dict[str, list[str]] = {}
    successors: dict[str, list[str]] = {}
    for edge in edges:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if source and target:
            successors.setdefault(source, []).append(target)
            predecessors.setdefault(target, []).append(source)

    dependency_ids: set[str] = set()
    frontier = [aggregation] if aggregation else []
    while frontier:
        current = frontier.pop()
        if not current or current in dependency_ids:
            continue
        dependency_ids.add(current)
        frontier.extend(predecessors.get(current, []))

    cards: list[dict[str, Any]] = []
    for card_index, agent in enumerate(agents):
        execution_index = card_index
        agent_id = str(agent.get("id") or "")
        trace = trace_by_id.get(agent_id, {})
        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
        output = _bounded_text(trace.get("output"), text_limit)
        declared_upstream = _string_list(
            metadata.get("declared_upstream_agent_ids", predecessors.get(agent_id, [])),
            limit=16,
            item_chars=64,
        )
        available_upstream = _string_list(
            metadata.get("available_upstream_agent_ids"), limit=16, item_chars=64
        )
        hard_signals: list[str] = []
        if not trace:
            hard_signals.append("missing_execution_trace")
        if trace.get("error"):
            hard_signals.append("execution_error")
        if trace and not output:
            hard_signals.append("empty_output")
        unavailable = [item for item in declared_upstream if item not in set(available_upstream)]
        if unavailable and metadata.get("dependency_prompt_mode") == "declared_edges":
            hard_signals.append("declared_upstream_unavailable")
        if agent_id == aggregation and agent_id not in trace_by_id:
            hard_signals.append("aggregation_not_executed")
        retrieval = _retrieval_summary(trace, text_limit=text_limit, card_index=card_index)
        if retrieval and retrieval.get("errors"):
            hard_signals.append("retrieval_error")
        if retrieval and retrieval.get("rounds_completed", 0) > 1:
            if retrieval.get("repeated_query_rounds") == retrieval.get("rounds_completed") - 1:
                hard_signals.append("retrieval_queries_repeated")

        card: dict[str, Any] = {
            "agent_id": agent_id,
            "declared_responsibility": _bounded_text(agent.get("role"), role_limit),
            "tools": _string_list(agent.get("tools"), limit=8, item_chars=64),
            "execution_mode": str(agent.get("execution_mode") or "llm_only"),
            "budget": agent.get("budget"),
            "predecessors": predecessors.get(agent_id, [])[:16],
            "successors": successors.get(agent_id, [])[:16],
            "is_aggregation": agent_id == aggregation,
            "on_final_dependency_path": agent_id in dependency_ids,
            "execution_index": metadata.get("execution_index", execution_index),
            "execution_status": "error" if trace.get("error") else "completed" if trace else "missing",
            "output_summary": output,
            "upstream_received": available_upstream,
            "upstream_missing": unavailable,
            "downstream_declared": successors.get(agent_id, [])[:16],
            "final_output_utilization": _utilization_signal(output, final_output),
            "hard_mismatch_signals": hard_signals,
            "trace_refs": {
                "card": f"node_cards[{card_index}]",
                "declared_responsibility": f"node_cards[{card_index}].declared_responsibility",
                "output_summary": f"node_cards[{card_index}].output_summary",
                "execution_status": f"node_cards[{card_index}].execution_status",
                "upstream_received": f"node_cards[{card_index}].upstream_received",
                "upstream_missing": f"node_cards[{card_index}].upstream_missing",
                "hard_mismatch_signals": f"node_cards[{card_index}].hard_mismatch_signals",
                "final_output_utilization": f"node_cards[{card_index}].final_output_utilization",
            },
        }
        if trace.get("error"):
            card["error"] = _bounded_text(trace.get("error"), 600)
            card["trace_refs"]["error"] = f"node_cards[{card_index}].error"
        if retrieval is not None:
            card["optional_capability_signals"] = {"retrieval": retrieval}
        cards.append(card)

    known_ids = {str(agent.get("id") or "") for agent in all_agents}
    edge_cards = []
    for edge_index, edge in enumerate(edges):
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        edge_cards.append(
            {
                "edge_id": f"{source}->{target}",
                "from": source,
                "to": target,
                "type": str(edge.get("type") or "data_flow"),
                "source_executed": source in trace_by_id,
                "target_executed": target in trace_by_id,
                "valid_endpoints": source in known_ids and target in known_ids and source != target,
                "on_final_dependency_path": source in dependency_ids and target in dependency_ids,
                "trace_refs": {
                    "card": f"edge_cards[{edge_index}]",
                    "source_executed": f"edge_cards[{edge_index}].source_executed",
                    "target_executed": f"edge_cards[{edge_index}].target_executed",
                    "valid_endpoints": f"edge_cards[{edge_index}].valid_endpoints",
                    "on_final_dependency_path": f"edge_cards[{edge_index}].on_final_dependency_path",
                },
            }
        )

    metadata = src_trajectory.get("metadata") if isinstance(src_trajectory.get("metadata"), dict) else {}
    global_signals: list[str] = []
    if aggregation and aggregation not in known_ids:
        global_signals.append("aggregation_unknown_agent")
    if src_trajectory.get("failure_nodes"):
        global_signals.append("reported_failure_nodes")
    if src_trajectory.get("error_message"):
        global_signals.append("trajectory_error")
    if not final_output:
        global_signals.append("empty_final_output")

    ordered_dependency_path = [
        card["agent_id"]
        for card in sorted(cards, key=lambda item: int(item.get("execution_index", 0)))
        if card["agent_id"] in dependency_ids
    ]
    diagnostic_hints = [
        {"target_type": "agent", "target_id": card["agent_id"], "signals": card["hard_mismatch_signals"]}
        for card in cards
        if card["hard_mismatch_signals"] and card["agent_id"] in dependency_ids
    ][:8]

    safe_constraints = sanitize_refiner_constraints(constraints)
    hide_outcome_score = os.getenv("MAS_LOOP_EDITOR_HIDE_OUTCOME_SCORE", "1") in {"1", "true", "True"}
    outcome = {
        "final_output": final_output,
        "failure_nodes": _string_list(src_trajectory.get("failure_nodes"), limit=16, item_chars=64),
        "error_message": _bounded_text(src_trajectory.get("error_message"), 600),
        "global_signals": global_signals,
        "trace_refs": {
            "final_output": "outcome.final_output",
            "failure_nodes": "outcome.failure_nodes",
            "error_message": "outcome.error_message",
            "global_signals": "outcome.global_signals",
        },
    }
    if not hide_outcome_score:
        outcome["success"] = bool(src_trajectory.get("success", False))
        outcome["score"] = metadata.get("score")
        outcome["trace_refs"]["success"] = "outcome.success"

    return {
        "task_summary": _bounded_text(task, max(512, text_limit)),
        "mas_summary": {
            "topology": src_mas.get("topology"),
            "aggregation": aggregation,
            "agent_count": len(all_agents),
            "diagnosed_agent_count": len(agents),
            "omitted_agent_count": max(0, len(all_agents) - len(agents)),
            "edge_count": len(edges),
            "max_cost": src_mas.get("max_cost"),
            "max_iterations": src_mas.get("max_iterations"),
        },
        "final_dependency_path": ordered_dependency_path,
        "diagnostic_hints": diagnostic_hints,
        "node_cards": cards,
        "edge_cards": edge_cards,
        "outcome": outcome,
        "constraints": safe_constraints,
    }
