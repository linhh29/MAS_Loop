from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from .local_client import LocalChatClient, LocalChatError, strip_reasoning
from .math_score import AIMEScorer
from .prompts import SUBAGENT_SYSTEM
from .schema import normalize_mas
from .sharded_jsonl import append_jsonl_shard
try:
    from .bcp_retrieval import format_evidence, retrieve_bcp_evidence, retrieve_bcp_evidence_with_meta
except Exception:  # optional dependency; non-BCP tasks should still run
    format_evidence = None
    retrieve_bcp_evidence = None
    retrieve_bcp_evidence_with_meta = None


_BCP_SEARCH_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string", "minLength": 1},
        "reason": {"type": "string"},
    },
    "required": ["search_query", "reason"],
    "additionalProperties": False,
}

_BCP_GATE_CONTROLLER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "final"]},
        "search_query": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "search_query", "reason"],
    "additionalProperties": False,
}

_BCP_INITIAL_QUERY_INSTRUCTION = (
    "You are a search-query generator for a keyword-based retrieval system (BM25). "
    "Your ONLY job is to produce a fluent, natural-language search query sentence — you must NOT "
    "attempt to answer the question or provide analysis.\n\n"
    "Guidelines:\n"
    "- Write a complete, grammatical sentence that reads like a search request, e.g. "
    "\"Find the Spanish TV series whose creator was born in 1973\" or "
    "\"Identify the actress who debuted on television in 1995\".\n"
    "- Embed the most discriminative concrete clues (names, places, dates, titles) from "
    "the information below directly into the sentence, not as a keyword list.\n"
    "- Keep the sentence under ~50 words. Do not add explanations, commentary, or formatting.\n"
    "- Output ONLY the search query sentence, nothing else."
)

_GATE_CANONICAL_CURRENT_TASK = (
    "Read GLOBAL TASK CONTEXT for the full task, constraints, and required output format. "
    "Use UPSTREAM DEPENDENCIES DATA when present. "
    "Execute only the sub-task described in ROLE INSTRUCTION."
)


def _reasoning_execution_guidance(dataset: str, is_aggregation_agent: bool) -> tuple[str, str]:
    """Return optional public-task reasoning guidance for difficult closed-book evals.

    The legacy generic prompt asks every node for a concise contribution, which
    produced very short AIME/GPQA answers despite an 8192-token operator limit.
    This opt-in profile keeps the model-authored MAS intact while asking solver
    nodes to reason rigorously and the final node to audit rather than echo.
    """
    profile = os.getenv("MAS_LOOP_REASONING_PROMPT_PROFILE", "legacy").strip().lower()
    if profile not in {"reasoning_consensus_v1", "deepscaler_reasoning_v1"}:
        return SUBAGENT_SYSTEM, ""
    dataset = str(dataset or "").strip().lower()
    if dataset == "aime":
        system = (
            "You are a rigorous olympiad mathematics sub-agent. Follow the assigned role and produce a concise, "
            "checkable derivation. Never treat 'after checking', 'known result', a guessed root, symmetry, or a "
            "listed collection as proof without showing the decisive verification. Check exact arithmetic and every "
            "original condition. Do not blindly copy an upstream conclusion or emit <think> tags. End any final-answer "
            "responsibility with exactly one \\boxed{integer}."
        )
        certificate_checks = (
            "For finite enumeration, show a completeness argument and verify each listed item plus every arrangement "
            "multiplier; for algebra, expand factors or substitute candidates into the original equation; for geometry "
            "or probability, derive the event under the actual conditioned sample space; check all bounds, signs, "
            "positivity, integrality, overlaps, and whether rounding is requested. "
        )
        extra = (
            "\n\nExecution requirement: compute the decisive quantity yourself. " + certificate_checks
            + (
                "Build a short claim-by-claim comparison of upstream candidates, reject every unsupported decisive "
                "claim, redo the shortest decisive calculation from the original problem, and make the final non-empty "
                "line exactly \\boxed{integer}."
                if is_aggregation_agent
                else "Give enough explicit certificate detail for a downstream auditor to falsify or confirm your "
                "candidate, and finish with your candidate in \\boxed{integer}."
            )
        )
        return system, extra
    if dataset in {"gpqa", "gpqa_diamond", "gpqa-diamond"}:
        system = (
            "You are a careful graduate-level science multiple-choice specialist. Follow the assigned role and solve "
            "from the exact question and options using established science. Keep the useful reasoning compact: identify "
            "the decisive fact or calculation, distinguish the closest alternative, and do not invent unstated conditions. "
            "Do not choose by confidence, verbosity, or majority, and do not emit <think> tags. Any answer responsibility "
            "must end with exactly one of \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
        )
        if is_aggregation_agent:
            extra = (
                "\n\nExecution requirement: act as AOrchestra's main reviewer. Check the original task yourself, then "
                "review the independent specialist attempts. If they agree, preserve the answer unless you can reproduce "
                "a specific contradiction. If they disagree, resolve the shortest governing definition, equation, mechanism, "
                "structural invariant, or experimental implication that separates the leading options. Do not perform a "
                "broad speculative audit and do not reward the longer explanation. End with one boxed uppercase option."
            )
        else:
            extra = (
                "\n\nExecution requirement: solve independently from scratch. Briefly state the checked supporting facts, "
                "perform the decisive calculation or scientific comparison, challenge the closest distractor, and end with "
                "one boxed uppercase option."
            )
        return system, extra
    return SUBAGENT_SYSTEM, ""

_AGENT_EXECUTOR_LOCK = threading.Lock()
_AGENT_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_AGENT_EXECUTOR_PID = 0


def _agent_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return a process-local pool for dependency-independent MAS agents.

    This pool only overlaps genuine executions.  It does not memoize, merge,
    deduplicate, singleflight, or otherwise share any Subagent result.
    """
    global _AGENT_EXECUTOR, _AGENT_EXECUTOR_PID
    pid = os.getpid()
    with _AGENT_EXECUTOR_LOCK:
        if _AGENT_EXECUTOR is None or _AGENT_EXECUTOR_PID != pid:
            workers = max(1, int(os.getenv("MAS_LOOP_DAG_AGENT_WORKERS_PER_PROCESS", "64")))
            _AGENT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="mas-loop-dag-agent",
            )
            _AGENT_EXECUTOR_PID = pid
        return _AGENT_EXECUTOR


def _bcp_retrieval_profile() -> str:
    profile = os.getenv("MAS_LOOP_BCP_RETRIEVAL_PROFILE", "legacy").strip().lower()
    if profile not in {"legacy", "controller"}:
        raise ValueError(
            "MAS_LOOP_BCP_RETRIEVAL_PROFILE must be 'legacy' or 'controller', "
            f"got {profile!r}"
        )
    return profile


def _parse_search_query(raw: str, fallback: str) -> tuple[str, str, str | None]:
    """Parse a legacy planner turn without preventing its fixed-round search."""
    visible = strip_reasoning(raw)
    try:
        obj = json.loads(visible)
        if isinstance(obj, dict):
            query = str(obj.get("search_query") or "").strip()
            reason = str(obj.get("reason") or "").strip()
            if query:
                return query[:4096], reason, None
    except Exception as exc:
        parse_error = f"{type(exc).__name__}:{exc}"
    else:
        parse_error = "missing_nonempty_search_query"
    query = visible.strip() or str(fallback).strip()
    return query[:4096], "planner_parse_fallback", parse_error


def _parse_gate_controller(raw: str) -> tuple[str, str, str, str | None]:
    """Parse controller output with retrieval controller's JSON-first, regex-fallback semantics."""
    visible = strip_reasoning(raw).strip()
    parse_error: str | None = None
    try:
        obj = json.loads(visible)
        if not isinstance(obj, dict):
            raise ValueError("controller output is not an object")
    except Exception as exc:
        parse_error = f"{type(exc).__name__}:{exc}"
        action_match = re.search(r'"?action"?\s*:\s*"?(search|final)"?', visible, re.I)
        query_match = re.search(
            r'"?search_query"?\s*:\s*"((?:[^"\\]|\\.)*)"', visible, re.I
        )
        query = ""
        if query_match:
            encoded = query_match.group(1)
            query = (
                bytes(encoded, "utf-8").decode("unicode_escape")
                if "\\" in encoded else encoded
            )
        obj = {
            "action": action_match.group(1).lower() if action_match else "final",
            "search_query": query.strip(),
            "reason": "",
        }
    action = str(obj.get("action") or "final").strip().lower()
    query = str(obj.get("search_query") or "").strip()[:4096]
    reason = str(obj.get("reason") or "").strip()
    if action not in {"search", "final"}:
        action = "final"
        query = ""
        parse_error = parse_error or "invalid_action"
    return action, query, reason, parse_error


def _normalize_bcp_query(query: str) -> str:
    return " ".join(
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", str(query or ""))
        if len(token) > 2
    )


def _truncate_whitespace_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if max_words <= 0 or len(words) <= max_words:
        return str(text or "")
    return " ".join(words[:max_words])


def _stable_execution_order(mas: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a stable topological order, falling back safely for cyclic policy output."""
    agents = list(mas.get("agents") or [])
    by_id = {str(agent.get("id")): agent for agent in agents}
    original_rank = {str(agent.get("id")): index for index, agent in enumerate(agents)}
    incoming = {agent_id: 0 for agent_id in by_id}
    outgoing: dict[str, list[str]] = {agent_id: [] for agent_id in by_id}
    for edge in mas.get("edges") or []:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if source not in by_id or target not in by_id or source == target:
            continue
        outgoing[source].append(target)
        incoming[target] += 1
    ready = sorted((agent_id for agent_id, degree in incoming.items() if degree == 0), key=original_rank.get)
    ordered_ids: list[str] = []
    while ready:
        agent_id = ready.pop(0)
        ordered_ids.append(agent_id)
        for target in outgoing.get(agent_id, []):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort(key=original_rank.get)
    # normalize_mas preserves a malformed cycle rather than inventing semantics.
    # Execute any unresolved nodes in authored order; their unavailable upstream
    # values are omitted instead of leaking unrelated outputs into the prompt.
    ordered_ids.extend(agent_id for agent_id in by_id if agent_id not in ordered_ids)
    return [by_id[agent_id] for agent_id in ordered_ids]


def _stable_execution_levels(mas: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], bool]:
    """Return stable dependency-ready levels and whether the graph is acyclic."""
    agents = list(mas.get("agents") or [])
    by_id = {str(agent.get("id")): agent for agent in agents}
    original_rank = {str(agent.get("id")): index for index, agent in enumerate(agents)}
    incoming = {agent_id: 0 for agent_id in by_id}
    outgoing: dict[str, list[str]] = {agent_id: [] for agent_id in by_id}
    seen_edges: set[tuple[str, str]] = set()
    for edge in mas.get("edges") or []:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        pair = (source, target)
        if (
            source not in by_id
            or target not in by_id
            or source == target
            or pair in seen_edges
        ):
            continue
        seen_edges.add(pair)
        outgoing[source].append(target)
        incoming[target] += 1

    ready = sorted(
        (agent_id for agent_id, degree in incoming.items() if degree == 0),
        key=original_rank.get,
    )
    levels: list[list[dict[str, Any]]] = []
    completed = 0
    while ready:
        level_ids = list(ready)
        levels.append([by_id[agent_id] for agent_id in level_ids])
        completed += len(level_ids)
        next_ready: list[str] = []
        for agent_id in level_ids:
            for target in outgoing.get(agent_id, []):
                incoming[target] -= 1
                if incoming[target] == 0:
                    next_ready.append(target)
        ready = sorted(next_ready, key=original_rank.get)

    if completed != len(by_id):
        # A malformed cycle has no safe parallel dependency interpretation.
        # Preserve the old deterministic serial fallback for every node.
        return [[agent] for agent in _stable_execution_order(mas)], False
    return levels, True


def _declared_upstream_ids(mas: dict[str, Any], agent_id: str) -> list[str]:
    """Return unique declared predecessors in authored edge order."""
    result: list[str] = []
    for edge in mas.get("edges") or []:
        if str(edge.get("to") or "") != agent_id:
            continue
        source = str(edge.get("from") or "")
        if source and source not in result:
            result.append(source)
    return result


def _format_upstream_dependencies(upstream_outputs: dict[str, str]) -> str:
    return "\n\n".join(
        f"[{agent_id.upper()}]:\n{output}" for agent_id, output in upstream_outputs.items()
    )


def _build_gate_workflow_prompt(
    *,
    task: str,
    agent: dict[str, Any],
    upstream_outputs: dict[str, str],
) -> str:
    """Build the canonical prompt_override used by retrieval controller generated workflows."""
    prompt = f"--- ROLE INSTRUCTION ---\n{str(agent.get('role') or '').strip()}\n\n"
    prompt += f"--- GLOBAL TASK CONTEXT ---\n{str(task or '').strip()}\n\n"
    if upstream_outputs:
        prompt += "--- UPSTREAM DEPENDENCIES DATA ---\n"
        prompt += _format_upstream_dependencies(upstream_outputs) + "\n\n"
    prompt += f"--- YOUR CURRENT TASK ---\n{_GATE_CANONICAL_CURRENT_TASK}"
    return prompt


def _build_gate_synthesis_prompt(
    *, agent: dict[str, Any], evidence: str, active_user_prompt: str
) -> str:
    """Mirror SubAgentRequest.execute_async phase-3 synthesis byte-for-byte."""
    return (
        f"[Sub-agent role]\n{str(agent.get('id') or '')}\n\n"
        f"[Sub-task instruction]\n{str(agent.get('role') or '')}\n\n"
        f"[Retrieved evidence]\n{evidence}\n\n"
        f"[Task input]\n{active_user_prompt}\n\n"
        "Using the retrieved evidence as grounding, answer the initial question."
    )


def _build_gate_llm_only_prompt(*, agent: dict[str, Any], active_user_prompt: str) -> str:
    """Mirror retrieval controller's no-tool SubAgentRequest prompt."""
    return (
        f"[Sub-agent role]\n{str(agent.get('id') or '')}\n\n"
        f"[Sub-task instruction]\n{str(agent.get('role') or '')}\n\n"
        f"[Task input]\n{active_user_prompt}"
    )


class MASExecutor:
    def __init__(
        self, base_url: str, model: str, max_agent_tokens: int = 1024,
        manage_sleep: bool | None = None, max_model_len: int | None = None,
    ):
        self.client = LocalChatClient(
            base_url=base_url,
            model=model,
            manage_sleep=manage_sleep,
            max_model_len=max_model_len,
            active_role="subagent",
        )
        self.max_agent_tokens = max(1, int(max_agent_tokens))

    @staticmethod
    def _profile(record: dict[str, Any]) -> None:
        path_s = os.getenv("MAS_LOOP_ROLLOUT_PROFILE_PATH", "")
        if not path_s:
            log_dir = os.getenv(
                "MAS_LOOP_LOG_DIR",
                "/path/to/checkpoints/unscoped/rollouts",
            )
            path_s = str(Path(log_dir) / "rollout_profile.jsonl")
        path = Path(path_s)
        # An explicit legacy *.jsonl path names a logical stream.  Actual data
        # is written to process-local shards beside it, avoiding cross-process
        # locks while retaining a discoverable, durable rollout profile.
        stream_dir = path.parent / f"{path.stem}_shards" if path.suffix == ".jsonl" else path
        append_jsonl_shard(stream_dir, record, shard_prefix="profile")

    def _agent_token_budget(self, agent: dict[str, Any]) -> int:
        """Resolve an untrusted model-authored completion budget safely.

        Orchestra output is policy output, not trusted configuration.  Clamp
        its optional per-agent ``max_tokens`` to the operator supplied budget
        and, when known, to what can fit inside the local model context.  This
        prevents malformed actions such as ``max_tokens=1000000`` from causing
        repeated HTTP 400s while preserving the configured 8192 context length.
        """
        try:
            requested = int(agent.get("max_tokens") or self.max_agent_tokens)
        except (TypeError, ValueError):
            requested = self.max_agent_tokens
        upper = self.max_agent_tokens
        if self.client.max_model_len is not None:
            # Leave at least one prompt token plus the client's truncation
            # safety margin.  Normal operation is tighter (1024 <= 8192).
            context_upper = max(
                1,
                int(self.client.max_model_len) - int(self.client.truncate_safety_margin) - 1,
            )
            upper = min(upper, context_upper)
        return max(1, min(requested, upper))

    @staticmethod
    def _compact_retrieval(
        documents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Keep bounded retrieval provenance in the rollout pair without disk caching.

        BrowseComp corpus documents can be several megabytes each.  The model
        already consumes a bounded evidence view during this execution, so
        persisting complete BM25 query results again is redundant and can grow
        checkpoints by many gigabytes.  Pair logs retain doc ids, scores,
        lengths and short previews; no standalone query/result artifact is
        created.
        """
        preview_chars = max(0, int(os.getenv("MAS_LOOP_BM25_DOC_PREVIEW_CHARS", "1200")))
        compact: list[dict[str, Any]] = []
        omitted_text_chars = 0
        for doc in documents:
            if not isinstance(doc, dict):
                value = str(doc)
                compact.append({"value_preview": value[:preview_chars], "value_length": len(value)})
                omitted_text_chars += max(0, len(value) - preview_chars)
                continue
            item = {k: v for k, v in doc.items() if k != "text"}
            text = str(doc.get("text") or "")
            item["text_preview"] = text[:preview_chars]
            item["text_length"] = len(text)
            compact.append(item)
            omitted_text_chars += max(0, len(text) - preview_chars)
        return compact, {
            "persisted": False,
            "reason": "bm25_query_result_cache_disabled",
            "document_count": len(documents),
            "preview_chars_per_document": preview_chars,
            "omitted_text_chars": omitted_text_chars,
        }

    @staticmethod
    def _compact_search_session(
        search_turns: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compact one agent search into its canonical pair; never write a cache."""
        compact_turns: list[dict[str, Any]] = []
        document_count = 0
        omitted_text_chars = 0
        for turn in search_turns:
            planner_prompt = str(turn.get("planner_prompt") or "")
            compact_turn = {k: v for k, v in turn.items() if k not in {"documents", "planner_prompt"}}
            compact_turn["planner_prompt_preview"] = planner_prompt[:2000]
            compact_turn["planner_prompt_length"] = len(planner_prompt)
            compact_turn["planner_prompt_sha256"] = hashlib.sha256(planner_prompt.encode("utf-8")).hexdigest()
            compact_docs, summary = MASExecutor._compact_retrieval(list(turn.get("documents", [])))
            document_count += int(summary["document_count"])
            omitted_text_chars += int(summary["omitted_text_chars"])
            compact_turn["documents"] = compact_docs
            compact_turns.append(compact_turn)
        return compact_turns, {
            "persisted": False,
            "reason": "bm25_query_result_cache_disabled",
            "schema": "bcp_multi_turn_search_summary_v2",
            "search_round_count": len(search_turns),
            "document_count_with_round_duplicates": document_count,
            "omitted_text_chars": omitted_text_chars,
        }

    @staticmethod
    def _bcp_search_planner_prompt(
        task: str,
        agent: dict[str, Any],
        round_index: int,
        rounds: int,
        prior_queries: list[str],
        evidence_digest: str,
        collaboration_context: str,
        *,
        current_query: str = "",
        current_evidence: str = "",
        profile: str = "legacy",
    ) -> str:
        previous = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(prior_queries)) or "(none)"
        if profile == "controller":
            delegation_context = str(collaboration_context or "").strip()
            delegation_block = (
                "\n\nDelegation / planning context (hints only; each new search_query must still be "
                "distinct from prior queries and from trivial paraphrases):\n"
                f"{delegation_context[:8000]}\n"
                if delegation_context else ""
            )
            return (
                "You improve lexical (BM25) retrieval across rounds. Stay concise.\n"
                "Rules:\n"
                "- Output JSON only with keys: action, search_query, reason.\n"
                "- action is \"search\" or \"final\".\n"
                "- If action is \"search\", search_query MUST be a NEW query distinct from ALL prior queries "
                "(different entities, synonyms, facets, spellings, dates, OR complementary gaps).\n"
                "- Do NOT repeat or trivially shorten previous queries (avoid overlapping keyword bundles).\n"
                "- Base the next query on the QUESTION plus WHAT IS STILL MISSING given Evidence digest.\n"
                "- Use action=\"final\" when additional retrieval is unlikely to change the answer materially "
                "or when snippets already cover all constraints.\n\n"
                f"Question:\n{task.strip()}\n"
                f"{delegation_block}"
                f"\nRound: {round_index}/{rounds}\n"
                f"Current BM25 query:\n{current_query}\n\n"
                f"Queries already used (do NOT repeat):\n{previous}\n\n"
                f"Evidence digest so far:\n{evidence_digest or '(no evidence yet)'}\n\n"
                f"Top snippets this round:\n{current_evidence or '(none)'}\n"
            )
        return (
            "You are the search planner inside a BrowseComp-Plus research agent. "
            "Generate the lexical BM25 query for this search turn.\n"
            "Return JSON only: {\"search_query\": \"...\", \"reason\": \"...\"}.\n"
            "The query should be concise, use concrete entities/aliases/dates, and cover a new evidence gap. "
            "Do not answer the question yet and do not copy a previous query verbatim.\n\n"
            f"Search turn: {round_index}/{rounds}\n"
            f"Original question:\n{task}\n\n"
            f"Agent id: {agent.get('id', '')}\nAgent role: {agent.get('role', '')}\n\n"
            f"Queries already executed:\n{previous}\n\n"
            f"Evidence digest from earlier turns:\n{evidence_digest or '(no evidence yet)'}\n\n"
            f"Relevant collaboration context:\n{collaboration_context or '(none)'}\n"
        )


    @staticmethod
    def _format_gate_brief_snippets(documents: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for doc in documents[:6]:
            snippet = _truncate_whitespace_words(str(doc.get("text") or ""), 24)
            lines.append(f"- {doc.get('docid', '')}: {snippet}")
        return "\n".join(lines)

    @staticmethod
    def _format_gate_evidence(
        documents: list[dict[str, Any]], search_turns: list[dict[str, Any]]
    ) -> str:
        contexts = [
            f"[docid={doc.get('docid', '')}] {str(doc.get('text') or '')}"
            for doc in documents
        ]
        if contexts:
            base = (
                "Retrieved evidence (ground answers in these snippets only):\n\n"
                + "\n\n---\n\n".join(contexts)
            )
        else:
            base = "(No documents retrieved from the corpus.)"
        if not search_turns:
            return base
        lines = ["", "--- Multi-round retrieval trace (controller-guided BM25) ---"]
        for turn in search_turns:
            action = str(turn.get("action") or "").strip()
            query = str(turn.get("search_query") or "").strip().replace("\n", " ")
            docids = turn.get("retrieved_docids") or []
            new_docids = turn.get("new_docids_first_seen_this_round") or []
            lines.append(
                f"Round {turn.get('round', '?')}: planner_action={action!r} | "
                f"bm25_query={query!r} | bm25_hits={len(docids)} | "
                f"new_distinct_snippets={len(new_docids)}"
            )
        lines.append(
            f"Evidence snippets shown above (accumulated distinct snippets): {len(contexts)}"
        )
        return base + "\n".join(lines)

    @staticmethod
    def _select_final_evidence(
        search_turns: list[dict[str, Any]], profile: str = "legacy"
    ) -> list[dict[str, Any]]:
        """Select bounded evidence while preserving each profile's historical semantics."""
        if profile == "controller":
            max_docs = max(1, int(os.getenv("MAS_LOOP_BCP_MAX_FINAL_SNIPPETS", "15")))
            max_words = max(1, int(os.getenv("MAS_LOOP_BCP_DOC_MAX_TOKENS", "512")))
            selected: list[dict[str, Any]] = []
            seen: set[str] = set()
            for turn in search_turns:
                for doc in list(turn.get("documents", [])):
                    docid = str(doc.get("docid", ""))
                    identity = docid or hashlib.sha256(
                        str(doc.get("text", "")).encode("utf-8")
                    ).hexdigest()
                    if identity in seen:
                        continue
                    text = _truncate_whitespace_words(str(doc.get("text") or ""), max_words)
                    if not text:
                        continue
                    item = dict(doc)
                    item["text"] = text
                    selected.append(item)
                    seen.add(identity)
                    if len(selected) >= max_docs:
                        return selected
            return selected

        max_docs = max(1, int(os.getenv("MAS_LOOP_BCP_FINAL_MAX_DOCS", "15")))
        max_chars = max(1, int(os.getenv("MAS_LOOP_BCP_FINAL_MAX_CHARS", "18000")))
        per_doc_chars = max(1, int(os.getenv("MAS_LOOP_BM25_DOC_MAX_CHARS", "1200")))
        rows = [list(turn.get("documents", [])) for turn in search_turns]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        used_chars = 0
        max_rank = max((len(x) for x in rows), default=0)
        for rank in range(max_rank):
            for docs in rows:
                if rank >= len(docs):
                    continue
                doc = docs[rank]
                docid = str(doc.get("docid", ""))
                identity = docid or hashlib.sha256(str(doc.get("text", "")).encode("utf-8")).hexdigest()
                if identity in seen:
                    continue
                text = str(doc.get("text") or "")[:per_doc_chars]
                if not text:
                    continue
                remaining = max_chars - used_chars
                if remaining <= 0 or len(selected) >= max_docs:
                    return selected
                item = dict(doc)
                item["text"] = text[:remaining]
                selected.append(item)
                seen.add(identity)
                used_chars += len(item["text"])
        return selected

    def _run_bcp_multi_turn_search(
        self,
        task: str,
        agent: dict[str, Any],
        collaboration_context: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, float], list[str]]:
        """Run either the legacy planner-first loop or retrieval controller-compatible retrieval."""
        if retrieve_bcp_evidence_with_meta is None or format_evidence is None:
            raise RuntimeError("BrowseComp BM25 retrieval dependencies are unavailable")
        profile = _bcp_retrieval_profile()
        default_rounds = "5"
        rounds = max(1, int(os.getenv("MAS_LOOP_BCP_SEARCH_ROUNDS", default_rounds)))
        if profile == "controller":
            rounds = min(5, rounds)
        planner_tokens = max(32, int(os.getenv("MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS", "256")))
        planner_temperature = float(os.getenv("MAS_LOOP_BCP_SEARCH_PLANNER_TEMPERATURE", "1.0"))
        digest_chars = max(256, int(os.getenv("MAS_LOOP_BCP_SEARCH_DIGEST_CHARS", "10000")))
        collaboration_chars = max(0, int(os.getenv("MAS_LOOP_BCP_SEARCH_CONTEXT_CHARS", "5000")))
        prior_queries: list[str] = []
        used_query_norms: set[str] = set()
        digest_blocks: list[str] = []
        turns: list[dict[str, Any]] = []
        errors: list[str] = []
        planner_sec = 0.0
        bm25_sec = 0.0
        early_stop_reason = "max_rounds"

        initial_query_raw = ""
        initial_query_error: str | None = None
        initial_query_source = "not_applicable"
        initial_query_sec = 0.0
        if profile == "controller":
            explicit_query = str(agent.get("initial_bm25_query") or "").strip()[:4096]
            if explicit_query:
                current_query = explicit_query
                initial_query_source = "agent_config"
            else:
                # retrieval controller's SubAgentRequest performs a role-aware query-generation
                # call before browsecomp_retrieval_tool. The retrieval tool only falls
                # back to full_task_question when that generated query is empty.
                original_task = str(task or "").strip()
                upstream_data = _format_upstream_dependencies(
                    dict(agent.get("_gate_upstream_outputs") or {})
                )
                current_task = _GATE_CANONICAL_CURRENT_TASK
                query_prompt = _BCP_INITIAL_QUERY_INSTRUCTION + "\n\n---\n\n"
                if original_task:
                    query_prompt += f"[Original question]\n{original_task}\n\n"
                if str(agent.get("role") or "").strip():
                    query_prompt += f"[Role / sub-agent purpose]\n{str(agent.get('role')).strip()}\n\n"
                if upstream_data:
                    query_prompt += f"[Upstream agent outputs]\n{upstream_data}\n\n"
                if current_task:
                    query_prompt += f"[Current sub-task]\n{current_task}\n\n"
                if not (original_task or upstream_data or current_task):
                    query_prompt += f"[Full task input]\n{collaboration_context}\n\n"
                query_prompt += "Generate the BM25 search query now."
                initial_start = time.time()
                try:
                    initial_query_raw = self.client.chat_once(
                        [{"role": "user", "content": query_prompt}],
                        max_tokens=max(
                            32,
                            int(os.getenv("MAS_LOOP_BCP_INITIAL_QUERY_MAX_TOKENS", str(self.max_agent_tokens))),
                        ),
                        temperature=float(os.getenv("MAS_LOOP_BCP_SUBAGENT_TEMPERATURE", "1.0")),
                    )
                    current_query = strip_reasoning(initial_query_raw).strip()[:4096]
                    initial_query_source = "controller_query_generator"
                    if not current_query:
                        initial_query_error = "empty_generated_query"
                        current_query = original_task[:4096]
                        initial_query_source = "full_task_question_fallback"
                except Exception as exc:
                    initial_query_error = f"{type(exc).__name__}:{exc}"
                    current_query = original_task[:4096]
                    initial_query_source = "full_task_question_fallback"
                finally:
                    initial_query_sec = time.time() - initial_start
            if not current_query:
                raise ValueError("retrieval controller retrieval requires a non-empty task or initial_bm25_query")
            normalized = _normalize_bcp_query(current_query)
            if normalized:
                used_query_norms.add(normalized)
        else:
            current_query = ""

        for round_index in range(1, rounds + 1):
            planner_prompt = ""
            planner_raw = ""
            planner_error: str | None = None
            action = "search"
            reason = ""

            if profile == "legacy":
                planner_prompt = self._bcp_search_planner_prompt(
                    task=task,
                    agent=agent,
                    round_index=round_index,
                    rounds=rounds,
                    prior_queries=prior_queries,
                    evidence_digest="\n\n".join(digest_blocks)[-digest_chars:],
                    collaboration_context=collaboration_context[-collaboration_chars:] if collaboration_chars else "",
                    profile=profile,
                )
                planner_start = time.time()
                fallback = task if round_index == 1 else f"{task} {agent.get('role', '')} evidence facet {round_index}"
                try:
                    planner_raw = self.client.chat_once(
                        [
                            {"role": "system", "content": "You plan BM25 searches. Output valid JSON only."},
                            {"role": "user", "content": planner_prompt},
                        ],
                        max_tokens=planner_tokens,
                        temperature=planner_temperature,
                        guided_json=_BCP_SEARCH_QUERY_SCHEMA,
                    )
                    current_query, reason, planner_error = _parse_search_query(planner_raw, fallback)
                except Exception as exc:
                    planner_error = f"{type(exc).__name__}:{exc}"
                    current_query, reason = fallback[:4096], "planner_call_fallback"
                planner_sec += time.time() - planner_start
                if planner_error:
                    errors.append(f"search_round_{round_index}:{planner_error}")

            retrieval_start = time.time()
            try:
                documents, cache_meta = retrieve_bcp_evidence_with_meta(current_query)
                retrieval_error = None
            except Exception as exc:
                documents = []
                retrieval_error = f"{type(exc).__name__}:{exc}"
                cache_meta = {
                    "cache_hit": False,
                    "cache_key_scope": "exact_query_topk_index",
                    "query": current_query,
                    "topk": int(os.getenv("MAS_LOOP_BM25_TOPK", "5")),
                    "error": retrieval_error,
                }
                errors.append(f"search_round_{round_index}:bm25:{retrieval_error}")
            bm25_sec += time.time() - retrieval_start

            if profile == "controller":
                current_brief = self._format_gate_brief_snippets(documents)
                # Match retrieval controller: digest the last 12 first-seen accumulated context
                # blocks, truncating each complete "[docid=...] text" block to 14 words.
                digest_seen: set[str] = set()
                accumulated_contexts: list[str] = []
                for old_turn in turns + [{"documents": documents}]:
                    for doc in old_turn.get("documents", []):
                        identity = str(doc.get("docid", "")) or hashlib.sha256(
                            str(doc.get("text", "")).encode("utf-8")
                        ).hexdigest()
                        if identity in digest_seen:
                            continue
                        digest_seen.add(identity)
                        accumulated_contexts.append(
                            f"[docid={doc.get('docid', '')}] {str(doc.get('text') or '')}"
                        )
                digest_lines = [
                    "- " + _truncate_whitespace_words(block.replace("\n", " "), 14)
                    for block in accumulated_contexts[-12:]
                ]
                evidence_digest = ("\n".join(digest_lines) or "(no evidence yet)")[-digest_chars:]
                planner_prompt = self._bcp_search_planner_prompt(
                    task=task,
                    agent=agent,
                    round_index=round_index,
                    rounds=rounds,
                    prior_queries=prior_queries + [current_query],
                    evidence_digest=evidence_digest,
                    collaboration_context=collaboration_context[-collaboration_chars:] if collaboration_chars else "",
                    current_query=current_query,
                    current_evidence=current_brief,
                    profile=profile,
                )
                planner_start = time.time()
                try:
                    planner_raw = self.client.chat_once(
                        [
                            {"role": "system", "content": "You are a retrieval planner. Output JSON only."},
                            {"role": "user", "content": planner_prompt},
                        ],
                        max_tokens=planner_tokens,
                        temperature=planner_temperature,
                        response_format={"type": "json_object"},
                    )
                    action, next_query, reason, planner_error = _parse_gate_controller(planner_raw)
                except Exception as exc:
                    action, next_query, reason = "final", "", "planner_call_fallback"
                    planner_error = f"{type(exc).__name__}:{exc}"
                planner_sec += time.time() - planner_start
                if planner_error:
                    errors.append(f"search_round_{round_index}:{planner_error}")
            else:
                next_query = ""

            previously_seen_docids = {
                str(doc.get("docid", ""))
                for old_turn in turns
                for doc in old_turn.get("documents", [])
                if str(doc.get("docid", ""))
            }
            retrieved_docids = [str(doc.get("docid", "")) for doc in documents]
            new_docids_first_seen_this_round: list[str] = []
            for docid in retrieved_docids:
                if docid and docid not in previously_seen_docids and docid not in new_docids_first_seen_this_round:
                    new_docids_first_seen_this_round.append(docid)

            turn = {
                "round": round_index,
                "planner_prompt": planner_prompt,
                "planner_raw_output": planner_raw,
                "planner_parse_error": planner_error,
                "search_query": current_query,
                "reason": reason,
                "action": action,
                "next_search_query": next_query,
                "query_repeated_from_prior_turn": current_query in prior_queries,
                "bm25_cache": cache_meta,
                "bm25_error": retrieval_error,
                "retrieved_docids": retrieved_docids,
                "new_docids_first_seen_this_round": new_docids_first_seen_this_round,
                "documents": documents,
            }
            turns.append(turn)
            prior_queries.append(current_query)

            if profile == "legacy":
                brief = format_evidence(
                    documents,
                    max_chars_per_doc=min(400, int(os.getenv("MAS_LOOP_BM25_DOC_MAX_CHARS", "1200"))),
                )
                digest_blocks.append(f"[turn {round_index}, query={current_query!r}]\n{brief}")
                continue

            if action != "search":
                early_stop_reason = "planner_final"
                break
            next_norm = _normalize_bcp_query(next_query)
            if not next_norm:
                turn["rejected_next_query"] = next_query
                turn["reject_reason"] = "empty_search_query"
                early_stop_reason = "empty_search_query"
                break
            if next_norm in used_query_norms:
                turn["rejected_next_query"] = next_query
                turn["reject_reason"] = "duplicate_normalized"
                early_stop_reason = "duplicate_normalized"
                break
            used_query_norms.add(next_norm)
            current_query = next_query[:4096]

        selected = self._select_final_evidence(turns, profile=profile)
        evidence = (
            self._format_gate_evidence(selected, turns)
            if profile == "controller" else format_evidence(selected)
        )
        compact_turns, artifact = self._compact_search_session(turns)
        artifact.update({
            "retrieval_profile": profile,
            "initial_query": prior_queries[0] if prior_queries else "",
            "initial_query_source": initial_query_source,
            "initial_query_raw_output": initial_query_raw[:1000],
            "initial_query_error": initial_query_error,
            "initial_query_generation_sec": initial_query_sec,
            "evidence_policy": "accumulated" if profile == "controller" else "rank_interleaved_accumulated",
            "rounds_requested": rounds,
            "rounds_completed": len(turns),
            "early_stop_reason": early_stop_reason,
            "max_final_snippets": (
                max(1, int(os.getenv("MAS_LOOP_BCP_MAX_FINAL_SNIPPETS", "15")))
                if profile == "controller" else max(1, int(os.getenv("MAS_LOOP_BCP_FINAL_MAX_DOCS", "15")))
            ),
            "doc_truncation_unit": "whitespace_words" if profile == "controller" else "characters",
            "doc_truncation_limit": (
                max(1, int(os.getenv("MAS_LOOP_BCP_DOC_MAX_TOKENS", "512")))
                if profile == "controller" else max(1, int(os.getenv("MAS_LOOP_BM25_DOC_MAX_CHARS", "1200")))
            ),
            "selected_document_count": len(selected),
        })
        timing = {"planner_sec": planner_sec, "bm25_sec": bm25_sec, "total_sec": planner_sec + bm25_sec}
        return evidence, compact_turns, artifact, timing, errors

    def execute_alfworld(
        self,
        task: str,
        mas: dict[str, Any],
        episode_row: dict[str, Any],
        *,
        data_root: str | Path,
        profile_phase: str = "",
        pair_id: str = "",
        episode_factory: Any | None = None,
        tool_factory: Any | None = None,
    ) -> dict[str, Any]:
        """Execute one authored MAS against one fresh, stateful ALFWorld episode.

        ALFWorld has a single mutable simulator state, so authored agents run in
        stable topological order rather than DAG-level parallelism. Every call
        creates and closes its own episode; source and updated rollouts therefore
        cannot accidentally share an already-mutated environment.
        """
        start = time.time()
        mas = normalize_mas(mas)
        row = dict(episode_row or {})
        game_file = Path(str(row.get("game_file") or ""))
        if not str(game_file) or game_file.is_absolute() or ".." in game_file.parts:
            # Security: manifests are data inputs; reject absolute/traversal paths
            # before the adapter joins them to the operator-controlled data root.
            raise ValueError(f"invalid ALFWorld game_file: {game_file}")
        resolved_root = Path(data_root).expanduser().resolve()
        if episode_factory is None or tool_factory is None:
            from datasets.alfworld.env_adapter import ALFWorldEpisode
            from datasets.interactive_env.interactive_tool import InteractiveEpisodeTool
            episode_factory = episode_factory or ALFWorldEpisode
            tool_factory = tool_factory or InteractiveEpisodeTool

        episode = episode_factory(row, data_root=resolved_root)
        execution_order = _stable_execution_order(mas)
        actions_per_node = max(1, int(os.getenv("MAS_LOOP_ALFWORLD_ACTIONS_PER_NODE", "4")))
        max_recovery_rounds = max(0, int(os.getenv("MAS_LOOP_ALFWORLD_MAX_RECOVERY_ROUNDS", "12")))
        traces: list[dict[str, Any]] = []
        outputs_by_id: dict[str, str] = {}
        failures: list[str] = []
        final = ""
        total_tokens = 0

        async def run_episode() -> dict[str, Any]:
            nonlocal final, total_tokens
            initial = await asyncio.to_thread(episode.reset)
            live_context = (
                f"Original task: {task}\n\n"
                "LIVE ALFWORLD ENVIRONMENT (authoritative):\n"
                f"Task: {initial.get('task_description', '')}\n"
                f"Initial observation: {initial.get('observation', '')}\n"
                f"Initial inventory: {initial.get('inventory', '')}\n"
                "Use only exact admissible environment commands. Do not claim success unless the live environment reports it."
            )

            async def action_text(prompt: str) -> tuple[str, dict[str, Any]]:
                diagnostics: dict[str, Any] = {}
                output = await asyncio.to_thread(
                    self.client.chat_once,
                    [{"role": "system", "content": SUBAGENT_SYSTEM}, {"role": "user", "content": str(prompt)}],
                    temperature=0.0,
                    max_tokens=self.max_agent_tokens,
                    response_format={"type": "json_object"},
                    diagnostics_out=diagnostics,
                )
                return output, diagnostics

            tool = tool_factory(episode, action_text, actions_per_node=actions_per_node)
            tool.started = True
            for execution_index, agent in enumerate(execution_order):
                before = await asyncio.to_thread(episode.snapshot)
                if before.get("done") or before.get("success"):
                    break
                upstream_ids = _declared_upstream_ids(mas, str(agent["id"]))
                upstream = "\n".join(
                    f"[{agent_id} output]\n{outputs_by_id[agent_id]}"
                    for agent_id in upstream_ids if agent_id in outputs_by_id
                )
                agent_start = time.time()
                error = None
                usage: dict[str, Any] = {}
                try:
                    summary, usage = await tool({
                        "role_name": str(agent["id"]),
                        "role_instruction": str(agent.get("role") or "Complete the task using the live environment."),
                        "task_input": live_context + (f"\n\nUPSTREAM RESULTS:\n{upstream}" if upstream else ""),
                    })
                    output = str(summary or "")
                except Exception as exc:
                    output = ""
                    error = f"{type(exc).__name__}: {exc}"
                    failures.append(str(agent["id"]))
                after = await asyncio.to_thread(episode.snapshot)
                for key in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"):
                    try:
                        total_tokens += int((usage or {}).get(key, 0) or 0)
                    except (TypeError, ValueError):
                        pass
                outputs_by_id[str(agent["id"])] = output
                traces.append({
                    "agent_id": str(agent["id"]),
                    "agent_role": str(agent.get("role") or ""),
                    "output": output,
                    "usage": usage or {},
                    "error": error,
                    "elapsed_sec": time.time() - agent_start,
                    "search_mode": "none",
                    "metadata": {
                        "phase": profile_phase, "pair_id": pair_id,
                        "execution_index": execution_index,
                        "interactive_environment": "alfworld",
                        "steps_before": int(before.get("steps", 0) or 0),
                        "steps_after": int(after.get("steps", 0) or 0),
                        "environment_success": bool(after.get("success")),
                    },
                })
                if str(agent["id"]) == str(mas.get("aggregation")):
                    final = output

            for recovery_index in range(max_recovery_rounds):
                before = await asyncio.to_thread(episode.snapshot)
                if before.get("done") or before.get("success"):
                    break
                summary, usage = await tool({
                    "role_name": "global_reactive_recovery_executor",
                    "role_instruction": (
                        "Complete the benchmark task from the current live state. Replan after every observation, "
                        "use only exact admissible commands, and stop when the environment succeeds."
                    ),
                    "task_input": live_context,
                })
                after = await asyncio.to_thread(episode.snapshot)
                output = str(summary or "")
                final = output or final
                traces.append({
                    "agent_id": f"recovery_{recovery_index + 1}",
                    "agent_role": "global_reactive_recovery_executor",
                    "output": output, "usage": usage or {}, "error": None,
                    "elapsed_sec": 0.0, "search_mode": "none",
                    "metadata": {
                        "phase": profile_phase, "pair_id": pair_id,
                        "interactive_environment": "alfworld",
                        "steps_before": int(before.get("steps", 0) or 0),
                        "steps_after": int(after.get("steps", 0) or 0),
                        "environment_success": bool(after.get("success")),
                        "recovery_round": recovery_index + 1,
                    },
                })
                if int(after.get("steps", 0) or 0) <= int(before.get("steps", 0) or 0):
                    break
            return await asyncio.to_thread(episode.result)

        try:
            with self.client.awake_session():
                environment_result = asyncio.run(run_episode())
        finally:
            episode.close()
        if not final and traces:
            final = str(traces[-1].get("output") or "")
        score = float(environment_result.get("goal_condition_success_rate", 0.0) or 0.0)
        elapsed = time.time() - start
        result = {
            "task_text": task, "mas_config": mas, "agent_traces": traces,
            "final_output": final, "success": bool(environment_result.get("won", 0)),
            "total_cost_usd": 0.0, "total_tokens": total_tokens,
            "elapsed_sec": elapsed, "failure_nodes": failures, "error_message": None,
            "metadata": {
                "score": score, "reward_source": "alfworld_environment",
                "environment_result": environment_result,
                "interactive_environment": "alfworld",
                "rollout_timing": {
                    "retrieval_sec": 0.0, "search_planner_sec": 0.0, "bm25_sec": 0.0,
                    "generation_sec": elapsed, "final_generation_sec": elapsed, "total_sec": elapsed,
                    "phase": profile_phase, "pair_id": pair_id,
                    "agent_count": len(mas["agents"]), "generated_agents": len(traces),
                    "dependency_prompt_mode": "declared_edges_stateful_sequential",
                    "dag_agent_parallel": False, "graph_is_acyclic": True,
                    "execution_level_count": len(execution_order),
                    "execution_level_widths": [1] * len(execution_order),
                },
            },
        }
        self._profile({
            "time": time.time(), "event": "mas_execute_alfworld",
            "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
            "model": getattr(self.client, "model", ""), "phase": profile_phase,
            "pair_id": pair_id, **result["metadata"]["rollout_timing"],
        })
        return result

    def execute(
        self,
        task: str,
        mas: dict[str, Any],
        expected_answer: str = "",
        profile_phase: str = "",
        pair_id: str = "",
    ) -> dict[str, Any]:
        start = time.time()
        mas = normalize_mas(mas)
        traces: list[dict[str, Any]] = []
        legacy_context = f"Original task:\n{task}\n"
        outputs_by_id: dict[str, str] = {}
        execution_order = _stable_execution_order(mas)
        execution_levels, graph_is_acyclic = _stable_execution_levels(mas)
        final = ""
        total_tokens = 0
        failures: list[str] = []
        generated_agents = 0
        retrieval_sessions: list[dict[str, Any]] = []
        retrieval_errors: list[str] = []
        retrieval_sec = 0.0
        planner_sec = 0.0
        bm25_sec = 0.0
        generation_start = time.time()
        final_generation_sec = 0.0
        use_bcp_search = os.getenv("MAS_LOOP_RETRIEVER", "").lower() in {"bm25", "bcp_bm25", "browsecomp_bm25", "hotpotqa_bm25", "hotpot_bm25"}
        retrieval_profile = _bcp_retrieval_profile() if use_bcp_search else "none"
        default_rounds = "5"
        configured_rounds = int(os.getenv("MAS_LOOP_BCP_SEARCH_ROUNDS", default_rounds)) if use_bcp_search else 0
        if retrieval_profile == "controller":
            configured_rounds = min(5, max(1, configured_rounds))
        subagent_temperature = float(os.getenv("MAS_LOOP_BCP_SUBAGENT_TEMPERATURE", "1.0"))
        dependency_prompt_mode = os.getenv(
            "MAS_LOOP_DEPENDENCY_PROMPT_MODE",
            "declared_edges" if retrieval_profile == "controller" else "legacy_cumulative",
        ).strip().lower()
        if dependency_prompt_mode not in {"legacy_cumulative", "declared_edges"}:
            raise ValueError(
                "MAS_LOOP_DEPENDENCY_PROMPT_MODE must be legacy_cumulative or declared_edges"
            )
        use_declared_dependency_prompt = dependency_prompt_mode == "declared_edges"
        parallel_requested = os.getenv("MAS_LOOP_DAG_AGENT_PARALLEL", "0") not in {
            "0", "false", "False",
        }
        parallel_agents = parallel_requested and use_declared_dependency_prompt and graph_is_acyclic

        def execute_one_agent(
            a: dict[str, Any],
            *,
            execution_index: int,
            execution_level: int,
            upstream_outputs: dict[str, str],
            legacy_context_snapshot: str,
        ) -> dict[str, Any]:
            agent_start = time.time()
            search_turns: list[dict[str, Any]] = []
            search_artifact: dict[str, Any] | None = None
            search_timing = {"planner_sec": 0.0, "bm25_sec": 0.0, "total_sec": 0.0}
            search_errors: list[str] = []
            evidence = ""
            execution_mode = str(a.get("execution_mode") or "llm_only").strip().lower()
            agent_uses_search = use_bcp_search and execution_mode == "multi_turn_search"
            upstream_ids = _declared_upstream_ids(mas, str(a["id"]))
            active_dependency_prompt = ""
            prompt_agent = a
            if use_declared_dependency_prompt:
                active_dependency_prompt = _build_gate_workflow_prompt(
                    task=task, agent=a, upstream_outputs=upstream_outputs
                )
                prompt_agent = dict(a)
                prompt_agent["_gate_upstream_outputs"] = upstream_outputs
            if agent_uses_search:
                try:
                    evidence, search_turns, search_artifact, search_timing, search_errors = self._run_bcp_multi_turn_search(
                        task,
                        prompt_agent,
                        (
                            f"[Role constraints]\n{str(a.get('role') or '').strip()}\n\n"
                            f"[Sub-task input]\n{active_dependency_prompt}"
                            if use_declared_dependency_prompt else legacy_context_snapshot
                        ),
                    )
                except Exception as exc:
                    search_errors = [f"search_session:{type(exc).__name__}:{exc}"]

            is_aggregation_agent = a["id"] == mas["aggregation"]
            if use_declared_dependency_prompt:
                if agent_uses_search:
                    prompt = _build_gate_synthesis_prompt(
                        agent=a,
                        evidence=evidence,
                        active_user_prompt=active_dependency_prompt,
                    )
                else:
                    prompt = _build_gate_llm_only_prompt(
                        agent=a, active_user_prompt=active_dependency_prompt
                    )
            else:
                prompt = legacy_context_snapshot + f"\nYour agent id: {a['id']}\nYour role: {a['role']}\n"
                if agent_uses_search:
                    prompt += (
                        f"\nYour independent BrowseComp-Plus search ({len(search_turns)} round(s)) has completed. "
                        "Use the accumulated evidence below and resolve conflicts.\n"
                        + (evidence or "(No BM25 evidence was retrieved.)") + "\n"
                    )
                prompt += "\nProduce your contribution to solving the original task."

            dataset_name = os.getenv("MAS_LOOP_DATASET_NAME", "")
            subagent_system, reasoning_guidance = _reasoning_execution_guidance(
                dataset_name, is_aggregation_agent
            )
            if reasoning_guidance:
                prompt += reasoning_guidance

            raw_out = ""
            chat_diagnostics: dict[str, Any] = {}
            generation_call_start = time.monotonic()
            try:
                # retrieval controller retrieval historically uses a user-only declared-edge
                # prompt.  Closed-book AIME/GPQA, however, benefit from the
                # dataset-specific public reasoning contract in the system
                # message; omitting it silently discarded the strongest
                # anti-copying and exact-answer instructions.
                reasoning_dataset = str(dataset_name or "").strip().lower() in {
                    "aime", "gpqa", "gpqa_diamond", "gpqa-diamond",
                }
                messages = (
                    [
                        {"role": "system", "content": subagent_system},
                        {"role": "user", "content": prompt},
                    ]
                    if (not use_declared_dependency_prompt or reasoning_dataset)
                    else [{"role": "user", "content": prompt}]
                )
                # The operator ceiling remains unchanged (8192 for HotpotQA),
                # while a valid model-authored per-agent budget may request a
                # smaller genuine generation.  This is not result reuse or
                # caching; the agent still executes exactly once.
                generation_max_tokens = (
                    self.max_agent_tokens
                    if retrieval_profile == "controller"
                    else self._agent_token_budget(a)
                )
                try:
                    raw_out = self.client.chat_once(
                        messages,
                        max_tokens=generation_max_tokens,
                        temperature=subagent_temperature,
                        diagnostics_out=chat_diagnostics,
                    )
                except TypeError as exc:
                    if "diagnostics_out" not in str(exc) or "unexpected keyword" not in str(exc):
                        raise
                    raw_out = self.client.chat_once(
                        messages,
                        max_tokens=generation_max_tokens,
                        temperature=subagent_temperature,
                    )
                out = strip_reasoning(raw_out)
                err = None
            except Exception as exc:
                out = ""
                err = str(exc)
                if isinstance(exc, LocalChatError):
                    chat_diagnostics = {
                        "attempt_count": len(exc.diagnostics),
                        "retry_count": max(0, len(exc.diagnostics) - 1),
                        "stream": any(bool(x.get("stream")) for x in exc.diagnostics),
                        "attempts": list(exc.diagnostics),
                    }
            generation_call_sec = time.monotonic() - generation_call_start
            trace = {
                "agent_id": a["id"],
                "agent_role": a["role"],
                "output": out,
                "usage": {},
                "error": err,
                "elapsed_sec": time.time() - agent_start,
                "search_mode": "bcp_multi_turn_bm25" if agent_uses_search else "none",
                "search_rounds": search_turns,
                "retrieval_artifact": search_artifact,
                "search_errors": search_errors,
                "metadata": {
                    "raw_output": raw_out if raw_out != out else "",
                    "phase": profile_phase,
                    "pair_id": pair_id,
                    "prompt_chars": len(prompt),
                    "prompt_word_estimate": len(prompt.split()),
                    "output_chars": len(out),
                    "generation_sec": generation_call_sec,
                    "chat_diagnostics": chat_diagnostics,
                    "retrieval_profile": retrieval_profile,
                    "execution_mode": execution_mode,
                    "is_aggregation_agent": is_aggregation_agent,
                    "subagent_temperature": subagent_temperature,
                    "execution_index": execution_index,
                    "execution_level": execution_level,
                    "declared_upstream_agent_ids": upstream_ids,
                    "available_upstream_agent_ids": list(upstream_outputs),
                    "dependency_prompt_mode": dependency_prompt_mode,
                    "dag_level_parallel": parallel_agents,
                },
            }
            retrieval_session = None
            if agent_uses_search:
                retrieval_session = {
                    "agent_id": a["id"],
                    "agent_role": a["role"],
                    "retrieval_profile": retrieval_profile,
                    "rounds_requested": configured_rounds,
                    "rounds_completed": len(search_turns),
                    "search_turns": search_turns,
                    "retrieval_artifact": search_artifact,
                    "timing": search_timing,
                    "errors": search_errors,
                }
            return {
                "agent_id": str(a["id"]),
                "output": out,
                "trace": trace,
                "prompt_words": len(prompt.split()),
                "output_words": len(out.split()),
                "failed": err is not None,
                "retrieval_session": retrieval_session,
                "retrieval_errors": [f"{a['id']}:{item}" for item in search_errors],
                "search_timing": search_timing,
                "generation_call_sec": generation_call_sec,
                "is_aggregation_agent": is_aggregation_agent,
            }

        index_by_id = {
            str(agent["id"]): index for index, agent in enumerate(execution_order)
        }
        levels_to_run = (
            execution_levels
            if parallel_agents else [[agent] for agent in execution_order]
        )

        # One wake cycle covers all planner turns and all final sub-agent calls.
        # Every submitted future performs its own real retrieval/planner/final
        # generation path; only dependency-independent nodes overlap in time.
        with self.client.awake_session():
            for execution_level, level_agents in enumerate(levels_to_run):
                level_upstream: dict[str, dict[str, str]] = {}
                for agent in level_agents:
                    upstream_ids = _declared_upstream_ids(mas, str(agent["id"]))
                    level_upstream[str(agent["id"])] = {
                        agent_id: outputs_by_id[agent_id]
                        for agent_id in upstream_ids
                        if agent_id in outputs_by_id
                    }
                legacy_snapshot = legacy_context
                if parallel_agents and len(level_agents) > 1:
                    futures = [
                        _agent_executor().submit(
                            execute_one_agent,
                            agent,
                            execution_index=index_by_id[str(agent["id"])],
                            execution_level=execution_level,
                            upstream_outputs=level_upstream[str(agent["id"])],
                            legacy_context_snapshot=legacy_snapshot,
                        )
                        for agent in level_agents
                    ]
                    level_results = [future.result() for future in futures]
                else:
                    level_results = [
                        execute_one_agent(
                            agent,
                            execution_index=index_by_id[str(agent["id"])],
                            execution_level=execution_level,
                            upstream_outputs=level_upstream[str(agent["id"])],
                            legacy_context_snapshot=legacy_context,
                        )
                        for agent in level_agents
                    ]

                # Merge in stable authored/topological order regardless of which
                # concurrent request completed first. This keeps traces and
                # Editor evidence deterministic without sharing model outputs.
                for item in level_results:
                    trace = item["trace"]
                    traces.append(trace)
                    total_tokens += int(item["prompt_words"]) + int(item["output_words"])
                    outputs_by_id[item["agent_id"]] = item["output"]
                    legacy_context += f"\n[{item['agent_id']} output]\n{item['output']}\n"
                    generated_agents += 1
                    final_generation_sec += float(item["generation_call_sec"])
                    timing = item["search_timing"]
                    retrieval_sec += float(timing["total_sec"])
                    planner_sec += float(timing["planner_sec"])
                    bm25_sec += float(timing["bm25_sec"])
                    retrieval_errors.extend(item["retrieval_errors"])
                    if item["retrieval_session"] is not None:
                        retrieval_sessions.append(item["retrieval_session"])
                    if item["failed"]:
                        failures.append(item["agent_id"])
                    if item["is_aggregation_agent"]:
                        final = item["output"]
        if not final and traces:
            final = traces[-1]["output"]
        score = score_output(
            final,
            expected_answer,
            dataset_name=os.getenv("MAS_LOOP_DATASET_NAME", ""),
        )
        elapsed = time.time() - start
        execution_level_widths = [len(level) for level in execution_levels]
        retrieval_mode = "none"
        if use_bcp_search and retrieval_profile == "controller":
            retrieval_mode = "mixed_execution_mode_controller_controller_bm25"
        elif use_bcp_search:
            retrieval_mode = (
                "per_agent_five_turn_llm_planned_bm25"
                if configured_rounds == 5
                else f"per_agent_{configured_rounds}_turn_llm_planned_bm25"
            )
        result = {
            "task_text": task,
            "mas_config": mas,
            "agent_traces": traces,
            "final_output": final,
            "success": score >= 0.999,
            "total_cost_usd": 0.0,
            "total_tokens": total_tokens,
            "elapsed_sec": elapsed,
            "failure_nodes": failures,
            "error_message": None,
            "metadata": {
                "score": score,
                "retriever": os.getenv("MAS_LOOP_RETRIEVER", ""),
                "retrieval_mode": retrieval_mode,
                "retrieval_profile": retrieval_profile,
                "retrieval_rounds": configured_rounds,
                "dependency_prompt_mode": dependency_prompt_mode,
                "dag_agent_parallel_requested": parallel_requested,
                "dag_agent_parallel": parallel_agents,
                "graph_is_acyclic": graph_is_acyclic,
                "execution_level_count": len(execution_levels),
                "execution_level_widths": execution_level_widths,
                "execution_level_max_width": max(execution_level_widths, default=0),
                "retrieval_topk": int(os.getenv("MAS_LOOP_BM25_TOPK", "5")) if use_bcp_search else 0,
                "retrieval_evidence_policy": (
                    "accumulated" if use_bcp_search and retrieval_profile == "controller" else
                    "rank_interleaved_accumulated" if use_bcp_search else "none"
                ),
                "retrieval_doc_truncation_unit": (
                    "whitespace_words" if use_bcp_search and retrieval_profile == "controller" else
                    "characters" if use_bcp_search else "none"
                ),
                "retrieval_doc_truncation_limit": (
                    int(os.getenv("MAS_LOOP_BCP_DOC_MAX_TOKENS", "512"))
                    if use_bcp_search and retrieval_profile == "controller" else
                    int(os.getenv("MAS_LOOP_BM25_DOC_MAX_CHARS", "1200")) if use_bcp_search else 0
                ),
                "retrieval_max_final_snippets": (
                    int(os.getenv("MAS_LOOP_BCP_MAX_FINAL_SNIPPETS", "15"))
                    if use_bcp_search and retrieval_profile == "controller" else
                    int(os.getenv("MAS_LOOP_BCP_FINAL_MAX_DOCS", "15")) if use_bcp_search else 0
                ),
                "search_planner_temperature": (
                    float(os.getenv("MAS_LOOP_BCP_SEARCH_PLANNER_TEMPERATURE", "1.0")) if use_bcp_search else None
                ),
                "subagent_temperature": subagent_temperature,
                "retrieval_sessions": retrieval_sessions,
                "retrieval_search_agent_count": len(retrieval_sessions),
                "retrieval_errors": retrieval_errors,
                # Compatibility keys remain, but complete BM25 results are
                # deliberately not persisted; bounded provenance is in search_rounds.
                "retrieved_docs": [],
                "retrieval_artifact": [x.get("retrieval_artifact") for x in retrieval_sessions],
                "rollout_timing": {
                    "retrieval_sec": retrieval_sec,
                    "search_planner_sec": planner_sec,
                    "bm25_sec": bm25_sec,
                    "generation_sec": time.time() - generation_start,
                    "final_generation_sec": final_generation_sec,
                    "total_sec": elapsed,
                    "phase": profile_phase,
                    "pair_id": pair_id,
                    "agent_count": len(mas["agents"]),
                    "generated_agents": generated_agents,
                    "dependency_prompt_mode": dependency_prompt_mode,
                    "dag_agent_parallel": parallel_agents,
                    "graph_is_acyclic": graph_is_acyclic,
                    "execution_level_count": len(execution_levels),
                    "execution_level_widths": execution_level_widths,
                },
            },
        }
        self._profile({
            "time": time.time(),
            "event": "mas_execute",
            "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
            "model": getattr(self.client, "model", ""),
            "phase": profile_phase,
            "pair_id": pair_id,
            **result["metadata"]["rollout_timing"],
        })
        return result



def _last_boxed(text: str) -> str:
    """Extract the last \boxed{...} payload with simple brace matching."""
    text = text or ""
    marker = r"\boxed"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    i = text.find("{", idx)
    if i < 0:
        return ""
    depth = 0
    start = i + 1
    for j in range(i, len(text)):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:j].strip()
    return ""


def _normalize_math_answer(text: str) -> str:
    import re
    text = str(text or "").strip()
    boxed = _last_boxed(text)
    if boxed:
        text = boxed
    # Prefer content after common final-answer markers, but only for short tails.
    m = re.search(r"(?i)(?:final answer|answer|答案)\s*[:：]?\s*(.+)$", text)
    if m and len(m.group(1)) <= 120:
        text = m.group(1)
    text = text.strip().strip("$ .。,")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("−", "-")
    # Convert simple LaTeX fractions to a canonical a/b string.
    def frac_repl(m):
        return f"({m.group(1)})/({m.group(2)})"
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", frac_repl, text)
    text = text.replace("\\", "")
    text = re.sub(r"\\?boxed", "", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("{}[]")
    return text.lower()


def _math_equal(a: str, b: str) -> bool:
    na, nb = _normalize_math_answer(a), _normalize_math_answer(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr
        return bool(simplify(parse_expr(na) - parse_expr(nb)) == 0)
    except Exception:
        return False



_MATH_DATASETS = frozenset({
    "aime", "hlemath", "deepscaler", "deep_scale_r", "deepscaler_math",
    "mas_loop_deepscaler", "mas_loop_aime", "mas_loop_hlemath",
})
_HOTPOTQA_DATASETS = frozenset({"hotpotqa", "hotpot_qa", "mas_loop_hotpotqa"})


def math_score(output: str, expected: str) -> float:
    """Strict binary mathematical equivalence for AIME/DeepScaler."""
    score, _ = AIMEScorer().calculate_score(str(expected or ""), str(output or ""))
    return float(score)


def score_output(output: str, expected: str, dataset_name: str | None = None) -> float:
    """Route task scoring without changing legacy open-text dataset semantics."""
    dataset = str(dataset_name or os.getenv("MAS_LOOP_DATASET_NAME", "")).strip().lower()
    if dataset in _MATH_DATASETS:
        return math_score(output, expected)
    if dataset in _HOTPOTQA_DATASETS:
        # Keep the existing official-style HotpotQA exact-match route intact.
        from .hotpotqa_score import answer_metrics
        return float(answer_metrics(str(output or ""), str(expected or ""))["exact_match"] or 0.0)
    return quality_score(output, expected)

def quality_score(output: str, expected: str) -> float:
    out_raw = output or ""
    exp_raw = expected or ""
    out = out_raw.strip().lower()
    exp = exp_raw.strip().lower()
    if not exp:
        return 1.0 if out else 0.0
    if _math_equal(out_raw, exp_raw):
        return 1.0
    boxed = _last_boxed(out_raw)
    if boxed and _math_equal(boxed, exp_raw):
        return 1.0
    norm_out = _normalize_math_answer(out_raw)
    norm_exp = _normalize_math_answer(exp_raw)
    if norm_exp and norm_exp in norm_out:
        return 0.7
    if out == exp:
        return 1.0
    if exp and exp in out:
        return 0.7
    os = set(out)
    es = set(exp)
    return (len(os & es) / len(os | es)) if os and es else 0.0
