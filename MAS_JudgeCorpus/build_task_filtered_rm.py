#!/usr/bin/env python3
"""Build task-stable, operation-deduplicated RM data at two levels.

Levels:
  * orchestration: task + MAS XML only
  * trajectory: task + MAS XML + bounded per-node execution outputs

Input is the full rich HotpotQA archive plus the 16-rollout task aggregation file.
Only task-level aligned non-tie rollouts are retained. Within each task and
canonical refiner operation, one rollout is selected deterministically.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def read_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def xml_tag(tag: str, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if value is None:
        value = ""
    return f"{pad}<{tag}>{escape(str(value))}</{tag}>"


def _first_sentence(text: Any, max_chars: int = 280) -> str:
    text = "" if text is None else str(text).strip()
    # Retrieval previews are usually "Title: ...\nbody". Remove the title line
    # before selecting a compact evidence sentence.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("title:"):
        lines = lines[1:]
    body = " ".join(lines).strip()
    if not body:
        return ""
    import re
    m = re.search(r"(?<=[.!?])\s+", body)
    sentence = body[:m.start()] if m else body
    return clip(sentence, max_chars)


def retrieval_summary_xml(trace: dict[str, Any], indent: int = 6, max_docs: int = 5, per_node_chars: int = 6000) -> list[str]:
    """Serialize one agent's bounded execution context inside <execution>."""
    lines: list[str] = []
    docs = []
    for rnd in trace.get("search_rounds", []) or []:
        docs.extend(rnd.get("documents", []) or [])
    seen = set()
    unique = []
    for doc in docs:
        did = str(doc.get("docid", ""))
        if did in seen:
            continue
        seen.add(did)
        preview = doc.get("text_preview", doc.get("text", ""))
        title = str(doc.get("title", "")).strip()
        if not title:
            first = next((ln.strip() for ln in str(preview).splitlines() if ln.strip()), "")
            title = first[6:].strip() if first.lower().startswith("title:") else first[:160]
        unique.append((title, _first_sentence(preview), doc.get("score")))
        if len(unique) >= max_docs:
            break

    lines.append(" " * indent + "<execution>")
    if trace.get("search_mode"):
        lines.append(xml_tag("search_mode", trace["search_mode"], indent + 2))
    if trace.get("search_rounds") is not None:
        lines.append(xml_tag("search_rounds", len(trace.get("search_rounds") or []), indent + 2))
    if unique:
        lines.append(" " * (indent + 2) + "<retrieval_context>")
        for rank, (title, sentence, score) in enumerate(unique, 1):
            lines.append(" " * (indent + 4) + "<document>")
            lines.append(xml_tag("rank", rank, indent + 6))
            lines.append(xml_tag("title", title, indent + 6))
            lines.append(xml_tag("first_sentence", sentence, indent + 6))
            if score is not None:
                lines.append(xml_tag("retrieval_score", round(float(score), 4), indent + 6))
            lines.append(" " * (indent + 4) + "</document>")
        lines.append(" " * (indent + 2) + "</retrieval_context>")
    if trace.get("error"):
        lines.append(xml_tag("error", clip(trace["error"], 1000), indent + 2))
    lines.append(" " * (indent + 2) + "<output>")
    # Escape free-form model output so the serialized XML is always well-formed.
    lines.append(escape(clip(trace.get("output", ""), per_node_chars)))
    lines.append(" " * (indent + 2) + "</output>")
    lines.append(" " * indent + "</execution>")
    return lines


def mas_to_orchestra_xml(mas: dict[str, Any], traj: dict[str, Any] | None = None, include_retrieval: bool = False, per_node_chars: int = 6000) -> str:
    """Canonical readable XML close to Orchestra's agent/edge prompt format.

    When include_retrieval is true, attach compact retrieved-document evidence
    under the corresponding agent. Full document bodies are intentionally omitted.
    """
    agents = mas.get("agents", mas.get("nodes", [])) or []
    edges = mas.get("edges", []) or []
    lines = ["<mas>"]
    for key in ("topology", "aggregation", "max_cost", "max_iterations"):
        if key in mas:
            lines.append(xml_tag(key, mas[key], 2))
    lines.append("  <agents>")
    for agent in agents:
        aid = agent.get("id", agent.get("agent_id", ""))
        lines.append("    <agent>")
        lines.append(xml_tag("agent_id", aid, 6))
        lines.append(xml_tag("agent_description", agent.get("role", agent.get("description", "")), 6))
        if agent.get("task") is not None:
            lines.append(xml_tag("task", agent.get("task"), 6))
        if agent.get("goal") is not None:
            lines.append(xml_tag("goal", agent.get("goal"), 6))
        lines.append("      <tools>")
        for tool in agent.get("tools", []) or []:
            lines.append(xml_tag("tool", tool, 8))
        lines.append("      </tools>")
        lines.append(xml_tag("budget", agent.get("budget", ""), 6))
        lines.append(xml_tag("execution_mode", agent.get("execution_mode", "llm_only"), 6))
        if include_retrieval and traj:
            agent_id = str(aid)
            trace = next((t for t in (traj.get("agent_traces", []) or []) if str(t.get("agent_id", "")) == agent_id), None)
            if trace:
                lines.extend(retrieval_summary_xml(trace, indent=6, per_node_chars=per_node_chars))
        lines.append("    </agent>")
    lines.append("  </agents>")
    lines.append("  <edges>")
    for edge in edges:
        lines += ["    <edge>", xml_tag("from", edge.get("from", edge.get("f", "")), 6),
                  xml_tag("to", edge.get("to", edge.get("t", "")), 6),
                  xml_tag("type", edge.get("type", "data_flow"), 6), "    </edge>"]
    lines += ["  </edges>"]
    if include_retrieval and traj:
        lines.append(xml_tag("final_output", clip(traj.get("final_output", ""), 6000), 2))
        lines.append(xml_tag("success", traj.get("success", ""), 2))
        lines.append(xml_tag("total_cost_usd", traj.get("total_cost_usd", ""), 2))
        lines.append(xml_tag("total_tokens", traj.get("total_tokens", ""), 2))
    lines.append("</mas>")
    return "\n".join(lines)


def clip(text: Any, n: int) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    return text if len(text) <= n else text[: max(0, n - 80)] + "\n...[TRUNCATED]..."


def trajectory_to_text(traj: dict[str, Any], per_node_chars: int = 6000, total_chars: int = 24000) -> str:
    """Serialize bounded per-node outputs; do not add trajectory correctness labels."""
    lines = ["<execution_trace>"]
    for i, trace in enumerate(traj.get("agent_traces", []) or []):
        lines += ["  <node>", xml_tag("agent_id", trace.get("agent_id", f"agent_{i}"), 4),
                  xml_tag("role", trace.get("agent_role", ""), 4)]
        if trace.get("search_mode"):
            lines.append(xml_tag("search_mode", trace["search_mode"], 4))
        if trace.get("search_rounds") is not None:
            lines.append(xml_tag("search_rounds", len(trace.get("search_rounds") or []), 4))
        if trace.get("error"):
            lines.append(xml_tag("error", clip(trace["error"], 1000), 4))
        lines += ["    <output>", clip(trace.get("output", ""), per_node_chars), "    </output>", "  </node>"]
    lines += ["  <final_output>", clip(traj.get("final_output", ""), per_node_chars), "  </final_output>",
              xml_tag("success", traj.get("success", ""), 2),
              xml_tag("total_cost_usd", traj.get("total_cost_usd", ""), 2),
              xml_tag("total_tokens", traj.get("total_tokens", ""), 2), "</execution_trace>"]
    return clip("\n".join(lines), total_chars)


def operation_signature(operation: dict[str, Any] | None) -> tuple[str, str]:
    op = operation or {}
    trans = op.get("translation") or {}
    ops = trans.get("canonical_operations") or op.get("operations") or op.get("patch") or []
    if not isinstance(ops, list):
        ops = [ops]
    canonical = json.dumps(ops, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not canonical or canonical == "[]":
        canonical = "__no_operation__"
    types = []
    for item in ops:
        types.append(str(item.get("op", "unknown")) if isinstance(item, dict) else str(item))
    return "+".join(types) if types else "no_operation", canonical


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--aggregation", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--relative-threshold", type=float, default=0.25)
    ap.add_argument("--absolute-gap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-node-chars", type=int, default=6000)
    ap.add_argument("--max-trace-chars", type=int, default=24000)
    args = ap.parse_args()

    task_meta = {}
    for row in read_jsonl(args.aggregation):
        rel = row.get("lower_denominator_relative_absolute_gap")
        if rel is None or rel < args.relative_threshold or row.get("absolute_gap", 0) < args.absolute_gap:
            continue
        if row.get("direction") in ("source", "update"):
            task_meta[row["task_id"]] = row

    grouped = defaultdict(list)
    for row in read_jsonl(args.archive):
        if row.get("task_id") in task_meta:
            grouped[row["task_id"]].append(row)

    selected = []
    stats = Counter()
    for tid, rows in grouped.items():
        direction = task_meta[tid]["direction"]
        candidates = []
        for row in rows:
            s, u = float(row.get("source_score") or 0), float(row.get("updated_score") or 0)
            if s == u:
                stats["pair_tie"] += 1
                continue
            aligned = (direction == "update" and u > s) or (direction == "source" and s > u)
            if not aligned:
                stats["pair_conflict"] += 1
                continue
            op_type, op_sig = operation_signature(row.get("editor_operation"))
            candidates.append((op_sig, op_type, row))
        by_op = defaultdict(list)
        for op_sig, op_type, row in candidates:
            by_op[op_sig].append((op_type, row))
        rng = random.Random(args.seed ^ stable_hash(tid))
        for op_sig, op_rows in sorted(by_op.items()):
            op_type, row = rng.choice(op_rows)
            row = dict(row)
            row["_op_type"], row["_op_sig"] = op_type, op_sig
            selected.append(row)
            stats["dedup_selected"] += 1
            stats["dedup_removed"] += max(0, len(op_rows) - 1)

    selected.sort(key=lambda x: (x["task_id"], x.get("pair_id", "")))
    out = Path(args.output_dir)
    orch_rows, traj_rows = [], []
    task_split = {}
    for row in selected:
        tid, meta, direction = row["task_id"], task_meta[row["task_id"]], task_meta[row["task_id"]]["direction"]
        src_mas, upd_mas = row["source_mas"], row["updated_mas"]
        src_traj, upd_traj = row.get("source_trajectory") or {}, row.get("updated_trajectory") or {}
        if direction == "update":
            chosen_mas, rejected_mas, chosen_traj, rejected_traj = upd_mas, src_mas, upd_traj, src_traj
            chosen_score, rejected_score = row.get("updated_score"), row.get("source_score")
        else:
            chosen_mas, rejected_mas, chosen_traj, rejected_traj = src_mas, upd_mas, src_traj, upd_traj
            chosen_score, rejected_score = row.get("source_score"), row.get("updated_score")
        split = row.get("split", "train")
        task_split.setdefault(tid, split)
        common = {
            "pair_id": row.get("pair_id"), "task_id": tid, "task_type": row.get("task_type"),
            "task_text": row.get("task_text"), "expected_answer": row.get("expected_answer"), "split": split,
            "task_direction": direction, "task_source_total_score": meta.get("source_total_score"),
            "task_update_total_score": meta.get("update_total_score"), "task_signed_gap": meta.get("signed_gap"),
            "task_absolute_gap": meta.get("absolute_gap"), "task_lower_total_score": meta.get("lower_total_score"),
            "task_lower_relative_signed_gap": meta.get("lower_denominator_relative_signed_gap"),
            "task_lower_relative_absolute_gap": meta.get("lower_denominator_relative_absolute_gap"),
            "chosen_score": chosen_score, "rejected_score": rejected_score,
            "pair_score_delta": abs(float(chosen_score or 0) - float(rejected_score or 0)),
            "refiner_operation_type": row.get("_op_type"), "refiner_operation_signature": row.get("_op_sig"),
        }
        # Strict separation of the two levels:
        # orchestration = task + MAS map only; trajectory = MAS map + compact
        # retrieval evidence + node execution trace.
        orch_chosen_xml = mas_to_orchestra_xml(chosen_mas)
        orch_rejected_xml = mas_to_orchestra_xml(rejected_mas)
        traj_chosen_xml = mas_to_orchestra_xml(chosen_mas, chosen_traj, include_retrieval=True, per_node_chars=args.per_node_chars)
        traj_rejected_xml = mas_to_orchestra_xml(rejected_mas, rejected_traj, include_retrieval=True, per_node_chars=args.per_node_chars)
        orch = dict(common, chosen_text=f"Task:\n{row.get('task_text','')}\n\nCandidate MAS:\n{orch_chosen_xml}",
                    rejected_text=f"Task:\n{row.get('task_text','')}\n\nCandidate MAS:\n{orch_rejected_xml}",
                    chosen_mas_xml=orch_chosen_xml, rejected_mas_xml=orch_rejected_xml,
                    source_mas=src_mas, updated_mas=upd_mas)
        if orch_chosen_xml != orch_rejected_xml:
            orch_rows.append(orch)
        else:
            stats["orchestration_same_structure_removed"] += 1
        prompt = f"Task:\n{row.get('task_text','')}\n\nCandidate MAS with per-agent execution:\n"
        # All node-level retrieval and generation outputs are already nested in
        # each <agent><execution> block. Do not append a second global trace.
        traj_rows.append(dict(common,
            chosen_text=prompt + traj_chosen_xml,
            rejected_text=prompt + traj_rejected_xml,
            chosen_mas_xml=traj_chosen_xml, rejected_mas_xml=traj_rejected_xml,
            chosen_trajectory=chosen_traj, rejected_trajectory=rejected_traj,
            source_mas=src_mas, updated_mas=upd_mas,
            trace_limits={"per_node_chars": args.per_node_chars, "max_trace_chars": args.max_trace_chars}))

    def compact_row(row):
        # OrchRM only needs the two preference strings. Keep a small audit
        # envelope so rows remain traceable without duplicating rich MAS/traces.
        keys = [
            "pair_id", "task_id", "task_type", "task_text", "expected_answer", "split",
            "task_direction", "task_source_total_score", "task_update_total_score",
            "task_signed_gap", "task_absolute_gap", "task_lower_total_score",
            "task_lower_relative_signed_gap", "task_lower_relative_absolute_gap",
            "chosen_score", "rejected_score", "pair_score_delta",
            "refiner_operation_type", "refiner_operation_signature",
            "chosen_text", "rejected_text",
        ]
        return {k: row.get(k) for k in keys}

    def write_split(rows, prefix):
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task_id"]].append(row)
        train, val = [], []
        for tid, rs in sorted(by_task.items()):
            (val if task_split.get(tid, "train") == "val" else train).extend(rs)
        dump_jsonl(out / f"{prefix}_train.jsonl", train)
        dump_jsonl(out / f"{prefix}_val.jsonl", val)
        dump_jsonl(out / f"{prefix}_all.jsonl", rows)
        # Compact compatibility exports for pairwise reward-model trainers.
        dump_jsonl(out / f"{prefix}_train_for_orchrm.jsonl", [compact_row(r) for r in train])
        dump_jsonl(out / f"{prefix}_val_for_orchrm.jsonl", [compact_row(r) for r in val])
        return len(train), len(val), len({r["task_id"] for r in train}), len({r["task_id"] for r in val})

    stats.update({"selected_tasks": len(task_meta), "matched_tasks": len(grouped),
                  "selected_rollouts_after_operation_dedup": len(selected),
                  "orchestration_rows": len(orch_rows), "trajectory_rows": len(traj_rows)})
    stats["orchestration_train_rows"], stats["orchestration_val_rows"], stats["orchestration_train_tasks"], stats["orchestration_val_tasks"] = write_split(orch_rows, "orchestration_level")
    stats["trajectory_train_rows"], stats["trajectory_val_rows"], stats["trajectory_train_tasks"], stats["trajectory_val_tasks"] = write_split(traj_rows, "trajectory_level")
    summary = {"config": vars(args), "stats": dict(stats), "task_directions": dict(Counter(m["direction"] for m in task_meta.values())),
               "notes": ["No hard set is emitted.", "Only task-level aligned non-tie rollouts are used.", "One rollout is randomly selected per task and canonical refiner operation.", "Orchestration rows with identical chosen/rejected MAS XML are removed; trajectory rows may retain same-structure/different-trace pairs."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "examples.json").write_text(json.dumps({"orchestration": orch_rows[:1], "trajectory": traj_rows[:1]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
