from __future__ import annotations

import json
import re
from typing import Any

from .accounting import sum_usage
from .scoring import is_correct, majority_vote

METHODS = [
    "mas_lb", "mas_majority_voting", "mas_oracle_ub",
    "deepseek_v4_flash", "deepseek_v4_flash_trajectory",
    "kimi_k3", "kimi_k3_trajectory",
    "qwen3_8_2_4t_a95b", "qwen3_8_2_4t_a95b_trajectory",
    "skywork_v2_qwen3_8b", "orch_rm", "mas_judgerm_orchestration", "mas_judgerm_trajectory",
]

LLM_JUDGE_METHODS = {
    "deepseek_v4_flash": ("deepseek_v4_flash", False),
    "deepseek_v4_flash_trajectory": ("deepseek_v4_flash", True),
    "kimi_k3": ("kimi_k3", False),
    "kimi_k3_trajectory": ("kimi_k3", True),
    "qwen3_8_2_4t_a95b": ("qwen3_8_2_4t_a95b", False),
    "qwen3_8_2_4t_a95b_trajectory": ("qwen3_8_2_4t_a95b", True),
}


def parse_selected_index(text: str, n: int) -> int:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    candidates: list[Any] = []
    try: candidates.append(json.loads(stripped))
    except Exception: pass
    match = re.search(r"\{.*?\}", stripped, flags=re.S)
    if match:
        try: candidates.append(json.loads(match.group(0)))
        except Exception: pass
    for obj in candidates:
        if isinstance(obj, dict):
            value = obj.get("selected_index", obj.get("index"))
            try:
                idx = int(value)
                if 0 <= idx < n: return idx
            except Exception: pass
    match = re.search(r"(?:selected[_ ]index|index)\D{0,10}(\d+)", stripped, flags=re.I)
    if match and 0 <= int(match.group(1)) < n: return int(match.group(1))
    raise ValueError(f"could not parse selected index from judge output: {text[:500]!r}")


def _usage_for_cot(rows: list[dict[str, Any]], selected_indices: list[int]) -> dict[str, Any]:
    parts = [rows[0].get("shared_retrieval_usage")]
    parts.extend(rows[i].get("candidate_usage") for i in selected_indices)
    return sum_usage(parts)


def cot_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = sorted(rows, key=lambda x: int(x["candidate_idx"]))
    idx, pred, votes = majority_vote([x.get("raw_output", "") for x in rows])
    correct_indices = [i for i,x in enumerate(rows) if bool(x.get("correct"))]
    oracle_idx = correct_indices[0] if correct_indices else 0
    return {
        "cot_lb": {"selected_index":0,"prediction":rows[0]["prediction"],"correct":bool(rows[0]["correct"]),"usage":_usage_for_cot(rows,[0])},
        "majority_voting": {"selected_index":idx,"prediction":pred,"correct":is_correct(pred,rows[0]["gold"]),"votes":votes,"usage":_usage_for_cot(rows,list(range(len(rows))))},
        "oracle_ub": {"selected_index":oracle_idx,"prediction":rows[oracle_idx]["prediction"],"correct":bool(correct_indices),"oracle_correct_indices":correct_indices,"usage":_usage_for_cot(rows,list(range(len(rows))))},
    }


def mas_baseline_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = sorted(rows, key=lambda x: int(x["candidate_idx"]))
    idx, pred, votes = majority_vote([x.get("raw_output", "") for x in rows])
    correct_indices = [i for i, x in enumerate(rows) if bool(x.get("correct"))]
    oracle_idx = correct_indices[0] if correct_indices else 0
    all_usage = mas_generation_usage(rows)
    return {
        "mas_lb": {
            "selected_index": 0,
            "prediction": rows[0]["prediction"],
            "correct": bool(rows[0]["correct"]),
            "usage": sum_usage([rows[0].get("candidate_usage")]),
        },
        "mas_majority_voting": {
            "selected_index": idx,
            "prediction": pred,
            "correct": is_correct(pred, rows[0]["gold"]),
            "votes": votes,
            "usage": all_usage,
        },
        "mas_oracle_ub": {
            "selected_index": oracle_idx,
            "prediction": rows[oracle_idx]["prediction"],
            "correct": bool(correct_indices),
            "oracle_correct_indices": correct_indices,
            "usage": all_usage,
        },
    }


def mas_generation_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sum_usage([x.get("candidate_usage") for x in rows])


async def llm_judge_decision(*, rows: list[dict[str, Any]], client: Any, method: str, trajectory: bool = False, max_tokens: int = 1024) -> dict[str, Any]:
    rows = sorted(rows, key=lambda x: int(x["candidate_idx"]))
    xml_key = "mas_xml_trajectory" if trajectory else "mas_xml_orchestration"
    blocks = []
    for row in rows:
        blocks.append(f"--- Candidate {row['candidate_idx']} ---\n{row[xml_key]}")
    level = "trajectory" if trajectory else "orchestration"
    criterion = (
        "Judge complete execution trajectories: orchestration, per-agent execution, tool/retrieval traces, "
        "intermediate outputs, final aggregation, and final answer."
        if trajectory else
        "Judge orchestration only: roles, decomposition, topology, tool use, and aggregation. "
        "You are not given executions or the gold answer."
    )
    prompt = (
        "Select the candidate multi-agent system most likely to answer the task correctly. "
        f"{criterion} You are not given the gold answer. Return JSON immediately, without analysis, as "
        "{\"selected_index\": <0-based integer>, \"reason\": \"brief reason\"}.\n\n"
        f"Task:\n{rows[0]['question']}\n\n" + "\n\n".join(blocks)
    )
    # Kimi and Qwen endpoints reliably honor prompt-level /no_think. Qwen's
    # gateway rejects enable_thinking=false, while Kimi accepts it but may still
    # emit long reasoning unless /no_think is present.
    no_think = "\n/no_think" if any(name in method.lower() for name in ("qwen", "kimi")) else ""
    judge_extra_body = None
    with client.capture_usage() as judge_ledger:
        raw, _ = await client.generate(user_prompt=prompt + no_think, system_prompt=f"You are a {level}-level MAS judge. Emit requested JSON immediately; do not reason before JSON.{no_think}", role=method, temperature=0.0, max_tokens=max_tokens, response_format={"type":"json_object"}, extra_body=judge_extra_body, metadata={"dataset":rows[0]["dataset"],"sample_id":rows[0]["sample_id"],"verification_level":level})
    idx = parse_selected_index(raw, len(rows))
    judge_usage = judge_ledger.totals()
    return {"selected_index":idx,"prediction":rows[idx]["prediction"],"correct":bool(rows[idx]["correct"]),"judge_raw":raw,"usage":sum_usage([mas_generation_usage(rows),judge_usage]),"selector_usage":judge_usage}
