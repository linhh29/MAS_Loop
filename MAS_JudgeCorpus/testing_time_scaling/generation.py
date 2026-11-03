from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any

from .accounting import sum_usage
from .formatting import artifacts_to_mas, make_xmls, state_to_trajectory, to_jsonable
from .retrieval import multi_round_retrieve
from .scoring import extract_answer, is_correct


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big") & 0x7fffffff


def _calls_since(client: Any, start: int) -> dict[str, Any]:
    calls = client.ledger.calls[start:]
    return {**sum_usage(calls), "calls": calls}


async def generate_cot_bank(*, sample: dict[str, Any], client: Any, retriever: Any, n: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = sample["dataset"]
    rounds = int(cfg["retrieval"][dataset]["rounds"])
    # Query planning is model-accounted and answer-blind.
    query_prompt = (
        "Produce one concise BM25 search query that preserves the key entities and relation in the question. "
        "Do not answer it. Output only the query.\n\nQuestion:\n" + sample["question"]
    )
    with client.capture_usage() as evidence_ledger:
        initial_query, _ = await client.generate(user_prompt=query_prompt, role="cot_retrieval_query", temperature=0.0, max_tokens=128, metadata={"dataset": dataset, "sample_id": sample["sample_id"]})
        evidence, retrieval_trace = await multi_round_retrieve(
            dataset=dataset, question=sample["question"], initial_query=initial_query, retriever=retriever, client=client,
            topk=int(cfg["retrieval"][dataset]["topk"]), rounds=rounds,
            max_chars=int(cfg["retrieval"][dataset]["max_evidence_chars"]), role="cot_retrieval_gap",
            metadata={"dataset": dataset, "sample_id": sample["sample_id"]},
        )
    evidence_usage = evidence_ledger.totals()
    system = "Answer the question using the retrieved evidence. Reason step by step, then end with a line 'Final Answer: <answer>'."
    prompt = f"Question:\n{sample['question']}\n\nRetrieved evidence:\n{evidence}"

    async def one(idx: int) -> dict[str, Any]:
        with client.capture_usage() as candidate_ledger:
            raw, _ = await client.generate(user_prompt=prompt, system_prompt=system, role="cot_answer", temperature=float(cfg["generation"]["temperature"]), max_tokens=int(cfg["generation"]["max_tokens"]), seed=stable_seed(dataset, sample["sample_id"], "cot", idx), metadata={"dataset": dataset, "sample_id": sample["sample_id"], "candidate_idx": idx})
        pred = extract_answer(raw)
        return {
            **{k: sample[k] for k in ("dataset","index","sample_id","question","gold")},
            "candidate_idx": idx, "raw_output": raw, "prediction": pred, "correct": is_correct(pred, sample["gold"]),
            "retrieval": {"initial_query": initial_query, "trajectory": retrieval_trace, "evidence": evidence},
            "shared_retrieval_usage": evidence_usage, "candidate_usage": candidate_ledger.totals(),
        }

    concurrency = max(1, int(cfg["experiment"].get("candidate_concurrency", 1)))
    rows: list[dict[str, Any]] = []
    for offset in range(0, n, concurrency):
        rows.extend(await asyncio.gather(*(one(idx) for idx in range(offset, min(n, offset + concurrency)))))
    return rows


async def generate_mas_candidate(*, sample: dict[str, Any], client: Any, retriever: Any, candidate_idx: int, cfg: dict[str, Any], mas_root: Path) -> dict[str, Any]:
    building = mas_root / "building"
    if str(mas_root) not in sys.path: sys.path.insert(0, str(mas_root))
    if str(building) not in sys.path: sys.path.insert(0, str(building))
    from building.skill_mas.build import run_mas_pipeline_with_retries

    sessions: list[dict[str, Any]] = []
    dataset = sample["dataset"]

    async def planner_call(prompt: str) -> str:
        text, _ = await client.generate(user_prompt=prompt, system_prompt="You are a MAS planner. Output JSON only.", role="mas_planner", temperature=float(cfg["mas"]["planner_temperature"]), max_tokens=int(cfg["mas"]["planner_max_tokens"]), seed=stable_seed(dataset, sample["sample_id"], "planner", candidate_idx), metadata={"dataset":dataset,"sample_id":sample["sample_id"],"candidate_idx":candidate_idx})
        return text

    async def text_call(prompt: str) -> tuple[str, dict[str, Any]]:
        return await client.generate(user_prompt=prompt, role="mas_subagent", temperature=float(cfg["generation"]["temperature"]), max_tokens=int(cfg["generation"]["max_tokens"]), seed=stable_seed(dataset, sample["sample_id"], "subagent", candidate_idx, len(client.ledger.calls)), metadata={"dataset":dataset,"sample_id":sample["sample_id"],"candidate_idx":candidate_idx})

    async def tool_call(req: Any) -> tuple[str, dict[str, Any]]:
        tc = getattr(req, "tool_context", None) or {}
        initial = str(tc.get("initial_bm25_query") or sample["question"])
        with client.capture_usage() as retrieval_ledger:
            evidence, trace = await multi_round_retrieve(
                dataset=dataset, question=sample["question"], initial_query=initial, retriever=retriever, client=client,
                topk=int(cfg["retrieval"][dataset]["topk"]), rounds=int(cfg["retrieval"][dataset]["rounds"]),
                max_chars=int(cfg["retrieval"][dataset]["max_evidence_chars"]), role="mas_retrieval_gap",
                metadata={"dataset":dataset,"sample_id":sample["sample_id"],"candidate_idx":candidate_idx,"role_name":getattr(req,"role_name","")},
            )
        usage = retrieval_ledger.totals()
        sessions.append({"role_name":str(getattr(req,"role_name","") or ""),"initial_bm25_query":initial,"trajectory":trace,"evidence":evidence,"usage":usage})
        return evidence, usage

    class_name = f"{dataset.title().replace('_','')}Task{sample['index']}Candidate{candidate_idx}MASWorkflow"
    with client.capture_usage() as candidate_ledger:
        result = await run_mas_pipeline_with_retries(
            task_text=sample["question"], class_name=class_name,
            init_skill_path=mas_root / "building/init_skill/SKILL.md", planner_call_fn=planner_call,
            text_call_fn=text_call, tool_call_fn=tool_call,
            max_generation_attempts=int(cfg["mas"]["max_generation_attempts"]), max_execution_attempts=int(cfg["mas"]["max_execution_attempts"]),
            dataset_name="bcp",  # isolated adapter: builder currently has no hotpotqa route
        )
    mas = artifacts_to_mas(result.artifacts, result.mas_code)
    trajectory = state_to_trajectory(result.state, sessions, result.final_output, result.success)
    orch_xml, traj_xml, orch_text, traj_text = make_xmls(sample["question"], mas, trajectory)
    pred = extract_answer(result.final_output)
    return {
        **{k: sample[k] for k in ("dataset","index","sample_id","question","gold")},
        "candidate_idx": candidate_idx, "success": result.success, "raw_output": result.final_output,
        "prediction": pred, "correct": is_correct(pred, sample["gold"]), "builder_dataset_alias": "bcp",
        "mas": mas, "mas_code": result.mas_code, "artifacts": to_jsonable(result.artifacts), "state": to_jsonable(result.state),
        "retrieval_sessions": sessions, "trajectory": trajectory,
        "mas_xml_orchestration": orch_xml, "mas_xml_trajectory": traj_xml,
        "rm_text_orchestration": orch_text, "rm_text_trajectory": traj_text,
        "retry": {"generation_attempts_used":result.generation_attempts_used,"execution_attempts_used":result.execution_attempts_used,"failure_stage":result.failure_stage,"failure_reason":result.failure_reason,"events":result.retry_events},
        "candidate_usage": candidate_ledger.totals(),
    }
