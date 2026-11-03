"""Offline BrowseComp-Plus evaluation with the same Qwen judge as GRPO reward.

Inputs may be MAS-Loop canonical rollout shard directories/files or simple JSONL
rows containing query/expected_answer/prediction. Every source/updated candidate
is judged independently and both the full per-request record and an aggregate
summary are written below the requested checkpoint evaluation directory.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from .bcp_judge import BCPJudge
from .sharded_jsonl import iter_jsonl, resolve_jsonl_inputs


def _candidate_rows(record: dict[str, Any], candidates: set[str]) -> Iterable[dict[str, Any]]:
    pair = record.get("structured_pair")
    if isinstance(pair, dict):
        task = pair.get("task") if isinstance(pair.get("task"), dict) else {}
        question = str(task.get("task_text") or record.get("task") or "")
        expected = str(task.get("expected_answer") or record.get("expected") or "")
        pair_id = str(pair.get("pair_id") or record.get("pair_id") or record.get("id") or "")
        optimizer_step = int(pair.get("optimizer_step") or record.get("optimizer_step") or 0)
        outer_round = int(pair.get("outer_adversarial_round") or record.get("outer_adversarial_round") or 1)
        for candidate in ("source", "updated"):
            if candidate not in candidates:
                continue
            value = pair.get(candidate) if isinstance(pair.get(candidate), dict) else {}
            trajectory = value.get("trajectory") if isinstance(value.get("trajectory"), dict) else {}
            response = str(trajectory.get("final_output") or value.get("final_output") or "")
            if response:
                yield {
                    "pair_id": pair_id,
                    "candidate": candidate,
                    "optimizer_step": optimizer_step,
                    "outer_adversarial_round": outer_round,
                    "question": question,
                    "correct_answer": expected,
                    "response": response,
                }
        return

    # Compatibility with older unstructured MAS-Loop records.
    question = str(record.get("task") or record.get("query") or record.get("question") or "")
    expected = str(record.get("expected") or record.get("expected_answer") or record.get("answer") or "")
    pair_id = str(record.get("pair_id") or record.get("id") or record.get("query_id") or "")
    emitted = False
    for candidate, trajectory_key, score_key in (
        ("source", "src_trajectory", "src_output"),
        ("updated", "updated_trajectory", "updated_output"),
    ):
        if candidate not in candidates:
            continue
        trajectory = record.get(trajectory_key) if isinstance(record.get(trajectory_key), dict) else {}
        response = str(trajectory.get("final_output") or record.get(score_key) or "")
        if response:
            emitted = True
            yield {
                "pair_id": pair_id,
                "candidate": candidate,
                "optimizer_step": int(record.get("optimizer_step") or 0),
                "outer_adversarial_round": int(record.get("outer_adversarial_round") or 1),
                "question": question,
                "correct_answer": expected,
                "response": response,
            }
    if emitted:
        return

    # Compatibility with standard prediction JSONL used by standalone evals.
    response = str(record.get("prediction") or record.get("response") or record.get("final_output") or "")
    if response and "source" in candidates:
        yield {
            "pair_id": pair_id,
            "candidate": "source",
            "optimizer_step": int(record.get("optimizer_step") or 0),
            "outer_adversarial_round": int(record.get("outer_adversarial_round") or 1),
            "question": question,
            "correct_answer": expected,
            "response": response,
        }


def collect_candidates(inputs: list[str], candidates: set[str], max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in iter_jsonl(inputs):
        for row in _candidate_rows(record, candidates):
            key = (row["pair_id"], row["candidate"], row["response"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                return rows
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_names = {x.strip() for x in args.candidates.split(",") if x.strip()}
    if not candidate_names or not candidate_names.issubset({"source", "updated"}):
        raise ValueError("--candidates must be source, updated, or source,updated")
    files = resolve_jsonl_inputs(args.input)
    if not files:
        raise FileNotFoundError(f"no JSONL inputs resolved from: {args.input}")
    rows = collect_candidates(args.input, candidate_names, args.max_samples)
    if not rows:
        raise ValueError("no judgeable BCP predictions found in the supplied inputs")

    os.environ["MAS_LOOP_JUDGE_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE"] = str(args.max_workers)
    judge = BCPJudge(
        base_url=args.judge_base_url,
        model=args.judge_model,
        manage_sleep=True,
        max_model_len=args.max_model_len,
        result_dir=output_dir / "judge_results",
    )

    started = time.time()

    def score_one(row: dict[str, Any]) -> dict[str, Any]:
        # Each request owns a sleep/admission lease. Concurrent leases share one
        # wake cycle and the last request puts the judge back to sleep.
        with judge.client.awake_session():
            result = judge.judge(
                question=row["question"],
                response=row["response"],
                correct_answer=row["correct_answer"],
                pair_id=row["pair_id"],
                candidate=row["candidate"],
                optimizer_step=row["optimizer_step"],
                outer_adversarial_round=row["outer_adversarial_round"],
                data_source="bcp_eval",
            )
        return {**row, "judge": result, "score": float(result["score"])}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers), thread_name_prefix="bcp-judge") as pool:
        futures = [pool.submit(score_one, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: (x["pair_id"], x["candidate"], x["optimizer_step"]))

    result_path = output_dir / "evaluation_results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    valid = [x for x in results if x["score"] >= 0]
    correct = sum(1 for x in valid if x["score"] >= 0.999)
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidate_names):
        subset = [x for x in results if x["candidate"] == candidate]
        subset_valid = [x for x in subset if x["score"] >= 0]
        subset_correct = sum(1 for x in subset_valid if x["score"] >= 0.999)
        by_candidate[candidate] = {
            "total": len(subset),
            "valid": len(subset_valid),
            "judge_failures": len(subset) - len(subset_valid),
            "correct": subset_correct,
            "accuracy": subset_correct / len(subset_valid) if subset_valid else None,
        }
    summary = {
        "schema_version": "mas_loop_bcp_eval_with_llm_judge_v1",
        "created_at": time.time(),
        "elapsed_sec": time.time() - started,
        "input_files": [str(path) for path in files],
        "output_dir": str(output_dir),
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_max_model_len": args.max_model_len,
        "judge_max_tokens": args.max_tokens,
        "max_workers": args.max_workers,
        "total": len(results),
        "valid": len(valid),
        "judge_failures": len(results) - len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else None,
        "by_candidate": by_candidate,
        "results_file": str(result_path),
        "judge_results_dir": str(output_dir / "judge_results"),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score BCP rollouts with the MAS-Loop Qwen LLM judge")
    parser.add_argument("--input", action="append", required=True, help="JSONL file, shard directory, or glob; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-base-url", default=os.getenv("MAS_LOOP_JUDGE_BASE_URL", "http://127.0.0.1:8513/v1"))
    parser.add_argument("--judge-model", default=os.getenv("MAS_LOOP_JUDGE_MODEL", "judge"))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("MAS_LOOP_JUDGE_MAX_MODEL_LEN", "23192")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAS_LOOP_JUDGE_MAX_TOKENS", "4096")))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("MAS_LOOP_JUDGE_EVAL_WORKERS", "128")))
    parser.add_argument("--candidates", default="source,updated")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all candidates")
    args = parser.parse_args()
    if args.max_tokens != 4096:
        raise ValueError("BCP judge max output tokens must remain 4096")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    return args


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
