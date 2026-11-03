#!/usr/bin/env python3
"""Strict completion audit for a testing-time-scaling run."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

DATASETS = {"hotpotqa": 200, "bcp": 168}
CANDIDATE_COUNT = 16
LOCAL_METHODS = [
    "mas_lb", "mas_majority_voting", "mas_oracle_ub",
    "skywork_v2_qwen3_8b", "orch_rm",
    "mas_judgerm_orchestration", "mas_judgerm_trajectory",
]
API_METHODS = [
    "deepseek_v4_flash", "deepseek_v4_flash_trajectory",
    "kimi_k3", "kimi_k3_trajectory",
    "qwen3_8_2_4t_a95b", "qwen3_8_2_4t_a95b_trajectory",
]
ALL_METHODS = LOCAL_METHODS[:3] + API_METHODS + LOCAL_METHODS[3:]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def audit(run_dir: Path, *, require_api: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"run_dir": str(run_dir), "candidate_banks": {}, "methods": {}}

    manifest_path = run_dir / "manifest.json"
    check(manifest_path.exists(), "missing manifest.json", errors)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    check(int(manifest.get("candidate_count", -1)) == CANDIDATE_COUNT, "manifest candidate_count is not 16", errors)
    retrieval = manifest.get("retrieval", {})
    check(retrieval.get("hotpotqa", {}).get("backend") == "hotpotqa_lucene", "HotpotQA manifest backend is not hotpotqa_lucene", errors)
    check(retrieval.get("bcp", {}).get("backend") == "controller_parquet", "BCP manifest backend is not controller_parquet", errors)

    for dataset, sample_count in DATASETS.items():
        for kind in ("cot", "mas"):
            path = run_dir / "candidates" / dataset / f"{kind}.jsonl"
            rows = read_jsonl(path)
            expected_rows = sample_count * CANDIDATE_COUNT
            pairs = collections.Counter((str(r.get("sample_id")), int(r.get("candidate_idx", -1))) for r in rows)
            by_sample: dict[str, set[int]] = collections.defaultdict(set)
            for row in rows:
                by_sample[str(row.get("sample_id"))].add(int(row.get("candidate_idx", -1)))
                check(row.get("candidate_usage") is not None, f"{dataset}/{kind} row lacks candidate_usage", errors)
                usage = row.get("candidate_usage") or {}
                check(int(usage.get("total_tokens", -1)) >= 0, f"{dataset}/{kind} row has invalid candidate total_tokens", errors)
                if kind == "cot":
                    check(row.get("shared_retrieval_usage") is not None, f"{dataset}/cot row lacks shared_retrieval_usage", errors)
                else:
                    check(bool(row.get("mas_xml_orchestration")), f"{dataset}/mas row lacks orchestration XML", errors)
                    check(bool(row.get("mas_xml_trajectory")), f"{dataset}/mas row lacks trajectory XML", errors)
            invalid_samples = [sid for sid, indices in by_sample.items() if indices != set(range(CANDIDATE_COUNT))]
            bank = {
                "rows": len(rows), "expected_rows": expected_rows, "unique_pairs": len(pairs),
                "duplicates": sum(v - 1 for v in pairs.values() if v > 1), "samples": len(by_sample),
                "samples_with_invalid_indices": len(invalid_samples),
            }
            report["candidate_banks"][f"{dataset}/{kind}"] = bank
            check(len(rows) == expected_rows, f"{dataset}/{kind}: {len(rows)} rows, expected {expected_rows}", errors)
            check(len(pairs) == expected_rows and max(pairs.values(), default=0) <= 1, f"{dataset}/{kind}: duplicate or missing candidate pairs", errors)
            check(len(by_sample) == sample_count, f"{dataset}/{kind}: {len(by_sample)} samples, expected {sample_count}", errors)
            check(not invalid_samples, f"{dataset}/{kind}: {len(invalid_samples)} samples do not contain indices 0..15", errors)

    required_methods = list(LOCAL_METHODS) + (API_METHODS if require_api else [])
    for method in ALL_METHODS:
        report["methods"][method] = {}
        for dataset, sample_count in DATASETS.items():
            path = run_dir / "methods" / method / dataset / "predictions.jsonl"
            rows = read_jsonl(path)
            sample_ids = [str(r.get("sample_id")) for r in rows]
            entry = {"rows": len(rows), "expected_rows": sample_count, "unique_samples": len(set(sample_ids))}
            report["methods"][method][dataset] = entry
            if method in required_methods:
                check(len(rows) == sample_count, f"{method}/{dataset}: {len(rows)} predictions, expected {sample_count}", errors)
                check(len(set(sample_ids)) == sample_count, f"{method}/{dataset}: duplicate or missing prediction sample_ids", errors)
                for row in rows:
                    usage = row.get("usage") or {}
                    inp, out, total = (int(usage.get(k, -1)) for k in ("input_tokens", "output_tokens", "total_tokens"))
                    check(inp >= 0 and out >= 0 and total == inp + out, f"{method}/{dataset}: invalid token usage", errors)
                    check(row.get("correct") is not None, f"{method}/{dataset}: unscored prediction", errors)
            elif not rows:
                warnings.append(f"{method}/{dataset}: API result not present")

    final_path = run_dir / "final_table.json"
    paper_path = run_dir / "paper_table.json"
    check(final_path.exists(), "missing final_table.json", errors)
    check(paper_path.exists(), "missing paper_table.json", errors)
    final_rows = json.loads(final_path.read_text()) if final_path.exists() else []
    paper_rows = json.loads(paper_path.read_text()) if paper_path.exists() else []
    final_keys = {(r.get("method_id"), r.get("dataset")) for r in final_rows}
    for method in required_methods:
        for dataset in DATASETS:
            check((method, dataset) in final_keys, f"final_table missing {method}/{dataset}", errors)
    check(len(paper_rows) == len(ALL_METHODS), f"paper_table has {len(paper_rows)} rows, expected {len(ALL_METHODS)}", errors)
    check([r.get("method_id") for r in paper_rows] == ALL_METHODS, "paper_table method order/content is incorrect", errors)
    for row in paper_rows:
        method = row.get("method_id")
        if method in required_methods:
            check(row.get("hotpotqa_accuracy") is not None and row.get("hotpotqa_token_m") is not None, f"paper_table missing HotpotQA metrics for {method}", errors)
            check(row.get("bcp_oot_accuracy") is not None and row.get("bcp_oot_token_m") is not None, f"paper_table missing BCP metrics for {method}", errors)

    report.update({"require_api": require_api, "errors": errors, "warnings": sorted(set(warnings)), "passed": not errors})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir.resolve(), require_api=args.require_api)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
