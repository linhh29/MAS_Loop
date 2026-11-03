#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
MAS_ROOT = WORKSPACE.parent
if str(WORKSPACE) not in sys.path: sys.path.insert(0, str(WORKSPACE))

from testing_time_scaling.clients import OpenAICompatibleClient
from testing_time_scaling.config import load_config
from testing_time_scaling.data import load_dataset
from testing_time_scaling.generation import generate_cot_bank, generate_mas_candidate
from testing_time_scaling.io_utils import append_jsonl, atomic_json, completed_keys, read_jsonl
from testing_time_scaling.retrieval import build_retriever
from testing_time_scaling.reward_models import RewardScorer, rm_decisions
from testing_time_scaling.selection import LLM_JUDGE_METHODS, METHODS, llm_judge_decision, mas_baseline_decisions, mas_generation_usage
from testing_time_scaling.accounting import sum_usage
from testing_time_scaling.summarize import write_final_tables


def redact_config(value: Any, key: str = "") -> Any:
    if isinstance(value, dict): return {k: redact_config(v, k) for k,v in value.items()}
    if isinstance(value, list): return [redact_config(v, key) for v in value]
    if key.lower() in {"api_key", "token", "secret", "password"} and value: return "***REDACTED***"
    return value


def endpoint_client(cfg: dict[str, Any], name: str) -> OpenAICompatibleClient:
    ep = cfg["endpoints"][name]
    return OpenAICompatibleClient(name=name, model=str(ep.get("model", "")), base_url=str(ep.get("base_url", "")), api_key=str(ep.get("api_key", "")), timeout_s=float(ep.get("timeout_s", 600)), max_retries=int(ep.get("max_retries", 3)), tokenizer=ep.get("tokenizer"), extra_body=ep.get("extra_body"), requests_per_minute=float(ep.get("requests_per_minute", 0)))


def grouped(path: Path) -> dict[tuple[str,str], list[dict[str, Any]]]:
    out: dict[tuple[str,str], list[dict[str,Any]]] = defaultdict(list)
    for row in read_jsonl(path): out[(str(row["dataset"]),str(row["sample_id"]))].append(row)
    return out


def manifest(cfg: dict[str, Any], provenance: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {"schema_version":"mas_testing_time_scaling/1","run_id":run_id,"created_at":datetime.now().astimezone().isoformat(),"python":sys.version,"platform":platform.platform(),"workspace":str(WORKSPACE),"mas_root_read_only":str(MAS_ROOT),"datasets":provenance,"candidate_count":int(cfg["experiment"]["candidates"]),"methods":METHODS,"token_policy":cfg["token_policy"],"retrieval":cfg["retrieval"]}


async def generate(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path) -> None:
    client = endpoint_client(cfg, "generation")
    n = int(cfg["experiment"]["candidates"])
    kinds = [args.kind] if args.kind != "all" else ["cot","mas"]
    sem = asyncio.Semaphore(int(cfg["experiment"].get("sample_concurrency", 1)))
    try:
        for dataset in args.datasets:
            samples, _ = load_dataset(dataset, MAS_ROOT, limit=args.limit)
            if args.num_shards > 1:
                samples = [sample for sample in samples if int(sample["index"]) % args.num_shards == args.shard_index]
            retriever = build_retriever(dataset, MAS_ROOT, allow_hotpot_fts_fallback=False)
            for kind in kinds:
                path = run_dir / f"candidates/{dataset}/{kind}.jsonl"
                done = completed_keys(path, ("dataset","sample_id","candidate_idx"))
                async def one(sample: dict[str, Any]) -> None:
                    async with sem:
                        if kind == "cot":
                            missing = [i for i in range(n) if (dataset,sample["sample_id"],i) not in done]
                            if not missing: return
                            rows = await generate_cot_bank(sample=sample,client=client,retriever=retriever,n=n,cfg=cfg)
                            for row in rows:
                                if row["candidate_idx"] in missing: append_jsonl(path,row)
                        else:
                            missing = [idx for idx in range(n) if (dataset,sample["sample_id"],idx) not in done]
                            concurrency = max(1, int(cfg["experiment"].get("candidate_concurrency", 1)))
                            for offset in range(0, len(missing), concurrency):
                                batch = missing[offset:offset + concurrency]
                                rows = await asyncio.gather(*(
                                    generate_mas_candidate(sample=sample,client=client,retriever=retriever,candidate_idx=idx,cfg=cfg,mas_root=MAS_ROOT)
                                    for idx in batch
                                ))
                                for row in rows:
                                    append_jsonl(path,row)
                # Keep retriever memory bounded and commits incremental.
                for start in range(0,len(samples),max(1,int(cfg["experiment"].get("sample_concurrency",1)))):
                    await asyncio.gather(*(one(s) for s in samples[start:start+int(cfg["experiment"].get("sample_concurrency",1))]))
    finally:
        await client.close()


async def select(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path) -> None:
    methods = METHODS if args.methods == ["all"] else args.methods
    method_rows: dict[str,dict[str,list[dict[str,Any]]]] = defaultdict(lambda:defaultdict(list))
    clients: dict[str,OpenAICompatibleClient] = {}
    try:
        for dataset in args.datasets:
            mas_groups = grouped(run_dir/f"candidates/{dataset}/mas.jsonl")
            for method in methods:
                out_path=run_dir/f"methods/{method}/{dataset}/predictions.jsonl"
                existing=read_jsonl(out_path); done={str(x["sample_id"]) for x in existing}; method_rows[method][dataset].extend(existing)
                if method in {"mas_lb","mas_majority_voting","mas_oracle_ub"}:
                    for (_,sid), rows in sorted(mas_groups.items(),key=lambda x:x[1][0]["index"]):
                        if sid in done: continue
                        if len(rows) != int(cfg["experiment"]["candidates"]): raise RuntimeError(f"incomplete MAS bank {dataset}/{sid}: {len(rows)}")
                        d=mas_baseline_decisions(rows)[method]; row={"method":method,"dataset":dataset,"sample_id":sid,"question":rows[0]["question"],"gold":rows[0]["gold"],**d}; append_jsonl(out_path,row); method_rows[method][dataset].append(row)
                elif method in LLM_JUDGE_METHODS:
                    ep_name, trajectory = LLM_JUDGE_METHODS[method]
                    if ep_name not in clients: clients[ep_name]=endpoint_client(cfg,ep_name)
                    semaphore = asyncio.Semaphore(max(1, int(cfg["judges"].get("concurrency", 16))))

                    async def judge_one(sid: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
                        async with semaphore:
                            attempts = max(1, int(cfg["judges"].get("sample_attempts", 3)))
                            judge_max_tokens = int(cfg["endpoints"][ep_name].get("judge_max_tokens", cfg["judges"]["max_tokens"]))
                            errors: list[str] = []
                            # Capture complete per-sample cost across parse retries. Inner
                            # llm_judge_decision captures still provide one-call details;
                            # ContextVar nesting also feeds every billed response here.
                            with clients[ep_name].capture_usage() as attempt_ledger:
                                for attempt in range(1, attempts + 1):
                                    try:
                                        d=await llm_judge_decision(rows=rows,client=clients[ep_name],method=method,trajectory=trajectory,max_tokens=judge_max_tokens)
                                        selector_usage = attempt_ledger.totals()
                                        d["selector_usage"] = selector_usage
                                        d["usage"] = sum_usage([mas_generation_usage(rows), selector_usage])
                                        d["judge_attempts"] = attempt
                                        return ({"method":method,"dataset":dataset,"sample_id":sid,"question":rows[0]["question"],"gold":rows[0]["gold"],**d}, None)
                                    except Exception as exc:
                                        errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                            return None, f"{dataset}/{sid}: " + " | ".join(errors)

                    pending = []
                    for (_,sid), rows in sorted(mas_groups.items(),key=lambda x:x[1][0]["index"]):
                        if sid in done: continue
                        if len(rows) != int(cfg["experiment"]["candidates"]): raise RuntimeError(f"incomplete MAS bank {dataset}/{sid}: {len(rows)}")
                        pending.append(asyncio.create_task(judge_one(sid, rows)))
                    failures: list[str] = []
                    for task in asyncio.as_completed(pending):
                        row, error = await task
                        if row is not None:
                            append_jsonl(out_path,row); method_rows[method][dataset].append(row)
                        elif error is not None:
                            failures.append(error)
                    if failures:
                        preview = "\n".join(failures[:20])
                        raise RuntimeError(
                            f"API judge {method} failed for {len(failures)} samples after retries; "
                            f"successful samples were checkpointed. First failures:\n{preview}"
                        )
                else:
                    spec=cfg["reward_models"][method]
                    scorer=RewardScorer(base_model=Path(spec["base_model"]),adapter=Path(spec["adapter"]) if spec.get("adapter") else None,max_length=int(spec.get("max_length",8192)),batch_size=int(spec.get("batch_size",2)))
                    try:
                        trajectory=method=="mas_judgerm_trajectory"
                        pending = []
                        for (_,sid), rows in sorted(mas_groups.items(),key=lambda x:x[1][0]["index"]):
                            if sid in done: continue
                            if len(rows) != int(cfg["experiment"]["candidates"]): raise RuntimeError(f"incomplete MAS bank {dataset}/{sid}: {len(rows)}")
                            pending.append((sid, rows))
                        sample_batch = max(1, int(spec.get("sample_batch_size", 4)))
                        for start in range(0, len(pending), sample_batch):
                            chunk = pending[start:start+sample_batch]
                            decisions = rm_decisions([rows for _, rows in chunk], scorer, trajectory=trajectory)
                            for (sid, rows), d in zip(chunk, decisions):
                                row={"method":method,"dataset":dataset,"sample_id":sid,"question":rows[0]["question"],"gold":rows[0]["gold"],**d}; append_jsonl(out_path,row); method_rows[method][dataset].append(row)
                    finally: scorer.close()
        # Rebuild the final table from every completed method artifact, including methods run in prior split invocations.
        all_rows: dict[str,dict[str,list[dict[str,Any]]]] = defaultdict(dict)
        for known_method in METHODS:
            # Always rebuild complete benchmark tables. A dataset-scoped resume
            # must not erase already completed metrics from the other dataset.
            for known_dataset in ("hotpotqa", "bcp"):
                rows=read_jsonl(run_dir/f"methods/{known_method}/{known_dataset}/predictions.jsonl")
                if rows: all_rows[known_method][known_dataset]=rows
        table=write_final_tables(run_dir,all_rows)
        print(json.dumps(table,ensure_ascii=False,indent=2))
    finally:
        await asyncio.gather(*(c.close() for c in clients.values()),return_exceptions=True)


def validate(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    report={"datasets":{},"retrieval":{},"paths":{},"endpoints":{}}
    for dataset in args.datasets:
        samples,prov=load_dataset(dataset,MAS_ROOT,limit=0); report["datasets"][dataset]=prov
        strict_error = None
        try:
            retriever=build_retriever(dataset,MAS_ROOT,allow_hotpot_fts_fallback=False)
        except Exception as exc:
            strict_error=f"{type(exc).__name__}: {exc}"
            retriever=build_retriever(dataset,MAS_ROOT,allow_hotpot_fts_fallback=True)
        docs=retriever.search(samples[0]["question"],2)
        report["retrieval"][dataset]={"backend":retriever.backend,"strict_backend_available":strict_error is None,"strict_backend_error":strict_error,"query":samples[0]["question"],"num_hits":len(docs),"docids":[x.get("docid") for x in docs]}
    for name,ep in cfg["endpoints"].items(): report["endpoints"][name]={"configured":bool(ep.get("base_url") and ep.get("model")),"model":ep.get("model","")}
    for method,spec in cfg["reward_models"].items(): report["paths"][method]={"base_model_exists":Path(spec["base_model"]).is_dir(),"adapter_exists":None if not spec.get("adapter") else Path(spec["adapter"]).is_dir()}
    return report


async def main_async() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["validate","generate","select","all"])
    p.add_argument("--config",type=Path,default=HERE/"config.example.yaml")
    p.add_argument("--secrets",type=Path,default=WORKSPACE/"api_secrets.yaml")
    p.add_argument("--run-id",default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--datasets",nargs="+",choices=["hotpotqa","bcp"],default=["hotpotqa","bcp"])
    p.add_argument("--kind",choices=["cot","mas","all"],default="all")
    p.add_argument("--methods",nargs="+",choices=["all",*METHODS],default=["all"])
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--num-shards", type=int, default=1, help="Partition samples into disjoint resume workers")
    p.add_argument("--shard-index", type=int, default=0, help="Zero-based sample shard index")
    args=p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    cfg=load_config(args.config,args.secrets)
    run_dir=HERE/"results"/args.run_id; run_dir.mkdir(parents=True,exist_ok=True)
    provenance={d:load_dataset(d,MAS_ROOT,limit=args.limit)[1] for d in args.datasets}
    atomic_json(run_dir/"config.resolved.json",redact_config(cfg)); atomic_json(run_dir/"manifest.json",manifest(cfg,provenance,args.run_id))
    if args.command=="validate":
        report=validate(cfg,args); atomic_json(run_dir/"validation.json",report); print(json.dumps(report,ensure_ascii=False,indent=2)); return
    if args.command in {"generate","all"}: await generate(args,cfg,run_dir)
    if args.command in {"select","all"}: await select(args,cfg,run_dir)

if __name__=="__main__": asyncio.run(main_async())
