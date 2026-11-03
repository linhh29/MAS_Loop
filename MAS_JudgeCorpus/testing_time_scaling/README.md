# MAS testing-time scaling experiment

This directory is an isolated, resumable implementation of the requested 16-candidate testing-time-scaling study. It reads the existing repository datasets, BM25 assets, MAS builder, base reward model, and LoRA adapters without modifying them. Every generated artifact is written under `testing_time_scaling/results/<run_id>/`.

## Evaluated data and methods

- HotpotQA test: exactly 200 rows from `hotpotqa/data/hotpotqa_test.jsonl`.
- BrowseComp-Plus OOT test: exactly 168 rows from `data/browsecomp_plus_test.jsonl`.
- Methods: MAS LB, 16-way MAS majority voting, MAS oracle upper bound, DeepSeek-V4-Flash/Kimi-K3/Qwen3.8-2.4T-A95B orchestration and trajectory judges, Skywork-Reward-V2-Qwen3-8B, Orch-RM, and MAS-JudgeRM orchestration/trajectory levels.

The two MAS-JudgeRM verification levels are deliberately separate output rows.

## Retrieval contract

- **HotpotQA:** the required HotpotQA Lucene corpus at `hotpotqa/indexes/bm25`, accessed with Pyserini 1.2.0 and Java 21. Install these dependencies in the experiment environment and point `config.example.yaml` to the local index. Full candidate generation refuses to silently fall back. The repository-provided SQLite FTS5 export remains validation-only. The pinned profile is one round, top-k 5.
- **BCP:** the exact retrieval controller parquet BM25 cache with `max_doc_chars=9000`, corpus fingerprint `09ef823e...`, k1=1.2, b=0.75. The pinned profile is five rounds, top-k 5. This avoids silently switching to Lucene or another fallback.

Retrieval uses only the public question and intermediate evidence; gold answers are never included in prompts.

## Token accounting

Every OpenAI-compatible generation call records API `prompt_tokens`, `completion_tokens`, and `total_tokens`. If an endpoint omits usage, the configured local tokenizer is used; if no tokenizer is configured, a clearly marked character estimate is used. Every call is stored in each candidate/decision token ledger.

Reward models are sequence classifiers rather than generative vLLM calls. Their exact non-padding tokenizer input count is reported as input tokens, with output tokens set to zero. The final `Token (M)` column includes:

- MAS LB: MAS candidate 0.
- MAS majority and MAS oracle: all 16 MAS candidates.
- Every MAS selector: all 16 MAS build/execution calls plus its own LLM-judge or RM input tokens.

Candidate calls can be physically cached, but each method is charged the full logical bank it requires. Accuracy uses deterministic answer extraction and normalized matching, so no hidden evaluation-judge calls contaminate token totals.

## Configuration

Create an untracked `testing_time_scaling/api_secrets.yaml` containing the
endpoint `base_url` and `api_key` overrides needed by the configured methods.
The file is merged over `config.example.yaml` at runtime.

`api_secrets.yaml` and result files are gitignored. `config.resolved.json` is always redacted.

## Commands

```bash
# Validate exact row counts, paths, checkpoints, and retrieve sample documents.
python testing_time_scaling/run.py validate --run-id validation

# Small end-to-end candidate smoke test after endpoints are configured.
python testing_time_scaling/run.py generate --run-id smoke --limit 1 --kind all
python testing_time_scaling/run.py select --run-id smoke --limit 1 --methods all

# Full requested run (resumes from JSONL candidate/decision keys).
python testing_time_scaling/run.py all --run-id full_200_168
```

You may split expensive phases:

```bash
python testing_time_scaling/run.py generate --run-id full_200_168 --kind cot
python testing_time_scaling/run.py generate --run-id full_200_168 --kind mas
python testing_time_scaling/run.py select --run-id full_200_168 --methods mas_lb mas_majority_voting mas_oracle_ub
python testing_time_scaling/run.py select --run-id full_200_168 --methods deepseek_v4_flash deepseek_v4_flash_trajectory kimi_k3 kimi_k3_trajectory qwen3_8_2_4t_a95b qwen3_8_2_4t_a95b_trajectory
python testing_time_scaling/run.py select --run-id full_200_168 --methods skywork_v2_qwen3_8b orch_rm mas_judgerm_orchestration mas_judgerm_trajectory
```

## Output layout

```text
results/<run_id>/
  manifest.json
  config.resolved.json
  validation.json
  candidates/<dataset>/{cot,mas}.jsonl
  methods/<method>/<dataset>/predictions.jsonl
  methods/<method>/<dataset>/summary.json
  final_table.{json,csv,tex}
```

MAS candidate rows contain the generated code, three-stage build artifacts, state, per-agent retrieval sessions, orchestration XML, trajectory XML, predictions, correctness, retries, and all model-call usage.

## Strict-Lucene runtime

The experiment expects Pyserini 1.2.0, Pyjnius 1.7.0, and Java 21 to be
available in the runtime environment. For sparse-only retrieval, two narrowly
scoped compatibility guards are supported:

- `PYSERINI_DISABLE_VECTOR_MODULE=1` for sparse BM25;
- `PYSERINI_SPARSE_ONLY=1` to avoid importing optional dense encoders.

`retrieval.py` sets both automatically before Pyjnius starts the JVM. `run.py validate` must report `hotpotqa_lucene_pyserini` and `strict_backend_available: true` before any full Hotpot run.

## External API configuration

Generation plus all three orchestration judges are configured in workspace-root `api_secrets.yaml`. Secret values stay outside resolved configs and result manifests. DeepSeek-V4-Flash, Kimi-K3, and Qwen3.8-2.4T-A95B are paced at 60 physical HTTP attempts/minute per endpoint. Qwen judge uses a larger output budget because its provider returns reasoning before structured JSON.
