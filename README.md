# MAS-Loop: Beyond One-Shot Orchestration towards Self-Evolving Multi-Agent Systems

<div align="center">

### Construct → Execute → Diagnose & Refine → Re-execute

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.38946)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/linhh29/MAS_Loop)
[![Project Blog](https://img.shields.io/badge/Blog-Project%20Page-0078D4?logo=readthedocs&logoColor=white)](https://linhh29.github.io/blog/MAS-Loop/index.html)
[![Online Demo](https://img.shields.io/badge/Demo-Online%20Demo-512BD4?logo=rocket&logoColor=white)](https://mas-loop-demo.hehailin.life/)

Execution-grounded feedback connects system construction, trajectory diagnosis, and independent re-execution.

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Method Pipeline](#2-method-pipeline)
3. [Repository Structure](#3-repository-structure)
4. [Environment Setup](#4-environment-setup)
5. [Configuration](#5-configuration)
6. [Running MAS-Loop](#6-running-mas-loop)
7. [Reward-Model Pipeline](#7-reward-model-pipeline)
8. [Testing-Time Scaling](#8-testing-time-scaling)
9. [Outputs and Reproducibility Records](#9-outputs-and-reproducibility-records)
10. [Tests](#10-tests)

---

## 1. Overview

MAS-Loop is a framework for improving a multi-agent system (MAS) through a
closed loop of construction, execution, diagnosis, and structured editing.
Given a task, the framework first constructs an executable agent graph, runs
the graph to obtain a trajectory, identifies weaknesses in the resulting
system, applies a graph-level edit, and evaluates the updated system against
the original one.

The implementation supports:

- structured MAS definitions with typed agents, dependencies, tools, budgets,
  execution modes, and aggregation rules;
- task decomposition and agent-role construction by an **Orchestra** model;
- compilation of MAS graphs into executable asynchronous Python workflows;
- trajectory-aware diagnosis and graph editing by an **Editor** model;
- source-versus-updated scoring with task reward, judge reward, token usage,
  latency, and cost accounting;
- paired trajectory construction for Orchestra, Editor, and reward-model
  training;
- distributed rollout and reinforcement learning through the bundled
  `verl` backend;
- testing-time scaling with multiple MAS candidates and learned selectors.

### Two implementation layers

| Layer | Intended use | Main location |
| --- | --- | --- |
| Formalized pipeline | Inspecting task-level construction, execution, and structured editing | `formalize/`, `mas_loop/` |
| Distributed training pipeline | Large-scale rollouts, Orchestra/Editor optimization, reward integration, and checkpoint evaluation | `verl/mas_loop_rl/`, `verl/scripts/` |

---

## 2. Method Pipeline

### 2.1 End-to-end loop

```text
Input task
   │
   ▼
Task decomposition
   │
   ▼
Agent-role and dependency construction
   │
   ▼
Initial MAS graph
   │
   ▼
Compile and execute ───────────────► Source trajectory and score
   │                                           │
   │                                           ▼
   │                                Diagnosis and graph patch
   │                                           │
   │                                           ▼
   └─────────────────────────────── Updated MAS graph
                                               │
                                               ▼
                                    Re-execute and compare
                                               │
                                               ▼
                              Evolution record / training pair
```

### 2.2 Processing stages

| Stage | Component | Input | Output |
| --- | --- | --- | --- |
| 1 | Task decomposition | Task text and dataset requirements | Goal, subtasks, and constraints |
| 2 | Agent engineering | Decomposition and task context | Agent roles, dependencies, tools, and budgets |
| 3 | Graph assembly | Agent specifications | Typed `MASGraph` / `MASConfig` |
| 4 | Compilation and execution | MAS graph and model/tool clients | Final answer, agent traces, usage, latency, and errors |
| 5 | Evaluation | Execution result and reference signal | Task score, judge result, and resource statistics |
| 6 | Structured editing | Source graph, trace, and evaluation evidence | `Patch` / `GraphPatch` |
| 7 | Re-execution and comparison | Updated graph | Updated trajectory and source-to-updated delta |

### 2.3 Principal data structures

| Type | Role |
| --- | --- |
| `MASConfig` / `MASGraph` | Canonical representation of agents, graph topology, edges, tools, constraints, and aggregation |
| `Patch` / `GraphPatch` | Validated graph-edit operations, including agent revision, addition, removal, and rewiring |
| `ExecutionTrajectory` | Per-agent outputs, errors, token usage, cost, elapsed time, and final output |
| `EvoTrajSample` | Complete source MAS → edit → updated MAS transition with outcome statistics |

Supported topology representations include sequential, directed acyclic,
hierarchical, and iterative agent systems.

---

## 3. Repository Structure

```text
MAS_Loop/
├── mas_loop/                       # Canonical MAS and trajectory schemas
│   ├── mas_schema.py
│   ├── mas_evo_traj.py
│   ├── dataset_runners/            # Dataset-specific execution adapters
│   └── training/                   # Lightweight training utilities
│
├── formalize/                      # Readable reference implementation
│   ├── orchestra.py                # Decomposition and agent construction
│   ├── compiler.py                 # MAS graph -> executable Python workflow
│   ├── executor.py                 # Graph execution and trace collection
│   ├── editor.py                   # Trajectory-conditioned graph editing
│   ├── schema.py                   # Formal graph and patch types
│   └── prompts.py                  # Structured construction/editing prompts
│
├── building/                       # MAS and Skill-MAS construction runtime
│   ├── skill_mas/                  # Fixed-phase builder and async model client
│   ├── template/                   # Generated workflow templates
│   ├── init_skill/                 # Initial construction skill
│   └── optimized_skill/            # Dataset-specific optimized skills
│
├── verl/
│   ├── mas_loop_rl/                # Distributed MAS-Loop integration
│   │   ├── loop_agent.py           # Orchestra, Editor, and replay agent loops
│   │   ├── executor.py             # Scalable MAS execution
│   │   ├── reward.py               # Training rewards and pair logging
│   │   ├── schema.py               # Runtime MAS/patch normalization
│   │   ├── bcp_retrieval.py        # Retrieval integration
│   │   ├── bcp_judge.py            # LLM-judge integration
│   │   ├── prepare_data.py          # Role-specific training-data conversion
│   │   └── eval_checkpoint.py       # Checkpoint evaluation
│   ├── scripts/                    # Dataset and training launchers
│   └── verl/                       # Bundled training backend
│
└── MAS_JudgeCorpus/                # Reward-model corpus and training
│   ├── build_task_filtered_rm.py
│   ├── train_lora_rm.py
│   └── testing_time_scaling/
```

### Recommended entry points

| Objective | Entry point |
| --- | --- |
| Understand the MAS representation | `mas_loop/mas_schema.py`, `formalize/schema.py` |
| Inspect graph construction | `formalize/orchestra.py` |
| Inspect graph compilation and execution | `formalize/compiler.py`, `formalize/executor.py` |
| Inspect Editor behavior | `formalize/editor.py` |
| Inspect distributed Orchestra/Editor rollouts | `verl/mas_loop_rl/loop_agent.py` |
| Inspect the learning signal | `verl/mas_loop_rl/reward.py` |
| Build reward-model data | `MAS_JudgeCorpus/build_task_filtered_rm.py` |

---

## 4. Environment Setup

### 4.1 Python environment

The bundled training backend requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 Core training backend

```bash
python -m pip install -e ./verl
```

The readable API-backed pipeline also requires the `openai` Python package.
Reward-model training additionally requires `datasets`, `transformers`,
`peft`, and `PyYAML`.

```bash
python -m pip install openai pyyaml datasets transformers peft
```

GPU experiments require versions of PyTorch, CUDA, vLLM or SGLang, and the
distributed runtime that are compatible with the selected launcher. Relevant
dependency manifests are provided in:

- `verl/requirements.txt`;
- `verl/requirements-npu.txt`.

### 4.3 Source-tree smoke test

```bash
python - <<'PY'
from mas_loop.mas_schema import MASConfig
from mas_loop.mas_evo_traj import EvoTrajSample

print("MAS-Loop imports succeeded")
PY
```

---

## 5. Configuration

### 5.1 Model endpoints

Model identifiers, endpoint defaults, pricing metadata, temperatures, and
token limits for the formalized pipeline are defined in:

```text
building/skill_mas/model_config.json
```

No usable credential is stored in this artifact. Credentials must be supplied
through environment variables or an untracked local secrets file.

```bash
export OPENAI_API_KEY="..."

export MAS_LOOP_ORCHESTRA_BASE_URL="http://host:port/v1"
export MAS_LOOP_EDITOR_BASE_URL="http://host:port/v1"
export MAS_LOOP_SUBAGENT_BASE_URL="http://host:port/v1"
export MAS_LOOP_JUDGE_BASE_URL="http://host:port/v1"

export MAS_LOOP_ORCHESTRA_API_KEY="..."
export MAS_LOOP_EDITOR_API_KEY="..."
export MAS_LOOP_SUB_AGENT_API_KEY="..."
```

For an unauthenticated loopback vLLM endpoint, the client supplies a non-secret
SDK placeholder automatically.

### 5.2 Data, retrieval, and output paths

The most frequently overridden path variables are:

| Variable | Purpose |
| --- | --- |
| `MAS_LOOP_ROOT` | Repository root |
| `MAS_LOOP_BCP_ROOT` | BrowseComp-Plus dataset/resource root |
| `MAS_LOOP_BM25_INDEX` | BM25 or Lucene retrieval index |
| `MAS_LOOP_LOG_DIR` | Rollout and reward logs |
| `MAS_LOOP_PAIR_DIR` | Source/updated training-pair records |
| `MAS_LOOP_CKPT_ROOT` | Checkpoints and run artifacts |

All `/path/to/...` values in the repository are anonymous placeholders and
must be replaced or overridden for the local environment.

### 5.3 External assets

This source-only artifact does not bundle large datasets, retrieval indexes, or
model checkpoints.

| Asset | Expected configuration |
| --- | --- |
| Benchmark data | Dataset-specific path in the runner, YAML, or launcher |
| Base models | Model path or OpenAI-compatible endpoint |
| Orchestra/Editor checkpoints | Checkpoint path in the selected launcher |
| BM25/Lucene resources | Retrieval root and index path |
| Reward-model adapters | Adapter path in the reward-model YAML |

---

## 6. Running MAS-Loop

### 6.1 Formalized task-level pipeline

The slim artifact exposes the task-level lifecycle as library code in
`formalize/`. Dataset-specific batch drivers are intentionally omitted. The
core implementation covers:

1. task decomposition and agent specifications;
2. initial MAS graph;
3. source execution trace and score;
4. Editor diagnosis and graph patch;
5. updated MAS graph;
6. updated execution trace and score;
7. source-versus-updated comparison.

### 6.2 Distributed Orchestra/Editor training

Convert task or evolution records into role-specific training data:

```bash
python -m mas_loop_rl.prepare_data \
  --input /path/to/tasks_or_evolution_trajectories.jsonl \
  --output-dir /path/to/training_data \
  --dataset hotpotqa \
  --role both
```

This produces Orchestra and Editor train/validation Parquet files together
with validation manifests.

The canonical launchers are:

| Dataset/workflow | Launcher |
| --- | --- |
| Shared adversarial GRPO logic | `verl/scripts/run_mas_loop_adversarial_grpo.sh` |
| HotpotQA | `verl/scripts/run_hotpotqa_mas_loop_full.sh` |
| BrowseComp-Plus | `verl/scripts/run_bcp_mas_loop_full.sh` |
| DeepScaleR | `verl/scripts/run_deepscaler_mas_loop_full.sh` |
| ALFWorld | `verl/scripts/run_alfworld_mas_loop_full.sh` |

These launchers expose environment-specific GPU topology, model placement,
vLLM concurrency, rollout length, batch size, and checkpoint settings. Review
and override their defaults before execution.

---

## 7. Reward-Model Pipeline

### 7.1 Constructing reward-model pairs

```bash
python -m MAS_JudgeCorpus.build_task_filtered_rm \
  --archive /path/to/paired_trajectories.jsonl \
  --aggregation /path/to/task_statistics.json \
  --output-dir /path/to/rm_data
```

The constructor filters task-level source/updated trajectories and emits
chosen/rejected examples suitable for Bradley–Terry training.

### 7.2 Training MAS-JudgeRM

Orchestration-level reward model:

```bash
python -m MAS_JudgeCorpus.train_lora_rm \
  MAS_JudgeCorpus/configs_orchestration_level_lora.yaml
```

Trajectory-level reward model:

```bash
python -m MAS_JudgeCorpus.train_lora_rm \
  MAS_JudgeCorpus/configs_trajectory_level_lora.yaml
```

Before training, replace the model, data, and output paths in the selected YAML
file.

---

## 8. Testing-Time Scaling

The testing-time-scaling pipeline generates multiple CoT or MAS candidates and
then evaluates selection methods based on voting, oracle analysis, LLM judges,
or learned reward models.

Configuration:

```text
MAS_JudgeCorpus/testing_time_scaling/config.example.yaml
```

Validate dataset, retrieval, endpoint, and model paths:

```bash
python -m MAS_JudgeCorpus.testing_time_scaling.run validate \
  --config MAS_JudgeCorpus/testing_time_scaling/config.example.yaml \
  --run-id validation
```

Run candidate generation and selection:

```bash
python -m MAS_JudgeCorpus.testing_time_scaling.run all \
  --config MAS_JudgeCorpus/testing_time_scaling/config.example.yaml \
  --run-id experiment \
  --datasets hotpotqa bcp \
  --kind all \
  --methods all
```

The `generate` and `select` subcommands can be executed independently. Both
phases are resumable using the same `--run-id`.

For remote endpoints, place overrides in the untracked file:

```text
MAS_JudgeCorpus/testing_time_scaling/api_secrets.yaml
```

See `MAS_JudgeCorpus/testing_time_scaling/README.md` for method definitions,
token accounting, sharding, and output details.

---

## 9. Outputs and Reproducibility Records

### 9.1 Distributed training

A distributed run may contain:

- sharded Orchestra and Editor rollout records;
- source/updated pair files;
- judge decisions and retrieval traces;
- training and validation manifests;
- timing and concurrency diagnostics;
- round- and role-specific checkpoints;
- checkpoint evaluation summaries.

Generated outputs are intentionally excluded from this artifact. They can be
reconstructed from the code when the required external data and models are
available.

### 9.2 Testing-time scaling

```text
results/<run_id>/
├── manifest.json
├── config.resolved.json
├── validation.json
├── candidates/<dataset>/{cot,mas}.jsonl
├── methods/<method>/<dataset>/predictions.jsonl
├── methods/<method>/<dataset>/summary.json
└── final_table.{json,csv,tex}
```

Resolved configuration files redact secret values.

---

## 10. Tests

MAS-Loop backend tests:

```bash
pytest -q verl/tests/mas_loop_rl
```

Testing-time-scaling unit tests:

```bash
pytest -q MAS_JudgeCorpus/testing_time_scaling/tests
```

CPU-oriented tests cover schemas, prompt formatting, scoring, checkpoint
evaluation, structured output, and data handling. End-to-end rollout and
training tests additionally require the configured GPU and model services.
