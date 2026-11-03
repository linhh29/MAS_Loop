#!/usr/bin/env bash
set -xeuo pipefail
# Local-only MAS-Loop adversarial GRPO launcher. Uses this repository's ./verl only.
# Checkpoints are written directly under MAS_LOOP_CKPT_ROOT; no runs/ or remote staging is used.

# Preserve the timestamp from the first launcher process across the immutable
# script-snapshot exec.  This lets reports distinguish model/service startup
# from the joint optimizer step itself.
export MAS_LOOP_LAUNCH_STARTED_AT_UNIX=${MAS_LOOP_LAUNCH_STARTED_AT_UNIX:-$(date +%s.%N)}

MAS_ROOT=${MAS_ROOT:-/path/to/MAS_Loop}
VERL_ROOT=${VERL_ROOT:-${MAS_ROOT}/verl}
# Keep each dataset's checkpoints and rollout artifacts isolated.  The dataset
# variable is also exported by the dataset-specific adapters before launch.
MAS_LOOP_DATASET_NAME=${MAS_LOOP_DATASET_NAME:-unscoped}
MAS_LOOP_CKPT_ROOT=${MAS_LOOP_CKPT_ROOT:-/path/to/checkpoints/${MAS_LOOP_DATASET_NAME}}
# vLLM sleep mode uses CUDA memory pools, which are incompatible with PyTorch
# expandable segments (PyTorch issue #147851). Clear inherited allocator settings
# before launching either the external peer server or the colocated rollout engine.
unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-120000}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}
# vLLM V1 multiproc_executor.execute_model uses this seconds env, not VLLM_RPC_TIMEOUT.
# Long 8k-token rollouts with TP=8 can exceed the upstream default 300s and falsely kill EngineCore.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}
# Also enlarge the PyTorch ProcessGroupNCCL watchdog default used by vLLM stateless TP groups.
export MAS_LOOP_PATCH_TORCH_DIST_TIMEOUT=${MAS_LOOP_PATCH_TORCH_DIST_TIMEOUT:-0}
export MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS=${MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS:-600}
# TransferQueue uses one short-lived ZMQ DEALER socket per completed put.  A
# bounded notification fan-out prevents libzmq's context socket ceiling from
# being hit by large Editor replay batches without reducing rollout.n.
export TQ_DATA_UPDATE_MAX_CONCURRENCY=${TQ_DATA_UPDATE_MAX_CONCURRENCY:-128}
export TQ_DATA_UPDATE_MAX_RETRIES=${TQ_DATA_UPDATE_MAX_RETRIES:-3}
export TQ_DATA_UPDATE_RETRY_BACKOFF_SEC=${TQ_DATA_UPDATE_RETRY_BACKOFF_SEC:-0.25}
# Keep Ray async actor RPC admission independent from vLLM max_num_seqs.  The
# latter may be very large for KV/scheduler capacity, while mirroring >12k into
# Ray max_concurrency has triggered TaskReceiver objects_valid crashes.
export MAS_LOOP_RAY_SERVER_MAX_CONCURRENCY=${MAS_LOOP_RAY_SERVER_MAX_CONCURRENCY:-1024}
# Global pre-RPC admission shared by all AgentLoop workers. This is separate
# from vLLM max_num_seqs/KV capacity and prevents a large GRPO batch from
# placing every generate RPC into the vLLM Ray actor mailbox at once.
export MAS_LOOP_RAY_GENERATE_MAX_OUTSTANDING=${MAS_LOOP_RAY_GENERATE_MAX_OUTSTANDING:-128}
# Model replicas can take longer than 15 minutes to load from shared storage.
# This bounds startup only; it is recorded separately from joint-step timing.
MAS_LOOP_VLLM_STARTUP_ATTEMPTS=${MAS_LOOP_VLLM_STARTUP_ATTEMPTS:-720}
if [[ ! "${MAS_LOOP_VLLM_STARTUP_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAS_LOOP_VLLM_STARTUP_ATTEMPTS must be a positive integer, got: ${MAS_LOOP_VLLM_STARTUP_ATTEMPTS}" >&2
  exit 2
fi
# Verl V1 multi-output TransferQueue requires tensordict>=0.10. Keep all
# compatibility packages project-local so the other Verl checkout is untouched.
if [[ "${MAS_LOOP_BOOTSTRAP_V1_DEPS:-1}" == "1" ]]; then
  flock /tmp/mas_loop_bootstrap_deps.lock env MAS_ROOT="${MAS_ROOT}" VERL_ROOT="${VERL_ROOT}" bash "${VERL_ROOT}/scripts/bootstrap_mas_loop_v1_deps.sh"
fi
MAS_LOOP_PYTHON_MIRROR=${MAS_LOOP_PYTHON_MIRROR:-}
if [[ -n "${MAS_LOOP_PYTHON_MIRROR}" ]]; then
  [[ "${MAS_LOOP_PYTHON_MIRROR}" == /tmp/* && -f "${MAS_LOOP_PYTHON_MIRROR}/mas_loop_rl/__init__.py" ]] || {
    echo "Invalid /tmp mas_loop_rl Python mirror: ${MAS_LOOP_PYTHON_MIRROR}" >&2
    exit 2
  }
  export PYTHONPATH="${MAS_LOOP_RUNTIME_DEPS_DIR:-${VERL_ROOT}/.mas_loop_runtime_deps}:${MAS_LOOP_PYTHON_MIRROR}:${VERL_ROOT}:${MAS_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${MAS_LOOP_RUNTIME_DEPS_DIR:-${VERL_ROOT}/.mas_loop_runtime_deps}:${VERL_ROOT}:${MAS_ROOT}:${PYTHONPATH:-}"
fi

ALL_GPUS=${ALL_GPUS:-8,9,10,11,12,13,14,15}
N_GPUS=$(awk -F, '{print NF}' <<< "${ALL_GPUS}")
case "${ALL_GPUS}" in
  "0,1,2,3,4,5,6,7"|"8,9,10,11,12,13,14,15") ;;
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    [[ "${MAS_LOOP_ENABLE_16GPU_ABLATION:-0}" == "1" ]] || { echo "[ERROR] full 16-GPU domain requires MAS_LOOP_ENABLE_16GPU_ABLATION=1" >&2; exit 2; } ;;
  *) echo "[ERROR] allowed GPU sets are one eight-GPU domain or explicit full-node ablation; current ALL_GPUS=${ALL_GPUS}" >&2; exit 2 ;;
esac

TASK_JSONL=${TASK_JSONL:-${MAS_ROOT}/datasets/aime/data/aime_test.jsonl}
RUN_DIR=${RUN_DIR:-${MAS_LOOP_CKPT_ROOT}}
DATA_DIR=${DATA_DIR:-${RUN_DIR}/data}
LOG_DIR=${LOG_DIR:-${RUN_DIR}/logs}
mkdir -p "${DATA_DIR}" "${LOG_DIR}"

JOINT_TIMELINE_FILE=${JOINT_TIMELINE_FILE:-${RUN_DIR}/timing/joint_timeline.jsonl}
record_joint_timing_event(){
  local event=$1 round=${2:-0} step=${3:-0} timestamp_override=${4:-}
  # Security: Pass all values as argv and append under the configured run
  # directory; no shell interpolation is used to construct executable code.
  python3 - "${JOINT_TIMELINE_FILE}" "${event}" "${round}" "${step}" "${timestamp_override}" <<'PY_JOINT_TIMING'
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

path_text, event, round_text, step_text, override = sys.argv[1:]
path = Path(path_text)
path.parent.mkdir(parents=True, exist_ok=True)
timestamp = float(override) if override else time.time()
row = {
    "event": event,
    "round": int(round_text),
    "joint_step": int(step_text),
    "timestamp_unix": timestamp,
    "timestamp_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
    "pid": os.getppid(),
}
with path.open("a", encoding="utf-8") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY_JOINT_TIMING
}

latest_orchestra_attempt_start(){
  local round=$1 step=$2
  # Security: Timeline path and numeric selectors are passed as argv. The parser
  # accepts only matching structured JSON records and fails closed if no finite,
  # positive attempt boundary exists.
  python3 - "${JOINT_TIMELINE_FILE}" "${round}" "${step}" <<'PY_ORCHESTRA_ATTEMPT_START'
import json
import math
import sys
from pathlib import Path

path_text, round_text, step_text = sys.argv[1:]
path = Path(path_text)
if not path.is_file():
    raise SystemExit(f"joint timeline does not exist: {path}")
target_round = int(round_text)
target_step = int(step_text)
matches = []
with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid joint timeline JSON at line {line_number}: {exc}") from exc
        if (
            row.get("event") != "orchestra_start"
            or int(row.get("round", -1)) != target_round
            or int(row.get("joint_step", -1)) != target_step
        ):
            continue
        try:
            timestamp = float(row["timestamp_unix"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid orchestra_start timestamp at line {line_number}") from exc
        if not math.isfinite(timestamp) or timestamp <= 0:
            raise SystemExit(f"non-positive or non-finite orchestra_start timestamp at line {line_number}")
        matches.append(timestamp)
if not matches:
    raise SystemExit(
        f"no orchestra_start boundary for round={target_round} joint_step={target_step}"
    )
print(max(matches))
PY_ORCHESTRA_ATTEMPT_START
}

if [[ "${MAS_LOOP_RUNNING_SCRIPT_SNAPSHOT:-0}" != "1" ]]; then
  record_joint_timing_event launcher_start 0 0 "${MAS_LOOP_LAUNCH_STARTED_AT_UNIX}"
fi

# Bash may read a script incrementally. Editing this shared launcher while a long
# phase is running can therefore make an otherwise successful job fail near EOF
# with a spurious syntax error. Execute an immutable per-run snapshot before any
# model/server work so later source edits only affect the next run.
if [[ "${MAS_LOOP_RUNNING_SCRIPT_SNAPSHOT:-0}" != "1" ]]; then
  SCRIPT_SNAPSHOT_DIR="${RUN_DIR}/script_snapshot"
  SCRIPT_SNAPSHOT="${SCRIPT_SNAPSHOT_DIR}/run_mas_loop_adversarial_grpo.sh"
  mkdir -p "${SCRIPT_SNAPSHOT_DIR}"
  # Resume reuses RUN_DIR. The previous immutable snapshot is mode 0555, so
  # make only that file owner-writable before atomically refreshing it with the
  # current launcher, then restore immutability for the running process.
  if [[ -e "${SCRIPT_SNAPSHOT}" ]]; then
    chmod u+w "${SCRIPT_SNAPSHOT}"
  fi
  cp -f "${BASH_SOURCE[0]}" "${SCRIPT_SNAPSHOT}"
  chmod 0555 "${SCRIPT_SNAPSHOT}"
  sha256sum "${SCRIPT_SNAPSHOT}" > "${SCRIPT_SNAPSHOT}.sha256"
  export MAS_LOOP_RUNNING_SCRIPT_SNAPSHOT=1
  exec bash "${SCRIPT_SNAPSHOT}" "$@"
fi

MODEL_PATH=${MODEL_PATH:-}
ORCHESTRA_MODEL=${ORCHESTRA_MODEL:-${MODEL_PATH}}
EDITOR_MODEL=${EDITOR_MODEL:-${MODEL_PATH}}
SUBAGENT_MODEL=${SUBAGENT_MODEL:-${MODEL_PATH}}
MAS_LOOP_USE_LLM_JUDGE=${MAS_LOOP_USE_LLM_JUDGE:-0}
JUDGE_MODEL=${JUDGE_MODEL:-${SUBAGENT_MODEL}}
REQUIRE_LOCAL_MODEL_PATHS=${REQUIRE_LOCAL_MODEL_PATHS:-1}

ADVERSARIAL_ROUNDS=${ADVERSARIAL_ROUNDS:-1}
RUN_ORCHESTRA=${RUN_ORCHESTRA:-1}
RUN_EDITOR=${RUN_EDITOR:-1}
EDITOR_DATA_FROM_ORCH_LOGS=${EDITOR_DATA_FROM_ORCH_LOGS:-1}
UPDATE_PEER_FROM_HF_CKPT=${UPDATE_PEER_FROM_HF_CKPT:-1}
# Default to strict step-level joint optimization when both trainable roles run.
# One Orchestra optimizer step is immediately followed by the matching Editor
# optimizer step before the next rollout batch is generated.
JOINT_STEP_TRAINING=${JOINT_STEP_TRAINING:-1}

# Pairwise v5: each task samples 32 Orchestra sources; each source receives
# exactly one Refiner reasoning+patch action and one code-derived updated MAS.
ROLLOUT_N=${ROLLOUT_N:-32}
export ROLLOUT_N
export MAS_LOOP_ROLLOUT_N=${MAS_LOOP_ROLLOUT_N:-${ROLLOUT_N}}
export MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE=${MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE:-1}
export MAS_LOOP_EDITOR_CANDIDATE_CONCURRENCY=${MAS_LOOP_EDITOR_CANDIDATE_CONCURRENCY:-128}
EDITOR_GROUP_SIZE=${EDITOR_GROUP_SIZE:-${ROLLOUT_N}}
export MAS_LOOP_EDITOR_GROUP_SIZE=${MAS_LOOP_EDITOR_GROUP_SIZE:-${EDITOR_GROUP_SIZE}}
if (( MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE != 1 )); then
  echo "pairwise v5 requires MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE=1" >&2; exit 2
fi
if (( EDITOR_GROUP_SIZE != ROLLOUT_N )); then
  echo "pairwise v5 requires EDITOR_GROUP_SIZE=${ROLLOUT_N}; got ${EDITOR_GROUP_SIZE}" >&2; exit 2
fi
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
# One Refiner GRPO group per task, with rollout.n source-conditioned members.
EDITOR_TASK_GROUPS_PER_STEP=${TRAIN_BATCH_SIZE}
EDITOR_TRAIN_BATCH_SIZE=${EDITOR_TRAIN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}
if (( EDITOR_TRAIN_BATCH_SIZE != TRAIN_BATCH_SIZE * ROLLOUT_N )); then
  echo "EDITOR_TRAIN_BATCH_SIZE must equal TRAIN_BATCH_SIZE*ROLLOUT_N" >&2; exit 2
fi
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TRAIN_DATA_SHUFFLE=${TRAIN_DATA_SHUFFLE:-True}
TRAIN_DROP_LAST=${TRAIN_DROP_LAST:-True}
PAD_EDITOR_REPLAY_TO_EXPECTED_ROWS=${PAD_EDITOR_REPLAY_TO_EXPECTED_ROWS:-True}
# Keep strict replay cardinality by default. Dataset wrappers with no-padding
# semantics may retain every complete Editor group and train on the exact rows.
ALLOW_INCOMPLETE_EDITOR_REPLAY=${ALLOW_INCOMPLETE_EDITOR_REPLAY:-False}
[[ "${ALLOW_INCOMPLETE_EDITOR_REPLAY}" == "True" || "${ALLOW_INCOMPLETE_EDITOR_REPLAY}" == "False" ]] || { echo "ALLOW_INCOMPLETE_EDITOR_REPLAY must be True or False" >&2; exit 2; }
ACTOR_TEMPERATURE=${ACTOR_TEMPERATURE:-0.7}
export MAS_LOOP_EDITOR_TEMPERATURE=${MAS_LOOP_EDITOR_TEMPERATURE:-0.7}
export MAS_LOOP_ACCEPT_REGRET_WEIGHT=${MAS_LOOP_ACCEPT_REGRET_WEIGHT:-0.0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-15000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN_DEFAULT=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-196608}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
EDITOR_PPO_MAX_TOKEN_LEN_PER_GPU=${EDITOR_PPO_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
EDITOR_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${EDITOR_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}}
[[ "${USE_DYNAMIC_BSZ}" == "True" || "${USE_DYNAMIC_BSZ}" == "False" ]] || { echo "USE_DYNAMIC_BSZ must be True or False" >&2; exit 2; }
[[ "${PPO_MAX_TOKEN_LEN_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "PPO_MAX_TOKEN_LEN_PER_GPU must be a positive integer" >&2; exit 2; }
[[ "${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "LOG_PROB_MAX_TOKEN_LEN_PER_GPU must be a positive integer" >&2; exit 2; }
[[ "${EDITOR_PPO_MAX_TOKEN_LEN_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "EDITOR_PPO_MAX_TOKEN_LEN_PER_GPU must be a positive integer" >&2; exit 2; }
[[ "${EDITOR_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "EDITOR_LOG_PROB_MAX_TOKEN_LEN_PER_GPU must be a positive integer" >&2; exit 2; }
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-${MAX_MODEL_LEN_DEFAULT}}
EXTERNAL_VLLM_MAX_MODEL_LEN=${EXTERNAL_VLLM_MAX_MODEL_LEN:-${MAX_MODEL_LEN_DEFAULT}}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
case "${ALL_GPUS}" in
  "0,1,2,3,4,5,6,7") MAS_LOOP_PORT_NAMESPACE_DEFAULT=80; MAS_LOOP_RAY_PORT_DEFAULT=8025 ;;
  "8,9,10,11,12,13,14,15") MAS_LOOP_PORT_NAMESPACE_DEFAULT=81; MAS_LOOP_RAY_PORT_DEFAULT=8125 ;;
  *) MAS_LOOP_PORT_NAMESPACE_DEFAULT=82; MAS_LOOP_RAY_PORT_DEFAULT=8225 ;;
esac
MAS_LOOP_PORT_NAMESPACE=${MAS_LOOP_PORT_NAMESPACE:-${MAS_LOOP_PORT_NAMESPACE_DEFAULT}}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/mas_loop_ray_${MAS_LOOP_PORT_NAMESPACE}}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-${MAS_LOOP_RAY_PORT_DEFAULT}}
ROLLOUT_TP=${ROLLOUT_TP:-8}
if (( ROLLOUT_TP != 8 || N_GPUS % ROLLOUT_TP != 0 )); then
  echo "[ERROR] MAS-Loop rollout requires TP=8 and selected GPU count divisible by 8, got GPUs=${N_GPUS} TP=${ROLLOUT_TP}" >&2
  exit 2
fi
# External HTTP engines must cover all selected GPUs. Eight-GPU runs remain
# TP=8/DP=1; explicit full-node ablations use TP=8/DP=2.
PEER_TENSOR_PARALLEL_SIZE=${PEER_TENSOR_PARALLEL_SIZE:-${N_GPUS}}
PEER_DATA_PARALLEL_SIZE=${PEER_DATA_PARALLEL_SIZE:-1}
SUB_AGENT_TENSOR_PARALLEL_SIZE=${SUB_AGENT_TENSOR_PARALLEL_SIZE:-${N_GPUS}}
SUB_AGENT_DATA_PARALLEL_SIZE=${SUB_AGENT_DATA_PARALLEL_SIZE:-1}
JUDGE_TENSOR_PARALLEL_SIZE=${JUDGE_TENSOR_PARALLEL_SIZE:-${N_GPUS}}
JUDGE_DATA_PARALLEL_SIZE=${JUDGE_DATA_PARALLEL_SIZE:-1}
validate_external_topology(){
  local role=$1 tp=$2 dp=$3
  if [[ ! "${tp}" =~ ^[1-9][0-9]*$ || ! "${dp}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${role} TP and DP must be positive integers, got TP=${tp} DP=${dp}" >&2
    exit 2
  fi
  if (( tp != 8 || tp * dp != N_GPUS )); then
    echo "[ERROR] ${role} must use TP=8 and TP*DP must cover ${N_GPUS} selected GPUs, got TP=${tp} DP=${dp}" >&2
    exit 2
  fi
}
validate_external_topology editor-peer "${PEER_TENSOR_PARALLEL_SIZE}" "${PEER_DATA_PARALLEL_SIZE}"
validate_external_topology subagent "${SUB_AGENT_TENSOR_PARALLEL_SIZE}" "${SUB_AGENT_DATA_PARALLEL_SIZE}"
validate_external_topology judge "${JUDGE_TENSOR_PARALLEL_SIZE}" "${JUDGE_DATA_PARALLEL_SIZE}"
# A sleeping vLLM releases its allocator-managed weights/KV, but CUDA contexts,
# the FSDP actor shard, and wake-up/remapping scratch space remain resident.
# Keep the colocated actor at 0.90, but default external services to 0.82 so the
# larger 30B sub-agent has several GiB of deterministic transient headroom.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.90}
EXTERNAL_VLLM_GPU_MEM_UTIL=${EXTERNAL_VLLM_GPU_MEM_UTIL:-0.82}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
# Maximum completion budget for each Qwen3-30B-A3B MAS node execution. BCP and
# DeepScaleR wrappers pass this explicitly; keep standalone runs consistent.
export MAS_LOOP_MAX_AGENT_TOKENS=${MAS_LOOP_MAX_AGENT_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
# Keep actor rollout scheduling independent from legacy supervisor values. The
# current loop submits one actor request per sample (64 per step), so 16 caused
# three quarters of requests to wait despite very low KV-cache use.
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
# The installed vLLM V1 backend rejects concurrent partial prefill for the
# colocated actor engine. Keep actor defaults at 1/1; external HTTP services
# retain their independent role-specific concurrent-prefill controls below.
ROLLOUT_MAX_NUM_PARTIAL_PREFILLS=${ROLLOUT_MAX_NUM_PARTIAL_PREFILLS:-1}
ROLLOUT_MAX_LONG_PARTIAL_PREFILLS=${ROLLOUT_MAX_LONG_PARTIAL_PREFILLS:-1}
ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD=${ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD:-1024}
# Optional per-service overrides. User requested Qwen30B sub-agent to use
# a smaller serving context/concurrency than the 8B peer model. If unset, keep
# the historical shared external vLLM settings.
PEER_GPU_MEMORY_UTILIZATION=${PEER_GPU_MEMORY_UTILIZATION:-${EXTERNAL_VLLM_GPU_MEM_UTIL}}
PEER_MAX_NUM_SEQS=${PEER_MAX_NUM_SEQS:-128}
PEER_MAX_NUM_BATCHED_TOKENS=${PEER_MAX_NUM_BATCHED_TOKENS:-32768}
PEER_MAX_MODEL_LEN=${PEER_MAX_MODEL_LEN:-${EXTERNAL_VLLM_MAX_MODEL_LEN:-${ROLLOUT_MAX_MODEL_LEN}}}
SUB_AGENT_GPU_MEMORY_UTILIZATION=${SUB_AGENT_GPU_MEMORY_UTILIZATION:-${EXTERNAL_VLLM_GPU_MEM_UTIL}}
SUB_AGENT_MAX_NUM_SEQS=${SUB_AGENT_MAX_NUM_SEQS:-128}
SUB_AGENT_MAX_NUM_BATCHED_TOKENS=${SUB_AGENT_MAX_NUM_BATCHED_TOKENS:-32768}
SUB_AGENT_MAX_MODEL_LEN=${SUB_AGENT_MAX_MODEL_LEN:-${EXTERNAL_VLLM_MAX_MODEL_LEN:-${ROLLOUT_MAX_MODEL_LEN}}}
# BCP correctness is graded by an independent Qwen3-30B vLLM phase. Keep its
# completion ceiling independent from actor/sub-agent output lengths; tune
# utilization/admission rather than shortening max output tokens.
JUDGE_GPU_MEMORY_UTILIZATION=${JUDGE_GPU_MEMORY_UTILIZATION:-0.78}
JUDGE_MAX_NUM_SEQS=${JUDGE_MAX_NUM_SEQS:-128}
JUDGE_MAX_NUM_BATCHED_TOKENS=${JUDGE_MAX_NUM_BATCHED_TOKENS:-49152}
JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-${EXTERNAL_VLLM_MAX_MODEL_LEN:-${ROLLOUT_MAX_MODEL_LEN}}}
JUDGE_MAX_NUM_PARTIAL_PREFILLS=${JUDGE_MAX_NUM_PARTIAL_PREFILLS:-1}
JUDGE_MAX_LONG_PARTIAL_PREFILLS=${JUDGE_MAX_LONG_PARTIAL_PREFILLS:-1}
JUDGE_LONG_PREFILL_TOKEN_THRESHOLD=${JUDGE_LONG_PREFILL_TOKEN_THRESHOLD:-1024}
export MAS_LOOP_JUDGE_MAX_TOKENS=${MAS_LOOP_JUDGE_MAX_TOKENS:-4096}
# Keep xgrammar guided decoding opt-in for Judge. The validator/retry path is
# still strict and fail-closed, without the high-concurrency scheduler stall.
export MAS_LOOP_JUDGE_GUIDED_JSON=${MAS_LOOP_JUDGE_GUIDED_JSON:-0}
# External Editor/Refiner uses the same fail-closed parser/retry contract as the
# Judge. Keep xgrammar opt-in because high-concurrency guided decoding can make
# negligible token progress while all requests remain marked running.
export MAS_LOOP_EDITOR_GUIDED_JSON=${MAS_LOOP_EDITOR_GUIDED_JSON:-0}
# Model-authored structures are XML-first in v6. Internally they are parsed to
# canonical dictionaries; legacy JSON remains accepted by the parser only.
export MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL=${MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL:-xml}
export MAS_LOOP_ACTOR_GUIDED_JSON=0
export MAS_LOOP_INVALID_ACTION_PENALTY=${MAS_LOOP_INVALID_ACTION_PENALTY:-0.5}
export MAS_LOOP_ORCH_ADV_PENALTY=${MAS_LOOP_ORCH_ADV_PENALTY:-0.3}
printf '[MAS-Loop] reward parameters eta=%s lambda_adv=%s\n' "${MAS_LOOP_INVALID_ACTION_PENALTY}" "${MAS_LOOP_ORCH_ADV_PENALTY}" | tee -a "${LOG_DIR}/vllm_launch_config.log" 2>/dev/null || true
# Repair only genuinely missing persisted Judge evidence before the strict
# phase validator runs. Accepted/no-op updated candidates reuse source evidence
# and are excluded by the same contract as validate_bcp_artifacts.py.
MAS_LOOP_AUTO_BACKFILL_JUDGES=${MAS_LOOP_AUTO_BACKFILL_JUDGES:-1}
MAS_LOOP_JUDGE_BACKFILL_MAX_WORKERS=${MAS_LOOP_JUDGE_BACKFILL_MAX_WORKERS:-128}
# Concurrent chunked-prefill lets multiple long MAS prompts share each vLLM
# scheduler iteration. This changes scheduling only: every source/updated MAS
# still performs a fresh stochastic model sample.
SUB_AGENT_MAX_NUM_PARTIAL_PREFILLS=${SUB_AGENT_MAX_NUM_PARTIAL_PREFILLS:-4}
SUB_AGENT_MAX_LONG_PARTIAL_PREFILLS=${SUB_AGENT_MAX_LONG_PARTIAL_PREFILLS:-2}
SUB_AGENT_LONG_PREFILL_TOKEN_THRESHOLD=${SUB_AGENT_LONG_PREFILL_TOKEN_THRESHOLD:-1024}
PEER_MAX_NUM_PARTIAL_PREFILLS=${PEER_MAX_NUM_PARTIAL_PREFILLS:-4}
PEER_MAX_LONG_PARTIAL_PREFILLS=${PEER_MAX_LONG_PARTIAL_PREFILLS:-2}
PEER_LONG_PREFILL_TOKEN_THRESHOLD=${PEER_LONG_PREFILL_TOKEN_THRESHOLD:-1024}
# Qwen3-30B-A3B is MoE. EP can improve all-8-GPU serving throughput, but is
# opt-in until a phase-boundary benchmark validates this platform/backend.
SUB_AGENT_ENABLE_EXPERT_PARALLEL=${SUB_AGENT_ENABLE_EXPERT_PARALLEL:-0}
EXTERNAL_VLLM_ENFORCE_EAGER=${EXTERNAL_VLLM_ENFORCE_EAGER:-0}
EXTERNAL_VLLM_ENFORCE_EAGER_FLAG=""
if [[ "${EXTERNAL_VLLM_ENFORCE_EAGER}" == "1" ]]; then EXTERNAL_VLLM_ENFORCE_EAGER_FLAG="--enforce-eager"; fi
EXTERNAL_VLLM_MAX_MODEL_LEN_FLAG=""
if [[ -n "${EXTERNAL_VLLM_MAX_MODEL_LEN:-}" ]]; then EXTERNAL_VLLM_MAX_MODEL_LEN_FLAG="--max-model-len ${EXTERNAL_VLLM_MAX_MODEL_LEN}"; fi
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-0}
ROLLOUT_ENFORCE_EAGER_BOOL=False
if [[ "${ROLLOUT_ENFORCE_EAGER}" == "1" ]]; then ROLLOUT_ENFORCE_EAGER_BOOL=True; fi
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-1}
ROLLOUT_ENABLE_SLEEP_MODE_BOOL=False
if [[ "${ROLLOUT_ENABLE_SLEEP_MODE}" == "1" ]]; then ROLLOUT_ENABLE_SLEEP_MODE_BOOL=True; fi
# Enable vLLM periodic scheduler/KV statistics for tuning runs. Keep the
# production default quiet because full 40k-sample runs would otherwise create
# very large logs; the one-step benchmark explicitly sets this to 0.
ROLLOUT_DISABLE_LOG_STATS=${ROLLOUT_DISABLE_LOG_STATS:-1}
ROLLOUT_DISABLE_LOG_STATS_BOOL=True
if [[ "${ROLLOUT_DISABLE_LOG_STATS}" == "0" ]]; then ROLLOUT_DISABLE_LOG_STATS_BOOL=False; fi
TEST_FREQ=${TEST_FREQ:--1}
SAVE_FREQ=${SAVE_FREQ:-250}
# In strict joint mode this is the retained evaluation-artifact count. The
# default means ten evenly spaced intermediate checkpoints plus the final one.
# Each newest checkpoint stays fully resumable until the next joint barrier;
# selected older checkpoints are compacted to HF inference artifacts.
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-11}
# null means VERL keeps every scheduled step checkpoint.
MAX_ACTOR_CKPTS_TO_KEEP=null
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-128}
# Full-parameter training remains enabled (lora_rank=0). These switches only
# move FSDP shards/Adam states to CPU while the colocated vLLM owns the GPUs.
# Without them, the first optimizer step leaves enough resident state that
# vLLM CuMem cannot remap its KV cache at the requested high utilization.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
# Role-specific Editor overrides allow optimizer experiments to remain isolated
# from the Orchestra phase. Defaults preserve the historical shared settings.
EDITOR_ACTOR_PARAM_OFFLOAD=${EDITOR_ACTOR_PARAM_OFFLOAD:-${ACTOR_PARAM_OFFLOAD}}
EDITOR_ACTOR_OPTIMIZER_OFFLOAD=${EDITOR_ACTOR_OPTIMIZER_OFFLOAD:-${ACTOR_OPTIMIZER_OFFLOAD}}
EDITOR_REF_PARAM_OFFLOAD=${EDITOR_REF_PARAM_OFFLOAD:-${REF_PARAM_OFFLOAD}}
export MAS_LOOP_MAX_CONCURRENCY=${MAS_LOOP_MAX_CONCURRENCY:-128}
# llm_server.py otherwise silently defaults the colocated Orchestra actor to
# only 1024 admitted requests even when max_num_seqs and the dispatcher are
# configured for a full rollout group (e.g. 4096). Keep one explicit admission
# ceiling across the actor and local peer/sub-agent clients.
export MAS_LOOP_VLLM_MAX_ACTIVE=${MAS_LOOP_VLLM_MAX_ACTIVE:-${MAS_LOOP_MAX_CONCURRENCY}}
export MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE=${MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE:-${MAS_LOOP_VLLM_MAX_ACTIVE}}
export MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE=${MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE:-${MAS_LOOP_VLLM_MAX_ACTIVE}}
export MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE=${MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE:-128}
export MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE=${MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE:-128}
export MAS_LOOP_VLLM_SLEEP=${MAS_LOOP_VLLM_SLEEP:-1}
# Reuse immutable external Subagent/Judge engines across joint steps. This is
# safe only with sleep mode because Editor/Orchestra FSDP must reclaim their
# allocator-managed weights and KV cache between inference phases.
export MAS_LOOP_PERSIST_EXTERNAL_VLLM=${MAS_LOOP_PERSIST_EXTERNAL_VLLM:-${MAS_LOOP_VLLM_SLEEP}}
if [[ "${MAS_LOOP_PERSIST_EXTERNAL_VLLM}" == "1" && "${MAS_LOOP_VLLM_SLEEP}" != "1" ]]; then
  echo "[ERROR] MAS_LOOP_PERSIST_EXTERNAL_VLLM=1 requires MAS_LOOP_VLLM_SLEEP=1" >&2
  exit 2
fi
export MAS_LOOP_EXCLUSIVE_VLLM=${MAS_LOOP_EXCLUSIVE_VLLM:-1}
export MAS_LOOP_ACTOR_OWNER=${MAS_LOOP_ACTOR_OWNER:-verl-actor-rollout}
export MAS_LOOP_VLLM_SLEEP_LEVEL=${MAS_LOOP_VLLM_SLEEP_LEVEL:-${VLLM_SLEEP_LEVEL}}
# Optimized control plane: request admission/refcounts live in RAM behind a
# run-specific Unix socket. Durable experiment artifacts are copied separately.
_control_tag=$(printf '%s' "${RUN_DIR}" | cksum | awk '{print $1}')
export MAS_LOOP_VLLM_CONTROL_DIR=${MAS_LOOP_VLLM_CONTROL_DIR:-/tmp/mas_loop_control_${USER:-user}_${_control_tag}}
export MAS_LOOP_RAY_WORKDIR_BUNDLE_ENABLE=${MAS_LOOP_RAY_WORKDIR_BUNDLE_ENABLE:-0}
RAY_RUNTIME_ENV_ARGS=()
if [[ "${MAS_LOOP_RAY_WORKDIR_BUNDLE_ENABLE}" == "1" ]]; then
  command -v rsync >/dev/null || { echo "[ERROR] rsync is required for isolated Ray working-dir bundle" >&2; exit 2; }
  MAS_LOOP_RAY_WORKDIR_BUNDLE=${MAS_LOOP_RAY_WORKDIR_BUNDLE:-${MAS_LOOP_VLLM_CONTROL_DIR}/ray_working_dir}
  runtime_deps_dir=${MAS_LOOP_RUNTIME_DEPS_DIR:-${VERL_ROOT}/.mas_loop_runtime_deps}
  bundle_staging="${MAS_LOOP_RAY_WORKDIR_BUNDLE}.tmp.${$}"
  mkdir -p "${bundle_staging}"
  rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyc.*' --exclude='sed*' "${VERL_ROOT}/verl/" "${bundle_staging}/verl/"
  rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyc.*' --exclude='sed*' "${VERL_ROOT}/mas_loop_rl/" "${bundle_staging}/mas_loop_rl/"
  for package in transfer_queue tensordict pyvers; do
    [[ -f "${runtime_deps_dir}/${package}/__init__.py" ]] || { echo "[ERROR] missing Ray runtime package: ${runtime_deps_dir}/${package}" >&2; exit 2; }
    rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyc.*' "${runtime_deps_dir}/${package}/" "${bundle_staging}/${package}/"
  done
  printf '%s\n' "${VERL_ROOT}" > "${bundle_staging}/SOURCE_ROOT"
  if [[ -e "${MAS_LOOP_RAY_WORKDIR_BUNDLE}" ]]; then
    echo "[ERROR] isolated Ray working-dir bundle already exists: ${MAS_LOOP_RAY_WORKDIR_BUNDLE}" >&2
    exit 2
  fi
  mv "${bundle_staging}" "${MAS_LOOP_RAY_WORKDIR_BUNDLE}"
  RAY_RUNTIME_ENV_ARGS+=("+ray_kwargs.ray_init.runtime_env.working_dir=${MAS_LOOP_RAY_WORKDIR_BUNDLE}")
  echo "[MAS-Loop] isolated Ray working-dir bundle: ${MAS_LOOP_RAY_WORKDIR_BUNDLE}"
fi
export MAS_LOOP_EXTERNAL_VLLM_LOG_DIR=${MAS_LOOP_EXTERNAL_VLLM_LOG_DIR:-${MAS_LOOP_VLLM_CONTROL_DIR}/vllm_logs}
export MAS_LOOP_VLLM_CONTROL_SOCKET=${MAS_LOOP_VLLM_CONTROL_SOCKET:-${MAS_LOOP_VLLM_CONTROL_DIR}/coordinator.sock}
export MAS_LOOP_VLLM_SLEEP_LOCK=${MAS_LOOP_VLLM_SLEEP_LOCK:-${MAS_LOOP_VLLM_CONTROL_DIR}/fallback.lock}
export MAS_LOOP_VLLM_SLEEP_STATE=${MAS_LOOP_VLLM_SLEEP_STATE:-${MAS_LOOP_VLLM_CONTROL_DIR}/state.json}
export MAS_LOOP_VLLM_SLEEP_EVENTS=${MAS_LOOP_VLLM_SLEEP_EVENTS:-${MAS_LOOP_VLLM_CONTROL_DIR}/events.jsonl}
export MAS_LOOP_VLLM_SLEEP_EVENTS_DURABLE=${MAS_LOOP_VLLM_SLEEP_EVENTS_DURABLE:-${RUN_DIR}/control_plane/vllm_sleep/events.jsonl}
export MAS_LOOP_PHASE_LEASE=${MAS_LOOP_PHASE_LEASE:-1}
export MAS_LOOP_PHASE_LEASE_GRACE_SEC=${MAS_LOOP_PHASE_LEASE_GRACE_SEC:-0.75}
# Final-owner safety dwell before /sleep. Even with mandatory TP8/DP1, the
# frontend can report completion before the engine has finished its final
# global-idle dummy wave; an immediate sleep can unmap allocations still in use.
export MAS_LOOP_VLLM_DP_QUIESCE_SEC=${MAS_LOOP_VLLM_DP_QUIESCE_SEC:-0}
# Bound CPU-heavy safetensor deserialization for each external TP8/DP1 vLLM.
# Without this, independent eight-GPU services can expand native thread pools
# across the whole host and make startup slower through oversubscription.
export MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS=${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS:-4}
# Bound native CPU pools in Ray AgentLoop/RewardLoop workers. This limits
# process/thread replication only; every trajectory and reward call still executes.
export MAS_LOOP_RAY_WORKER_CPU_THREADS=${MAS_LOOP_RAY_WORKER_CPU_THREADS:-1}
# Reuse one inference engine only when Subagent and Judge point at the same
# immutable checkpoint. Requests and outputs are never cached or deduplicated.
export MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM=${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM:-0}
export MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN=${MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN:-0}
export MAS_LOOP_EXTERNAL_VLLM_CPUSET=${MAS_LOOP_EXTERNAL_VLLM_CPUSET:-}
# Give simultaneous GPU domains disjoint bases for vLLM's internal TCPStore
# allocation. Public HTTP ports alone do not isolate DP process-group ports.
export MAS_LOOP_EXTERNAL_VLLM_PORT_BASE=${MAS_LOOP_EXTERNAL_VLLM_PORT_BASE:-}
[[ "${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS must be a positive integer" >&2; exit 2; }
[[ "${MAS_LOOP_RAY_WORKER_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] MAS_LOOP_RAY_WORKER_CPU_THREADS must be a positive integer" >&2; exit 2; }
[[ "${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM}" == "0" || "${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM}" == "1" ]] || { echo "[ERROR] MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM must be 0 or 1" >&2; exit 2; }
[[ "${MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN}" =~ ^[0-9]+$ ]] || { echo "[ERROR] MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN must be a non-negative integer" >&2; exit 2; }
if [[ -n "${MAS_LOOP_EXTERNAL_VLLM_CPUSET}" && ! "${MAS_LOOP_EXTERNAL_VLLM_CPUSET}" =~ ^[0-9,-]+$ ]]; then
  echo "[ERROR] MAS_LOOP_EXTERNAL_VLLM_CPUSET must contain only CPU ids, commas, and ranges" >&2
  exit 2
fi
if [[ -n "${MAS_LOOP_EXTERNAL_VLLM_PORT_BASE}" ]]; then
  [[ "${MAS_LOOP_EXTERNAL_VLLM_PORT_BASE}" =~ ^[0-9]+$ ]] && (( MAS_LOOP_EXTERNAL_VLLM_PORT_BASE >= 1024 && MAS_LOOP_EXTERNAL_VLLM_PORT_BASE <= 65000 )) || {
    echo "[ERROR] MAS_LOOP_EXTERNAL_VLLM_PORT_BASE must be an integer in [1024, 65000]" >&2
    exit 2
  }
fi
export MAS_LOOP_DISPATCH_WORKERS_PER_PROCESS=${MAS_LOOP_DISPATCH_WORKERS_PER_PROCESS:-128}
# Run dependency-ready MAS agents concurrently in a separate bounded pool.
# Each future still performs its own planner/retrieval/generation calls; this
# is execution overlap only, never result caching or request deduplication.
export MAS_LOOP_DEPENDENCY_PROMPT_MODE=${MAS_LOOP_DEPENDENCY_PROMPT_MODE:-legacy_cumulative}
export MAS_LOOP_DAG_AGENT_PARALLEL=${MAS_LOOP_DAG_AGENT_PARALLEL:-0}
export MAS_LOOP_DAG_AGENT_WORKERS_PER_PROCESS=${MAS_LOOP_DAG_AGENT_WORKERS_PER_PROCESS:-64}
export MAS_LOOP_BCP_SEARCH_ROUNDS=${MAS_LOOP_BCP_SEARCH_ROUNDS:-5}
export MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS=${MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS:-256}
# AgentLoop workers are CPU/RAM-heavy Python actors. Four processes still
# dispatch the same asynchronous HTTP workload and preserve TP=8, while avoiding
# the duplicated per-process state that pushed the host above Ray's OOM limit.
export MAS_LOOP_AGENT_LOOP_WORKER_PROCESSES=${MAS_LOOP_AGENT_LOOP_WORKER_PROCESSES:-4}
export MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER=${MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER:-128}
export MAS_LOOP_AGENT_LOOP_HEALTH_TIMEOUT_SEC=${MAS_LOOP_AGENT_LOOP_HEALTH_TIMEOUT_SEC:-30}
export MAS_LOOP_PHASE_SOCKET_BACKLOG=${MAS_LOOP_PHASE_SOCKET_BACKLOG:-16384}
export MAS_LOOP_DISPATCH_METRICS_DIR=${MAS_LOOP_DISPATCH_METRICS_DIR:-${MAS_LOOP_VLLM_CONTROL_DIR}/dispatch_metrics}
export MAS_LOOP_DISPATCH_METRICS_DURABLE_DIR=${MAS_LOOP_DISPATCH_METRICS_DURABLE_DIR:-${RUN_DIR}/control_plane/dispatch_metrics}
export MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC=${MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC:-30}
export MAS_LOOP_RAY_LB_METRICS_DIR=${MAS_LOOP_RAY_LB_METRICS_DIR:-${MAS_LOOP_VLLM_CONTROL_DIR}/ray_lb_metrics}
export MAS_LOOP_RAY_LB_METRICS_DURABLE_DIR=${MAS_LOOP_RAY_LB_METRICS_DURABLE_DIR:-${RUN_DIR}/control_plane/ray_lb_metrics}
export MAS_LOOP_RAY_LB_METRICS_INTERVAL_SEC=${MAS_LOOP_RAY_LB_METRICS_INTERVAL_SEC:-5}
export MAS_LOOP_CONTROL_PLANE_PERSIST_INTERVAL_SEC=${MAS_LOOP_CONTROL_PLANE_PERSIST_INTERVAL_SEC:-30}
export MAS_LOOP_HTTP_TIMEOUT=${MAS_LOOP_HTTP_TIMEOUT:-120}
# Process-wide urllib3 keep-alive pools prevent thousands of one-shot TCP
# connections from overflowing the local vLLM/Uvicorn accept queues. Role
# limits are per Ray worker process and can be tuned independently.
export MAS_LOOP_HTTP_TRANSPORT=${MAS_LOOP_HTTP_TRANSPORT:-urllib3}
export MAS_LOOP_HTTP_NUM_POOLS=${MAS_LOOP_HTTP_NUM_POOLS:-8}
export MAS_LOOP_HTTP_POOL_MAXSIZE=${MAS_LOOP_HTTP_POOL_MAXSIZE:-128}
export MAS_LOOP_ORCHESTRA_HTTP_POOL_MAXSIZE=${MAS_LOOP_ORCHESTRA_HTTP_POOL_MAXSIZE:-${MAS_LOOP_HTTP_POOL_MAXSIZE}}
export MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE=${MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE:-${MAS_LOOP_HTTP_POOL_MAXSIZE}}
export MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE=${MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE:-${MAS_LOOP_HTTP_POOL_MAXSIZE}}
export MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE=${MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE:-128}
export MAS_LOOP_HTTP_CONNECT_TIMEOUT_SEC=${MAS_LOOP_HTTP_CONNECT_TIMEOUT_SEC:-30}
export MAS_LOOP_HTTP_POOL_TIMEOUT_SEC=${MAS_LOOP_HTTP_POOL_TIMEOUT_SEC:-${MAS_LOOP_HTTP_TIMEOUT}}
# Large Editor/sub-agent bursts previously exceeded Uvicorn's default 2048
# accept backlog and failed before reaching vLLM. The MAS-Loop-local server
# wrapper passes this value directly to Uvicorn without patching system vLLM.
export MAS_LOOP_VLLM_HTTP_BACKLOG=${MAS_LOOP_VLLM_HTTP_BACKLOG:-16384}
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-60}
export MAS_LOOP_DISABLE_THINKING=${MAS_LOOP_DISABLE_THINKING:-1}
export MAS_LOOP_EDITOR_MAX_TOKENS=${MAS_LOOP_EDITOR_MAX_TOKENS:-768}
# Disable Qwen3 thinking in Verl actor prompt tokenization too. Helper OpenAI calls
# already pass chat_template_kwargs; without this, actor rollouts still start from
# a thinking preamble and can spend thousands of tokens before emitting JSON.
DATA_ENABLE_THINKING=${DATA_ENABLE_THINKING:-False}
MAS_LOOP_REWARD_MANAGE_SLEEP=${MAS_LOOP_REWARD_MANAGE_SLEEP:-true}
export MAS_LOOP_VLLM_LOCK_TTL_SEC=${MAS_LOOP_VLLM_LOCK_TTL_SEC:-0}
mkdir -p "${MAS_LOOP_VLLM_CONTROL_DIR}" "${MAS_LOOP_EXTERNAL_VLLM_LOG_DIR}" "${MAS_LOOP_DISPATCH_METRICS_DIR}" "${MAS_LOOP_DISPATCH_METRICS_DURABLE_DIR}" \
  "${MAS_LOOP_RAY_LB_METRICS_DIR}" "${MAS_LOOP_RAY_LB_METRICS_DURABLE_DIR}" "$(dirname "${MAS_LOOP_VLLM_SLEEP_EVENTS_DURABLE}")"
# Fresh run-specific local state; never share a coordinator with another project.
rm -f "${MAS_LOOP_VLLM_CONTROL_SOCKET}" "${MAS_LOOP_VLLM_SLEEP_STATE}" "${MAS_LOOP_VLLM_SLEEP_EVENTS}"
export MAS_LOOP_PAIR_DIR=${MAS_LOOP_PAIR_DIR:-${RUN_DIR}/rollouts/pairs}
export MAS_LOOP_BM25_DOC_PREVIEW_CHARS=${MAS_LOOP_BM25_DOC_PREVIEW_CHARS:-1200}

ORCHESTRA_PORT=${ORCHESTRA_PORT:-${MAS_LOOP_PORT_NAMESPACE}10}
EDITOR_PORT=${EDITOR_PORT:-${MAS_LOOP_PORT_NAMESPACE}11}
SUBAGENT_PORT=${SUBAGENT_PORT:-${MAS_LOOP_PORT_NAMESPACE}12}
JUDGE_PORT=${JUDGE_PORT:-${MAS_LOOP_PORT_NAMESPACE}13}
if [[ "${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM}" == "1" ]]; then
  [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" ]] || { echo "[ERROR] shared Subagent/Judge vLLM requires MAS_LOOP_USE_LLM_JUDGE=1" >&2; exit 2; }
  [[ "$(readlink -f "${SUBAGENT_MODEL}")" == "$(readlink -f "${JUDGE_MODEL}")" ]] || { echo "[ERROR] shared Subagent/Judge vLLM requires identical model paths" >&2; exit 2; }
  (( SUB_AGENT_TENSOR_PARALLEL_SIZE == JUDGE_TENSOR_PARALLEL_SIZE && SUB_AGENT_DATA_PARALLEL_SIZE == JUDGE_DATA_PARALLEL_SIZE )) || { echo "[ERROR] shared Subagent/Judge vLLM requires identical TP/DP topology" >&2; exit 2; }
  (( JUDGE_MAX_MODEL_LEN <= SUB_AGENT_MAX_MODEL_LEN )) || { echo "[ERROR] shared Judge max model length exceeds Subagent engine capacity" >&2; exit 2; }
  JUDGE_PORT="${SUBAGENT_PORT}"
fi

if [[ -z "${ORCHESTRA_MODEL}" || -z "${EDITOR_MODEL}" || -z "${SUBAGENT_MODEL}" ]]; then
  echo "Please set MODEL_PATH or ORCHESTRA_MODEL, EDITOR_MODEL and SUBAGENT_MODEL to local model/checkpoint paths." >&2
  exit 2
fi
if [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" && -z "${JUDGE_MODEL}" ]]; then
  echo "MAS_LOOP_USE_LLM_JUDGE=1 requires JUDGE_MODEL." >&2
  exit 2
fi
if [[ "${REQUIRE_LOCAL_MODEL_PATHS}" == "1" ]]; then
  model_paths=("${ORCHESTRA_MODEL}" "${EDITOR_MODEL}" "${SUBAGENT_MODEL}")
  if [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" ]]; then model_paths+=("${JUDGE_MODEL}"); fi
  for p in "${model_paths[@]}"; do
    [[ -e "$p" ]] || { echo "Model path does not exist: $p . Set REQUIRE_LOCAL_MODEL_PATHS=0 only if it is already cached/resolvable locally." >&2; exit 2; }
  done
fi

MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAS_LOOP_DATASET_NAME=${MAS_LOOP_DATASET_NAME:-}
if [[ -z "${MAS_LOOP_DATASET_NAME}" ]]; then
  case "${TASK_JSONL}" in
    *BrowseComp-Plus*|*browsecomp*|*bcp*) MAS_LOOP_DATASET_NAME=bcp ;;
    *DeepScaleR*|*deepscaler*) MAS_LOOP_DATASET_NAME=deepscaler ;;
    *) MAS_LOOP_DATASET_NAME=unknown ;;
  esac
fi
MAS_LOOP_CKPT_PREFIX=${MAS_LOOP_CKPT_PREFIX:-mas-loop}
DATA_PAD_TO_MULTIPLE=${DATA_PAD_TO_MULTIPLE:-${TRAIN_BATCH_SIZE}}
DATA_VAL_RATIO=${DATA_VAL_RATIO:-0.05}
# Dataset-specific wrappers already prepare role parquet files with their own
# retrieval/math settings. Avoid parsing multi-GB BCP JSONL twice.
if [[ "${SKIP_DATA_PREP:-0}" != "1" ]]; then
  python3 -m mas_loop_rl.prepare_data --input "${TASK_JSONL}" --output-dir "${DATA_DIR}" --role both --seed "${DATA_PREP_SEED:-0}" --val-ratio "${DATA_VAL_RATIO}" --max-rows "${MAX_TRAIN_ROWS}" --pad-train-to-multiple "${DATA_PAD_TO_MULTIPLE}"
else
  [[ -s "${DATA_DIR}/orchestra_train.parquet" && -s "${DATA_DIR}/editor_train.parquet" ]] || {
    echo "SKIP_DATA_PREP=1 but prepared parquet files are missing under ${DATA_DIR}" >&2
    exit 2
  }
fi

# Keep micro-batch settings valid for 8-way FSDP even when users downscale train_batch_size.
# Verl shards the global batch across GPUs for log-prob/actor forward, so per-GPU
# batch must be divisible by the corresponding micro batch.
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by N_GPUS=${N_GPUS}" >&2
  exit 2
fi
PER_GPU_BATCH=$((TRAIN_BATCH_SIZE / N_GPUS))
if (( LOG_PROB_MICRO_BATCH_SIZE_PER_GPU > PER_GPU_BATCH )); then
  echo "[WARN] LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} > per_gpu_batch=${PER_GPU_BATCH}; lowering to ${PER_GPU_BATCH}" >&2
  LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${PER_GPU_BATCH}
fi
if (( PER_GPU_BATCH % LOG_PROB_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  echo "[WARN] per_gpu_batch=${PER_GPU_BATCH} is not divisible by LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}; using 1" >&2
  LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
fi
if (( PPO_MICRO_BATCH_SIZE_PER_GPU > PER_GPU_BATCH )); then
  echo "[WARN] PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU} > per_gpu_batch=${PER_GPU_BATCH}; lowering to ${PER_GPU_BATCH}" >&2
  PPO_MICRO_BATCH_SIZE_PER_GPU=${PER_GPU_BATCH}
fi
if (( PER_GPU_BATCH % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  echo "[WARN] per_gpu_batch=${PER_GPU_BATCH} is not divisible by PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}; using 1" >&2
  PPO_MICRO_BATCH_SIZE_PER_GPU=1
fi
if (( PPO_MINI_BATCH_SIZE > TRAIN_BATCH_SIZE )); then
  echo "[WARN] PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} > TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; lowering to train batch" >&2
  PPO_MINI_BATCH_SIZE=${TRAIN_BATCH_SIZE}
fi

PIDS=()
COORDINATOR_PID=""
EDITOR_SERVER_PID=""
SUBAGENT_SERVER_PID=""
JUDGE_SERVER_PID=""
EDITOR_SERVER_MODEL=""
SUBAGENT_SERVER_MODEL=""
JUDGE_SERVER_MODEL=""
SERVICE_WATCHDOG_PID=""
SERVICE_WATCHDOG_FAILURE_FILE=""
CONTROL_PLANE_PERSISTER_PID=""

terminate_pid_bounded(){
  local pid=${1:-} label=${2:-process} timeout_sec=${3:-15}
  [[ -n "${pid}" ]] || return 0
  kill -0 "${pid}" 2>/dev/null || return 0
  kill -TERM "${pid}" 2>/dev/null || true
  local deadline=$((SECONDS + timeout_sec))
  while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.25; done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[MAS-Loop] ${label} pid=${pid} ignored TERM for ${timeout_sec}s; sending KILL" >&2
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

remove_registered_pid(){
  local target=${1:-} p
  local -a kept=()
  [[ -n "${target}" ]] || return 0
  for p in "${PIDS[@]}"; do
    [[ "${p}" == "${target}" ]] || kept+=("${p}")
  done
  if ((${#kept[@]})); then
    PIDS=("${kept[@]}")
  else
    PIDS=()
  fi
}

stop_vllm_role(){
  local name=$1 pid=""
  case "${name}" in
    editor-peer) pid="${EDITOR_SERVER_PID}" ;;
    subagent) pid="${SUBAGENT_SERVER_PID}" ;;
    judge) pid="${JUDGE_SERVER_PID}" ;;
    *) echo "[ERROR] unknown external vLLM role: ${name}" >&2; return 2 ;;
  esac
  if [[ -n "${pid}" ]]; then
    terminate_pid_bounded "${pid}" "${name}-vllm" 20
    remove_registered_pid "${pid}"
  fi
  if [[ "${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM}" == "1" && ( "${name}" == "subagent" || "${name}" == "judge" ) ]]; then
    SUBAGENT_SERVER_PID=""; SUBAGENT_SERVER_MODEL=""
    JUDGE_SERVER_PID=""; JUDGE_SERVER_MODEL=""
  else
    case "${name}" in
      editor-peer) EDITOR_SERVER_PID=""; EDITOR_SERVER_MODEL="" ;;
      subagent) SUBAGENT_SERVER_PID=""; SUBAGENT_SERVER_MODEL="" ;;
      judge) JUDGE_SERVER_PID=""; JUDGE_SERVER_MODEL="" ;;
    esac
  fi
}

cleanup(){
  if [[ -n "${SERVICE_WATCHDOG_PID}" ]]; then
    terminate_pid_bounded "${SERVICE_WATCHDOG_PID}" service-watchdog 5
    SERVICE_WATCHDOG_PID=""
  fi
  # Full cleanup is reserved for launcher exit. Between joint steps the
  # immutable Subagent/Judge services stay resident in sleep mode; only the
  # Editor peer is recycled when its checkpoint path changes.
  stop_vllm_role editor-peer
  stop_vllm_role subagent
  stop_vllm_role judge
  PIDS=()
}
persist_control_plane(){
  # The coordinator and worker metrics live on local ephemeral storage for low
  # latency. Snapshot them periodically into this run's ckpt tree so a node or
  # process failure does not lose the control-plane history. The persister has
  # its own PID and is intentionally not added to PIDS (external vLLM servers).
  mkdir -p "$(dirname "${MAS_LOOP_VLLM_SLEEP_EVENTS_DURABLE}")"
  if [[ -f "${MAS_LOOP_VLLM_SLEEP_EVENTS}" ]]; then cp -f "${MAS_LOOP_VLLM_SLEEP_EVENTS}" "${MAS_LOOP_VLLM_SLEEP_EVENTS_DURABLE}" 2>/dev/null || true; fi
  if [[ -f "${MAS_LOOP_VLLM_SLEEP_STATE}" ]]; then cp -f "${MAS_LOOP_VLLM_SLEEP_STATE}" "$(dirname "${MAS_LOOP_VLLM_SLEEP_EVENTS_DURABLE}")/state.json" 2>/dev/null || true; fi
  # Keep live vLLM output on local ephemeral storage: the shared checkpoint
  # filesystem can make the Python process exit with status 120 during startup.
  # Security: role names are fixed by the start_vllm_sleeping allowlist below.
  mkdir -p "${LOG_DIR}"
  if compgen -G "${MAS_LOOP_EXTERNAL_VLLM_LOG_DIR}/*_vllm.log" >/dev/null; then
    cp -f "${MAS_LOOP_EXTERNAL_VLLM_LOG_DIR}"/*_vllm.log "${LOG_DIR}/" 2>/dev/null || true
  fi
  mkdir -p "${MAS_LOOP_DISPATCH_METRICS_DURABLE_DIR}"
  if compgen -G "${MAS_LOOP_DISPATCH_METRICS_DIR}/dispatch-*.json" >/dev/null; then
    cp -f "${MAS_LOOP_DISPATCH_METRICS_DIR}"/dispatch-*.json "${MAS_LOOP_DISPATCH_METRICS_DURABLE_DIR}/" 2>/dev/null || true
  fi
  mkdir -p "${MAS_LOOP_RAY_LB_METRICS_DURABLE_DIR}"
  if compgen -G "${MAS_LOOP_RAY_LB_METRICS_DIR}/ray-lb-*.json" >/dev/null; then
    cp -f "${MAS_LOOP_RAY_LB_METRICS_DIR}"/ray-lb-*.json "${MAS_LOOP_RAY_LB_METRICS_DURABLE_DIR}/" 2>/dev/null || true
  fi
}
start_control_plane_persister(){
  local interval=${MAS_LOOP_CONTROL_PLANE_PERSIST_INTERVAL_SEC}
  if [[ ! "${interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAS_LOOP_CONTROL_PLANE_PERSIST_INTERVAL_SEC must be a positive integer, got: ${interval}" >&2
    return 2
  fi
  (
    while true; do
      persist_control_plane
      sleep "${interval}"
    done
  ) &
  CONTROL_PLANE_PERSISTER_PID=$!
}
stop_control_plane_persister(){
  if [[ -n "${CONTROL_PLANE_PERSISTER_PID}" ]]; then
    terminate_pid_bounded "${CONTROL_PLANE_PERSISTER_PID}" control-plane-persister 5
    CONTROL_PLANE_PERSISTER_PID=""
  fi
}
cleanup_all(){
  stop_control_plane_persister
  cleanup
  if [[ -n "${COORDINATOR_PID}" ]]; then
    terminate_pid_bounded "${COORDINATOR_PID}" phase-coordinator 15
    COORDINATOR_PID=""
  fi
  persist_control_plane
}
trap cleanup_all EXIT
start_control_plane_persister

start_phase_coordinator(){
  if [[ "${MAS_LOOP_PHASE_LEASE}" != "1" ]]; then return 0; fi
  PYTHONPATH="${PYTHONPATH}" python3 -m mas_loop_rl.phase_coordinator \
    --socket "${MAS_LOOP_VLLM_CONTROL_SOCKET}" --state "${MAS_LOOP_VLLM_SLEEP_STATE}" \
    --events "${MAS_LOOP_VLLM_SLEEP_EVENTS}" --grace-sec "${MAS_LOOP_PHASE_LEASE_GRACE_SEC}" --backlog "${MAS_LOOP_PHASE_SOCKET_BACKLOG}" \
    >"${LOG_DIR}/phase_coordinator.log" 2>&1 &
  COORDINATOR_PID=$!
  for _ in $(seq 1 100); do
    if [[ -S "${MAS_LOOP_VLLM_CONTROL_SOCKET}" ]]; then
      python3 - "${MAS_LOOP_VLLM_CONTROL_SOCKET}" <<'PYCOORD' && return 0
import json,socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.settimeout(2); s.connect(sys.argv[1]); s.sendall(b'{"op":"ping"}\n');
r=b''
while not r.endswith(b'\n'): r += s.recv(4096)
assert json.loads(r).get('ok')
PYCOORD
    fi
    if ! kill -0 "${COORDINATOR_PID}" 2>/dev/null; then
      # Security: only downgrade on the specific sandbox/host policy failure that
      # prevents local socket creation.  All other coordinator failures remain
      # fail-closed because silently losing phase ownership can cause GPU OOM.
      if grep -qF "PermissionError: [Errno 1] Operation not permitted" "${LOG_DIR}/phase_coordinator.log" 2>/dev/null; then
        echo "[WARN] phase coordinator socket creation was denied; using the existing flock fallback" >&2
        MAS_LOOP_PHASE_LEASE=0
        MAS_LOOP_VLLM_CONTROL_SOCKET=""
        export MAS_LOOP_PHASE_LEASE MAS_LOOP_VLLM_CONTROL_SOCKET
        COORDINATOR_PID=""
        return 0
      fi
      tail -n 80 "${LOG_DIR}/phase_coordinator.log" >&2
      return 1
    fi
    sleep 0.1
  done
  if grep -qF "PermissionError: [Errno 1] Operation not permitted" "${LOG_DIR}/phase_coordinator.log" 2>/dev/null; then
    echo "[WARN] phase coordinator socket creation was denied; using the existing flock fallback" >&2
    MAS_LOOP_PHASE_LEASE=0
    MAS_LOOP_VLLM_CONTROL_SOCKET=""
    export MAS_LOOP_PHASE_LEASE MAS_LOOP_VLLM_CONTROL_SOCKET
    COORDINATOR_PID=""
    return 0
  fi
  echo "phase coordinator failed to create ${MAS_LOOP_VLLM_CONTROL_SOCKET}" >&2
  return 1
}
start_phase_coordinator

ray_temp_owned_pids(){
  local temp_dir=$1
  local owner_sid pid sid cmd fd target
  owner_sid=$(ps -o sid= -p "$$" | tr -d ' ')
  while read -r pid sid cmd; do
    [[ -n "${pid}" && "${sid}" == "${owner_sid}" && "${pid}" != "$$" ]] || continue
    if [[ "${cmd}" == *"${temp_dir}"* ]]; then
      echo "${pid}"
      continue
    fi
    # Ray task names such as ray::SimpleStorageUnit omit the session path from
    # argv.  Verify ownership through an open fd into this exact phase temp dir;
    # the same-session check prevents touching sibling-project Ray processes.
    for fd in "/proc/${pid}/fd"/*; do
      [[ -L "${fd}" ]] || continue
      target=$(readlink "${fd}" 2>/dev/null || true)
      if [[ "${target}" == "${temp_dir}"/* ]]; then
        echo "${pid}"
        break
      fi
    done
  done < <(ps -eo pid=,sid=,cmd=)
}

cleanup_ray_temp(){
  local temp_dir=$1
  # Do NOT call `ray stop --force`: it may kill sibling-project Ray jobs in the same container.
  # Only terminate processes in this launcher's session that are tied to this
  # exact MAS_Loop phase temp dir through argv or an open file descriptor.
  local pids
  pids=$(ray_temp_owned_pids "${temp_dir}" | sort -nu | tr '\n' ' ' || true)
  if [[ -n "${pids// /}" ]]; then
    echo "[MAS-Loop] cleaning Ray processes for ${temp_dir}: ${pids}" >&2
    local p
    for p in ${pids}; do terminate_pid_bounded "${p}" ray-phase-process 10; done
  fi
}

sleep_vllm(){
  local port=$1
  if [[ "${MAS_LOOP_VLLM_SLEEP}" != "1" ]]; then return 0; fi
  curl -fsS -X POST "http://127.0.0.1:${port}/sleep?level=${VLLM_SLEEP_LEVEL}" >/dev/null
}

vllm_distributed_port_base(){
  local name=$1 offset distributed_port_base
  [[ -n "${MAS_LOOP_EXTERNAL_VLLM_PORT_BASE}" ]] || return 0
  # Security: role names are selected from a fixed whitelist; never interpolate
  # arbitrary input into the vLLM environment or launch command.
  case "${name}" in
    editor-peer) offset=0 ;;
    subagent) offset=100 ;;
    judge) offset=200 ;;
    *) echo "[ERROR] unknown external vLLM role for distributed port allocation: ${name}" >&2; return 2 ;;
  esac
  distributed_port_base=$((MAS_LOOP_EXTERNAL_VLLM_PORT_BASE + offset))
  if (( distributed_port_base < 1024 || distributed_port_base > 65535 )); then
    echo "[ERROR] distributed port base ${distributed_port_base} for ${name} is outside [1024, 65535]" >&2
    return 2
  fi
  printf '%s\n' "${distributed_port_base}"
}

start_vllm_sleeping(){
  local port=$1 model=$2 name=$3
  local sleep_flag="" gpu_util max_num_seqs max_num_batched_tokens max_model_len max_partial max_long_partial long_threshold tensor_parallel_size data_parallel_size
  local -a parallel_args=()
  local -a extra_flags=()
  local -a served_model_names=("${name}")
  if [[ "${MAS_LOOP_VLLM_SLEEP}" == "1" ]]; then sleep_flag="--enable-sleep-mode"; fi
  if [[ "$name" == "subagent" ]]; then
    gpu_util="${SUB_AGENT_GPU_MEMORY_UTILIZATION}"
    max_num_seqs="${SUB_AGENT_MAX_NUM_SEQS}"
    max_num_batched_tokens="${SUB_AGENT_MAX_NUM_BATCHED_TOKENS}"
    max_model_len="${SUB_AGENT_MAX_MODEL_LEN}"
    max_partial="${SUB_AGENT_MAX_NUM_PARTIAL_PREFILLS}"
    max_long_partial="${SUB_AGENT_MAX_LONG_PARTIAL_PREFILLS}"
    long_threshold="${SUB_AGENT_LONG_PREFILL_TOKEN_THRESHOLD}"
    tensor_parallel_size="${SUB_AGENT_TENSOR_PARALLEL_SIZE}"
    data_parallel_size="${SUB_AGENT_DATA_PARALLEL_SIZE}"
    if [[ "${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM}" == "1" ]]; then served_model_names+=("judge"); fi
    if [[ "${SUB_AGENT_ENABLE_EXPERT_PARALLEL}" == "1" ]]; then extra_flags+=(--enable-expert-parallel); fi
  elif [[ "$name" == "judge" ]]; then
    gpu_util="${JUDGE_GPU_MEMORY_UTILIZATION}"
    max_num_seqs="${JUDGE_MAX_NUM_SEQS}"
    max_num_batched_tokens="${JUDGE_MAX_NUM_BATCHED_TOKENS}"
    max_model_len="${JUDGE_MAX_MODEL_LEN}"
    max_partial="${JUDGE_MAX_NUM_PARTIAL_PREFILLS}"
    max_long_partial="${JUDGE_MAX_LONG_PARTIAL_PREFILLS}"
    long_threshold="${JUDGE_LONG_PREFILL_TOKEN_THRESHOLD}"
    tensor_parallel_size="${JUDGE_TENSOR_PARALLEL_SIZE}"
    data_parallel_size="${JUDGE_DATA_PARALLEL_SIZE}"
    if [[ "${SUB_AGENT_ENABLE_EXPERT_PARALLEL}" == "1" ]]; then extra_flags+=(--enable-expert-parallel); fi
  else
    gpu_util="${PEER_GPU_MEMORY_UTILIZATION}"
    max_num_seqs="${PEER_MAX_NUM_SEQS}"
    max_num_batched_tokens="${PEER_MAX_NUM_BATCHED_TOKENS}"
    max_model_len="${PEER_MAX_MODEL_LEN}"
    max_partial="${PEER_MAX_NUM_PARTIAL_PREFILLS}"
    max_long_partial="${PEER_MAX_LONG_PARTIAL_PREFILLS}"
    long_threshold="${PEER_LONG_PREFILL_TOKEN_THRESHOLD}"
    tensor_parallel_size="${PEER_TENSOR_PARALLEL_SIZE}"
    data_parallel_size="${PEER_DATA_PARALLEL_SIZE}"
  fi
  parallel_args=(--tensor-parallel-size "${tensor_parallel_size}")
  if (( data_parallel_size > 1 )); then
    parallel_args+=(--data-parallel-size "${data_parallel_size}" --data-parallel-size-local "${data_parallel_size}")
  fi
  extra_flags+=(--max-num-partial-prefills "${max_partial}" --max-long-partial-prefills "${max_long_partial}" --long-prefill-token-threshold "${long_threshold}")
  extra_flags+=(--no-enable-prefix-caching)
  local distributed_port_base
  distributed_port_base=$(vllm_distributed_port_base "${name}") || return 2
  echo "[MAS-Loop] start ${name}: TP=${tensor_parallel_size} DP=${data_parallel_size} world_size=$((tensor_parallel_size * data_parallel_size)) gpu_memory_utilization=${gpu_util} max_num_seqs_per_replica=${max_num_seqs} max_num_batched_tokens_per_replica=${max_num_batched_tokens} partial_prefills=${max_partial}/${max_long_partial} long_threshold=${long_threshold} expert_parallel=${SUB_AGENT_ENABLE_EXPERT_PARALLEL} prefix_cache=disabled distributed_port_base=${distributed_port_base:-automatic} http_backlog=${MAS_LOOP_VLLM_HTTP_BACKLOG} keep_alive=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE}s cpu_threads=${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS} cpuset=${MAS_LOOP_EXTERNAL_VLLM_CPUSET:-unbound}" | tee -a "${LOG_DIR}/vllm_launch_config.log"
  # Fixed argv arrays avoid shell interpolation while constraining native CPU pools.
  local -a launch_cmd=(env
    "OMP_NUM_THREADS=${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS}"
    "OMP_DYNAMIC=FALSE"
    "MKL_NUM_THREADS=${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS}"
    "MKL_DYNAMIC=FALSE"
    "OPENBLAS_NUM_THREADS=${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS}"
    "NUMEXPR_NUM_THREADS=${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS}"
    "RAYON_NUM_THREADS=${MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS}"
    "TOKENIZERS_PARALLELISM=false"
    "CUDA_VISIBLE_DEVICES=${ALL_GPUS}")
  if [[ -n "${distributed_port_base}" ]]; then
    launch_cmd+=("VLLM_PORT=${distributed_port_base}")
  fi
  if [[ -n "${MAS_LOOP_EXTERNAL_VLLM_CPUSET}" ]]; then
    launch_cmd=(taskset --cpu-list "${MAS_LOOP_EXTERNAL_VLLM_CPUSET}" "${launch_cmd[@]}")
  fi
  mkdir -p "${MAS_LOOP_EXTERNAL_VLLM_LOG_DIR}"
  local server_log="${MAS_LOOP_EXTERNAL_VLLM_LOG_DIR}/${name}_vllm.log"
  "${launch_cmd[@]}" python3 -m mas_loop_rl.vllm_api_server \
    --model "$model" --served-model-name "${served_model_names[@]}" --host 127.0.0.1 --port "$port" \
    "${parallel_args[@]}" --gpu-memory-utilization "${gpu_util}" --max-num-seqs "${max_num_seqs}" --max-num-batched-tokens "${max_num_batched_tokens}" --max-model-len "${max_model_len}" ${sleep_flag} ${EXTERNAL_VLLM_ENFORCE_EAGER_FLAG} "${extra_flags[@]}" \
    >"${server_log}" 2>&1 &
  local server_pid=$!
  PIDS+=("${server_pid}")
  if [[ "$name" == "editor-peer" ]]; then
    EDITOR_SERVER_PID="${server_pid}"
    EDITOR_SERVER_MODEL="${model}"
  elif [[ "$name" == "subagent" ]]; then
    SUBAGENT_SERVER_PID="${server_pid}"
    SUBAGENT_SERVER_MODEL="${model}"
    if [[ "${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM}" == "1" ]]; then
      JUDGE_SERVER_PID="${server_pid}"
      JUDGE_SERVER_MODEL="${model}"
    fi
  elif [[ "$name" == "judge" ]]; then
    JUDGE_SERVER_PID="${server_pid}"
    JUDGE_SERVER_MODEL="${model}"
  fi
  local i
  for ((i=1; i<=MAS_LOOP_VLLM_STARTUP_ATTEMPTS; i++)); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      sleep_vllm "$port"
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      local server_status=0
      # Security: server_pid comes only from Bash $! for the fixed argv launch above;
      # wait consumes no shell text and preserves the exact exit/signal status.
      wait "${server_pid}" || server_status=$?
      echo "vLLM server ${name} exited during startup with status ${server_status}; last log lines:" >&2
      tail -n 80 "${server_log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "vLLM server ${name} failed to start after $((MAS_LOOP_VLLM_STARTUP_ATTEMPTS * 5)) seconds" >&2
  stop_vllm_role "${name}" || true
  return 1
}

ensure_vllm_sleeping(){
  local port=$1 model=$2 name=$3 pid="" loaded_model=""
  case "${name}" in
    editor-peer) pid="${EDITOR_SERVER_PID}"; loaded_model="${EDITOR_SERVER_MODEL}" ;;
    subagent) pid="${SUBAGENT_SERVER_PID}"; loaded_model="${SUBAGENT_SERVER_MODEL}" ;;
    judge) pid="${JUDGE_SERVER_PID}"; loaded_model="${JUDGE_SERVER_MODEL}" ;;
    *) echo "[ERROR] unknown external vLLM role: ${name}" >&2; return 2 ;;
  esac

  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    if [[ "${loaded_model}" == "${model}" ]] && curl -fsS --max-time 10 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      sleep_vllm "${port}" || true
      echo "[MAS-Loop] reuse sleeping ${name} pid=${pid} model=${model}"
      return 0
    fi
    echo "[MAS-Loop] recycle ${name}: loaded_model=${loaded_model:-unknown} desired_model=${model}"
    stop_vllm_role "${name}"
  elif [[ -n "${pid}" ]]; then
    remove_registered_pid "${pid}"
    case "${name}" in
      editor-peer) EDITOR_SERVER_PID=""; EDITOR_SERVER_MODEL="" ;;
      subagent) SUBAGENT_SERVER_PID=""; SUBAGENT_SERVER_MODEL="" ;;
      judge) JUDGE_SERVER_PID=""; JUDGE_SERVER_MODEL="" ;;
    esac
  fi
  start_vllm_sleeping "${port}" "${model}" "${name}"
}

sleep_all_external_vllms(){
  [[ -z "${EDITOR_SERVER_PID}" ]] || sleep_vllm "${EDITOR_PORT}" || true
  [[ -z "${SUBAGENT_SERVER_PID}" ]] || sleep_vllm "${SUBAGENT_PORT}" || true
  if [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" && -n "${JUDGE_SERVER_PID}" && "${MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM}" != "1" ]]; then
    sleep_vllm "${JUDGE_PORT}" || true
  fi
}

release_external_vllms_after_phase(){
  if [[ "${MAS_LOOP_PERSIST_EXTERNAL_VLLM}" == "1" ]]; then
    sleep_all_external_vllms
  else
    cleanup
  fi
}

smoke_test_bcp_judge(){
  local round_log=$1
  local smoke_dir="${round_log}/judge_smoke"
  [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" ]] || return 0
  mkdir -p "${smoke_dir}"
  echo "[MAS-Loop] running Qwen BCP judge completion smoke test (max_tokens=${MAS_LOOP_JUDGE_MAX_TOKENS})" | tee -a "${LOG_DIR}/vllm_launch_config.log"

  # Exercise the production coordinator-managed wake/request/quiesce/sleep
  # lifecycle rather than bypassing it with a raw wake and immediate sleep.
  set +e
  MAS_LOOP_USE_LLM_JUDGE=1 \
  MAS_LOOP_JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1" \
  MAS_LOOP_JUDGE_MODEL=judge \
  MAS_LOOP_JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN}" \
  MAS_LOOP_JUDGE_MAX_TOKENS="${MAS_LOOP_JUDGE_MAX_TOKENS}" \
  MAS_LOOP_JUDGE_RESULTS_DIR="${smoke_dir}" \
  MAS_LOOP_LOG_DIR="${round_log}" \
  python3 - <<'PY_JUDGE_SMOKE'
import json
from mas_loop_rl.bcp_judge import BCPJudge

result = BCPJudge(manage_sleep=True).judge(
    question="What is 2 + 2?",
    response="4",
    correct_answer="4",
    pair_id="startup-smoke",
    candidate="smoke",
    optimizer_step=0,
    outer_adversarial_round=0,
    data_source="bcp_smoke",
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
if result.get("score") != 1.0:
    raise SystemExit(f"BCP judge smoke test failed: {result}")
PY_JUDGE_SMOKE
  local smoke_rc=$?
  set -e
  if (( smoke_rc != 0 )); then
    echo "[MAS-Loop] Qwen BCP judge completion smoke test failed rc=${smoke_rc}; refusing to start training" >&2
    return "${smoke_rc}"
  fi
  echo "[MAS-Loop] Qwen BCP judge completion smoke test passed; result_dir=${smoke_dir}" | tee -a "${LOG_DIR}/vllm_launch_config.log"
}

# A vLLM V1 DP API parent can remain alive and keep /v1/models healthy after an
# EngineCore child exits. Detect direct zombie children explicitly; otherwise a
# later /wake_up blocks on the missing DP rank until every rollout request times
# out. /proc is used instead of process-name matching so unrelated jobs cannot
# affect this check.
vllm_zombie_child(){
  local server_pid=$1 child state
  [[ "${server_pid}" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/${server_pid}/task/${server_pid}/children" ]] || return 1
  for child in $(<"/proc/${server_pid}/task/${server_pid}/children"); do
    [[ "${child}" =~ ^[0-9]+$ ]] || continue
    [[ -r "/proc/${child}/status" ]] || continue
    state=$(awk '/^State:/{print $2; exit}' "/proc/${child}/status" 2>/dev/null || true)
    if [[ "${state}" == "Z" ]]; then
      printf '%s' "${child}"
      return 0
    fi
  done
  return 1
}

# Fail fast when an external vLLM engine dies during Orchestra rollout.
# Without this guard, Ray workers can wait forever on the exclusive phase lease
# after the dead service leaves the coordinator owner in sleep_failed state.
start_service_watchdog(){
  local trainer_pid=$1 round_log=$2 phase_ray_dir=${3:-}
  SERVICE_WATCHDOG_FAILURE_FILE="${round_log}/service_watchdog_failure.txt"
  rm -f "${SERVICE_WATCHDOG_FAILURE_FILE}"
  (
    # A missing server PID is definitive. Health endpoints may briefly stall
    # while a TP=8 engine wakes, so require ~1 minute of consecutive failures.
    local health_failures=0 reason="" loop_count=0 zombie_pid=""
    while kill -0 "${trainer_pid}" 2>/dev/null; do
      loop_count=$((loop_count + 1))
      reason=""
      zombie_pid=""
      if [[ "${RUN_EDITOR}" == "1" ]] && { [[ -z "${EDITOR_SERVER_PID}" ]] || ! kill -0 "${EDITOR_SERVER_PID}" 2>/dev/null; }; then
        reason="editor-peer process exited (pid=${EDITOR_SERVER_PID:-unset})"
        health_failures=12
      elif [[ "${RUN_EDITOR}" == "1" ]] && zombie_pid=$(vllm_zombie_child "${EDITOR_SERVER_PID}"); then
        reason="editor-peer DP child became zombie (server_pid=${EDITOR_SERVER_PID}, child_pid=${zombie_pid})"
        health_failures=12
      elif [[ -z "${SUBAGENT_SERVER_PID}" ]] || ! kill -0 "${SUBAGENT_SERVER_PID}" 2>/dev/null; then
        reason="subagent process exited (pid=${SUBAGENT_SERVER_PID:-unset})"
        health_failures=12
      elif zombie_pid=$(vllm_zombie_child "${SUBAGENT_SERVER_PID}"); then
        reason="subagent DP child became zombie (server_pid=${SUBAGENT_SERVER_PID}, child_pid=${zombie_pid})"
        health_failures=12
      elif [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" ]] && { [[ -z "${JUDGE_SERVER_PID}" ]] || ! kill -0 "${JUDGE_SERVER_PID}" 2>/dev/null; }; then
        reason="judge process exited (pid=${JUDGE_SERVER_PID:-unset})"
        health_failures=12
      elif [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" && "${JUDGE_SERVER_PID}" != "${SUBAGENT_SERVER_PID}" ]] && zombie_pid=$(vllm_zombie_child "${JUDGE_SERVER_PID}"); then
        reason="judge DP child became zombie (server_pid=${JUDGE_SERVER_PID}, child_pid=${zombie_pid})"
        health_failures=12
      elif [[ "${RUN_EDITOR}" == "1" ]] && ! curl -fsS --max-time 5 "http://127.0.0.1:${EDITOR_PORT}/v1/models" >/dev/null 2>&1; then
        reason="editor-peer health endpoint unavailable on port ${EDITOR_PORT}"
        health_failures=$((health_failures + 1))
      elif ! curl -fsS --max-time 5 "http://127.0.0.1:${SUBAGENT_PORT}/v1/models" >/dev/null 2>&1; then
        reason="subagent health endpoint unavailable on port ${SUBAGENT_PORT}"
        health_failures=$((health_failures + 1))
      elif [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" ]] && ! curl -fsS --max-time 5 "http://127.0.0.1:${JUDGE_PORT}/v1/models" >/dev/null 2>&1; then
        reason="judge health endpoint unavailable on port ${JUDGE_PORT}"
        health_failures=$((health_failures + 1))
      elif [[ -n "${phase_ray_dir}" ]] && (( loop_count % 6 == 0 )) && \
           find "${phase_ray_dir}" -path '*/logs/worker-*.out' -type f -mmin -2 -print0 2>/dev/null | \
             xargs -0 -r grep -Ilm1 -E 'EngineCore fatal error|EngineDeadError|CUDA out of memory' 2>/dev/null | grep -q .; then
        reason="integrated rollout EngineCore reported a fatal error under ${phase_ray_dir}"
        # Unlike a transient HTTP wake stall, EngineDeadError is terminal.
        health_failures=12
      else
        health_failures=0
      fi
      if (( health_failures >= 12 )); then
        printf '[%s] fatal rollout service failure: %s\n' "$(date -Is)" "${reason}" | tee -a "${SERVICE_WATCHDOG_FAILURE_FILE}" >&2
        kill -TERM "${trainer_pid}" 2>/dev/null || true
        sleep 10
        kill -KILL "${trainer_pid}" 2>/dev/null || true
        exit 70
      fi
      sleep 5
    done
  ) &
  SERVICE_WATCHDOG_PID=$!
  echo "[MAS-Loop] service watchdog pid=${SERVICE_WATCHDOG_PID} trainer_pid=${trainer_pid}"
}

finish_service_watchdog(){
  local watchdog_rc=0
  if [[ -n "${SERVICE_WATCHDOG_PID}" ]]; then
    kill "${SERVICE_WATCHDOG_PID}" 2>/dev/null || true
    wait "${SERVICE_WATCHDOG_PID}" 2>/dev/null || watchdog_rc=$?
    SERVICE_WATCHDOG_PID=""
  fi
  if [[ -s "${SERVICE_WATCHDOG_FAILURE_FILE:-}" ]]; then
    echo "[ERROR] External vLLM watchdog detected a fatal service failure:" >&2
    cat "${SERVICE_WATCHDOG_FAILURE_FILE}" >&2
    return 70
  fi
  return 0
}

latest_hf_ckpt(){
  local root=$1
  python3 - "$root" <<'PY'
import re, sys
from pathlib import Path
root=Path(sys.argv[1]); best=None; best_step=-1
if root.exists():
    for p in root.glob('global_step_*/actor/huggingface'):
        m=re.search(r'global_step_(\d+)', str(p))
        if m and (p/'config.json').exists():
            step=int(m.group(1))
            if step>best_step: best_step=step; best=p
print(best or '')
PY
}

latest_hf_step(){
  local root=$1
  python3 - "$root" <<'PY'
import re, sys
from pathlib import Path
root=Path(sys.argv[1]); best_step=-1
if root.exists():
    for p in root.glob('global_step_*/actor/huggingface'):
        m=re.search(r'global_step_(\d+)', str(p))
        if m and (p/'config.json').exists():
            best_step=max(best_step, int(m.group(1)))
print(best_step if best_step >= 0 else '')
PY
}

maybe_update_model_from_ckpt(){
  local role=$1 root=$2 latest
  if [[ "${UPDATE_PEER_FROM_HF_CKPT}" != "1" ]]; then return 0; fi
  latest=$(latest_hf_ckpt "$root")
  if [[ -n "$latest" ]]; then
    if [[ "$role" == "orchestra" ]]; then ORCHESTRA_MODEL="$latest"; else EDITOR_MODEL="$latest"; fi
    echo "Updated ${role} model for next adversarial calls: ${latest}"
  else
    echo "No HF checkpoint found under ${root}; keeping existing ${role} model path."
  fi
}

validate_bcp_phase_artifacts(){
  local phase_dir=$1 phase=$2
  case "${MAS_LOOP_DATASET_NAME,,}" in
    bcp|browsecomp|browsecomp_plus) ;;
    *) return 0 ;;
  esac
  echo "[MAS-Loop] validating persisted BCP ${phase} metrics/rollouts/judge artifacts under ${phase_dir}"
  python3 -m mas_loop_rl.validate_bcp_artifacts \
    --phase-dir "${phase_dir}" --phase "${phase}" \
    --output "${phase_dir}/artifact_validation.json"
}

backfill_bcp_judges_if_needed(){
  local phase_dir=$1 missing
  [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" ]] || return 0
  [[ "${MAS_LOOP_AUTO_BACKFILL_JUDGES}" == "1" ]] || return 0
  case "${MAS_LOOP_DATASET_NAME,,}" in
    bcp|browsecomp|browsecomp_plus) ;;
    *) return 0 ;;
  esac
  missing=$(python3 - "${phase_dir}" <<'PY_BACKFILL_SCAN'
import sys
from mas_loop_rl.backfill_bcp_judges import scan_missing_candidates

targets, report = scan_missing_candidates(sys.argv[1])
print(len(targets))
PY_BACKFILL_SCAN
  )
  if (( missing == 0 )); then
    echo "[MAS-Loop] Judge evidence scan complete: no backfill needed under ${phase_dir}"
    return 0
  fi
  echo "[MAS-Loop] targeted Judge backfill: ${missing} candidate(s) under ${phase_dir}"
  curl -fsS -X POST "http://127.0.0.1:${JUDGE_PORT}/wake_up" >/dev/null
  MAS_LOOP_JUDGE_MAX_TOKENS="${MAS_LOOP_JUDGE_MAX_TOKENS}" \
  MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE="${MAS_LOOP_JUDGE_BACKFILL_MAX_WORKERS}" \
  python3 -m mas_loop_rl.backfill_bcp_judges \
    --phase-dir "${phase_dir}" \
    --judge-base-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --judge-model judge \
    --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
    --max-workers "${MAS_LOOP_JUDGE_BACKFILL_MAX_WORKERS}" \
    --audit "${phase_dir}/judge_backfill_audit.json"
  sleep_vllm "${JUDGE_PORT}" || true
}

prepare_phase_ckpt_dir(){
  local round=$1 role=$2
  echo "${MAS_LOOP_CKPT_ROOT}/round_${round}/${role}"
}

parquet_training_steps(){
  local parquet=$1 batch_size=$2 epochs=$3
  python3 - "$parquet" "$batch_size" "$epochs" "$TRAIN_DROP_LAST" <<'PY_STEPS'
import sys
import pyarrow.parquet as pq
rows = pq.ParquetFile(sys.argv[1]).metadata.num_rows
batch = int(sys.argv[2]); epochs = int(sys.argv[3]); drop_last = sys.argv[4] == "True"
steps_per_epoch = rows // batch if drop_last else (rows + batch - 1) // batch
steps = steps_per_epoch * epochs
if steps <= 0:
    raise SystemExit(f"No training steps: rows={rows} batch={batch} epochs={epochs}")
print(steps)
PY_STEPS
}

run_orchestra_phase(){
  local round=$1; shift
  local round_log
  local phase_ray_dir="/dev/shm/mlr_${$}_${round}o"
  local ckpt_dir trainer_pid peer_base_url=""
  ckpt_dir=$(prepare_phase_ckpt_dir "$round" orchestra)
  if [[ "${RUN_EDITOR}" == "1" ]]; then
    peer_base_url="http://127.0.0.1:${EDITOR_PORT}/v1"
  fi
  round_log="${ckpt_dir}/rollouts"
  mkdir -p "$round_log" "$ckpt_dir/metrics" "$ckpt_dir/hydra" "$phase_ray_dir"
  # Pure-orchestrator ablation: do not start or contact the Refiner peer.
  # The default two-role path remains unchanged.
  if [[ "${RUN_EDITOR}" == "1" ]]; then
    ensure_vllm_sleeping "${EDITOR_PORT}" "${EDITOR_MODEL}" editor-peer
  fi
  ensure_vllm_sleeping "${SUBAGENT_PORT}" "${SUBAGENT_MODEL}" subagent
  if [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" ]]; then ensure_vllm_sleeping "${JUDGE_PORT}" "${JUDGE_MODEL}" judge; fi
  sleep_all_external_vllms
  # A real Judge smoke test is useful on first launch. Repeating it for an
  # unchanged resident server on every optimizer step only adds wake/sleep and
  # generation overhead.
  if [[ "${MAS_LOOP_USE_LLM_JUDGE}" == "1" && "${round}" == "1" && "${JUDGE_SMOKE_TESTED:-0}" != "1" ]]; then
    smoke_test_bcp_judge "${round_log}"
    JUDGE_SMOKE_TESTED=1
  fi
  set +e
  CUDA_VISIBLE_DEVICES="${ALL_GPUS}" \
  MAS_LOOP_ROLE=orchestra MAS_LOOP_ACTOR_VLLM_MAX_ACTIVE="${MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE}" MAS_LOOP_OUTER_ROUND="$round" MAS_LOOP_PEER_BASE_URL="${peer_base_url}" MAS_LOOP_PEER_MODEL=editor-peer MAS_LOOP_PEER_MAX_TOKENS=${MAS_LOOP_EDITOR_MAX_TOKENS} \
  MAS_LOOP_SUBAGENT_BASE_URL=http://127.0.0.1:${SUBAGENT_PORT}/v1 MAS_LOOP_SUBAGENT_MODEL=subagent MAS_LOOP_SUBAGENT_MAX_MODEL_LEN=${SUB_AGENT_MAX_MODEL_LEN} MAS_LOOP_PEER_MAX_MODEL_LEN=${PEER_MAX_MODEL_LEN} MAS_LOOP_USE_LLM_JUDGE=${MAS_LOOP_USE_LLM_JUDGE} MAS_LOOP_JUDGE_BASE_URL=http://127.0.0.1:${JUDGE_PORT}/v1 MAS_LOOP_JUDGE_MODEL=judge MAS_LOOP_JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN} MAS_LOOP_JUDGE_RESULTS_DIR="$round_log/judge_results" MAS_LOOP_LOG_DIR="$round_log" MAS_LOOP_PAIR_DIR="$round_log/pairs" MAS_LOOP_RL_METRICS_FILE="$ckpt_dir/metrics/training_metrics.jsonl" \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="['${DATA_DIR}/orchestra_train.parquet']" data.val_files="['${DATA_DIR}/orchestra_val.parquet']" data.train_batch_size=${TRAIN_BATCH_SIZE} data.shuffle=${TRAIN_DATA_SHUFFLE} data.drop_last=${TRAIN_DROP_LAST} data.max_prompt_length=${MAX_PROMPT_LENGTH} data.max_response_length=${MAX_RESPONSE_LENGTH} data.filter_overlong_prompts=True data.truncation=left +data.apply_chat_template_kwargs.enable_thinking=${DATA_ENABLE_THINKING} \
    actor_rollout_ref.model.path="${ORCHESTRA_MODEL}" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.lora_rank=0 actor_rollout_ref.model.lora_adapter_path=null \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.optim.lr=${ORCH_LR:-1e-6} actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" actor_rollout_ref.actor.checkpoint.load_contents="[]" \
    actor_rollout_ref.rollout.name=${INFER_BACKEND:-vllm} actor_rollout_ref.rollout.temperature=${ACTOR_TEMPERATURE} actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_partial_prefills=${ROLLOUT_MAX_NUM_PARTIAL_PREFILLS} +actor_rollout_ref.rollout.engine_kwargs.vllm.max_long_partial_prefills=${ROLLOUT_MAX_LONG_PARTIAL_PREFILLS} +actor_rollout_ref.rollout.engine_kwargs.vllm.long_prefill_token_threshold=${ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.agent.num_workers=${MAS_LOOP_AGENT_LOOP_WORKER_PROCESSES} actor_rollout_ref.rollout.agent.default_agent_loop=mas_loop_orchestra actor_rollout_ref.rollout.agent.agent_loop_config_path="${VERL_ROOT}/mas_loop_rl/loop_agent.yaml" +actor_rollout_ref.rollout.agent.multi_output_grpo_mode=final_broadcast +actor_rollout_ref.rollout.agent.multi_output_group_size=1 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE_BOOL} actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER_BOOL:-False} actor_rollout_ref.rollout.disable_log_stats=${ROLLOUT_DISABLE_LOG_STATS_BOOL} actor_rollout_ref.rollout.enable_prefix_caching=False \
    reward.custom_reward_function.path="${VERL_ROOT}/mas_loop_rl/reward.py" reward.custom_reward_function.name=compute_score reward.reward_manager.name=naive reward.num_workers=${REWARD_NUM_WORKERS} \
    +reward.custom_reward_function.reward_kwargs.role=orchestra +reward.custom_reward_function.reward_kwargs.log_dir="$round_log" +reward.custom_reward_function.reward_kwargs.peer_base_url="${peer_base_url}" +reward.custom_reward_function.reward_kwargs.peer_model=editor-peer +reward.custom_reward_function.reward_kwargs.subagent_base_url=http://127.0.0.1:${SUBAGENT_PORT}/v1 +reward.custom_reward_function.reward_kwargs.subagent_model=subagent +reward.custom_reward_function.reward_kwargs.manage_vllm_sleep=${MAS_LOOP_REWARD_MANAGE_SLEEP} \
    trainer.use_v1=True trainer.v1.trainer_mode=sync trainer.logger='["console"]' trainer.project_name=mas_loop_grpo trainer.experiment_name=orchestra_grpo_round_${round} trainer.n_gpus_per_node=${N_GPUS} trainer.nnodes=1 trainer.total_epochs=${TOTAL_EPOCHS} trainer.save_freq=${SAVE_FREQ} +trainer.num_checkpoints=${NUM_CHECKPOINTS} trainer.test_freq=${TEST_FREQ} trainer.val_before_train=${VAL_BEFORE_TRAIN} trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP} trainer.default_local_dir="$ckpt_dir" hydra.run.dir="$ckpt_dir/hydra" hydra.output_subdir=null +ray_kwargs.ray_init._temp_dir="${phase_ray_dir}" +ray_kwargs.ray_init.dashboard_port=${RAY_DASHBOARD_PORT} "${RAY_RUNTIME_ENV_ARGS[@]}" +ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.OMP_DYNAMIC="'FALSE'" +ray_kwargs.ray_init.runtime_env.env_vars.MKL_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MKL_DYNAMIC="'FALSE'" +ray_kwargs.ray_init.runtime_env.env_vars.OPENBLAS_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.NUMEXPR_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.RAYON_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM="'false'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_SERVER_MAX_CONCURRENCY="'${MAS_LOOP_RAY_SERVER_MAX_CONCURRENCY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_GENERATE_MAX_OUTSTANDING="'${MAS_LOOP_RAY_GENERATE_MAX_OUTSTANDING}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER="'${MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE="'${MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_CANDIDATE_CONCURRENCY="'${MAS_LOOP_EDITOR_CANDIDATE_CONCURRENCY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_LB_METRICS_DIR="'${MAS_LOOP_RAY_LB_METRICS_DIR}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_LB_METRICS_INTERVAL_SEC="'${MAS_LOOP_RAY_LB_METRICS_INTERVAL_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'${VLLM_USE_V1}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RPC_TIMEOUT="'${VLLM_RPC_TIMEOUT}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENGINE_ITERATION_TIMEOUT_S="'${VLLM_ENGINE_ITERATION_TIMEOUT_S}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="'${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}'" +ray_kwargs.ray_init.runtime_env.env_vars.TQ_DATA_UPDATE_MAX_CONCURRENCY="'${TQ_DATA_UPDATE_MAX_CONCURRENCY}'" +ray_kwargs.ray_init.runtime_env.env_vars.TQ_DATA_UPDATE_MAX_RETRIES="'${TQ_DATA_UPDATE_MAX_RETRIES}'" +ray_kwargs.ray_init.runtime_env.env_vars.TQ_DATA_UPDATE_RETRY_BACKOFF_SEC="'${TQ_DATA_UPDATE_RETRY_BACKOFF_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_VLLM_MAX_ACTIVE="'${MAS_LOOP_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ACTOR_VLLM_MAX_ACTIVE="'${MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE="'${MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE="'${MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE="'${MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE="'${MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_USE_LLM_JUDGE="'${MAS_LOOP_USE_LLM_JUDGE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_BASE_URL="'http://127.0.0.1:${JUDGE_PORT}/v1'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_MODEL="'judge'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_MAX_MODEL_LEN="'${JUDGE_MAX_MODEL_LEN}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_MAX_TOKENS="'${MAS_LOOP_JUDGE_MAX_TOKENS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_GUIDED_JSON="'${MAS_LOOP_JUDGE_GUIDED_JSON}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_INVALID_ACTION_PENALTY="'${MAS_LOOP_INVALID_ACTION_PENALTY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ORCH_ADV_PENALTY="'${MAS_LOOP_ORCH_ADV_PENALTY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_RESULTS_DIR="'${round_log}/judge_results'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_TRANSPORT="'${MAS_LOOP_HTTP_TRANSPORT}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_NUM_POOLS="'${MAS_LOOP_HTTP_NUM_POOLS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_POOL_MAXSIZE="'${MAS_LOOP_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ORCHESTRA_HTTP_POOL_MAXSIZE="'${MAS_LOOP_ORCHESTRA_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE="'${MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE="'${MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE="'${MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_VLLM_DP_QUIESCE_SEC="'${MAS_LOOP_VLLM_DP_QUIESCE_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_CONNECT_TIMEOUT_SEC="'${MAS_LOOP_HTTP_CONNECT_TIMEOUT_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_POOL_TIMEOUT_SEC="'${MAS_LOOP_HTTP_POOL_TIMEOUT_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DISPATCH_METRICS_DIR="'${MAS_LOOP_DISPATCH_METRICS_DIR}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC="'${MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DEPENDENCY_PROMPT_MODE="'${MAS_LOOP_DEPENDENCY_PROMPT_MODE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DAG_AGENT_PARALLEL="'${MAS_LOOP_DAG_AGENT_PARALLEL}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DAG_AGENT_WORKERS_PER_PROCESS="'${MAS_LOOP_DAG_AGENT_WORKERS_PER_PROCESS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_BCP_SEARCH_ROUNDS="'${MAS_LOOP_BCP_SEARCH_ROUNDS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS="'${MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_PATCH_TORCH_DIST_TIMEOUT="'${MAS_LOOP_PATCH_TORCH_DIST_TIMEOUT}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS="'${MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS}'" +ray_kwargs.ray_init.include_dashboard=False "$@" &
  trainer_pid=$!
  start_service_watchdog "$trainer_pid" "$round_log" "$phase_ray_dir"
  wait "$trainer_pid"
  local rc=$? watchdog_rc=0
  finish_service_watchdog || watchdog_rc=$?
  if (( rc == 0 && watchdog_rc != 0 )); then rc=$watchdog_rc; fi
  set -e
  # Snapshot process-local metrics while Ray workers still exist, then again
  # after phase cleanup. This survives both normal phase transitions and EXIT.
  persist_control_plane
  # Release the colocated actor/Ray workers before waking the external Judge
  # for any targeted repair, while keeping that Judge server alive.
  cleanup_ray_temp "${phase_ray_dir}"
  if (( rc == 0 )); then
    backfill_bcp_judges_if_needed "${ckpt_dir}" || rc=$?
  fi
  # Keep external services resident across joint steps. They are asleep here,
  # so Editor FSDP can use nearly all device memory without paying 30B model
  # reload costs on the next Orchestra step.
  release_external_vllms_after_phase
  persist_control_plane
  if (( rc != 0 )); then
    echo "[ERROR] Orchestra phase round ${round} failed with exit code ${rc}." >&2
    return ${rc}
  fi
  validate_bcp_phase_artifacts "${ckpt_dir}" orchestra
  maybe_update_model_from_ckpt orchestra "$ckpt_dir"
  local canonical_logs="${round_log}/canonical_rollouts"
  if [[ "${RUN_EDITOR}" == "1" && "${EDITOR_DATA_FROM_ORCH_LOGS}" == "1" && -d "${canonical_logs}" ]] && compgen -G "${canonical_logs}/*.jsonl" >/dev/null; then
    python3 -m mas_loop_rl.build_editor_data_from_logs --logs "${canonical_logs}" --output-dir "${DATA_DIR}" --group-size "${EDITOR_GROUP_SIZE}" --val-ratio 0 --pad-train-to-multiple "${EDITOR_TRAIN_BATCH_SIZE}" || { echo "[WARN] Could not build editor data from canonical Orchestra shards; keep existing ${DATA_DIR}/editor_train.parquet if present." >&2; test -s "${DATA_DIR}/editor_train.parquet"; }
  fi
}

run_editor_phase(){
  local round=$1; shift
  local editor_train_batch_size=${CURRENT_EDITOR_TRAIN_BATCH_SIZE:-${EDITOR_TRAIN_BATCH_SIZE}}
  local editor_ppo_mini_batch_size=${CURRENT_EDITOR_PPO_MINI_BATCH_SIZE:-${PPO_MINI_BATCH_SIZE}}
  local editor_ppo_micro_batch_size_per_gpu=${CURRENT_EDITOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-${PPO_MICRO_BATCH_SIZE_PER_GPU}}
  local editor_log_prob_micro_batch_size_per_gpu=${CURRENT_EDITOR_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}}
  local round_log
  local phase_ray_dir="/dev/shm/mlr_${$}_${round}e"
  local ckpt_dir trainer_pid
  ckpt_dir=$(prepare_phase_ckpt_dir "$round" editor)
  round_log="${ckpt_dir}/rollouts"
  mkdir -p "$round_log" "$ckpt_dir/metrics" "$ckpt_dir/hydra" "$phase_ray_dir"
  # Editor consumes the exact E_k prompt/output/reward collected for each O_k.
  # No Orchestra-peer or sub-agent server is needed during replay training.
  set +e
  CUDA_VISIBLE_DEVICES="${ALL_GPUS}" \
  MAS_LOOP_ROLE=editor MAS_LOOP_ACTOR_VLLM_MAX_ACTIVE="${MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE}" MAS_LOOP_OUTER_ROUND="$round" MAS_LOOP_LOG_DIR="$round_log" MAS_LOOP_PAIR_DIR="$round_log/pairs" MAS_LOOP_RL_METRICS_FILE="$ckpt_dir/metrics/training_metrics.jsonl" \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="['${DATA_DIR}/editor_train.parquet']" data.val_files="['${DATA_DIR}/editor_val.parquet']" data.train_batch_size=${editor_train_batch_size} data.drop_last=True data.max_prompt_length=${MAX_PROMPT_LENGTH} data.max_response_length=${MAX_RESPONSE_LENGTH} data.shuffle=False data.validation_shuffle=False data.filter_overlong_prompts=False data.truncation=left +data.apply_chat_template_kwargs.enable_thinking=${DATA_ENABLE_THINKING} \
    actor_rollout_ref.model.path="${EDITOR_MODEL}" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.lora_rank=0 actor_rollout_ref.model.lora_adapter_path=null \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.param_offload=${EDITOR_ACTOR_PARAM_OFFLOAD} actor_rollout_ref.actor.fsdp_config.optimizer_offload=${EDITOR_ACTOR_OPTIMIZER_OFFLOAD} actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 actor_rollout_ref.ref.fsdp_config.param_offload=${EDITOR_REF_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.optim.lr=${EDITOR_LR:-1e-6} actor_rollout_ref.actor.ppo_mini_batch_size=${editor_ppo_mini_batch_size} actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${editor_ppo_micro_batch_size_per_gpu} actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${EDITOR_PPO_MAX_TOKEN_LEN_PER_GPU} actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" actor_rollout_ref.actor.checkpoint.load_contents="[]" \
    actor_rollout_ref.rollout.name=${INFER_BACKEND:-vllm} actor_rollout_ref.rollout.temperature=${ACTOR_TEMPERATURE} actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_partial_prefills=${ROLLOUT_MAX_NUM_PARTIAL_PREFILLS} +actor_rollout_ref.rollout.engine_kwargs.vllm.max_long_partial_prefills=${ROLLOUT_MAX_LONG_PARTIAL_PREFILLS} +actor_rollout_ref.rollout.engine_kwargs.vllm.long_prefill_token_threshold=${ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD} actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.agent.num_workers=${MAS_LOOP_AGENT_LOOP_WORKER_PROCESSES} actor_rollout_ref.rollout.agent.default_agent_loop=mas_loop_editor_replay actor_rollout_ref.rollout.agent.agent_loop_config_path="${VERL_ROOT}/mas_loop_rl/loop_agent.yaml" +actor_rollout_ref.rollout.agent.multi_output_grpo_mode=external_group +actor_rollout_ref.rollout.agent.multi_output_group_size=${EDITOR_GROUP_SIZE} actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${editor_log_prob_micro_batch_size_per_gpu} actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${EDITOR_LOG_PROB_MAX_TOKEN_LEN_PER_GPU} actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${editor_log_prob_micro_batch_size_per_gpu} actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${EDITOR_LOG_PROB_MAX_TOKEN_LEN_PER_GPU} +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE_BOOL} actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER_BOOL:-False} actor_rollout_ref.rollout.disable_log_stats=${ROLLOUT_DISABLE_LOG_STATS_BOOL} actor_rollout_ref.rollout.enable_prefix_caching=False \
    reward.custom_reward_function.path="${VERL_ROOT}/mas_loop_rl/reward.py" reward.custom_reward_function.name=compute_score reward.reward_manager.name=naive reward.num_workers=${REWARD_NUM_WORKERS} \
    +reward.custom_reward_function.reward_kwargs.role=editor +reward.custom_reward_function.reward_kwargs.log_dir="$round_log" \
    trainer.use_v1=True trainer.v1.trainer_mode=sync trainer.logger='["console"]' trainer.project_name=mas_loop_grpo trainer.experiment_name=editor_grpo_round_${round} trainer.n_gpus_per_node=${N_GPUS} trainer.nnodes=1 trainer.total_epochs=${TOTAL_EPOCHS} trainer.save_freq=${SAVE_FREQ} +trainer.num_checkpoints=${NUM_CHECKPOINTS} trainer.test_freq=${TEST_FREQ} trainer.val_before_train=${VAL_BEFORE_TRAIN} trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP} trainer.default_local_dir="$ckpt_dir" hydra.run.dir="$ckpt_dir/hydra" hydra.output_subdir=null +ray_kwargs.ray_init._temp_dir="${phase_ray_dir}" +ray_kwargs.ray_init.dashboard_port=$((RAY_DASHBOARD_PORT+1)) "${RAY_RUNTIME_ENV_ARGS[@]}" +ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.OMP_DYNAMIC="'FALSE'" +ray_kwargs.ray_init.runtime_env.env_vars.MKL_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MKL_DYNAMIC="'FALSE'" +ray_kwargs.ray_init.runtime_env.env_vars.OPENBLAS_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.NUMEXPR_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.RAYON_NUM_THREADS="'${MAS_LOOP_RAY_WORKER_CPU_THREADS}'" +ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM="'false'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_SERVER_MAX_CONCURRENCY="'${MAS_LOOP_RAY_SERVER_MAX_CONCURRENCY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_GENERATE_MAX_OUTSTANDING="'${MAS_LOOP_RAY_GENERATE_MAX_OUTSTANDING}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER="'${MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE="'${MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_CANDIDATE_CONCURRENCY="'${MAS_LOOP_EDITOR_CANDIDATE_CONCURRENCY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_LB_METRICS_DIR="'${MAS_LOOP_RAY_LB_METRICS_DIR}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_RAY_LB_METRICS_INTERVAL_SEC="'${MAS_LOOP_RAY_LB_METRICS_INTERVAL_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'${VLLM_USE_V1}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RPC_TIMEOUT="'${VLLM_RPC_TIMEOUT}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENGINE_ITERATION_TIMEOUT_S="'${VLLM_ENGINE_ITERATION_TIMEOUT_S}'" +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="'${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}'" +ray_kwargs.ray_init.runtime_env.env_vars.TQ_DATA_UPDATE_MAX_CONCURRENCY="'${TQ_DATA_UPDATE_MAX_CONCURRENCY}'" +ray_kwargs.ray_init.runtime_env.env_vars.TQ_DATA_UPDATE_MAX_RETRIES="'${TQ_DATA_UPDATE_MAX_RETRIES}'" +ray_kwargs.ray_init.runtime_env.env_vars.TQ_DATA_UPDATE_RETRY_BACKOFF_SEC="'${TQ_DATA_UPDATE_RETRY_BACKOFF_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_VLLM_MAX_ACTIVE="'${MAS_LOOP_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ACTOR_VLLM_MAX_ACTIVE="'${MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE="'${MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE="'${MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE="'${MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE="'${MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_USE_LLM_JUDGE="'${MAS_LOOP_USE_LLM_JUDGE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_BASE_URL="'http://127.0.0.1:${JUDGE_PORT}/v1'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_MODEL="'judge'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_MAX_MODEL_LEN="'${JUDGE_MAX_MODEL_LEN}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_MAX_TOKENS="'${MAS_LOOP_JUDGE_MAX_TOKENS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_GUIDED_JSON="'${MAS_LOOP_JUDGE_GUIDED_JSON}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_INVALID_ACTION_PENALTY="'${MAS_LOOP_INVALID_ACTION_PENALTY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ORCH_ADV_PENALTY="'${MAS_LOOP_ORCH_ADV_PENALTY}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_RESULTS_DIR="'${round_log}/judge_results'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_TRANSPORT="'${MAS_LOOP_HTTP_TRANSPORT}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_NUM_POOLS="'${MAS_LOOP_HTTP_NUM_POOLS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_POOL_MAXSIZE="'${MAS_LOOP_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_ORCHESTRA_HTTP_POOL_MAXSIZE="'${MAS_LOOP_ORCHESTRA_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE="'${MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE="'${MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE="'${MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_VLLM_DP_QUIESCE_SEC="'${MAS_LOOP_VLLM_DP_QUIESCE_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_CONNECT_TIMEOUT_SEC="'${MAS_LOOP_HTTP_CONNECT_TIMEOUT_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_HTTP_POOL_TIMEOUT_SEC="'${MAS_LOOP_HTTP_POOL_TIMEOUT_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DISPATCH_METRICS_DIR="'${MAS_LOOP_DISPATCH_METRICS_DIR}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC="'${MAS_LOOP_DISPATCH_METRICS_INTERVAL_SEC}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DEPENDENCY_PROMPT_MODE="'${MAS_LOOP_DEPENDENCY_PROMPT_MODE}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DAG_AGENT_PARALLEL="'${MAS_LOOP_DAG_AGENT_PARALLEL}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_DAG_AGENT_WORKERS_PER_PROCESS="'${MAS_LOOP_DAG_AGENT_WORKERS_PER_PROCESS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_BCP_SEARCH_ROUNDS="'${MAS_LOOP_BCP_SEARCH_ROUNDS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS="'${MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS}'" +ray_kwargs.ray_init.runtime_env.env_vars.MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS="'${MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS}'" +ray_kwargs.ray_init.include_dashboard=False "$@" &
  trainer_pid=$!
  wait "$trainer_pid"
  local rc=$?
  set -e
  # Snapshot process-local metrics while Ray workers still exist, then again
  # after phase cleanup. This survives both normal phase transitions and EXIT.
  persist_control_plane
  cleanup_ray_temp "${phase_ray_dir}"
  release_external_vllms_after_phase
  persist_control_plane
  if (( rc != 0 )); then
    echo "[ERROR] Editor phase round ${round} failed with exit code ${rc}." >&2
    return ${rc}
  fi
  validate_bcp_phase_artifacts "${ckpt_dir}" editor
  maybe_update_model_from_ckpt editor "$ckpt_dir"
  # The external Editor peer does not share the in-process FSDP/vLLM weight
  # synchronizer. Recycle only this role after E_k changes; immutable
  # Subagent/Judge engines remain sleeping and are reused by step k+1.
  if [[ -n "${EDITOR_SERVER_PID}" && "${EDITOR_SERVER_MODEL}" != "${EDITOR_MODEL}" ]]; then
    stop_vllm_role editor-peer
  fi
}

assert_joint_checkpoint(){
  local role=$1 root=$2 step=$3 allow_evaluation=${4:-0}
  python3 - "$role" "$root" "$step" "$N_GPUS" "$allow_evaluation" <<'PY_JOINT_CKPT'
import json
import sys
from pathlib import Path

role, root, step_text, world_text, allow_evaluation_text = sys.argv[1:]
step = int(step_text)
world = int(world_text)
allow_evaluation = allow_evaluation_text == "1"
folder = Path(root) / f"global_step_{step}"
actor = folder / "actor"
hf = actor / "huggingface"
evaluation_marker = folder / ".mas_loop_evaluation_checkpoint.json"
if allow_evaluation and evaluation_marker.is_file():
    payload = json.loads(evaluation_marker.read_text(encoding="utf-8"))
    if payload.get("protocol") != "mas-loop-evaluation-checkpoint-v1" or int(payload.get("global_step", -1)) != step:
        raise SystemExit(f"invalid compacted {role} checkpoint marker at step {step}: {payload}")
    weights = list(hf.glob("*.safetensors")) + list(hf.glob("pytorch_model*.bin"))
    if not (hf / "config.json").is_file() or not weights:
        raise SystemExit(f"incomplete compacted {role} checkpoint at step {step}: {hf}")
    print(folder)
    raise SystemExit(0)
marker = folder / ".mas_loop_checkpoint_complete.json"
required = [folder / "data.pt", hf / "config.json", marker]
missing = [str(path) for path in required if not path.is_file()]
for prefix in ("model", "optim", "extra_state"):
    shards = sorted(actor.glob(f"{prefix}_world_size_{world}_rank_*.pt"))
    if len(shards) != world:
        missing.append(f"{actor}/{prefix}_world_size_{world}_rank_*.pt expected={world} actual={len(shards)}")
hf_weights = list((actor / "huggingface").glob("*.safetensors")) + list(
    (actor / "huggingface").glob("pytorch_model*.bin")
)
if not hf_weights:
    missing.append(f"{actor}/huggingface model weights")
if missing:
    raise SystemExit(f"incomplete {role} checkpoint at step {step}: " + "; ".join(missing))
payload = json.loads(marker.read_text(encoding="utf-8"))
if int(payload.get("global_step", -1)) != step:
    raise SystemExit(f"checkpoint marker step mismatch for {role}: {payload}")
print(folder)
PY_JOINT_CKPT
}

assert_latest_joint_checkpoint(){
  local role=$1 root=$2 expected_step=$3 latest
  latest=$(latest_hf_step "$root")
  latest=${latest:--1}
  if (( latest != expected_step )); then
    echo "[ERROR] ${role} trainer crossed the joint-step barrier: expected latest=${expected_step}, actual=${latest}" >&2
    return 2
  fi
}

prepare_joint_resume_checkpoint(){
  local role=$1 root=$2 expected_previous=$3 target_step=$4
  if (( target_step != expected_previous + 1 )); then
    echo "[ERROR] invalid ${role} joint-step transition: previous=${expected_previous}, target=${target_step}" >&2
    return 2
  fi

  # Never let a one-step trainer silently restart from global step 0 or
  # overwrite an existing target. The HF directory alone is not a sufficient
  # resume signal: Verl also consumes latest_checkpointed_iteration.txt.
  if (( expected_previous > 0 )); then
    assert_joint_checkpoint "$role" "$root" "$expected_previous" >/dev/null || return $?
  fi

  python3 - "$role" "$root" "$expected_previous" "$target_step" <<'PY_JOINT_RESUME_PREP'
import os
import re
import sys
from pathlib import Path

role, root_text, previous_text, target_text = sys.argv[1:]
root = Path(root_text)
previous = int(previous_text)
target = int(target_text)
root.mkdir(parents=True, exist_ok=True)

steps = []
for path in root.glob("global_step_*"):
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match:
        steps.append(int(match.group(1)))

target_dir = root / f"global_step_{target}"
if target_dir.exists():
    raise SystemExit(
        f"refusing to overwrite existing {role} target checkpoint: {target_dir}"
    )

if previous == 0:
    if steps:
        raise SystemExit(
            f"dirty {role} step-1 root {root}: existing checkpoint steps={sorted(steps)}"
        )
    pointer = root / "latest_checkpointed_iteration.txt"
    if pointer.exists():
        raw = pointer.read_text(encoding="utf-8").strip()
        if raw not in ("", "0"):
            raise SystemExit(
                f"dirty {role} step-1 resume pointer {pointer}: value={raw!r}"
            )
        pointer.unlink()
    print(f"prepared clean {role} step 1 root: {root}")
    raise SystemExit(0)

if not steps or max(steps) != previous:
    raise SystemExit(
        f"unsafe {role} resume root {root}: expected latest directory step "
        f"{previous}, found {sorted(steps)}"
    )

pointer = root / "latest_checkpointed_iteration.txt"
tmp = root / f".{pointer.name}.tmp.{os.getpid()}"
with tmp.open("w", encoding="utf-8") as handle:
    handle.write(f"{previous}\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, pointer)
dir_fd = os.open(root, os.O_RDONLY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)

actual = pointer.read_text(encoding="utf-8").strip()
if actual != str(previous):
    raise SystemExit(
        f"failed to synchronize {role} resume pointer: expected={previous}, actual={actual!r}"
    )
print(f"prepared {role} resume: previous={previous}, target={target}, root={root}")
PY_JOINT_RESUME_PREP
}

write_joint_step_manifest(){
  local round=$1 step=$2 total_steps=$3 orchestra_root=$4 editor_root=$5 replay_dir=$6
  local manifest_dir="${MAS_LOOP_CKPT_ROOT}/round_${round}/joint_steps"
  mkdir -p "${manifest_dir}"
  python3 - "$manifest_dir" "$round" "$step" "$total_steps" "$orchestra_root" "$editor_root" "$replay_dir" "${CURRENT_MANIFEST_GROUPS:-$TRAIN_BATCH_SIZE}" "${CURRENT_MANIFEST_ROWS:-$EDITOR_TRAIN_BATCH_SIZE}" "$ROLLOUT_N" "$ALL_GPUS" "$RUN_DIR" "${DATA_PREP_SEED:-0}" "$ORCHESTRA_MODEL" "$EDITOR_MODEL" <<'PY_JOINT_MANIFEST'
import json
import os
import sys
import time
from pathlib import Path

(manifest_dir, round_text, step_text, total_text, orchestra_root, editor_root, replay_dir,
 groups_text, rows_text, group_size_text, gpu_namespace, run_dir, seed_text,
 initial_orchestra_model, initial_editor_model) = sys.argv[1:]
step = int(step_text)
rollout_orchestra = initial_orchestra_model if step == 1 else str(Path(orchestra_root) / f"global_step_{step - 1}")
rollout_editor = initial_editor_model if step == 1 else str(Path(editor_root) / f"global_step_{step - 1}")
payload = {
    "protocol": "mas-loop-source-adversarial-joint-step-v3",
    "state": "complete",
    "round": int(round_text),
    "joint_step": step,
    "total_steps": int(total_text),
    "rollout_orchestra_checkpoint": rollout_orchestra,
    "rollout_editor_checkpoint": rollout_editor,
    "resulting_orchestra_checkpoint": str(Path(orchestra_root) / f"global_step_{step}"),
    "resulting_editor_checkpoint": str(Path(editor_root) / f"global_step_{step}"),
    # Backward-compatible aliases for existing resume/audit tooling.
    "orchestra_checkpoint": str(Path(orchestra_root) / f"global_step_{step}"),
    "editor_checkpoint": str(Path(editor_root) / f"global_step_{step}"),
    "editor_replay_dir": replay_dir,
    "task_group_size": int(group_size_text),
    "gpu_namespace": gpu_namespace,
    "run_dir": run_dir,
    "data_seed": int(seed_text),
    "source_executions": int(rows_text),
    "updated_executions": int(rows_text),
    "orchestra_reward_version": "source_minus_refiner_exploit_v1",
    "refiner_reward_version": "updated_minus_source_v2",
    "structured_output_protocol": os.getenv("MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL", "xml"),
    "xml_protocol_version": "mas_loop_xml_v1",
    "patch_compiler_version": "mas_patch_intent_compiler_v1",
    "orchestra_adversarial_penalty": float(os.getenv("MAS_LOOP_ORCH_ADV_PENALTY", "0.3")),
    "canonical_group_count": int(groups_text),
    "canonical_pair_count": int(rows_text),
    "editor_replay_rows": int(rows_text),
    "completed_at_unix": time.time(),
}
out = Path(manifest_dir) / f"joint_step_{step}.json"
tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
with tmp.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, out)
print(out)
PY_JOINT_MANIFEST
}

assert_joint_step_manifest_complete(){
  local manifest=$1 round=$2 step=$3 total_steps=$4
  python3 - "$manifest" "$round" "$step" "$total_steps" <<'PY_JOINT_RESUME_MANIFEST'
import json
import sys
from pathlib import Path

path, round_text, step_text, total_text = sys.argv[1:]
payload = json.loads(Path(path).read_text(encoding="utf-8"))
expected = {
    "protocol": "mas-loop-source-adversarial-joint-step-v3",
    "state": "complete",
    "round": int(round_text),
    "joint_step": int(step_text),
    "total_steps": int(total_text),
}
mismatches = {
    key: {"expected": value, "actual": payload.get(key)}
    for key, value in expected.items()
    if payload.get(key) != value
}
if mismatches:
    raise SystemExit(f"invalid completed joint-step manifest {path}: {mismatches}")
PY_JOINT_RESUME_MANIFEST
}

validate_completed_joint_step_retention(){
  local orchestra_root=$1 editor_root=$2 step=$3 total_steps=$4 retained
  retained=$(python3 - "$orchestra_root" "$editor_root" "$step" "$total_steps" "$NUM_CHECKPOINTS" <<'PY_JOINT_RETENTION_VALIDATE'
import sys
from verl.trainer.ppo.checkpoint_retention import validate_completed_step_retention

roots = sys.argv[1:3]
step, total_steps, total_checkpoints = map(int, sys.argv[3:])
print("1" if validate_completed_step_retention(roots, step, total_steps, total_checkpoints) else "0")
PY_JOINT_RETENTION_VALIDATE
  )
  for role_root in "orchestra:${orchestra_root}" "editor:${editor_root}"; do
    local role=${role_root%%:*} root=${role_root#*:}
    if [[ -d "${root}/global_step_${step}" ]]; then
      assert_joint_checkpoint "$role" "$root" "$step" 1 >/dev/null
    elif [[ "$retained" == "1" ]]; then
      echo "[ERROR] retained ${role} checkpoint is missing at step ${step}" >&2
      return 2
    fi
  done
}

run_joint_training_round(){
  local round=$1; shift
  local orchestra_root editor_root canonical_logs total_steps step replay_dir latest_o latest_e manifest completed_this_run=0 arg attempt_start
  local replay_groups replay_rows editor_ppo_mini editor_ppo_micro editor_log_micro per_gpu_replay
  local -a phase_overrides=() replay_cardinality_args=()
  # The joint-step scheduler owns these Hydra keys. Keep user arguments as a
  # fixed argv array and drop conflicting duplicates instead of using eval or
  # shell interpolation.
  for arg in "$@"; do
    case "${arg}" in
      trainer.total_training_steps=*|trainer.total_epochs=*|+trainer.max_steps_this_run=*|trainer.num_checkpoints=*|trainer.resume_mode=*)
        echo "[MAS-Loop] ignoring scheduler-owned override in joint mode: ${arg%%=*}"
        ;;
      *) phase_overrides+=("${arg}") ;;
    esac
  done
  orchestra_root=$(prepare_phase_ckpt_dir "$round" orchestra)
  editor_root=$(prepare_phase_ckpt_dir "$round" editor)
  total_steps=$(parquet_training_steps "${DATA_DIR}/orchestra_train.parquet" "${TRAIN_BATCH_SIZE}" "${TOTAL_EPOCHS}")
  if (( total_steps <= 0 )); then
    echo "[ERROR] joint training requires at least one optimizer step" >&2
    return 2
  fi
  echo "[MAS-Loop] strict joint-step schedule: total_steps=${total_steps}, rollout.n=${ROLLOUT_N}, orchestra_task_groups=${TRAIN_BATCH_SIZE}, editor_task_groups=${EDITOR_TASK_GROUPS_PER_STEP}, pair_rows=${EDITOR_TRAIN_BATCH_SIZE}"

  for step in $(seq 1 "${total_steps}"); do
    echo "[MAS-Loop] joint step ${step}/${total_steps}: Orchestra update -> paired Editor update"
    manifest="${MAS_LOOP_CKPT_ROOT}/round_${round}/joint_steps/joint_step_${step}.json"
    if [[ -f "${manifest}" ]]; then
      assert_joint_step_manifest_complete "${manifest}" "${round}" "${step}" "${total_steps}"
      validate_completed_joint_step_retention "$orchestra_root" "$editor_root" "$step" "$total_steps"
      echo "[MAS-Loop] joint step ${step}/${total_steps} already complete; its manifest and retention state are valid."
      continue
    fi
    current_groups=${TRAIN_BATCH_SIZE}
    if [[ "${TRAIN_DROP_LAST}" == "False" && "${step}" == "${total_steps}" ]]; then
      current_groups=$(python3 - "${DATA_DIR}/orchestra_train.parquet" "${TRAIN_BATCH_SIZE}" <<'PY_LAST_BATCH'
import sys, pyarrow.parquet as pq
rows=pq.ParquetFile(sys.argv[1]).metadata.num_rows; batch=int(sys.argv[2]); print(rows % batch or batch)
PY_LAST_BATCH
      )
    fi
    current_editor_rows=$((current_groups * ROLLOUT_N))
    record_joint_timing_event joint_step_start "${round}" "${step}"
    latest_o=$(latest_hf_step "$orchestra_root")
    latest_e=$(latest_hf_step "$editor_root")
    latest_o=${latest_o:--1}; latest_e=${latest_e:--1}
    if (( latest_o > step || latest_e > step )); then
      echo "[ERROR] checkpoint roots are ahead of requested joint step ${step}: orchestra=${latest_o}, editor=${latest_e}" >&2
      return 2
    fi

    if (( latest_o < step )); then
      prepare_joint_resume_checkpoint orchestra "$orchestra_root" "$((step - 1))" "$step"
      record_joint_timing_event orchestra_start "${round}" "${step}"
      EDITOR_DATA_FROM_ORCH_LOGS=0 run_orchestra_phase "$round" ${phase_overrides[@]+"${phase_overrides[@]}"} \
        trainer.total_training_steps="${total_steps}" trainer.total_epochs="${TOTAL_EPOCHS}" \
        +trainer.max_steps_this_run=1 trainer.num_checkpoints="${total_steps}" trainer.resume_mode=auto \
        actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
        actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']"
      record_joint_timing_event orchestra_complete "${round}" "${step}"
    fi
    assert_latest_joint_checkpoint orchestra "$orchestra_root" "$step"
    assert_joint_checkpoint orchestra "$orchestra_root" "$step" >/dev/null
    maybe_update_model_from_ckpt orchestra "$orchestra_root"

    canonical_logs="${orchestra_root}/rollouts/canonical_rollouts"
    replay_dir="${DATA_DIR}/joint_step_${step}"
    record_joint_timing_event replay_build_start "${round}" "${step}"
    attempt_start=$(latest_orchestra_attempt_start "${round}" "${step}")
    replay_cardinality_args=()
    if [[ "${ALLOW_INCOMPLETE_EDITOR_REPLAY}" == "False" ]]; then
      replay_cardinality_args=(--expected-groups "${current_groups}" --expected-rows "${current_editor_rows}")
    fi
    python3 -m mas_loop_rl.build_editor_data_from_logs \
      --logs "${canonical_logs}" --output-dir "${replay_dir}" \
      --group-size "${EDITOR_GROUP_SIZE}" --optimizer-step "${step}" \
      --min-record-time "${attempt_start}" \
      ${replay_cardinality_args[@]+"${replay_cardinality_args[@]}"} \
      --val-ratio 0 --pad-train-to-multiple 0
    read -r replay_groups replay_rows < <(python3 - "${replay_dir}/editor_replay_audit.json" "${current_groups}" "${current_editor_rows}" "${EDITOR_GROUP_SIZE}" "${ALLOW_INCOMPLETE_EDITOR_REPLAY}" <<'PY_REPLAY_CARDINALITY'
import json
import sys
from pathlib import Path

audit_path, expected_groups_text, expected_rows_text, group_size_text, allow_incomplete_text = sys.argv[1:]
payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
groups = int(payload.get("final_groups", 0))
rows = int(payload.get("final_rows", 0))
expected_groups = int(expected_groups_text)
expected_rows = int(expected_rows_text)
group_size = int(group_size_text)
allow_incomplete = allow_incomplete_text == "True"
if groups <= 0 or rows <= 0 or rows != groups * group_size:
    raise SystemExit(f"invalid Editor replay cardinality: groups={groups} rows={rows} group_size={group_size}")
if groups > expected_groups or rows > expected_rows:
    raise SystemExit(
        f"Editor replay exceeds source batch: groups={groups}/{expected_groups} rows={rows}/{expected_rows}"
    )
if not allow_incomplete and (groups != expected_groups or rows != expected_rows):
    raise SystemExit(
        f"strict Editor replay mismatch: groups={groups}/{expected_groups} rows={rows}/{expected_rows}"
    )
if bool(payload.get("exact_target_padding_enabled")) or int(payload.get("padded_rows", 0)) != 0:
    raise SystemExit(f"no-padding replay unexpectedly contains duplicates: {payload}")
print(groups, rows)
PY_REPLAY_CARDINALITY
    )
    read -r editor_ppo_mini editor_ppo_micro editor_log_micro per_gpu_replay < <(python3 - "${replay_rows}" "${N_GPUS}" "${EDITOR_GROUP_SIZE}" "${PPO_MINI_BATCH_SIZE}" "${PPO_MICRO_BATCH_SIZE_PER_GPU}" "${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" <<'PY_EDITOR_BATCHES'
import math
import sys

rows, world, group_size, configured_mini, configured_ppo_micro, configured_log_micro = map(int, sys.argv[1:])
if rows % world:
    raise SystemExit(f"Editor replay rows={rows} must be divisible by world_size={world}")
per_gpu = rows // world
mini_candidates = [
    value
    for value in range(min(configured_mini, rows), group_size - 1, -1)
    if value % group_size == 0 and rows % value == 0
]
if not mini_candidates:
    raise SystemExit(
        f"no group-aligned Editor PPO mini batch divides rows={rows}; group_size={group_size}"
    )
mini = mini_candidates[0]
ppo_micro = math.gcd(configured_ppo_micro, per_gpu)
log_micro = math.gcd(configured_log_micro, per_gpu)
if ppo_micro <= 0 or log_micro <= 0:
    raise SystemExit("invalid Editor micro-batch derivation")
print(mini, ppo_micro, log_micro, per_gpu)
PY_EDITOR_BATCHES
    )
    echo "[MAS-Loop] Editor replay step ${step}: complete_groups=${replay_groups}/${current_groups}, rows=${replay_rows}/${current_editor_rows}, no_padding=1, ppo_mini=${editor_ppo_mini}, ppo_micro_per_gpu=${editor_ppo_micro}, log_micro_per_gpu=${editor_log_micro}, per_gpu_rows=${per_gpu_replay}"
    record_joint_timing_event replay_build_complete "${round}" "${step}"

    latest_e=$(latest_hf_step "$editor_root"); latest_e=${latest_e:--1}
    if (( latest_e < step )); then
      prepare_joint_resume_checkpoint editor "$editor_root" "$((step - 1))" "$step"
      local saved_data_dir="${DATA_DIR}"
      DATA_DIR="${replay_dir}"
      record_joint_timing_event editor_start "${round}" "${step}"
      CURRENT_EDITOR_TRAIN_BATCH_SIZE="${replay_rows}" \
      CURRENT_EDITOR_PPO_MINI_BATCH_SIZE="${editor_ppo_mini}" \
      CURRENT_EDITOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${editor_ppo_micro}" \
      CURRENT_EDITOR_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${editor_log_micro}" \
      MAS_LOOP_SKIP_INITIAL_ROLLOUT_WEIGHT_SYNC=1 \
      MAS_LOOP_SKIP_POST_STEP_ROLLOUT_WEIGHT_SYNC=1 \
      run_editor_phase "$round" ${phase_overrides[@]+"${phase_overrides[@]}"} \
        trainer.total_training_steps="${total_steps}" trainer.total_epochs="${total_steps}" \
        +trainer.max_steps_this_run=1 trainer.num_checkpoints="${total_steps}" trainer.resume_mode=auto \
        actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
        actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']"
      DATA_DIR="${saved_data_dir}"
      record_joint_timing_event editor_complete "${round}" "${step}"
    fi
    assert_latest_joint_checkpoint editor "$editor_root" "$step"
    assert_joint_checkpoint editor "$editor_root" "$step" >/dev/null
    maybe_update_model_from_ckpt editor "$editor_root"
    CURRENT_MANIFEST_GROUPS="${replay_groups}" CURRENT_MANIFEST_ROWS="${replay_rows}" write_joint_step_manifest "$round" "$step" "$total_steps" "$orchestra_root" "$editor_root" "$replay_dir"
    # Security: checkpoint roots and numeric bounds are passed as argv to a Python
    # retention helper that validates generated paths before bounded deletion.
    python3 -m verl.trainer.ppo.checkpoint_retention \
      --role-root "$orchestra_root" --role-root "$editor_root" \
      --completed-step "$step" --total-steps "$total_steps" \
      --total-checkpoints "$NUM_CHECKPOINTS"
    record_joint_timing_event joint_step_complete "${round}" "${step}"
    echo "[MAS-Loop] joint step ${step}/${total_steps} complete; next rollout will use O_${step} + E_${step}."
    completed_this_run=$((completed_this_run + 1))
    if (( MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN > 0 && completed_this_run >= MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN )); then
      echo "[MAS-Loop] reached MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN=${MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN}; stopping cleanly after joint step ${step}."
      break
    fi
  done
}

for round in $(seq 1 "${ADVERSARIAL_ROUNDS}"); do
  if [[ "${JOINT_STEP_TRAINING}" == "1" && "${RUN_ORCHESTRA}" == "1" && "${RUN_EDITOR}" == "1" ]]; then
    run_joint_training_round "$round" "$@"
  else
    if [[ "${RUN_ORCHESTRA}" == "1" ]]; then run_orchestra_phase "$round" "$@"; fi
    if [[ "${RUN_EDITOR}" == "1" ]]; then run_editor_phase "$round" "$@"; fi
  fi
done
