#!/usr/bin/env bash
set -xeuo pipefail

# Disable core dumps: large vLLM/Ray crashes must not fill /tmp with core files.
ulimit -c 0
ulimit -n 1048576 2>/dev/null || true
# DeepScaleR (math) MAS-Loop adversarial GRPO launcher.
# Only uses this repository's ./verl code. All local vLLM services use GPUs 8-15
# together with tensor-parallel=8 and vLLM sleep mode; the active model is woken
# on demand by mas_loop_rl.local_client and put back to sleep when idle.
#
# Differences vs run_bcp_mas_loop_adversarial_grpo.sh:
#   * MAS_LOOP_RETRIEVER=none  -> executor.py skips BM25 (pure math, no corpus).
#   * DeepScaleR rows are {problem, answer, solution}; prepare_data routes them
#     to a math branch and executor.quality_score uses \boxed{} + sympy equality.
#   * Qwen3 hidden thinking is disabled for helper OpenAI calls; otherwise tiny
#     math sub-agent/refiner requests emit long <think> traces, balloon prompt
#     context, and make the reward loop look hung.
# RL hyper-parameters are aligned with the BCP training table:
#   train_batch=64, ppo_mini=64 (<=train_batch), ppo_micro=2, logprob_micro=4,
#   max_prompt=15000, max_response=8192, rollout_n=8 with one-to-one Orchestra/Editor pairs, max_concurrency=128.

MAS_ROOT=${MAS_ROOT:-/path/to/MAS_Loop}
VERL_ROOT=${VERL_ROOT:-${MAS_ROOT}/verl}
export VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
# vLLM V1 TP multiproc execute_model has a separate seconds timeout (default 300s).
# Keep long-context DeepScaleR rollouts from being killed as EngineDeadError.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}
# Also enlarge the PyTorch ProcessGroupNCCL watchdog default used by vLLM stateless TP groups.
export MAS_LOOP_PATCH_TORCH_DIST_TIMEOUT=${MAS_LOOP_PATCH_TORCH_DIST_TIMEOUT:-0}
export MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS=${MAS_LOOP_TORCH_DIST_TIMEOUT_SECONDS:-600}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-120000}
export PYTHONPATH="${MAS_LOOP_RUNTIME_DEPS_DIR:-${VERL_ROOT}/.mas_loop_runtime_deps}:${VERL_ROOT}:${MAS_ROOT}:${PYTHONPATH:-}"

export ALL_GPUS=${ALL_GPUS:-8,9,10,11,12,13,14,15}
TASK_JSONL=${TASK_JSONL:-${MAS_ROOT}/datasets/DeepScaleR/deepscaler_train.json}
RUN_DIR=${RUN_DIR:-/path/to/checkpoints/benchmarks/mas_loop_grpo_deepscaler}
DATA_DIR=${DATA_DIR:-${RUN_DIR}/data}
LOG_DIR=${LOG_DIR:-${RUN_DIR}/logs}
mkdir -p "${DATA_DIR}" "${LOG_DIR}"

# Pure math: no retrieval. executor.py only activates BM25 for MAS_LOOP_RETRIEVER in {bm25,...}.
export MAS_LOOP_RETRIEVER=${MAS_LOOP_RETRIEVER:-none}

# Smoke/debug can set MAX_TRAIN_ROWS=64. Full DeepScaleR training keeps 0 (=all 40315 rows).
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAS_LOOP_DATASET_NAME=${MAS_LOOP_DATASET_NAME:-deepscaler}

# Model: all three roles (orchestra/editor/subagent) hail from the same local base
# model (local Qwen3-8B under /path/to/models); the launcher updates
# orchestra/editor from their HF checkpoints each round.
MODEL_PATH=${MODEL_PATH:-/path/to/models/Qwen3-8B}
SUBAGENT_MODEL=${SUBAGENT_MODEL:-/path/to/models/Qwen3-30B-A3B}

# --- RL hyper-parameters aligned with the BCP training table ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
# Table says 256, but vanilla Verl requires ppo_mini_batch_size <= train_batch_size.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
export MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE=${MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE:-8}
EDITOR_GROUP_SIZE=${EDITOR_GROUP_SIZE:-${MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE}}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-15000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
ADVERSARIAL_ROUNDS=${ADVERSARIAL_ROUNDS:-1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
# Save every 250 optimizer steps and always save the mandatory final step.
# Each completed HuggingFace checkpoint is atomically published under
# MAS_LOOP_CKPT_ROOT; no secondary staging or upload is used.
TEST_FREQ=${TEST_FREQ:--1}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-10}
SAVE_FREQ=${SAVE_FREQ:-250}
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-1}
DATA_PREP_SEED=${DATA_PREP_SEED:-0}
DATA_VAL_RATIO=${DATA_VAL_RATIO:-0}
DATA_PAD_TO_MULTIPLE=${DATA_PAD_TO_MULTIPLE:-${TRAIN_BATCH_SIZE}}
export MAS_LOOP_MAX_CONCURRENCY=${MAS_LOOP_MAX_CONCURRENCY:-128}
export MAS_LOOP_DISABLE_THINKING=${MAS_LOOP_DISABLE_THINKING:-1}
export MAS_LOOP_EDITOR_MAX_TOKENS=${MAS_LOOP_EDITOR_MAX_TOKENS:-768}
export DATA_ENABLE_THINKING=${DATA_ENABLE_THINKING:-False}

# --- Exclusive-sleep defaults for 96GB cards (override via env) ---
# All serving engines use TP=8, but the arbiter keeps exactly one awake so the
# active engine can reserve 92% while the others offload/discard allocations.
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.90}
export EXTERNAL_VLLM_GPU_MEM_UTIL=${EXTERNAL_VLLM_GPU_MEM_UTIL:-0.82}
export PEER_GPU_MEMORY_UTILIZATION=${PEER_GPU_MEMORY_UTILIZATION:-${EXTERNAL_VLLM_GPU_MEM_UTIL}}
export SUB_AGENT_GPU_MEMORY_UTILIZATION=${SUB_AGENT_GPU_MEMORY_UTILIZATION:-${EXTERNAL_VLLM_GPU_MEM_UTIL}}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
export SUB_AGENT_MAX_NUM_SEQS=${SUB_AGENT_MAX_NUM_SEQS:-128}
export SUB_AGENT_MAX_MODEL_LEN=${SUB_AGENT_MAX_MODEL_LEN:-8192}
# Disable CUDA graph on every vLLM engine. The BCP run died at step 2 with
# "TimeoutError: RPC call to execute_model timed out" - a multiproc worker stall
# triggered by CUDA-graph re-capture across sleep/wake. enforce_eager removes it.
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-1}
export EXTERNAL_VLLM_ENFORCE_EAGER=${EXTERNAL_VLLM_ENFORCE_EAGER:-1}
# Use a distinct Ray dashboard port so a still-running BCP/Gate job is not disturbed.
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8325}
export RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/mlr}
export MAS_LOOP_PAIR_DIR=${MAS_LOOP_PAIR_DIR:-${LOG_DIR}/pairs}

python3 -m mas_loop_rl.prepare_data \
  --input "${TASK_JSONL}" \
  --output-dir "${DATA_DIR}" \
  --role both \
  --seed "${DATA_PREP_SEED}" \
  --val-ratio "${DATA_VAL_RATIO}" \
  --max-rows "${MAX_TRAIN_ROWS}" --pad-train-to-multiple "${DATA_PAD_TO_MULTIPLE}"

# Delegate the alternating adversarial GRPO phases to the generic MAS-Loop script.
MAS_ROOT="${MAS_ROOT}" VERL_ROOT="${VERL_ROOT}" ALL_GPUS="${ALL_GPUS}" \
TASK_JSONL="${TASK_JSONL}" RUN_DIR="${RUN_DIR}" DATA_DIR="${DATA_DIR}" LOG_DIR="${LOG_DIR}" \
MODEL_PATH="${MODEL_PATH}" ORCHESTRA_MODEL="${MODEL_PATH}" EDITOR_MODEL="${MODEL_PATH}" SUBAGENT_MODEL="${SUBAGENT_MODEL}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
ROLLOUT_N="${ROLLOUT_N}" MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE="${MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE}" EDITOR_GROUP_SIZE="${EDITOR_GROUP_SIZE}" PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
ADVERSARIAL_ROUNDS="${ADVERSARIAL_ROUNDS}" VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN}" NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" SAVE_FREQ="${SAVE_FREQ}" TEST_FREQ="${TEST_FREQ}" \
DATA_PREP_SEED="${DATA_PREP_SEED}" DATA_VAL_RATIO="${DATA_VAL_RATIO}" DATA_PAD_TO_MULTIPLE="${DATA_PAD_TO_MULTIPLE}" \
MAS_LOOP_DISABLE_THINKING="${MAS_LOOP_DISABLE_THINKING}" MAS_LOOP_EDITOR_MAX_TOKENS="${MAS_LOOP_EDITOR_MAX_TOKENS}" DATA_ENABLE_THINKING="${DATA_ENABLE_THINKING}" \
MAS_LOOP_MAX_CONCURRENCY="${MAS_LOOP_MAX_CONCURRENCY}" \
PEER_GPU_MEMORY_UTILIZATION="${PEER_GPU_MEMORY_UTILIZATION}" SUB_AGENT_GPU_MEMORY_UTILIZATION="${SUB_AGENT_GPU_MEMORY_UTILIZATION}" \
SUB_AGENT_MAX_NUM_SEQS="${SUB_AGENT_MAX_NUM_SEQS}" SUB_AGENT_MAX_MODEL_LEN="${SUB_AGENT_MAX_MODEL_LEN}" SKIP_DATA_PREP=1 \
"${VERL_ROOT}/scripts/run_mas_loop_adversarial_grpo.sh" "$@"
