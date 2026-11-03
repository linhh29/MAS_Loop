#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ulimit -n 1048576 2>/dev/null || true

# Standalone MAS-Loop BCP evaluation. It starts one TP=8 Qwen3-30B-A3B judge
# on physical GPUs 8-15, scores canonical source/updated rollout pairs, and
# writes every judgement plus summary directly under the BCP checkpoint tree.
BCP_EVAL_MAS_ROOT=${BCP_EVAL_MAS_ROOT:-/path/to/MAS_Loop}
BCP_EVAL_VERL_ROOT=${BCP_EVAL_VERL_ROOT:-${BCP_EVAL_MAS_ROOT}/verl}
BCP_EVAL_GPUS=${BCP_EVAL_GPUS:-8,9,10,11,12,13,14,15}
BCP_EVAL_INPUT=${BCP_EVAL_INPUT:-}
BCP_EVAL_TIMESTAMP=${BCP_EVAL_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
BCP_EVAL_OUTPUT_DIR=${BCP_EVAL_OUTPUT_DIR:-/path/to/checkpoints/bcp/eval/${BCP_EVAL_TIMESTAMP}}
BCP_EVAL_JUDGE_MODEL=${BCP_EVAL_JUDGE_MODEL:-/path/to/models/Qwen3-30B-A3B}
BCP_EVAL_JUDGE_PORT=${BCP_EVAL_JUDGE_PORT:-8513}
BCP_EVAL_JUDGE_GPU_MEMORY_UTILIZATION=${BCP_EVAL_JUDGE_GPU_MEMORY_UTILIZATION:-0.78}
BCP_EVAL_JUDGE_MAX_NUM_SEQS=${BCP_EVAL_JUDGE_MAX_NUM_SEQS:-128}
BCP_EVAL_JUDGE_MAX_NUM_BATCHED_TOKENS=${BCP_EVAL_JUDGE_MAX_NUM_BATCHED_TOKENS:-49152}
BCP_EVAL_JUDGE_MAX_MODEL_LEN=${BCP_EVAL_JUDGE_MAX_MODEL_LEN:-23192}
BCP_EVAL_JUDGE_MAX_TOKENS=${BCP_EVAL_JUDGE_MAX_TOKENS:-4096}
BCP_EVAL_JUDGE_MAX_WORKERS=${BCP_EVAL_JUDGE_MAX_WORKERS:-128}
BCP_EVAL_JUDGE_MAX_ACTIVE=${BCP_EVAL_JUDGE_MAX_ACTIVE:-128}
BCP_EVAL_JUDGE_HTTP_POOL_MAXSIZE=${BCP_EVAL_JUDGE_HTTP_POOL_MAXSIZE:-128}
BCP_EVAL_WAIT_INTERVAL_SEC=${BCP_EVAL_WAIT_INTERVAL_SEC:-15}
BCP_EVAL_STABLE_CHECKS=${BCP_EVAL_STABLE_CHECKS:-2}
BCP_EVAL_WAIT_TIMEOUT_SEC=${BCP_EVAL_WAIT_TIMEOUT_SEC:-0}
BCP_EVAL_CONTROL_DIR=${BCP_EVAL_CONTROL_DIR:-/tmp/mas_loop_bcp_eval_${BCP_EVAL_TIMESTAMP}}

[[ "${BCP_EVAL_MAS_ROOT}" == "/path/to/MAS_Loop" ]] || { echo "[BCP-EVAL] wrong MAS root" >&2; exit 2; }
[[ "${BCP_EVAL_GPUS}" == "8,9,10,11,12,13,14,15" ]] || { echo "[BCP-EVAL] only physical GPUs 8-15 are allowed" >&2; exit 2; }
[[ -n "${BCP_EVAL_INPUT}" ]] || { echo "[BCP-EVAL] set BCP_EVAL_INPUT to a canonical_rollouts directory/file/glob" >&2; exit 2; }
[[ -f "${BCP_EVAL_JUDGE_MODEL}/config.json" ]] || { echo "[BCP-EVAL] missing judge model: ${BCP_EVAL_JUDGE_MODEL}" >&2; exit 2; }
(( BCP_EVAL_JUDGE_MAX_TOKENS == 4096 )) || { echo "[BCP-EVAL] judge max output tokens must remain 4096" >&2; exit 2; }
mkdir -p "${BCP_EVAL_OUTPUT_DIR}" "${BCP_EVAL_CONTROL_DIR}"
MAS_ROOT="${BCP_EVAL_MAS_ROOT}" VERL_ROOT="${BCP_EVAL_VERL_ROOT}" bash "${BCP_EVAL_VERL_ROOT}/scripts/bootstrap_mas_loop_v1_deps.sh"

cat > "${BCP_EVAL_OUTPUT_DIR}/launch_config.env" <<CFG
BCP_EVAL_GPUS=${BCP_EVAL_GPUS}
BCP_EVAL_INPUT=${BCP_EVAL_INPUT}
BCP_EVAL_OUTPUT_DIR=${BCP_EVAL_OUTPUT_DIR}
BCP_EVAL_JUDGE_MODEL=${BCP_EVAL_JUDGE_MODEL}
BCP_EVAL_JUDGE_PORT=${BCP_EVAL_JUDGE_PORT}
BCP_EVAL_JUDGE_GPU_MEMORY_UTILIZATION=${BCP_EVAL_JUDGE_GPU_MEMORY_UTILIZATION}
BCP_EVAL_JUDGE_MAX_NUM_SEQS=${BCP_EVAL_JUDGE_MAX_NUM_SEQS}
BCP_EVAL_JUDGE_MAX_NUM_BATCHED_TOKENS=${BCP_EVAL_JUDGE_MAX_NUM_BATCHED_TOKENS}
BCP_EVAL_JUDGE_MAX_MODEL_LEN=${BCP_EVAL_JUDGE_MAX_MODEL_LEN}
BCP_EVAL_JUDGE_MAX_TOKENS=${BCP_EVAL_JUDGE_MAX_TOKENS}
BCP_EVAL_JUDGE_MAX_WORKERS=${BCP_EVAL_JUDGE_MAX_WORKERS}
CFG

if [[ "${BCP_EVAL_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "[BCP-EVAL] config-only validation passed; no GPU service started."
  exit 0
fi

PYTHONPATH="${MAS_LOOP_RUNTIME_DEPS_DIR:-${BCP_EVAL_VERL_ROOT}/.mas_loop_runtime_deps}:${BCP_EVAL_VERL_ROOT}:${BCP_EVAL_MAS_ROOT}:${PYTHONPATH:-}" \
python3 -m mas_loop_rl.wait_for_gpu_isolation \
  --gpus "${BCP_EVAL_GPUS}" --interval "${BCP_EVAL_WAIT_INTERVAL_SEC}" \
  --stable-checks "${BCP_EVAL_STABLE_CHECKS}" --timeout "${BCP_EVAL_WAIT_TIMEOUT_SEC}" \
  | tee "${BCP_EVAL_OUTPUT_DIR}/gpu_isolation.log"

judge_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${judge_pid}" ]]; then
    kill -TERM "${judge_pid}" 2>/dev/null || true
    for _ in {1..30}; do kill -0 "${judge_pid}" 2>/dev/null || break; sleep 1; done
    kill -KILL "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${BCP_EVAL_GPUS}" PYTHONPATH="${MAS_LOOP_RUNTIME_DEPS_DIR:-${BCP_EVAL_VERL_ROOT}/.mas_loop_runtime_deps}:${BCP_EVAL_VERL_ROOT}:${BCP_EVAL_MAS_ROOT}:${PYTHONPATH:-}" \
python3 -m mas_loop_rl.vllm_api_server \
  --model "${BCP_EVAL_JUDGE_MODEL}" --served-model-name judge --host 127.0.0.1 --port "${BCP_EVAL_JUDGE_PORT}" \
  --tensor-parallel-size 8 --gpu-memory-utilization "${BCP_EVAL_JUDGE_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${BCP_EVAL_JUDGE_MAX_NUM_SEQS}" --max-num-batched-tokens "${BCP_EVAL_JUDGE_MAX_NUM_BATCHED_TOKENS}" \
  --max-model-len "${BCP_EVAL_JUDGE_MAX_MODEL_LEN}" --max-num-partial-prefills 1 --max-long-partial-prefills 1 \
  --long-prefill-token-threshold 1024 --enable-sleep-mode --enforce-eager \
  > "${BCP_EVAL_OUTPUT_DIR}/judge_vllm.log" 2>&1 &
judge_pid=$!
for _ in {1..180}; do
  if curl -fsS "http://127.0.0.1:${BCP_EVAL_JUDGE_PORT}/v1/models" >/dev/null 2>&1; then break; fi
  if ! kill -0 "${judge_pid}" 2>/dev/null; then tail -n 100 "${BCP_EVAL_OUTPUT_DIR}/judge_vllm.log" >&2; exit 1; fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${BCP_EVAL_JUDGE_PORT}/v1/models" >/dev/null
curl -fsS -X POST "http://127.0.0.1:${BCP_EVAL_JUDGE_PORT}/sleep?level=1" >/dev/null

MAS_LOOP_VLLM_SLEEP=1 MAS_LOOP_PHASE_LEASE=0 MAS_LOOP_USE_LLM_JUDGE=1 \
MAS_LOOP_VLLM_SLEEP_LOCK="${BCP_EVAL_CONTROL_DIR}/judge.lock" MAS_LOOP_VLLM_SLEEP_STATE="${BCP_EVAL_CONTROL_DIR}/state.json" \
MAS_LOOP_JUDGE_BASE_URL="http://127.0.0.1:${BCP_EVAL_JUDGE_PORT}/v1" MAS_LOOP_JUDGE_MODEL=judge \
MAS_LOOP_JUDGE_MAX_MODEL_LEN="${BCP_EVAL_JUDGE_MAX_MODEL_LEN}" MAS_LOOP_JUDGE_MAX_TOKENS="${BCP_EVAL_JUDGE_MAX_TOKENS}" \
MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE="${BCP_EVAL_JUDGE_MAX_ACTIVE}" MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE="${BCP_EVAL_JUDGE_HTTP_POOL_MAXSIZE}" \
PYTHONPATH="${MAS_LOOP_RUNTIME_DEPS_DIR:-${BCP_EVAL_VERL_ROOT}/.mas_loop_runtime_deps}:${BCP_EVAL_VERL_ROOT}:${BCP_EVAL_MAS_ROOT}:${PYTHONPATH:-}" \
python3 -m mas_loop_rl.eval_bcp_with_judge \
  --input "${BCP_EVAL_INPUT}" --output-dir "${BCP_EVAL_OUTPUT_DIR}" \
  --judge-base-url "http://127.0.0.1:${BCP_EVAL_JUDGE_PORT}/v1" --judge-model judge \
  --max-model-len "${BCP_EVAL_JUDGE_MAX_MODEL_LEN}" --max-tokens "${BCP_EVAL_JUDGE_MAX_TOKENS}" \
  --max-workers "${BCP_EVAL_JUDGE_MAX_WORKERS}" "$@" \
  2>&1 | tee "${BCP_EVAL_OUTPUT_DIR}/evaluation.log"
