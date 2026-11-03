#!/usr/bin/env bash
set -euo pipefail

ROOT=$(readlink -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")
BASE_LAUNCHER="${ROOT}/verl/scripts/run_hotpotqa_mas_loop_full.sh"
[[ -x "${BASE_LAUNCHER}" || -f "${BASE_LAUNCHER}" ]] || {
  echo "Missing base launcher: ${BASE_LAUNCHER}" >&2
  exit 2
}

if (( $# != 1 )); then
  echo "Usage: $0 {editor1024-gpu0-7|editor2048-gpu8-15}" >&2
  exit 2
fi

# The arm name is an explicit whitelist. Do not eval or interpolate arbitrary
# command text into the training command.
case "$1" in
  editor1024-gpu0-7)
    gpu_domain="0,1,2,3,4,5,6,7"
    arm_tag="gpu0-7_editor_tp8_concurrency1024"
    editor_max_active=1024
    editor_max_num_seqs=1024
    editor_http_pool=1024
    ;;
  editor2048-gpu8-15)
    gpu_domain="8,9,10,11,12,13,14,15"
    arm_tag="gpu8-15_editor_tp8_concurrency2048"
    editor_max_active=2048
    editor_max_num_seqs=2048
    editor_http_pool=2048
    ;;
  *)
    echo "Unknown A/B arm: $1" >&2
    echo "Usage: $0 {editor1024-gpu0-7|editor2048-gpu8-15}" >&2
    exit 2
    ;;
esac

base_ts=${HP_AB_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
run_root=${ROOT}/runs/hotpotqa/editor_scheduler_ab_${base_ts}/${arm_tag}

printf '[AB] arm=%s gpus=%s run_root=%s\n' "${arm_tag}" "${gpu_domain}" "${run_root}"
printf '[AB] fixed train_batch=512 rollout_n=16 editor_batch=8192 editor_max_output=8192 editor_tp=8 editor_dp=1 editor_batched_tokens_per_tp8_engine=196608\n'
printf '[AB] variable editor_max_active=%s editor_max_num_seqs_per_tp8_engine=%s editor_http_pool_per_process=%s\n' \
  "${editor_max_active}" "${editor_max_num_seqs}" "${editor_http_pool}"

exec env \
  HP_ALL_GPUS="${gpu_domain}" \
  HP_TIMESTAMP="${base_ts}_${arm_tag}" \
  HP_RUN_ROOT="${run_root}" \
  HP_MAX_TRAIN_ROWS=512 \
  HP_MAX_JOINT_STEPS_THIS_RUN=1 \
  HP_TOTAL_EPOCHS=1 \
  HP_ADVERSARIAL_ROUNDS=1 \
  HP_SAVE_FREQ=1 \
  HP_NUM_CHECKPOINTS=1 \
  HP_AGENT_LOOP_WORKER_PROCESSES=16 \
  HP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER=128 \
  HP_RAY_GENERATE_MAX_OUTSTANDING=2048 \
  HP_DISPATCH_WORKERS_PER_PROCESS=256 \
  HP_PEER_TENSOR_PARALLEL_SIZE=8 \
  HP_PEER_DATA_PARALLEL_SIZE=1 \
  HP_REFINER_VLLM_MAX_ACTIVE="${editor_max_active}" \
  HP_REFINER_MAX_NUM_SEQS="${editor_max_num_seqs}" \
  HP_REFINER_HTTP_POOL_MAXSIZE="${editor_http_pool}" \
  HP_REFINER_MAX_BATCHED_TOKENS=196608 \
  HP_REFINER_GPU_MEMORY_UTILIZATION=0.86 \
  HP_REFINER_CANDIDATE_CONCURRENCY=1 \
  HP_PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
  HP_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
  HP_ROLLOUT_MAX_NUM_SEQS=1024 \
  HP_ROLLOUT_MAX_BATCHED_TOKENS=196608 \
  HP_ROLLOUT_GPU_MEM_UTIL=0.88 \
  HP_ORCHESTRA_VLLM_MAX_ACTIVE=1024 \
  HP_SUBAGENT_VLLM_MAX_ACTIVE=1024 \
  HP_JUDGE_VLLM_MAX_ACTIVE=512 \
  bash "${BASE_LAUNCHER}"
