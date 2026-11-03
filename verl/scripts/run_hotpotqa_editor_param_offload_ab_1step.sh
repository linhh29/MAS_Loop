#!/usr/bin/env bash
set -euo pipefail

ROOT=$(readlink -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")
BASE_LAUNCHER="${ROOT}/verl/scripts/run_hotpotqa_mas_loop_full.sh"
[[ -x "${BASE_LAUNCHER}" || -f "${BASE_LAUNCHER}" ]] || {
  echo "Missing base launcher: ${BASE_LAUNCHER}" >&2
  exit 2
}

if (( $# != 1 )); then
  echo "Usage: $0 {editor-param-on-gpu0-7|editor-param-off-gpu8-15}" >&2
  exit 2
fi

# Security: Restrict experiment selection to fixed arm names so no user-provided
# command text or arbitrary environment assignment reaches the training shell.
case "$1" in
  editor-param-on-gpu0-7)
    gpu_domain="0,1,2,3,4,5,6,7"
    editor_actor_param_offload=True
    arm_tag="gpu0-7_editor_tp8_c1024_paramon"
    ;;
  editor-param-off-gpu8-15)
    gpu_domain="8,9,10,11,12,13,14,15"
    editor_actor_param_offload=False
    arm_tag="gpu8-15_editor_tp8_c1024_paramoff"
    ;;
  *)
    echo "Unknown A/B arm: $1" >&2
    echo "Usage: $0 {editor-param-on-gpu0-7|editor-param-off-gpu8-15}" >&2
    exit 2
    ;;
esac

base_ts=${HP_AB_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
[[ "${base_ts}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
  echo "HP_AB_TIMESTAMP must match YYYYMMDD_HHMMSS, got: ${base_ts}" >&2
  exit 2
}
run_root=${ROOT}/runs/hotpotqa/editor_param_offload_ab_${base_ts}/${arm_tag}

printf '[AB] arm=%s gpus=%s run_root=%s\n' "${arm_tag}" "${gpu_domain}" "${run_root}"
printf '[AB] fixed orchestra_batch=512 rollout_n=16 editor_batch=8192 editor_max_output=8192 editor_tp=8 editor_dp=1 editor_concurrency=1024 editor_batched_tokens_per_tp8_engine=196608 prefix_cache=disabled\n'
printf '[AB] fixed mini_batch=256 micro_batch_per_gpu=8 log_prob_micro_batch_per_gpu=8 ppo/log_prob_token_cap_per_gpu=32768\n'
printf '[AB] fixed orchestra_offload actor_param=True actor_optimizer=True ref_param=True; editor actor_optimizer=True ref_param=True\n'
printf '[AB] variable editor_actor_param_offload=%s\n' "${editor_actor_param_offload}"

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
  HP_REFINER_VLLM_MAX_ACTIVE=1024 \
  HP_REFINER_MAX_NUM_SEQS=1024 \
  HP_REFINER_HTTP_POOL_MAXSIZE=1024 \
  HP_REFINER_MAX_BATCHED_TOKENS=196608 \
  HP_REFINER_GPU_MEMORY_UTILIZATION=0.86 \
  HP_REFINER_CANDIDATE_CONCURRENCY=1 \
  HP_PPO_MINI_BATCH_SIZE=256 \
  HP_PPO_MICRO_BATCH_SIZE_PER_GPU=8 \
  HP_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
  HP_PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
  HP_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
  HP_EDITOR_PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
  HP_EDITOR_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
  HP_ACTOR_PARAM_OFFLOAD=True \
  HP_ACTOR_OPTIMIZER_OFFLOAD=True \
  HP_REF_PARAM_OFFLOAD=True \
  HP_EDITOR_ACTOR_PARAM_OFFLOAD="${editor_actor_param_offload}" \
  HP_EDITOR_ACTOR_OPTIMIZER_OFFLOAD=True \
  HP_EDITOR_REF_PARAM_OFFLOAD=True \
  HP_ROLLOUT_MAX_NUM_SEQS=1024 \
  HP_ROLLOUT_MAX_BATCHED_TOKENS=196608 \
  HP_ROLLOUT_GPU_MEM_UTIL=0.88 \
  HP_ORCHESTRA_VLLM_MAX_ACTIVE=1024 \
  HP_SUBAGENT_VLLM_MAX_ACTIVE=1024 \
  HP_JUDGE_VLLM_MAX_ACTIVE=512 \
  bash "${BASE_LAUNCHER}"
