#!/usr/bin/env bash
set -euo pipefail

# Reproducible two-optimizer-step benchmark for MAS-Loop optimizations 1-3.
# It is intentionally DeepScaleR-only, full-parameter (lora_rank=0 downstream),
# and hard-pinned to this project's GPUs 8-15 and local /tmp models.
ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || {
  echo "Refusing to run outside ${ROOT}" >&2; exit 2;
}
TS=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${DS_RUN_ROOT:-/path/to/checkpoints/benchmarks/deepscaler_phase123_2step_${TS}}
exec env \
  DS_MAS_ROOT="${ROOT}" DS_VERL_ROOT="${ROOT}/verl" DS_ALL_GPUS=8,9,10,11,12,13,14,15 \
  DS_TIMESTAMP="${TS}" DS_RUN_ROOT="${RUN_ROOT}" DS_RUN_DIR="${RUN_ROOT}/train" \
  DS_TRAIN_BATCH_SIZE=64 DS_PPO_MINI_BATCH_SIZE=64 DS_PPO_MICRO_BATCH_SIZE_PER_GPU=2 \
  DS_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4 DS_ROLLOUT_N=8 DS_EDITOR_GROUP_SIZE=8 \
  DS_MAX_PROMPT_LENGTH=15000 DS_MAX_RESPONSE_LENGTH=8192 DS_ROLLOUT_MAX_MODEL_LEN=23192 \
  DS_ROLLOUT_TP=8 DS_ROLLOUT_GPU_MEM_UTIL=0.90 DS_PEER_GPU_MEMORY_UTILIZATION=0.90 \
  DS_SUB_AGENT_GPU_MEMORY_UTILIZATION=0.90 DS_ROLLOUT_MAX_NUM_SEQS=64 DS_PEER_MAX_NUM_SEQS=64 \
  DS_SUB_AGENT_MAX_NUM_SEQS=64 DS_SUB_AGENT_MAX_MODEL_LEN=8192 DS_MAX_CONCURRENCY=192 \
  DS_MAX_TRAIN_ROWS=128 DS_DATA_VAL_RATIO=0 DS_DATA_PAD_TO_MULTIPLE=64 DS_TOTAL_EPOCHS=1 \
  DS_RUN_ORCHESTRA=1 DS_RUN_EDITOR=0 DS_SAVE_FREQ=-1 \
  DS_PHASE_LEASE_GRACE_SEC="${DS_PHASE_LEASE_GRACE_SEC:-0.75}" \
  DS_DISPATCH_WORKERS_PER_PROCESS="${DS_DISPATCH_WORKERS_PER_PROCESS:-24}" \
  bash "${ROOT}/verl/scripts/run_deepscaler_only_supervised.sh" "$@"
