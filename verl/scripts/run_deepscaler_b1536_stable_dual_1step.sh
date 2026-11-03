#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ulimit -n 1048576 2>/dev/null || true

# Coarse batch-only probe against the b1024 baseline:
#   task batch 1024 -> 1536 (+50%), while keeping the validated admission caps.
# This isolates optimizer/barrier amortization from Ray actor-concurrency crashes.
ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || exit 2
TS=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${DS_RUN_ROOT:-/path/to/checkpoints/benchmarks/deepscaler_b1536_o4352_e2816_s6400_dual_s1_${TS}}
exec env \
  DS_TIMESTAMP="${TS}" DS_RUN_ROOT="${RUN_ROOT}" \
  BENCH_NUM_STEPS=1 BENCH_ROLLOUT_N=8 BENCH_RUN_EDITOR=1 \
  BENCH_EDITOR_TRAIN_BATCH_SIZE=12288 \
  BENCH_PPO_MINI_BATCH_SIZE=256 \
  BENCH_ROLLOUT_MAX_NUM_SEQS=8192 \
  BENCH_PEER_MAX_NUM_SEQS=8192 \
  BENCH_SUB_AGENT_MAX_NUM_SEQS=8192 \
  BENCH_MAX_CONCURRENCY=6912 BENCH_DISPATCH_WORKERS=864 \
  BENCH_ORCHESTRA_VLLM_MAX_ACTIVE=4352 \
  BENCH_EDITOR_VLLM_MAX_ACTIVE=2816 \
  BENCH_SUB_AGENT_VLLM_MAX_ACTIVE=6400 \
  BENCH_ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 \
  BENCH_PEER_MAX_NUM_BATCHED_TOKENS=131072 \
  BENCH_SUB_AGENT_MAX_NUM_BATCHED_TOKENS=98304 \
  BENCH_ROLLOUT_GPU_MEM_UTIL=0.85 \
  BENCH_PEER_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_SUB_AGENT_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_RUNTIME_MONITOR=1 BENCH_RUNTIME_MONITOR_INTERVAL=5 \
  GPU_ISOLATION_CHECK_INTERVAL=5 GPU_ISOLATION_STABLE_CHECKS=2 \
  bash "${ROOT}/verl/scripts/run_deepscaler_vllm_concurrency_trial.sh" 1536 "$@"
