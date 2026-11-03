#!/usr/bin/env bash
set -euo pipefail

# Two-complete-step DeepScaleR benchmark after the b1024 role audit.
# Measured b1024 peaks at active O/E/S=4352/3200/4480:
#   Orchestra: 8.96% KV, no waiting
#   Editor:   100.0% KV, waiting and preemption pressure
#   Sub-agent: 41.3% KV, no waiting/preemption
# Increase only roles with measured KV headroom. Keep dispatch wider than the
# 6144 role cap because synchronous sessions can occupy workers while waiting
# for the next model phase; equal dispatch/role caps can starve the current role.
ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || {
  echo "Refusing to run outside ${ROOT}" >&2; exit 2;
}
TS=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${DS_RUN_ROOT:-/path/to/checkpoints/benchmarks/deepscaler_b1024_c6144_s2_${TS}}

exec env \
  DS_TIMESTAMP="${TS}" DS_RUN_ROOT="${RUN_ROOT}" \
  BENCH_NUM_STEPS=2 BENCH_ROLLOUT_N=8 \
  BENCH_PPO_MINI_BATCH_SIZE=256 \
  BENCH_ROLLOUT_MAX_NUM_SEQS=8192 \
  BENCH_PEER_MAX_NUM_SEQS=8192 \
  BENCH_SUB_AGENT_MAX_NUM_SEQS=8192 \
  BENCH_MAX_CONCURRENCY=6656 BENCH_DISPATCH_WORKERS=832 \
  BENCH_ORCHESTRA_VLLM_MAX_ACTIVE=6144 \
  BENCH_EDITOR_VLLM_MAX_ACTIVE=3072 \
  BENCH_SUB_AGENT_VLLM_MAX_ACTIVE=6144 \
  BENCH_ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 \
  BENCH_PEER_MAX_NUM_BATCHED_TOKENS=131072 \
  BENCH_SUB_AGENT_MAX_NUM_BATCHED_TOKENS=98304 \
  BENCH_ROLLOUT_GPU_MEM_UTIL=0.85 \
  BENCH_PEER_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_SUB_AGENT_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_RUNTIME_MONITOR=1 BENCH_RUNTIME_MONITOR_INTERVAL=5 \
  GPU_ISOLATION_CHECK_INTERVAL=5 GPU_ISOLATION_STABLE_CHECKS=2 \
  bash "${ROOT}/verl/scripts/run_deepscaler_vllm_concurrency_trial.sh" 1024 "$@"
