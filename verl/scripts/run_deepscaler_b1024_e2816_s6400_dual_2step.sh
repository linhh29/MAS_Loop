#!/usr/bin/env bash
set -euo pipefail

# Two-step, dual-role DeepScaleR benchmark selected from measured KV pressure:
#   Orchestra active=4352: low KV (~9% peak), already fast; keep stable because
#     a direct 6144 jump previously hit Ray out-of-order actor scheduling.
#   Editor active=2816: reduce from 3072 because its 65.17 GiB/GPU KV pool was
#     at 99.9% p95 / 100% peak with 582 preemptions; lower admission should
#     reduce recomputation and can improve throughput despite lower concurrency.
#   Sub-agent active=6400: a conservative +4.2% from the stable 6144 point;
#     previous run had 95.93% p95 / 99% peak KV but zero waiting/preemption.
# Each task produces rollout.n=8 independent one-to-one O_k/E_k pairs.
# Orchestra: 1024 tasks x 8 samples per optimizer step.
# Editor replay: 8192 rows per optimizer step, so both roles run exactly 2 steps.
ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || {
  echo "Refusing to run outside ${ROOT}" >&2; exit 2;
}
TS=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${DS_RUN_ROOT:-/path/to/checkpoints/benchmarks/deepscaler_b1024_e2816_s6400_dual_s2_${TS}}

exec env \
  DS_TIMESTAMP="${TS}" DS_RUN_ROOT="${RUN_ROOT}" \
  BENCH_NUM_STEPS=2 BENCH_ROLLOUT_N=8 BENCH_RUN_EDITOR=1 \
  BENCH_EDITOR_TRAIN_BATCH_SIZE=8192 \
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
  bash "${ROOT}/verl/scripts/run_deepscaler_vllm_concurrency_trial.sh" 1024 "$@"
