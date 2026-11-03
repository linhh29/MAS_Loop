#!/usr/bin/env bash
set -euo pipefail

# Role-aware b896 two-step throughput benchmark.
# Evidence from the preceding b768 run:
#   Orchestra: 4.63% KV peak at active=2560 -> safely raise to 4352.
#   Editor:    99.99% KV peak and 83 preemptions at active=3328 -> lower to 3200.
#   Sub-agent: 89.39% KV peak, no waiting/preemption at active=4352 -> cautiously test 4480.
# Increasing train batch from 768 to 896 reduces a 40,315-example epoch from
# 53 to 45 optimizer steps without changing rollout_n=8 or sequence limits.
ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || {
  echo "Refusing to run outside ${ROOT}" >&2; exit 2;
}
TS=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${DS_RUN_ROOT:-/path/to/checkpoints/benchmarks/deepscaler_b896_role_tuned_s2_${TS}}

exec env \
  DS_TIMESTAMP="${TS}" DS_RUN_ROOT="${RUN_ROOT}" \
  BENCH_NUM_STEPS=2 BENCH_ROLLOUT_N=8 \
  BENCH_ROLLOUT_MAX_NUM_SEQS=7168 \
  BENCH_PEER_MAX_NUM_SEQS=7424 \
  BENCH_SUB_AGENT_MAX_NUM_SEQS=7168 \
  BENCH_MAX_CONCURRENCY=4480 BENCH_DISPATCH_WORKERS=560 \
  BENCH_ORCHESTRA_VLLM_MAX_ACTIVE=4352 \
  BENCH_EDITOR_VLLM_MAX_ACTIVE=3200 \
  BENCH_SUB_AGENT_VLLM_MAX_ACTIVE=4480 \
  BENCH_ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 \
  BENCH_PEER_MAX_NUM_BATCHED_TOKENS=131072 \
  BENCH_SUB_AGENT_MAX_NUM_BATCHED_TOKENS=98304 \
  BENCH_ROLLOUT_GPU_MEM_UTIL=0.85 \
  BENCH_PEER_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_SUB_AGENT_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_RUNTIME_MONITOR=1 BENCH_RUNTIME_MONITOR_INTERVAL=5 \
  GPU_ISOLATION_CHECK_INTERVAL=5 GPU_ISOLATION_STABLE_CHECKS=2 \
  bash "${ROOT}/verl/scripts/run_deepscaler_vllm_concurrency_trial.sh" 896 "$@"
