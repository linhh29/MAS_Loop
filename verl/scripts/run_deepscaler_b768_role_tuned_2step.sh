#!/usr/bin/env bash
set -euo pipefail

# Role-aware two-step benchmark.  Sub-agent had only ~30% KV occupancy at
# active=3584, while Editor reached 100% and preempted.  Increase only the
# former and lower the latter; keep batch=768 for an attributable A/B result.
ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || {
  echo "Refusing to run outside ${ROOT}" >&2; exit 2;
}
TS=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${DS_RUN_ROOT:-/path/to/checkpoints/benchmarks/deepscaler_b768_role_tuned_s2_${TS}}

exec env \
  DS_TIMESTAMP="${TS}" DS_RUN_ROOT="${RUN_ROOT}" \
  BENCH_NUM_STEPS=2 BENCH_ROLLOUT_N=8 \
  BENCH_ROLLOUT_MAX_NUM_SEQS=6144 \
  BENCH_PEER_MAX_NUM_SEQS=6400 \
  BENCH_SUB_AGENT_MAX_NUM_SEQS=6144 \
  BENCH_MAX_CONCURRENCY=4352 BENCH_DISPATCH_WORKERS=544 \
  BENCH_ORCHESTRA_VLLM_MAX_ACTIVE=2560 \
  BENCH_EDITOR_VLLM_MAX_ACTIVE=3328 \
  BENCH_SUB_AGENT_VLLM_MAX_ACTIVE=4352 \
  BENCH_ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 \
  BENCH_PEER_MAX_NUM_BATCHED_TOKENS=131072 \
  BENCH_SUB_AGENT_MAX_NUM_BATCHED_TOKENS=98304 \
  BENCH_ROLLOUT_GPU_MEM_UTIL=0.85 \
  BENCH_PEER_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_SUB_AGENT_GPU_MEMORY_UTILIZATION=0.90 \
  BENCH_RUNTIME_MONITOR=1 BENCH_RUNTIME_MONITOR_INTERVAL=5 \
  GPU_ISOLATION_CHECK_INTERVAL=5 GPU_ISOLATION_STABLE_CHECKS=2 \
  bash "${ROOT}/verl/scripts/run_deepscaler_vllm_concurrency_trial.sh" 768 "$@"
