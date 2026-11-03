#!/usr/bin/env bash
set -euo pipefail

# Two-complete-step benchmark for the next DeepScaleR throughput point.
# The preceding b768 run proved that dispatch and phase admission are truly
# concurrent (4352 live sub-agent sessions) and that role switching occurs in
# four bulk phases without wake/sleep thrashing.  This trial therefore targets
# fewer optimizer barriers while keeping the risky KV-heavy roles bounded:
#   - batch 1024 -> 8192 independent GRPO trajectories per step;
#   - Editor stays at 3200 (b768 reached 100% KV and preempted at 3328);
#   - Sub-agent stays at 4480 for the batch-size comparison (~92-93% projected
#     KV); a separate 4608 trial is only warranted if this run has no preemption.
ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || {
  echo "Refusing to run outside ${ROOT}" >&2; exit 2;
}
TS=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${DS_RUN_ROOT:-/path/to/checkpoints/benchmarks/deepscaler_b1024_role_tuned_s2_${TS}}

exec env \
  DS_TIMESTAMP="${TS}" DS_RUN_ROOT="${RUN_ROOT}" \
  BENCH_NUM_STEPS=2 BENCH_ROLLOUT_N=8 \
  BENCH_PPO_MINI_BATCH_SIZE=256 \
  BENCH_ROLLOUT_MAX_NUM_SEQS=8192 \
  BENCH_PEER_MAX_NUM_SEQS=8192 \
  BENCH_SUB_AGENT_MAX_NUM_SEQS=8192 \
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
  bash "${ROOT}/verl/scripts/run_deepscaler_vllm_concurrency_trial.sh" 1024 "$@"
