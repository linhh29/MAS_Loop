#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ulimit -n 1048576 2>/dev/null || true

ROOT=/path/to/MAS_Loop
[[ "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" == "${ROOT}" ]] || exit 2
BATCH=${1:-128}
[[ "${BATCH}" =~ ^[0-9]+$ ]] && (( BATCH > 0 && BATCH % 8 == 0 )) || { echo "batch must be a positive multiple of 8" >&2; exit 2; }
ROLLOUT_N=${BENCH_ROLLOUT_N:-8}
NUM_STEPS=${BENCH_NUM_STEPS:-1}
RUN_EDITOR=${BENCH_RUN_EDITOR:-1}
EXPECTED_SAMPLES=$((BATCH * NUM_STEPS))
TRAJECTORIES=$((BATCH * ROLLOUT_N))
TS=${BCP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${BCP_SESSION:-masloop_bcp_b${BATCH}_${TS}}
RUN_ROOT=${BCP_RUN_ROOT:-/path/to/checkpoints/benchmarks/bcp_concurrency_b${BATCH}_${TS}}
LOG=${BCP_LAUNCH_LOG:-${RUN_ROOT}/trial_launcher.log}
STATUS_LOG=${BCP_STATUS_LOG:-/tmp/${SESSION}.status.log}
CONTROL_DIR=${BCP_CONTROL_DIR:-/tmp/mas_loop_bcp_control_${TS}}
mkdir -p "${RUN_ROOT}" "/path/to/checkpoints"

TMP="/path/to/checkpoints/benchmarks/current_mas_loop_job.env.tmp.$$"
cat > "${TMP}" <<MON
RUN_DIR=${RUN_ROOT}/train
RUN_NAME=$(basename "${RUN_ROOT}")
GPU_SET=8,9,10,11,12,13,14,15
LAUNCH_SCRIPT=${ROOT}/verl/scripts/run_bcp_vllm_concurrency_trial.sh
START_TIME=$(date +%s)
SESSION=${SESSION}
DATASET=bcp
EXPECTED_SAMPLES=${EXPECTED_SAMPLES}
PAIR_GROUP_SIZE=${ROLLOUT_N}
MODE=concurrency_trial
NUM_STEPS=${NUM_STEPS}
WAIT_LOG=${STATUS_LOG}
VLLM_SLEEP_STATE=${CONTROL_DIR}/state.json
VLLM_SLEEP_EVENTS=${CONTROL_DIR}/events.jsonl
MON
mv -f "${TMP}" "/path/to/checkpoints/benchmarks/current_mas_loop_job.env"

cat > "${RUN_ROOT}/trial_config.env" <<CFG
BATCH=${BATCH}
NUM_STEPS=${NUM_STEPS}
RUN_EDITOR=${RUN_EDITOR}
EXPECTED_SAMPLES=${EXPECTED_SAMPLES}
ROLLOUT_N=${ROLLOUT_N}
TRAJECTORIES=${TRAJECTORIES}
SESSION=${SESSION}
RUN_ROOT=${RUN_ROOT}
ROLLOUT_MAX_NUM_SEQS=${BENCH_ROLLOUT_MAX_NUM_SEQS:-512}
PEER_MAX_NUM_SEQS=${BENCH_PEER_MAX_NUM_SEQS:-768}
SUB_AGENT_MAX_NUM_SEQS=${BENCH_SUB_AGENT_MAX_NUM_SEQS:-1024}
MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
ORCHESTRA_VLLM_MAX_ACTIVE=${BENCH_ORCHESTRA_VLLM_MAX_ACTIVE:-768}
EDITOR_VLLM_MAX_ACTIVE=${BENCH_EDITOR_VLLM_MAX_ACTIVE:-768}
SUB_AGENT_VLLM_MAX_ACTIVE=${BENCH_SUB_AGENT_VLLM_MAX_ACTIVE:-1024}
CFG

echo "[BCP trial] batch=${BATCH} trajectories=${TRAJECTORIES} session=${SESSION} run_root=${RUN_ROOT}" | tee "${STATUS_LOG}"
echo "[BCP trial] real BM25 search_rounds=5; waiting for exclusive GPUs 8-15" | tee -a "${STATUS_LOG}"
PYTHONPATH="${ROOT}/verl" python3 -m mas_loop_rl.wait_for_gpu_isolation \
  --gpus 8,9,10,11,12,13,14,15 --interval "${GPU_ISOLATION_CHECK_INTERVAL:-5}" \
  --stable-checks "${GPU_ISOLATION_STABLE_CHECKS:-2}" | tee -a "${STATUS_LOG}"

set +e
env BCP_TIMESTAMP="${TS}" BCP_SESSION="${SESSION}" BCP_RUN_ROOT="${RUN_ROOT}" BCP_CONTROL_DIR="${CONTROL_DIR}" \
  BENCH_TRAIN_BATCH_SIZE="${BATCH}" BENCH_NUM_STEPS="${NUM_STEPS}" BENCH_RUN_EDITOR="${RUN_EDITOR}" \
  bash "${ROOT}/verl/scripts/run_bcp_vllm_concurrency_1step.sh" >"${LOG}" 2>&1 &
TRAIN_PID=$!

PYTHONPATH="${ROOT}/verl" python3 -m mas_loop_rl.monitor_vllm_metrics \
  --launcher-log "${LOG}" --output "${RUN_ROOT}/train/logs/orchestra_vllm_metrics.log" \
  --pid "${TRAIN_PID}" --interval "${ORCHESTRA_METRICS_INTERVAL:-2}" &
MONITOR_PID=$!

RUNTIME_MONITOR_PID=
if [[ "${BENCH_RUNTIME_MONITOR:-1}" == "1" ]]; then
  MAS_LOOP_VLLM_CONTROL_SOCKET="${CONTROL_DIR}/coordinator.sock" \
  PYTHONPATH="${ROOT}/verl" python3 -m mas_loop_rl.monitor_runtime_state \
    --output "${RUN_ROOT}/runtime_metrics.jsonl" --pid "${TRAIN_PID}" \
    --interval "${BENCH_RUNTIME_MONITOR_INTERVAL:-5}" --gpus 8,9,10,11,12,13,14,15 \
    --editor-url http://127.0.0.1:8511 --subagent-url http://127.0.0.1:8512 \
    --dispatch-dir "${CONTROL_DIR}/dispatch_metrics" --ray-lb-dir "${CONTROL_DIR}/ray_lb_metrics" &
  RUNTIME_MONITOR_PID=$!
fi
wait "${TRAIN_PID}"; RC=$?
wait "${MONITOR_PID}" || true
[[ -z "${RUNTIME_MONITOR_PID}" ]] || wait "${RUNTIME_MONITOR_PID}" || true
set -e
printf 'EXIT_CODE=%s\nFINISHED_AT=%s\n' "${RC}" "$(date '+%F %T %Z')" > "${RUN_ROOT}/trial_exit.status"
if [[ -d "${RUN_ROOT}/train" ]]; then
  PYTHONPATH="${ROOT}/verl" python3 -m mas_loop_rl.analyze_vllm_concurrency_run \
    "${RUN_ROOT}" --output "${RUN_ROOT}/concurrency_summary.json" || true
fi
exit "${RC}"
